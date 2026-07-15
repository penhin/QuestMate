use serde::Serialize;
use tauri::{LogicalSize, Manager, PhysicalPosition, Position, Size};

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ActiveGame {
    process_name: Option<String>,
    window_title: Option<String>,
    detected_game: Option<String>,
}

#[tauri::command]
fn get_active_game() -> ActiveGame {
    platform::get_active_game()
}

#[tauri::command]
fn list_processes() -> Vec<String> {
    platform::list_processes()
}

#[tauri::command]
fn set_overlay_layout(
    app: tauri::AppHandle,
    mode: String,
    placement: String,
) -> Result<(), String> {
    let window = app
        .get_webview_window("main")
        .ok_or_else(|| "main window not found".to_string())?;

    let (width, height) = match mode.as_str() {
        "bubble" => (96.0, 96.0),
        "popover" => (400.0, 640.0),
        "drawer" => (460.0, 900.0),
        _ => return Err(format!("unsupported overlay mode: {mode}")),
    };

    window
        .set_size(Size::Logical(LogicalSize { width, height }))
        .map_err(|err| err.to_string())?;
    position_overlay(&window, &placement)?;
    window
        .set_always_on_top(true)
        .map_err(|err| err.to_string())?;
    // Re-apply this after resizing as Windows can recreate the native surface
    // while switching between the bubble and panel layouts.
    #[cfg(target_os = "windows")]
    window.set_shadow(false).map_err(|err| err.to_string())?;
    window.show().map_err(|err| err.to_string())?;

    Ok(())
}

fn position_overlay(window: &tauri::WebviewWindow, placement: &str) -> Result<(), String> {
    let monitor = window
        .current_monitor()
        .map_err(|err| err.to_string())?
        .or(window.primary_monitor().map_err(|err| err.to_string())?)
        .ok_or_else(|| "no monitor available".to_string())?;
    let work_area = monitor.work_area();
    let window_size = window.outer_size().map_err(|err| err.to_string())?;
    let horizontal_space = work_area.size.width.saturating_sub(window_size.width);
    let vertical_space = work_area.size.height.saturating_sub(window_size.height);
    let inset = 24;

    let (offset_x, offset_y) = match placement {
        "bottom-right" => (
            horizontal_space.saturating_sub(inset),
            vertical_space.saturating_sub(inset),
        ),
        "bottom-left" => (
            inset.min(horizontal_space),
            vertical_space.saturating_sub(inset),
        ),
        "center" => (horizontal_space / 2, vertical_space / 2),
        _ => return Err(format!("unsupported overlay placement: {placement}")),
    };

    window
        .set_position(Position::Physical(PhysicalPosition::new(
            work_area.position.x + offset_x as i32,
            work_area.position.y + offset_y as i32,
        )))
        .map_err(|err| err.to_string())
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .invoke_handler(tauri::generate_handler![
            get_active_game,
            list_processes,
            set_overlay_layout
        ])
        .run(tauri::generate_context!())
        .expect("error while running QuestMate overlay");
}

#[cfg(target_os = "windows")]
mod platform {
    use super::ActiveGame;
    use serde::Deserialize;
    use std::collections::{BTreeSet, HashMap};
    use std::ffi::OsString;
    use std::os::windows::ffi::OsStringExt;
    use std::sync::OnceLock;
    use windows_sys::Win32::Foundation::{CloseHandle, BOOL, HWND, LPARAM};
    use windows_sys::Win32::Graphics::Dwm::{DwmGetWindowAttribute, DWMWA_CLOAKED};
    use windows_sys::Win32::System::Threading::{
        OpenProcess, QueryFullProcessImageNameW, PROCESS_QUERY_LIMITED_INFORMATION,
    };
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        EnumChildWindows, EnumWindows, GetForegroundWindow, GetWindow, GetWindowLongW,
        GetWindowTextLengthW, GetWindowTextW, GetWindowThreadProcessId, IsWindowVisible,
        GWL_EXSTYLE, GW_OWNER, WS_EX_APPWINDOW, WS_EX_TOOLWINDOW,
    };

    pub fn get_active_game() -> ActiveGame {
        unsafe {
            let hwnd = GetForegroundWindow();
            if hwnd.is_null() {
                return empty();
            }

            let mut process_id = 0;
            GetWindowThreadProcessId(hwnd, &mut process_id);

            let window_title = read_window_title(hwnd);
            let process_name = read_process_name(process_id);
            let detected_game = process_name.as_deref().and_then(detect_game);

            ActiveGame {
                process_name,
                window_title,
                detected_game: detected_game.map(str::to_string),
            }
        }
    }

    pub fn list_processes() -> Vec<String> {
        unsafe {
            let mut processes = BTreeSet::new();
            EnumWindows(
                Some(collect_taskbar_process),
                &mut processes as *mut BTreeSet<String> as LPARAM,
            );
            processes.into_iter().collect()
        }
    }

    unsafe extern "system" fn collect_taskbar_process(hwnd: HWND, lparam: LPARAM) -> BOOL {
        if !is_taskbar_window(hwnd) {
            return 1;
        }

        if let Some(process_name) = read_window_process_name(hwnd) {
            let name = strip_exe_suffix(&process_name);
            if !is_overlay_process(&name) {
                (*(lparam as *mut BTreeSet<String>)).insert(name);
            }
        }
        1
    }

    unsafe fn is_taskbar_window(hwnd: HWND) -> bool {
        if IsWindowVisible(hwnd) == 0 {
            return false;
        }

        let mut cloaked = 0u32;
        if DwmGetWindowAttribute(
            hwnd,
            DWMWA_CLOAKED as u32,
            &mut cloaked as *mut u32 as *mut _,
            std::mem::size_of::<u32>() as u32,
        ) >= 0
            && cloaked != 0
        {
            return false;
        }

        let extended_style = GetWindowLongW(hwnd, GWL_EXSTYLE) as u32;
        let is_app_window = extended_style & WS_EX_APPWINDOW != 0;
        let is_tool_window = extended_style & WS_EX_TOOLWINDOW != 0;

        // Windows uses this owner/style combination to decide whether a
        // top-level window receives its own taskbar button.
        (!is_tool_window && GetWindow(hwnd, GW_OWNER).is_null()) || is_app_window
    }

    unsafe fn read_window_process_name(hwnd: HWND) -> Option<String> {
        let mut process_id = 0;
        GetWindowThreadProcessId(hwnd, &mut process_id);
        let process_name = read_process_name(process_id)?;

        if !process_name.eq_ignore_ascii_case("ApplicationFrameHost.exe") {
            return Some(process_name);
        }

        // UWP apps are hosted by ApplicationFrameHost. Their actual process is
        // attached as a child window, so prefer it when available.
        let mut context = ChildProcessContext {
            host_process_id: process_id,
            process_name: None,
        };
        EnumChildWindows(
            hwnd,
            Some(find_uwp_child_process),
            &mut context as *mut ChildProcessContext as LPARAM,
        );
        context.process_name.or(Some(process_name))
    }

    struct ChildProcessContext {
        host_process_id: u32,
        process_name: Option<String>,
    }

    unsafe extern "system" fn find_uwp_child_process(hwnd: HWND, lparam: LPARAM) -> BOOL {
        let context = &mut *(lparam as *mut ChildProcessContext);
        let mut process_id = 0;
        GetWindowThreadProcessId(hwnd, &mut process_id);

        if process_id == 0 || process_id == context.host_process_id {
            return 1;
        }

        if let Some(process_name) = read_process_name(process_id) {
            context.process_name = Some(process_name);
            return 0;
        }
        1
    }

    fn is_overlay_process(name: &str) -> bool {
        name.eq_ignore_ascii_case("questmate-overlay")
    }

    fn strip_exe_suffix(value: &str) -> String {
        value
            .strip_suffix(".exe")
            .or_else(|| value.strip_suffix(".EXE"))
            .unwrap_or(value)
            .to_string()
    }

    unsafe fn read_window_title(hwnd: HWND) -> Option<String> {
        let len = GetWindowTextLengthW(hwnd);
        if len <= 0 {
            return None;
        }

        let mut buffer = vec![0u16; (len + 1) as usize];
        let copied = GetWindowTextW(hwnd, buffer.as_mut_ptr(), buffer.len() as i32);
        if copied <= 0 {
            return None;
        }

        Some(
            OsString::from_wide(&buffer[..copied as usize])
                .to_string_lossy()
                .to_string(),
        )
    }

    unsafe fn read_process_name(process_id: u32) -> Option<String> {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, process_id);
        if handle.is_null() {
            return None;
        }

        let mut buffer = vec![0u16; 32_768];
        let mut copied = buffer.len() as u32;
        let succeeded = QueryFullProcessImageNameW(handle, 0, buffer.as_mut_ptr(), &mut copied);
        let _ = CloseHandle(handle);

        if succeeded == 0 || copied == 0 {
            return None;
        }

        let path = OsString::from_wide(&buffer[..copied as usize])
            .to_string_lossy()
            .to_string();
        path.rsplit(['\\', '/']).next().map(str::to_string)
    }

    #[derive(Deserialize)]
    struct GameRegistry {
        games: Vec<GameRegistryEntry>,
    }

    #[derive(Deserialize)]
    struct GameRegistryEntry {
        name: String,
        processes: Vec<String>,
    }

    fn game_process_map() -> &'static HashMap<String, String> {
        static PROCESS_MAP: OnceLock<HashMap<String, String>> = OnceLock::new();
        PROCESS_MAP.get_or_init(|| {
            let registry: GameRegistry =
                serde_json::from_str(include_str!("../../src/config/games.json"))
                    .expect("embedded game registry must be valid JSON");
            registry
                .games
                .into_iter()
                .flat_map(|game| {
                    game.processes
                        .into_iter()
                        .map(move |process| (process.to_ascii_lowercase(), game.name.clone()))
                })
                .collect()
        })
    }

    fn detect_game(process_name: &str) -> Option<&'static str> {
        game_process_map()
            .get(&process_name.to_ascii_lowercase())
            .map(String::as_str)
    }

    fn empty() -> ActiveGame {
        ActiveGame {
            process_name: None,
            window_title: None,
            detected_game: None,
        }
    }
}

#[cfg(not(target_os = "windows"))]
mod platform {
    use super::ActiveGame;

    pub fn get_active_game() -> ActiveGame {
        ActiveGame {
            process_name: None,
            window_title: None,
            detected_game: None,
        }
    }

    pub fn list_processes() -> Vec<String> {
        Vec::new()
    }
}
