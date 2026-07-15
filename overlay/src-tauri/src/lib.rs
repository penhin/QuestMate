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
    use std::collections::BTreeSet;
    use std::ffi::OsString;
    use std::os::windows::ffi::OsStringExt;
    use windows_sys::Win32::Foundation::{CloseHandle, HWND, MAX_PATH};
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{
        CreateToolhelp32Snapshot, Process32FirstW, Process32NextW, PROCESSENTRY32W,
        TH32CS_SNAPPROCESS,
    };
    use windows_sys::Win32::System::ProcessStatus::K32GetModuleBaseNameW;
    use windows_sys::Win32::System::Threading::{
        OpenProcess, PROCESS_QUERY_INFORMATION, PROCESS_VM_READ,
    };
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        GetForegroundWindow, GetWindowTextLengthW, GetWindowTextW, GetWindowThreadProcessId,
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
            let snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
            if snapshot == windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE {
                return Vec::new();
            }

            let mut entry = PROCESSENTRY32W {
                dwSize: std::mem::size_of::<PROCESSENTRY32W>() as u32,
                cntUsage: 0,
                th32ProcessID: 0,
                th32DefaultHeapID: 0,
                th32ModuleID: 0,
                cntThreads: 0,
                th32ParentProcessID: 0,
                pcPriClassBase: 0,
                dwFlags: 0,
                szExeFile: [0; 260],
            };
            let mut processes = BTreeSet::new();
            if Process32FirstW(snapshot, &mut entry) != 0 {
                loop {
                    let end = entry
                        .szExeFile
                        .iter()
                        .position(|value| *value == 0)
                        .unwrap_or(entry.szExeFile.len());
                    let name = String::from_utf16_lossy(&entry.szExeFile[..end]);
                    if !name.is_empty() {
                        processes.insert(strip_exe_suffix(&name));
                    }
                    if Process32NextW(snapshot, &mut entry) == 0 {
                        break;
                    }
                }
            }
            let _ = CloseHandle(snapshot);
            processes.into_iter().collect()
        }
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
        let handle = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, 0, process_id);
        if handle.is_null() {
            return None;
        }

        let mut buffer = vec![0u16; MAX_PATH as usize];
        let copied = K32GetModuleBaseNameW(
            handle,
            std::ptr::null_mut(),
            buffer.as_mut_ptr(),
            buffer.len() as u32,
        );
        let _ = CloseHandle(handle);

        if copied == 0 {
            return None;
        }

        Some(
            OsString::from_wide(&buffer[..copied as usize])
                .to_string_lossy()
                .to_string(),
        )
    }

    fn detect_game(process_name: &str) -> Option<&'static str> {
        match process_name.to_ascii_lowercase().as_str() {
            "eldenring.exe" => Some("Elden Ring"),
            "eldenringnightreign.exe" => Some("Elden Ring Nightreign"),
            "blackmythwukong.exe" => Some("Black Myth: Wukong"),
            "sekiro.exe" => Some("Sekiro: Shadows Die Twice"),
            "monsterhunterwilds.exe" => Some("Monster Hunter Wilds"),
            "monsterhunterworld.exe" => Some("Monster Hunter: World"),
            "re4.exe" => Some("Resident Evil 4"),
            "cyberpunk2077.exe" => Some("Cyberpunk 2077"),
            "baldursgate3.exe" => Some("Baldur's Gate 3"),
            "genshinimpact.exe" => Some("Genshin Impact"),
            "starrail.exe" => Some("Honkai: Star Rail"),
            _ => None,
        }
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
