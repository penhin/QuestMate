use serde::Serialize;
use tauri::{LogicalSize, Manager, Size};

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
fn set_overlay_mode(app: tauri::AppHandle, mode: String) -> Result<(), String> {
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
    window
        .set_always_on_top(true)
        .map_err(|err| err.to_string())?;
    window.show().map_err(|err| err.to_string())?;

    Ok(())
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .invoke_handler(tauri::generate_handler![get_active_game, set_overlay_mode])
        .run(tauri::generate_context!())
        .expect("error while running QuestMate overlay");
}

#[cfg(target_os = "windows")]
mod platform {
    use super::ActiveGame;
    use std::ffi::OsString;
    use std::os::windows::ffi::OsStringExt;
    use windows_sys::Win32::Foundation::{CloseHandle, MAX_PATH};
    use windows_sys::Win32::System::ProcessStatus::K32GetModuleBaseNameW;
    use windows_sys::Win32::System::Threading::{OpenProcess, PROCESS_QUERY_INFORMATION, PROCESS_VM_READ};
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        GetForegroundWindow, GetWindowTextLengthW, GetWindowTextW, GetWindowThreadProcessId,
    };

    pub fn get_active_game() -> ActiveGame {
        unsafe {
            let hwnd = GetForegroundWindow();
            if hwnd == 0 {
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

    unsafe fn read_window_title(hwnd: isize) -> Option<String> {
        let len = GetWindowTextLengthW(hwnd);
        if len <= 0 {
            return None;
        }

        let mut buffer = vec![0u16; (len + 1) as usize];
        let copied = GetWindowTextW(hwnd, buffer.as_mut_ptr(), buffer.len() as i32);
        if copied <= 0 {
            return None;
        }

        Some(OsString::from_wide(&buffer[..copied as usize]).to_string_lossy().to_string())
    }

    unsafe fn read_process_name(process_id: u32) -> Option<String> {
        let handle = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, 0, process_id);
        if handle == 0 {
            return None;
        }

        let mut buffer = vec![0u16; MAX_PATH as usize];
        let copied = K32GetModuleBaseNameW(handle, 0, buffer.as_mut_ptr(), buffer.len() as u32);
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
}
