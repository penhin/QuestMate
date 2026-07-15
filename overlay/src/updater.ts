import { check } from "@tauri-apps/plugin-updater";
import { relaunch } from "@tauri-apps/plugin-process";

/**
 * Checks once during desktop-app startup. Browser Vite sessions and missing
 * release metadata are intentionally treated as no-update conditions.
 */
export async function installStartupUpdate(): Promise<void> {
  try {
    const update = await check();
    if (!update) {
      return;
    }
    await update.downloadAndInstall();
    await relaunch();
  } catch {
    // Updates must never prevent the overlay from opening.
  }
}
