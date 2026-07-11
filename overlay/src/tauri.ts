import { invoke } from "@tauri-apps/api/core";
import { detectGameFromProcess } from "./config/games";

export type OverlayMode = "bubble" | "popover" | "drawer";

export type ActiveGame = {
  processName: string | null;
  windowTitle: string | null;
  detectedGame: string | null;
};

export async function getActiveGame(): Promise<ActiveGame> {
  try {
    const activeGame = await invoke<ActiveGame>("get_active_game");
    return {
      ...activeGame,
      detectedGame: activeGame.detectedGame || detectGameFromProcess(activeGame.processName),
    };
  } catch {
    return {
      processName: null,
      windowTitle: null,
      detectedGame: null,
    };
  }
}

export async function setOverlayMode(mode: OverlayMode): Promise<void> {
  try {
    await invoke("set_overlay_mode", { mode });
  } catch {
    // Browser-only Vite development cannot call Tauri commands.
  }
}
