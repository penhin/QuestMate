import { invoke } from "@tauri-apps/api/core";
import { detectGameFromProcess } from "./config/games";

export type OverlayMode = "bubble" | "popover" | "drawer";
export type OverlayPlacement = "bottom-right" | "bottom-left" | "center";

const OVERLAY_PLACEMENT_STORAGE_KEY = "questmate.overlayPlacement";

type ActiveGame = {
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

export async function listProcesses(): Promise<string[]> {
  try {
    return await invoke<string[]>("list_processes");
  } catch {
    return [];
  }
}

export function getOverlayPlacement(): OverlayPlacement {
  if (typeof window === "undefined") {
    return "bottom-right";
  }

  const stored = localStorage.getItem(OVERLAY_PLACEMENT_STORAGE_KEY);
  return stored === "bottom-left" || stored === "center" ? stored : "bottom-right";
}

export function setOverlayPlacement(placement: OverlayPlacement): void {
  localStorage.setItem(OVERLAY_PLACEMENT_STORAGE_KEY, placement);
}

export async function setOverlayLayout(mode: OverlayMode, placement: OverlayPlacement): Promise<void> {
  try {
    await invoke("set_overlay_layout", { mode, placement });
  } catch {
    // Browser-only Vite development cannot call Tauri commands.
  }
}
