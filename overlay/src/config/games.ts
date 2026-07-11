export const GAME_PROCESS_MAP: Record<string, string> = {
  "eldenring.exe": "Elden Ring",
  "eldenringnightreign.exe": "Elden Ring Nightreign",
  "blackmythwukong.exe": "Black Myth: Wukong",
  "sekiro.exe": "Sekiro: Shadows Die Twice",
  "monsterhunterwilds.exe": "Monster Hunter Wilds",
  "monsterhunterworld.exe": "Monster Hunter: World",
  "re4.exe": "Resident Evil 4",
  "cyberpunk2077.exe": "Cyberpunk 2077",
  "baldursgate3.exe": "Baldur's Gate 3",
  "genshinimpact.exe": "Genshin Impact",
  "starrail.exe": "Honkai: Star Rail",
};

export function detectGameFromProcess(processName?: string | null): string {
  if (!processName) {
    return "";
  }

  return GAME_PROCESS_MAP[processName.toLowerCase()] ?? "";
}
