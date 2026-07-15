import registry from "./games.json";

type GameRegistryEntry = {
  name: string;
  processes: string[];
};

const games = registry.games satisfies GameRegistryEntry[];

const GAME_PROCESS_MAP: Record<string, string> = Object.fromEntries(
  games.flatMap((game) =>
    game.processes.map((process) => [process.toLowerCase(), game.name] as const),
  ),
);

export function detectGameFromProcess(processName?: string | null): string {
  if (!processName) {
    return "";
  }

  return GAME_PROCESS_MAP[processName.toLowerCase()] ?? "";
}
