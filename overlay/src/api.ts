export type Source = {
  title: string;
  url: string;
  snippet?: string | null;
  score?: number | null;
};

export type ChatResponse = {
  session_id: string;
  answer: string;
  sources: Source[];
};

export type AiSettings = {
  provider: "anthropic";
  apiKey: string;
  model: string;
  baseUrl: string;
};

const API_BASE_URL_STORAGE_KEY = "questmate.apiBaseUrl";
export const DEFAULT_API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";
const AI_SETTINGS_STORAGE_KEY = "questmate.aiSettings";
export const DEFAULT_AI_MODEL = "claude-sonnet-4-5";

export function normalizeApiBaseUrl(value: string): string {
  return value.trim().replace(/\/+$/, "");
}

export function getApiBaseUrl(): string {
  if (typeof window === "undefined") {
    return DEFAULT_API_BASE_URL;
  }

  return normalizeApiBaseUrl(localStorage.getItem(API_BASE_URL_STORAGE_KEY) ?? DEFAULT_API_BASE_URL);
}

export function setApiBaseUrl(value: string): string {
  const normalized = normalizeApiBaseUrl(value) || DEFAULT_API_BASE_URL;
  localStorage.setItem(API_BASE_URL_STORAGE_KEY, normalized);
  return normalized;
}

export function getAiSettings(): AiSettings {
  if (typeof window === "undefined") {
    return { provider: "anthropic", apiKey: "", model: DEFAULT_AI_MODEL, baseUrl: "" };
  }

  try {
    const stored = JSON.parse(localStorage.getItem(AI_SETTINGS_STORAGE_KEY) ?? "{}") as Partial<AiSettings>;
    return {
      provider: "anthropic",
      apiKey: stored.apiKey ?? "",
      model: stored.model ?? DEFAULT_AI_MODEL,
      baseUrl: stored.baseUrl ?? "",
    };
  } catch {
    return { provider: "anthropic", apiKey: "", model: DEFAULT_AI_MODEL, baseUrl: "" };
  }
}

export function setAiSettings(settings: AiSettings): AiSettings {
  const normalized = {
    provider: "anthropic" as const,
    apiKey: settings.apiKey.trim(),
    model: settings.model.trim() || DEFAULT_AI_MODEL,
    baseUrl: normalizeApiBaseUrl(settings.baseUrl),
  };
  localStorage.setItem(AI_SETTINGS_STORAGE_KEY, JSON.stringify(normalized));
  return normalized;
}

export async function askQuestMate(input: {
  game: string;
  question: string;
  sessionId?: string;
  aiSettings?: AiSettings;
}): Promise<ChatResponse> {
  const response = await fetch(`${getApiBaseUrl()}/api/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      game: input.game,
      question: input.question,
      session_id: input.sessionId,
      stream: false,
      ai_provider: input.aiSettings?.provider ?? "anthropic",
      ai_api_key: input.aiSettings?.apiKey || undefined,
      ai_model: input.aiSettings?.model || undefined,
      ai_base_url: input.aiSettings?.baseUrl || undefined,
    }),
  });

  if (!response.ok) {
    throw new Error(`QuestMate API request failed with status ${response.status}`);
  }

  return response.json() as Promise<ChatResponse>;
}

export async function checkBackend(): Promise<boolean> {
  try {
    const response = await fetch(`${getApiBaseUrl()}/health`);
    return response.ok;
  } catch {
    return false;
  }
}
