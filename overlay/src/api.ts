export type Source = {
  title: string;
  url: string;
  snippet?: string | null;
  score?: number | null;
};

export type GameCandidate = {
  name: string;
  aliases: string[];
  tags: string[];
  platform_urls: string[];
  database_domains: string[];
  confidence: number;
};

export type ChatResponse = {
  session_id: string;
  answer: string;
  sources: Source[];
  title?: string | null;
  is_new: boolean;
  needs_game_confirmation?: boolean;
  game_candidates?: GameCandidate[];
};

export type SessionMessage = {
  role: "user" | "assistant";
  content: string;
  created_at: string;
  sources: Source[];
};

export type SessionSummary = {
  session_id: string;
  title: string;
  message_count: number;
  updated_at?: string | null;
};

export type SessionResponse = {
  session_id: string;
  messages: SessionMessage[];
};

export type AiSettings = {
  provider: "anthropic" | "deepseek";
  apiKey: string;
  model: string;
  baseUrl: string;
};

export const DEFAULT_API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";
const AI_SETTINGS_STORAGE_KEY = "questmate.aiSettings";
export const DEFAULT_AI_MODEL = "claude-sonnet-4-5";
export const DEFAULT_DEEPSEEK_MODEL = "deepseek-chat";

export function normalizeApiBaseUrl(value: string): string {
  return value.trim().replace(/\/+$/, "");
}

export function getApiBaseUrl(): string {
  if (typeof window !== "undefined") {
    localStorage.removeItem("questmate.apiBaseUrl");
  }

  return normalizeApiBaseUrl(DEFAULT_API_BASE_URL);
}

export function getAiSettings(): AiSettings {
  if (typeof window === "undefined") {
    return { provider: "anthropic", apiKey: "", model: DEFAULT_AI_MODEL, baseUrl: "" };
  }

  try {
    const stored = JSON.parse(localStorage.getItem(AI_SETTINGS_STORAGE_KEY) ?? "{}") as Partial<AiSettings>;
    return {
      provider: stored.provider === "deepseek" ? "deepseek" : "anthropic",
      apiKey: stored.apiKey ?? "",
      model: stored.model ?? (stored.provider === "deepseek" ? DEFAULT_DEEPSEEK_MODEL : DEFAULT_AI_MODEL),
      baseUrl: stored.baseUrl ?? "",
    };
  } catch {
    return { provider: "anthropic", apiKey: "", model: DEFAULT_AI_MODEL, baseUrl: "" };
  }
}

export function setAiSettings(settings: AiSettings): AiSettings {
  const normalized = {
    provider: settings.provider,
    apiKey: settings.apiKey.trim(),
    model: settings.model.trim() || (settings.provider === "deepseek" ? DEFAULT_DEEPSEEK_MODEL : DEFAULT_AI_MODEL),
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
  metadata?: Record<string, unknown>;
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
      metadata: input.metadata ?? {},
    }),
  });

  if (!response.ok) {
    throw new Error(`QuestMate API request failed with status ${response.status}`);
  }

  return response.json() as Promise<ChatResponse>;
}

export async function streamQuestMate(
  input: {
    game: string;
    question: string;
    sessionId?: string;
    aiSettings?: AiSettings;
    metadata?: Record<string, unknown>;
  },
  handlers: {
    onStatus?: (status: string) => void;
    onChunk: (chunk: string) => void;
    onDone: (response: ChatResponse) => void;
  },
): Promise<void> {
  const response = await fetch(`${getApiBaseUrl()}/api/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify({
      game: input.game,
      question: input.question,
      session_id: input.sessionId,
      stream: true,
      ai_provider: input.aiSettings?.provider ?? "anthropic",
      ai_api_key: input.aiSettings?.apiKey || undefined,
      ai_model: input.aiSettings?.model || undefined,
      ai_base_url: input.aiSettings?.baseUrl || undefined,
      metadata: input.metadata ?? {},
    }),
  });

  if (!response.ok || !response.body) {
    throw new Error(`QuestMate API request failed with status ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() ?? "";
    for (const eventBlock of events) {
      handleStreamEvent(eventBlock, handlers);
    }
  }

  if (buffer.trim()) {
    handleStreamEvent(buffer, handlers);
  }
}

function handleStreamEvent(
  eventBlock: string,
  handlers: {
    onStatus?: (status: string) => void;
    onChunk: (chunk: string) => void;
    onDone: (response: ChatResponse) => void;
  },
) {
  const event = eventBlock
    .split("\n")
    .find((line) => line.startsWith("event:"))
    ?.replace("event:", "")
    .trim();
  const data = eventBlock
    .split("\n")
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.replace("data:", "").trim())
    .join("\n");

  if (!event || !data) {
    return;
  }

  const payload = JSON.parse(data) as { value?: string; message?: string } | ChatResponse;
  if (event === "status" && "value" in payload && typeof payload.value === "string") {
    handlers.onStatus?.(payload.value);
  }
  if (event === "chunk" && "value" in payload && typeof payload.value === "string") {
    handlers.onChunk(payload.value);
  }
  if (event === "done") {
    handlers.onDone(payload as ChatResponse);
  }
  if (event === "error") {
    throw new Error("message" in payload && payload.message ? payload.message : "QuestMate stream failed");
  }
}

export async function checkBackend(): Promise<boolean> {
  try {
    const response = await fetch(`${getApiBaseUrl()}/health`);
    return response.ok;
  } catch {
    return false;
  }
}

export async function listSessions(): Promise<SessionSummary[]> {
  const response = await fetch(`${getApiBaseUrl()}/api/sessions`);

  if (!response.ok) {
    throw new Error(`QuestMate sessions request failed with status ${response.status}`);
  }

  const body = (await response.json()) as { sessions: SessionSummary[] };
  return body.sessions;
}

export async function getSession(sessionId: string): Promise<SessionResponse> {
  const response = await fetch(`${getApiBaseUrl()}/api/sessions/${sessionId}`);

  if (!response.ok) {
    throw new Error(`QuestMate session request failed with status ${response.status}`);
  }

  return response.json() as Promise<SessionResponse>;
}

export async function renameSession(sessionId: string, title: string): Promise<SessionSummary> {
  const response = await fetch(`${getApiBaseUrl()}/api/sessions/${sessionId}`, {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ title }),
  });

  if (!response.ok) {
    throw new Error(`QuestMate session rename failed with status ${response.status}`);
  }

  return response.json() as Promise<SessionSummary>;
}

export async function deleteSession(sessionId: string): Promise<void> {
  const response = await fetch(`${getApiBaseUrl()}/api/sessions/${sessionId}`, {
    method: "DELETE",
  });

  if (!response.ok) {
    throw new Error(`QuestMate session delete failed with status ${response.status}`);
  }
}
