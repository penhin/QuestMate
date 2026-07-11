import { FormEvent, useEffect, useMemo, useState } from "react";
import {
  DEFAULT_AI_MODEL,
  type AiSettings,
  askQuestMate,
  checkBackend,
  getAiSettings,
  setAiSettings,
  type ChatResponse,
} from "./api";
import { getActiveGame, setOverlayMode, type ActiveGame, type OverlayMode } from "./tauri";

type Message = {
  role: "user" | "assistant";
  content: string;
  sources?: ChatResponse["sources"];
};

type Language = "zh" | "en";
type SettingsTab = "preferences" | "api" | "session" | "process";

type Copy = {
  open: string;
  title: string;
  subtitle: string;
  settings: string;
  closeSettings: string;
  preferences: string;
  displayMode: string;
  compactMode: string;
  drawerMode: string;
  minimize: string;
  apiOnline: string;
  apiOffline: string;
  apiSettings: string;
  aiProvider: string;
  aiApiKey: string;
  aiModel: string;
  aiBaseUrl: string;
  required: string;
  apiRequiredError: string;
  saveApi: string;
  resetApi: string;
  apiSaved: string;
  notDetected: string;
  detectGame: string;
  sessionManagement: string;
  clearSession: string;
  sessionEmpty: string;
  processDetection: string;
  activeWindow: string;
  language: string;
  game: string;
  question: string;
  gamePlaceholder: string;
  questionPlaceholder: string;
  loading: string;
  ask: string;
  sources: string;
  requestFailed: string;
};

const COPY = {
  zh: {
    open: "打开 QuestMate",
    title: "攻略速查",
    subtitle: "游戏中即时问答",
    settings: "设置",
    closeSettings: "关闭设置",
    preferences: "偏好",
    displayMode: "显示模式",
    compactMode: "小弹窗",
    drawerMode: "右侧抽屉",
    minimize: "最小化到悬浮球",
    apiOnline: "API 已连接",
    apiOffline: "API 离线",
    apiSettings: "模型",
    aiProvider: "服务商",
    aiApiKey: "API Key",
    aiModel: "模型",
    aiBaseUrl: "Base URL",
    required: "必填",
    apiRequiredError: "请填写必填项",
    saveApi: "保存",
    resetApi: "清空",
    apiSaved: "已保存",
    notDetected: "未识别",
    detectGame: "识别",
    sessionManagement: "会话",
    clearSession: "清空当前会话",
    sessionEmpty: "暂无会话",
    processDetection: "进程",
    activeWindow: "当前窗口",
    language: "语言",
    game: "游戏",
    question: "问题",
    gamePlaceholder: "艾尔登法环",
    questionPlaceholder: "这个 Boss 怎么打？",
    loading: "查询中...",
    ask: "提问",
    sources: "来源",
    requestFailed: "请求失败",
  },
  en: {
    open: "Open QuestMate",
    title: "Quick Guide",
    subtitle: "In-game answers on demand",
    settings: "Settings",
    closeSettings: "Close settings",
    preferences: "Preferences",
    displayMode: "Display mode",
    compactMode: "Popover",
    drawerMode: "Drawer",
    minimize: "Minimize to bubble",
    apiOnline: "API Online",
    apiOffline: "API Offline",
    apiSettings: "Model",
    aiProvider: "Provider",
    aiApiKey: "API Key",
    aiModel: "Model",
    aiBaseUrl: "Base URL",
    required: "Required",
    apiRequiredError: "Fill in the required fields",
    saveApi: "Save",
    resetApi: "Clear",
    apiSaved: "Saved",
    notDetected: "Not detected",
    detectGame: "Detect",
    sessionManagement: "Session",
    clearSession: "Clear session",
    sessionEmpty: "No session yet",
    processDetection: "Process",
    activeWindow: "Active window",
    language: "Language",
    game: "Game",
    question: "Question",
    gamePlaceholder: "Elden Ring",
    questionPlaceholder: "How do I beat this boss?",
    loading: "Searching...",
    ask: "Ask",
    sources: "Sources",
    requestFailed: "Request failed",
  },
} satisfies Record<Language, Copy>;

type IconName = "settings" | "history" | "activity" | "minimize" | "api";

const SETTINGS_TABS: Array<{ id: SettingsTab; label: keyof Copy; icon: IconName }> = [
  { id: "preferences", label: "preferences", icon: "settings" },
  { id: "api", label: "apiSettings", icon: "api" },
  { id: "session", label: "sessionManagement", icon: "history" },
  { id: "process", label: "processDetection", icon: "activity" },
];

export default function App() {
  const [language, setLanguage] = useState<Language>("zh");
  const [mode, setMode] = useState<OverlayMode>("bubble");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settingsTab, setSettingsTab] = useState<SettingsTab>("preferences");
  const [backendOnline, setBackendOnline] = useState(false);
  const [activeGame, setActiveGame] = useState<ActiveGame | null>(null);
  const [game, setGame] = useState("");
  const [question, setQuestion] = useState("");
  const [sessionId, setSessionId] = useState<string>();
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [aiSettings, setLocalAiSettings] = useState<AiSettings>(() => getAiSettings());
  const [apiSaved, setApiSaved] = useState(false);
  const [apiError, setApiError] = useState("");
  const [shakePanel, setShakePanel] = useState(false);
  const text = COPY[language];

  const detectedLabel = useMemo(() => {
    if (activeGame?.detectedGame) {
      return activeGame.detectedGame;
    }
    if (activeGame?.processName) {
      return activeGame.processName;
    }
    return text.notDetected;
  }, [activeGame, text.notDetected]);

  useEffect(() => {
    void checkBackend().then(setBackendOnline);
    void refreshActiveGame();
  }, []);

  async function refreshActiveGame() {
    const value = await getActiveGame();
    setActiveGame(value);
    if (value.detectedGame) {
      setGame(value.detectedGame);
    }
  }

  async function switchMode(nextMode: OverlayMode) {
    setSettingsOpen(false);
    setMode(nextMode);
    await setOverlayMode(nextMode);
  }

  async function changeDisplayMode(nextMode: Extract<OverlayMode, "popover" | "drawer">) {
    setMode(nextMode);
    await setOverlayMode(nextMode);
  }

  function clearSession() {
    setMessages([]);
    setSessionId(undefined);
    setError("");
  }

  function saveApiSettings() {
    if (!aiSettings.apiKey.trim() || !aiSettings.model.trim()) {
      setApiError(text.apiRequiredError);
      setShakePanel(false);
      window.setTimeout(() => setShakePanel(true), 0);
      window.setTimeout(() => setShakePanel(false), 420);
      return;
    }

    const nextSettings = setAiSettings(aiSettings);
    setLocalAiSettings(nextSettings);
    setApiError("");
    setApiSaved(true);
    window.setTimeout(() => setApiSaved(false), 1600);
  }

  function resetApiSettings() {
    const nextSettings = setAiSettings({
      provider: "anthropic",
      apiKey: "",
      model: DEFAULT_AI_MODEL,
      baseUrl: "",
    });
    setLocalAiSettings(nextSettings);
    setApiError("");
    setApiSaved(true);
    window.setTimeout(() => setApiSaved(false), 1600);
  }

  function toggleSettings() {
    setSettingsOpen((open) => !open);
  }

  function openApiSettings() {
    setSettingsOpen(true);
    setSettingsTab("api");
  }

  async function submit(event?: FormEvent) {
    event?.preventDefault();
    const trimmedGame = game.trim();
    const trimmedQuestion = question.trim();

    if (!trimmedGame || !trimmedQuestion || loading) {
      return;
    }

    setError("");
    setLoading(true);
    setMessages((current) => [...current, { role: "user", content: trimmedQuestion }]);
    setQuestion("");

    try {
      const response = await askQuestMate({
        game: trimmedGame,
        question: trimmedQuestion,
        sessionId,
        aiSettings,
      });
      setSessionId(response.session_id);
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: response.answer,
          sources: response.sources,
        },
      ]);
    } catch (err) {
      const message = err instanceof Error ? err.message : text.requestFailed;
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  if (mode === "bubble") {
    return (
      <button className="bubble" onClick={() => void switchMode("popover")} aria-label={text.open}>
        <span className="bubble-mark">Q</span>
        <span className={`bubble-status ${backendOnline ? "online" : ""}`} />
      </button>
    );
  }

  return (
    <main className={`${mode === "drawer" ? "panel drawer" : "panel popover"} ${shakePanel ? "shake" : ""}`}>
      <header className="panel-header">
        <div className="brand-lockup">
          <span className="app-mark">Q</span>
          <div>
            <p>QuestMate</p>
            <h1>{text.title}</h1>
          </div>
        </div>

        <div className="header-status">
          <button
            type="button"
            className={`connection ${backendOnline ? "online" : ""}`}
            onClick={openApiSettings}
            aria-label={text.apiSettings}
            title={text.apiSettings}
          >
            {backendOnline ? text.apiOnline : text.apiOffline}
          </button>
          <button
            type="button"
            className={settingsOpen ? "icon-button active" : "icon-button"}
            onClick={toggleSettings}
            aria-label={settingsOpen ? text.closeSettings : text.settings}
            title={settingsOpen ? text.closeSettings : text.settings}
          >
            <Icon name="settings" />
          </button>
          <button
            type="button"
            className="icon-button"
            aria-label={text.minimize}
            title={text.minimize}
            onClick={() => void switchMode("bubble")}
          >
            <Icon name="minimize" />
          </button>
        </div>
      </header>

      {settingsOpen ? (
        <section className="settings-layout">
          <nav className="settings-menu" aria-label={text.settings}>
            {SETTINGS_TABS.map((tab) => (
              <button
                key={tab.id}
                type="button"
                className={settingsTab === tab.id ? "active" : ""}
                onClick={() => setSettingsTab(tab.id)}
                aria-current={settingsTab === tab.id ? "page" : undefined}
              >
                <span className="menu-icon">
                  <Icon name={tab.icon} />
                </span>
                <span>{text[tab.label]}</span>
              </button>
            ))}
          </nav>

          <div className="settings-detail">
            {settingsTab === "preferences" && (
              <section className="settings-section">
                <div className="section-heading">
                  <h2>{text.preferences}</h2>
                  <p>{text.displayMode}</p>
                </div>
                <div className="segmented-control">
                  <button
                    type="button"
                    className={mode === "popover" ? "active" : ""}
                    onClick={() => void changeDisplayMode("popover")}
                    aria-pressed={mode === "popover"}
                  >
                    {text.compactMode}
                  </button>
                  <button
                    type="button"
                    className={mode === "drawer" ? "active" : ""}
                    onClick={() => void changeDisplayMode("drawer")}
                    aria-pressed={mode === "drawer"}
                  >
                    {text.drawerMode}
                  </button>
                </div>
                <div className="setting-row">
                  <div>
                    <h3>{text.language}</h3>
                    <p>{language === "zh" ? "中文" : "English"}</p>
                  </div>
                  <label className="language-switch">
                    <span>中文</span>
                    <input
                      type="checkbox"
                      checked={language === "en"}
                      onChange={(event) => setLanguage(event.target.checked ? "en" : "zh")}
                    />
                    <i />
                    <span>EN</span>
                  </label>
                </div>
              </section>
            )}

            {settingsTab === "api" && (
              <section className="settings-section">
                <div className="section-heading">
                  <h2>{text.apiSettings}</h2>
                </div>
                <div className="readonly-field">
                  <span>{text.aiProvider}</span>
                  <strong>Anthropic</strong>
                </div>
                <label className="field-stack">
                  <span>
                    {text.aiApiKey} <em>({text.required})</em>
                  </span>
                  <input
                    className={apiError && !aiSettings.apiKey.trim() ? "invalid" : ""}
                    value={aiSettings.apiKey}
                    onChange={(event) => {
                      setLocalAiSettings((current) => ({ ...current, apiKey: event.target.value }));
                      setApiSaved(false);
                      setApiError("");
                    }}
                    placeholder="sk-ant-..."
                    type="password"
                    required
                    spellCheck={false}
                  />
                </label>
                <label className="field-stack">
                  <span>
                    {text.aiModel} <em>({text.required})</em>
                  </span>
                  <input
                    className={apiError && !aiSettings.model.trim() ? "invalid" : ""}
                    value={aiSettings.model}
                    onChange={(event) => {
                      setLocalAiSettings((current) => ({ ...current, model: event.target.value }));
                      setApiSaved(false);
                      setApiError("");
                    }}
                    placeholder={DEFAULT_AI_MODEL}
                    required
                    spellCheck={false}
                  />
                </label>
                <label className="field-stack">
                  <span>{text.aiBaseUrl}</span>
                  <input
                    value={aiSettings.baseUrl}
                    onChange={(event) => {
                      setLocalAiSettings((current) => ({ ...current, baseUrl: event.target.value }));
                      setApiSaved(false);
                    }}
                    placeholder="https://api.anthropic.com"
                    inputMode="url"
                    spellCheck={false}
                  />
                </label>
                {apiError && (
                  <p className="field-error" role="alert">
                    {apiError}
                  </p>
                )}
                <div className="action-row">
                  <button type="button" className="secondary-action" onClick={resetApiSettings}>
                    {text.resetApi}
                  </button>
                  <button type="button" className="primary-action" onClick={saveApiSettings}>
                    {apiSaved ? text.apiSaved : text.saveApi}
                  </button>
                </div>
              </section>
            )}

            {settingsTab === "session" && (
              <section className="settings-section">
                <div className="section-heading">
                  <h2>{text.sessionManagement}</h2>
                  <p>{sessionId ?? text.sessionEmpty}</p>
                </div>
                <button type="button" className="secondary-action" onClick={clearSession}>
                  {text.clearSession}
                </button>
              </section>
            )}

            {settingsTab === "process" && (
              <section className="settings-section">
                <div className="section-heading">
                  <h2>{text.processDetection}</h2>
                  <p title={activeGame?.windowTitle ?? ""}>
                    {text.activeWindow}: {detectedLabel}
                  </p>
                </div>
                <GameField
                  game={game}
                  label={text.game}
                  placeholder={text.gamePlaceholder}
                  detectLabel={text.detectGame}
                  windowTitle={activeGame?.windowTitle ?? ""}
                  onGameChange={setGame}
                  onDetect={() => void refreshActiveGame()}
                />
              </section>
            )}

          </div>
        </section>
      ) : (
        <>
          <section className="workspace-bar">
            <div className="game-meta">
              <span>{text.subtitle}</span>
              <strong title={activeGame?.windowTitle ?? ""}>{detectedLabel}</strong>
            </div>
            <GameField
              game={game}
              label={text.game}
              placeholder={text.gamePlaceholder}
              detectLabel={text.detectGame}
              windowTitle={activeGame?.windowTitle ?? ""}
              onGameChange={setGame}
              onDetect={() => void refreshActiveGame()}
            />
          </section>

          <section className="messages" aria-live="polite">
            {messages.length > 0 &&
              messages.map((message, index) => (
                <article key={`${message.role}-${index}`} className={`message ${message.role}`}>
                  <p>{message.content}</p>
                  {message.sources && message.sources.length > 0 && (
                    <div className="sources">
                      <span>{text.sources}</span>
                      <ul>
                        {message.sources.map((source) => (
                          <li key={source.url}>
                            <a href={source.url} target="_blank" rel="noreferrer">
                              {source.title}
                            </a>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </article>
              ))}
          </section>

          {error && (
            <p className="error" role="alert">
              {error}
            </p>
          )}

          <form className="ask-form" onSubmit={(event) => void submit(event)}>
            <label>
              <span>{text.question}</span>
              <textarea
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder={text.questionPlaceholder}
                rows={mode === "drawer" ? 4 : 3}
              />
            </label>
            <button
              className="submit"
              disabled={loading || !game.trim() || !question.trim()}
              aria-busy={loading}
            >
              {loading ? text.loading : text.ask}
            </button>
          </form>
        </>
      )}
    </main>
  );
}

function Icon({ name }: { name: IconName }) {
  const paths: Record<IconName, string[]> = {
    settings: [
      "M12 15.5A3.5 3.5 0 1 0 12 8a3.5 3.5 0 0 0 0 7.5Z",
      "M19.4 15a1.7 1.7 0 0 0 .34 1.87l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.7 1.7 0 0 0-1.87-.34 1.7 1.7 0 0 0-1.04 1.57V21a2 2 0 0 1-4 0v-.09A1.7 1.7 0 0 0 8.96 19.4a1.7 1.7 0 0 0-1.87.34l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.7 1.7 0 0 0 4.6 15a1.7 1.7 0 0 0-1.57-1H3a2 2 0 0 1 0-4h.09A1.7 1.7 0 0 0 4.6 8.96a1.7 1.7 0 0 0-.34-1.87l-.06-.06A2 2 0 1 1 7.03 4.2l.06.06A1.7 1.7 0 0 0 8.96 4.6 1.7 1.7 0 0 0 10 3.03V3a2 2 0 0 1 4 0v.09a1.7 1.7 0 0 0 1.04 1.51 1.7 1.7 0 0 0 1.87-.34l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.7 1.7 0 0 0-.34 1.87A1.7 1.7 0 0 0 20.97 10H21a2 2 0 0 1 0 4h-.09A1.7 1.7 0 0 0 19.4 15Z",
    ],
    history: ["M3 12a9 9 0 1 0 3-6.7", "M3 4v5h5", "M12 7v5l3 2"],
    activity: ["M22 12h-4l-3 8-6-16-3 8H2"],
    minimize: ["M6 12h12"],
    api: ["M4 12h4", "M16 12h4", "M9 7l-4 5 4 5", "M15 7l4 5-4 5", "M11 18l2-12"],
  };

  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" className="ui-icon">
      {paths[name].map((path) => (
        <path key={path} d={path} />
      ))}
    </svg>
  );
}

function GameField(props: {
  game: string;
  label: string;
  placeholder: string;
  detectLabel: string;
  windowTitle: string;
  onGameChange: (value: string) => void;
  onDetect: () => void;
}) {
  return (
    <div className="game-control">
      <label>
        <span>{props.label}</span>
        <input
          value={props.game}
          onChange={(event) => props.onGameChange(event.target.value)}
          placeholder={props.placeholder}
          title={props.windowTitle}
        />
      </label>
      <button type="button" onClick={props.onDetect}>
        {props.detectLabel}
      </button>
    </div>
  );
}
