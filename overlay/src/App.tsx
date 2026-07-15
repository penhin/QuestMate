import { FormEvent, useEffect, useRef, useState } from "react";
import {
  DEFAULT_AI_MODEL,
  DEFAULT_DEEPSEEK_MODEL,
  type AiSettings,
  checkBackend,
  deleteSession as deleteQuestSession,
  getAiSettings,
  getSession,
  listSessions,
  renameSession,
  setAiSettings,
  streamQuestMate,
  type ChatResponse,
  type GameCandidate,
  type SessionSummary,
} from "./api";
import {
  getActiveGame,
  getOverlayPlacement,
  listProcesses,
  setOverlayLayout,
  setOverlayPlacement,
  type OverlayMode,
  type OverlayPlacement,
} from "./tauri";
import { installStartupUpdate } from "./updater";
import { COPY, GameField, Icon, SETTINGS_TABS, displayProcessName, formatCandidateHost, formatMessageCount, isOverlayProcess, providerLabel, updateLastAssistantMessage, type Language, type Message, type SettingsTab } from "./ui";

export default function App() {
  const [language, setLanguage] = useState<Language>("zh");
  const [mode, setMode] = useState<OverlayMode>("bubble");
  const [placement, setPlacement] = useState<OverlayPlacement>(() => getOverlayPlacement());
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settingsTab, setSettingsTab] = useState<SettingsTab>("preferences");
  const [backendOnline, setBackendOnline] = useState(false);
  const [processes, setProcesses] = useState<string[]>([]);
  const [game, setGame] = useState("");
  const [question, setQuestion] = useState("");
  const [sessionId, setSessionId] = useState<string>();
  const [draftSession, setDraftSession] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [editingSessionId, setEditingSessionId] = useState<string>();
  const [editingTitle, setEditingTitle] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [aiSettings, setLocalAiSettings] = useState<AiSettings>(() => getAiSettings());
  const [apiSaved, setApiSaved] = useState(false);
  const [apiError, setApiError] = useState("");
  const [shakePanel, setShakePanel] = useState(false);
  const [providerOpen, setProviderOpen] = useState(false);
  const [gameOpen, setGameOpen] = useState(false);
  const manualGameOverrideRef = useRef(false);
  const text = COPY[language];

  useEffect(() => {
    void setOverlayLayout("bubble", placement);
    void checkBackend().then(setBackendOnline);
    void installStartupUpdate();
    void refreshActiveGame();
    void refreshProcesses();
    void refreshSessions();

    const activeGameTimer = window.setInterval(() => {
      void refreshActiveGame();
      void refreshProcesses();
    }, 5000);

    return () => window.clearInterval(activeGameTimer);
  }, []);

  useEffect(() => {
    if (settingsOpen && settingsTab === "session") {
      void refreshSessions();
    }
  }, [settingsOpen, settingsTab]);

  async function refreshActiveGame() {
    const value = await getActiveGame();
    if (value.processName && !manualGameOverrideRef.current && !isOverlayProcess(value.processName)) {
      setGame(value.detectedGame || displayProcessName(value.processName));
    }
  }

  async function refreshProcesses() {
    setProcesses((await listProcesses()).filter((process) => !isOverlayProcess(process)));
  }

  function changeGame(value: string) {
    manualGameOverrideRef.current = Boolean(value.trim());
    setGame(value);
  }

  async function switchMode(nextMode: OverlayMode) {
    setSettingsOpen(false);
    setMode(nextMode);
    await setOverlayLayout(nextMode, placement);
  }

  async function changeDisplayMode(nextMode: Extract<OverlayMode, "popover" | "drawer">) {
    setMode(nextMode);
    await setOverlayLayout(nextMode, placement);
  }

  async function changePlacement(nextPlacement: OverlayPlacement) {
    setPlacement(nextPlacement);
    setOverlayPlacement(nextPlacement);
    await setOverlayLayout(mode, nextPlacement);
  }

  function clearSession() {
    setMessages([]);
    setSessionId(undefined);
    setError("");
    setDraftSession(true);
    setSettingsTab("session");
  }

  async function refreshSessions() {
    try {
      const nextSessions = await listSessions();
      setSessions(nextSessions);
    } catch {
      setSessions([]);
    }
  }

  async function openSession(nextSessionId: string) {
    try {
      const session = await getSession(nextSessionId);
      setSessionId(session.session_id);
      setDraftSession(false);
      setMessages(
        session.messages
          .filter((message) => message.role === "user" || message.role === "assistant")
          .map((message) => ({
            role: message.role,
            content: message.content,
            sources: message.sources,
          })),
      );
      setError("");
    } catch (err) {
      const message = err instanceof Error ? err.message : text.requestFailed;
      setError(message);
    }
  }

  async function saveSessionTitle(targetSessionId: string) {
    const title = editingTitle.trim();
    if (!title) {
      return;
    }

    try {
      const updated = await renameSession(targetSessionId, title);
      setSessions((current) =>
        current.map((session) => (session.session_id === targetSessionId ? updated : session)),
      );
      setEditingSessionId(undefined);
      setEditingTitle("");
    } catch (err) {
      const message = err instanceof Error ? err.message : text.requestFailed;
      setError(message);
    }
  }

  function startSessionEdit(targetSessionId: string, title: string) {
    setEditingSessionId(targetSessionId);
    setEditingTitle(title);
  }

  function cancelSessionEdit() {
    setEditingSessionId(undefined);
    setEditingTitle("");
  }

  async function removeSession(targetSessionId: string) {
    try {
      await deleteQuestSession(targetSessionId);
      setSessions((current) => current.filter((session) => session.session_id !== targetSessionId));
      if (sessionId === targetSessionId) {
        setMessages([]);
        setSessionId(undefined);
        setError("");
        setDraftSession(false);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : text.requestFailed;
      setError(message);
    }
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

  function changeProvider(provider: AiSettings["provider"]) {
    setLocalAiSettings((current) => ({
      ...current,
      provider,
      model:
        current.model === DEFAULT_AI_MODEL || current.model === DEFAULT_DEEPSEEK_MODEL
          ? provider === "deepseek"
            ? DEFAULT_DEEPSEEK_MODEL
            : DEFAULT_AI_MODEL
          : current.model,
      baseUrl:
        current.baseUrl === "" ||
        current.baseUrl === "https://api.anthropic.com" ||
        current.baseUrl === "https://api.deepseek.com"
          ? provider === "deepseek"
            ? "https://api.deepseek.com"
            : ""
          : current.baseUrl,
    }));
    setApiSaved(false);
    setApiError("");
    setProviderOpen(false);
  }

  async function submit(event?: FormEvent) {
    event?.preventDefault();
    await submitQuestion(question, game);
  }

  async function submitQuestion(
    rawQuestion: string,
    rawGame: string,
    metadata: Record<string, unknown> = {},
    options: { appendUserMessage?: boolean } = {},
  ) {
    const trimmedGame = rawGame.trim();
    const trimmedQuestion = rawQuestion.trim();

    if (!trimmedGame || !trimmedQuestion || loading) {
      return;
    }

    const appendUserMessage = options.appendUserMessage ?? true;
    setError("");
    setLoading(true);
    setMessages((current) => {
      const next = appendUserMessage ? [...current, { role: "user" as const, content: trimmedQuestion }] : current;
      const pendingMessage: Message = {
        role: "assistant",
        content: "",
        status: true,
        progress: ["准备查询"],
      };
      if (!appendUserMessage && next[next.length - 1]?.role === "assistant") {
        return [...next.slice(0, -1), pendingMessage];
      }
      return [...next, pendingMessage];
    });
    setQuestion("");

    try {
      let streamedAnswer = "";
      let finalResponse: ChatResponse | undefined;
      await streamQuestMate(
        {
          game: trimmedGame,
          question: trimmedQuestion,
          sessionId,
          aiSettings,
          metadata,
        },
        {
          onStatus: (status) => {
            setMessages((current) => {
              const lastAssistant = current[current.length - 1];
              const progress = lastAssistant?.role === "assistant" ? [...(lastAssistant.progress ?? [])] : [];
              if (progress[progress.length - 1] !== status) {
                progress.push(status);
              }
              return updateLastAssistantMessage(current, { content: "", progress, status: true });
            });
          },
          onChunk: (chunk) => {
            streamedAnswer += chunk;
          },
          onDone: (response) => {
            finalResponse = response;
          },
        },
      );
      const response = finalResponse;
      if (!response) {
        throw new Error(text.requestFailed);
      }
      setSessionId(response.session_id);
      setSessions((current) => {
        const existing = current.find((session) => session.session_id === response.session_id);
        const summary = {
          session_id: response.session_id,
          title: response.title || existing?.title || trimmedQuestion.slice(0, 28),
          message_count: (existing?.message_count ?? 0) + 2,
          updated_at: new Date().toISOString(),
        };
        const withoutCurrent = current.filter((session) => session.session_id !== response.session_id);
        return [summary, ...withoutCurrent];
      });
      setDraftSession(false);
      setMessages((current) => {
        return updateLastAssistantMessage(current, {
          content: response.answer || streamedAnswer,
          sources: response.sources,
          status: false,
          progress: undefined,
          gameCandidates: response.game_candidates,
          pendingQuestion: response.needs_game_confirmation ? trimmedQuestion : undefined,
        });
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : text.requestFailed;
      setError(message);
      setMessages((current) =>
        updateLastAssistantMessage(current, {
          content: text.requestFailed,
          status: false,
          progress: undefined,
        }),
      );
    } finally {
      setLoading(false);
    }
  }

  function confirmGameCandidate(candidate: GameCandidate, pendingQuestion?: string) {
    const nextGame = candidate.name;
    setGame(nextGame);
    manualGameOverrideRef.current = true;
    setMessages((current) =>
      updateLastAssistantMessage(current, {
        content: `已确认游戏：${candidate.name}`,
        gameCandidates: [],
        pendingQuestion: undefined,
        status: true,
        progress: [`已确认游戏：${candidate.name}`],
      }),
    );
    void submitQuestion(
      pendingQuestion || question,
      nextGame,
      {
        confirmed_game: true,
        game_aliases: [candidate.name, ...candidate.aliases],
        database_domains: candidate.database_domains,
      },
      { appendUserMessage: false },
    );
  }

  function rejectGameCandidates(pendingQuestion?: string) {
    setQuestion(pendingQuestion || question);
    setMessages((current) =>
      updateLastAssistantMessage(current, {
        content: "没有匹配到正确游戏。请补充 Steam/itch.io 链接、英文名或开发商后再查。",
        gameCandidates: [],
        pendingQuestion: undefined,
        status: false,
      }),
    );
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
                <div className="section-heading placement-heading">
                  <h3>{text.windowPosition}</h3>
                </div>
                <div className="segmented-control placement-control" role="group" aria-label={text.windowPosition}>
                  <button
                    type="button"
                    className={placement === "bottom-right" ? "active" : ""}
                    onClick={() => void changePlacement("bottom-right")}
                    aria-pressed={placement === "bottom-right"}
                  >
                    {text.bottomRight}
                  </button>
                  <button
                    type="button"
                    className={placement === "bottom-left" ? "active" : ""}
                    onClick={() => void changePlacement("bottom-left")}
                    aria-pressed={placement === "bottom-left"}
                  >
                    {text.bottomLeft}
                  </button>
                  <button
                    type="button"
                    className={placement === "center" ? "active" : ""}
                    onClick={() => void changePlacement("center")}
                    aria-pressed={placement === "center"}
                  >
                    {text.center}
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
                <div className="field-stack provider-field">
                  <span>{text.aiProvider}</span>
                  <div
                    className="custom-select"
                    onBlur={(event) => {
                      if (!event.currentTarget.contains(event.relatedTarget)) {
                        setProviderOpen(false);
                      }
                    }}
                  >
                    <button
                      type="button"
                      className={providerOpen ? "custom-select-trigger open" : "custom-select-trigger"}
                      onClick={() => setProviderOpen((open) => !open)}
                      aria-haspopup="listbox"
                      aria-expanded={providerOpen}
                    >
                      <span>{providerLabel(aiSettings.provider)}</span>
                      <Icon name="chevron" />
                    </button>
                    {providerOpen && (
                      <div className="custom-select-menu" role="listbox">
                        {(["anthropic", "deepseek"] as const).map((provider) => (
                          <button
                            key={provider}
                            type="button"
                            className={aiSettings.provider === provider ? "selected" : ""}
                            onClick={() => changeProvider(provider)}
                            role="option"
                            aria-selected={aiSettings.provider === provider}
                          >
                            <span>{providerLabel(provider)}</span>
                            {aiSettings.provider === provider && <span className="select-check">✓</span>}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
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
                    placeholder={aiSettings.provider === "deepseek" ? "sk-..." : "sk-ant-..."}
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
                    placeholder={aiSettings.provider === "deepseek" ? DEFAULT_DEEPSEEK_MODEL : DEFAULT_AI_MODEL}
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
                    placeholder={
                      aiSettings.provider === "deepseek" ? "https://api.deepseek.com" : "https://api.anthropic.com"
                    }
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
                </div>
                <div className="session-panel">
                  <div className="session-list">
                    <button
                      type="button"
                      className="session-row session-new-row"
                      onClick={clearSession}
                      aria-label={text.newSession}
                    >
                      <Icon name="plus" />
                    </button>
                    {draftSession && (
                      <div className="session-row session-draft active">
                        <button type="button" className="session-open" onClick={clearSession}>
                          <strong>{text.newSession}</strong>
                          <span>{formatMessageCount(0, language)}</span>
                        </button>
                      </div>
                    )}
                    {sessions.map((session) => (
                      <div
                        key={session.session_id}
                        className={session.session_id === sessionId ? "session-row active" : "session-row"}
                        onBlur={(event) => {
                          if (editingSessionId !== session.session_id) {
                            return;
                          }
                          const nextTarget = event.relatedTarget;
                          if (!(nextTarget instanceof Node) || !event.currentTarget.contains(nextTarget)) {
                            cancelSessionEdit();
                          }
                        }}
                      >
                        {editingSessionId === session.session_id ? (
                          <>
                            <input
                              className="session-title-input"
                              value={editingTitle}
                              onChange={(event) => setEditingTitle(event.target.value)}
                              onKeyDown={(event) => {
                                if (event.key === "Enter") {
                                  event.preventDefault();
                                  void saveSessionTitle(session.session_id);
                                }
                                if (event.key === "Escape") {
                                  cancelSessionEdit();
                                }
                              }}
                              autoFocus
                            />
                            <div className="session-actions">
                              <button type="button" onClick={() => void saveSessionTitle(session.session_id)}>
                                {text.saveSession}
                              </button>
                              <button type="button" onClick={cancelSessionEdit}>
                                {text.cancelEdit}
                              </button>
                            </div>
                          </>
                        ) : (
                          <>
                            <button
                              type="button"
                              className="session-open"
                              onClick={() => void openSession(session.session_id)}
                            >
                              <strong>{session.title}</strong>
                              <span>{formatMessageCount(session.message_count, language)}</span>
                            </button>
                            <div className="session-actions">
                              <button
                                type="button"
                                onClick={() => startSessionEdit(session.session_id, session.title)}
                              >
                                {text.editSession}
                              </button>
                              <button type="button" onClick={() => void removeSession(session.session_id)}>
                                {text.deleteSession}
                              </button>
                            </div>
                          </>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              </section>
            )}

          </div>
        </section>
      ) : (
        <>
          <section className="workspace-bar">
            <GameField
              game={game}
              label={text.game}
              placeholder={text.gamePlaceholder}
              processes={processes}
              onGameChange={changeGame}
              open={gameOpen}
              onOpenChange={setGameOpen}
            />
          </section>

          <section className="messages" aria-live="polite">
            {messages.length > 0 &&
              messages.map((message, index) => (
                <article
                  key={`${message.role}-${index}`}
                  className={`message ${message.role}${message.status ? " status" : ""}`}
                >
                  {message.status && message.progress && message.progress.length > 0 ? (
                    <section className="search-progress" role="status" aria-live="polite" aria-atomic="true">
                      <div className="search-progress-heading">
                        <span className="progress-spinner" aria-hidden="true" />
                        <h2>{text.searchProgress}</h2>
                      </div>
                      <ol>
                        {message.progress.map((step, stepIndex) => {
                          const isCurrent = stepIndex === message.progress!.length - 1;
                          return (
                            <li key={`${step}-${stepIndex}`} className={isCurrent ? "current" : "complete"}>
                              <span className="progress-marker" aria-hidden="true" />
                              <span>{step}</span>
                            </li>
                          );
                        })}
                      </ol>
                    </section>
                  ) : (
                    <p>{message.content}</p>
                  )}
                  {message.gameCandidates && message.gameCandidates.length > 0 && (
                    <div className="game-candidates" aria-label={text.chooseGame}>
                      <span>{text.chooseGame}</span>
                      <div className="candidate-list">
                        {message.gameCandidates.map((candidate) => (
                          <button
                            type="button"
                            key={`${candidate.name}-${candidate.platform_urls[0] ?? ""}`}
                            className="candidate-card"
                            onClick={() => confirmGameCandidate(candidate, message.pendingQuestion)}
                          >
                            <strong>{candidate.name}</strong>
                            {candidate.tags.length > 0 && (
                              <small>{candidate.tags.join(" / ")}</small>
                            )}
                            {candidate.platform_urls[0] && <em>{formatCandidateHost(candidate.platform_urls[0])}</em>}
                          </button>
                        ))}
                        <button
                          type="button"
                          className="candidate-card none"
                          onClick={() => rejectGameCandidates(message.pendingQuestion)}
                        >
                          <strong>{text.noneOfThese}</strong>
                        </button>
                      </div>
                    </div>
                  )}
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
