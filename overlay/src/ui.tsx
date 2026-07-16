import type { AiSettings, ChatResponse, GameCandidate } from "./api";

export type Message = {
  role: "user" | "assistant";
  content: string;
  status?: boolean;
  sources?: ChatResponse["sources"];
  gameCandidates?: GameCandidate[];
  pendingQuestion?: string;
  progress?: string[];
};

export type Language = "zh" | "en";
export type SettingsTab = "preferences" | "api" | "session";

export type Copy = {
  open: string;
  title: string;
  subtitle: string;
  settings: string;
  closeSettings: string;
  preferences: string;
  displayMode: string;
  windowPosition: string;
  bottomRight: string;
  bottomLeft: string;
  center: string;
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
  sessionManagement: string;
  clearSession: string;
  newSession: string;
  editSession: string;
  deleteSession: string;
  saveSession: string;
  cancelEdit: string;
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
  chooseGame: string;
  noneOfThese: string;
  searchProgress: string;
};

export const COPY = {
  zh: {
    open: "打开 QuestMate", title: "攻略速查", subtitle: "游戏中即时问答", settings: "设置",
    closeSettings: "关闭设置", preferences: "偏好", displayMode: "显示模式", windowPosition: "窗口位置",
    bottomRight: "右下角", bottomLeft: "左下角", center: "屏幕中央", compactMode: "小弹窗",
    drawerMode: "右侧抽屉", minimize: "最小化到悬浮球", apiOnline: "API 已连接", apiOffline: "API 离线",
    apiSettings: "模型", aiProvider: "服务商", aiApiKey: "API Key", aiModel: "模型", aiBaseUrl: "Base URL",
    required: "必填", apiRequiredError: "请填写必填项", saveApi: "保存", resetApi: "清空", apiSaved: "已保存",
    notDetected: "未识别", sessionManagement: "会话", clearSession: "清空当前会话", newSession: "新会话",
    editSession: "改名", deleteSession: "删除", saveSession: "保存", cancelEdit: "取消", activeWindow: "当前窗口",
    language: "语言", game: "游戏", question: "问题", gamePlaceholder: "艾尔登法环",
    questionPlaceholder: "这个 Boss 怎么打？", loading: "查询中...", ask: "提问", sources: "来源",
    requestFailed: "请求失败", chooseGame: "请选择要查询的游戏", noneOfThese: "都不是", searchProgress: "检索进度",
  },
  en: {
    open: "Open QuestMate", title: "Quick Guide", subtitle: "In-game answers on demand", settings: "Settings",
    closeSettings: "Close settings", preferences: "Preferences", displayMode: "Display mode",
    windowPosition: "Window position", bottomRight: "Bottom right", bottomLeft: "Bottom left", center: "Center",
    compactMode: "Popover", drawerMode: "Drawer", minimize: "Minimize to bubble", apiOnline: "API Online",
    apiOffline: "API Offline", apiSettings: "Model", aiProvider: "Provider", aiApiKey: "API Key", aiModel: "Model",
    aiBaseUrl: "Base URL", required: "Required", apiRequiredError: "Fill in the required fields", saveApi: "Save",
    resetApi: "Clear", apiSaved: "Saved", notDetected: "Not detected", sessionManagement: "Session",
    clearSession: "Clear current chat", newSession: "New chat", editSession: "Rename", deleteSession: "Delete",
    saveSession: "Save", cancelEdit: "Cancel", activeWindow: "Active window", language: "Language", game: "Game",
    question: "Question", gamePlaceholder: "Elden Ring", questionPlaceholder: "How do I beat this boss?",
    loading: "Searching...", ask: "Ask", sources: "Sources", requestFailed: "Request failed",
    chooseGame: "Choose the game", noneOfThese: "None of these", searchProgress: "Search progress",
  },
} satisfies Record<Language, Copy>;

type IconName = "settings" | "sliders" | "history" | "minimize" | "api" | "chevron" | "plus";

export const SETTINGS_TABS: Array<{ id: SettingsTab; label: keyof Copy; icon: IconName }> = [
  { id: "preferences", label: "preferences", icon: "sliders" },
  { id: "api", label: "apiSettings", icon: "api" },
  { id: "session", label: "sessionManagement", icon: "history" },
];

export function Icon({ name }: { name: IconName }) {
  const paths: Record<IconName, string[]> = {
    settings: [
      "M12 15.5A3.5 3.5 0 1 0 12 8a3.5 3.5 0 0 0 0 7.5Z",
      "M19.4 15a1.7 1.7 0 0 0 .34 1.87l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.7 1.7 0 0 0-1.87-.34 1.7 1.7 0 0 0-1.04 1.57V21a2 2 0 0 1-4 0v-.09A1.7 1.7 0 0 0 8.96 19.4a1.7 1.7 0 0 0-1.87.34l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.7 1.7 0 0 0 4.6 15a1.7 1.7 0 0 0-1.57-1H3a2 2 0 0 1 0-4h.09A1.7 1.7 0 0 0 4.6 8.96a1.7 1.7 0 0 0-.34-1.87l-.06-.06A2 2 0 1 1 7.03 4.2l.06.06A1.7 1.7 0 0 0 8.96 4.6 1.7 1.7 0 0 0 10 3.03V3a2 2 0 0 1 4 0v.09a1.7 1.7 0 0 0 1.04 1.51 1.7 1.7 0 0 0 1.87-.34l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.7 1.7 0 0 0-.34 1.87A1.7 1.7 0 0 0 20.97 10H21a2 2 0 0 1 0 4h-.09A1.7 1.7 0 0 0 19.4 15Z",
    ],
    sliders: ["M4 6h10", "M18 6h2", "M4 12h2", "M10 12h10", "M4 18h12", "M20 18h0", "M14 4v4", "M6 10v4", "M16 16v4"],
    history: ["M3 12a9 9 0 1 0 3-6.7", "M3 4v5h5", "M12 7v5l3 2"], minimize: ["M6 12h12"],
    api: ["M4 12h4", "M16 12h4", "M9 7l-4 5 4 5", "M15 7l4 5-4 5", "M11 18l2-12"],
    chevron: ["M6 9l6 6 6-6"], plus: ["M12 5v14", "M5 12h14"],
  };
  return <svg aria-hidden="true" viewBox="0 0 24 24" className="ui-icon">{paths[name].map((path) => <path key={path} d={path} />)}</svg>;
}

export const providerLabel = (provider: AiSettings["provider"]) => provider === "deepseek" ? "DeepSeek" : "Anthropic";
export const formatMessageCount = (count: number, language: Language) => language === "zh" ? `${count} 条消息` : `${count} messages`;
export const updateLastAssistantMessage = (messages: Message[], patch: Partial<Message>) => messages.map((message, index) => index === messages.length - 1 && message.role === "assistant" ? { ...message, ...patch } : message);
export function formatCandidateHost(url: string) { try { return new URL(url).hostname.replace(/^www\./, ""); } catch { return url; } }
export const displayProcessName = (processName: string) => processName.replace(/\.exe$/i, "");
export const isOverlayProcess = (processName: string) => /questmate-overlay/i.test(processName);

export function GameField(props: { game: string; label: string; placeholder: string; processes: string[]; onGameChange: (value: string) => void; open: boolean; onOpenChange: (open: boolean) => void; }) {
  return (
    <div className="game-control" onBlur={(event) => { if (!event.currentTarget.contains(event.relatedTarget)) props.onOpenChange(false); }}>
      <label><span>{props.label}</span><div className="game-combobox">
        <input value={props.game} onChange={(event) => { props.onGameChange(event.target.value); props.onOpenChange(true); }} onFocus={() => props.onOpenChange(true)} placeholder={props.placeholder} title={props.game} role="combobox" aria-expanded={props.open} aria-controls="questmate-game-options" autoComplete="off" />
        <button type="button" className={props.open ? "game-combobox-toggle open" : "game-combobox-toggle"} onClick={() => props.onOpenChange(!props.open)} aria-label={props.label}><Icon name="chevron" /></button>
        {props.open && props.processes.length > 0 && <div className="custom-select-menu game-options" id="questmate-game-options" role="listbox">{props.processes.map((name) => <button key={name} type="button" className={props.game === name ? "selected" : ""} onClick={() => { props.onGameChange(name); props.onOpenChange(false); }} role="option" aria-selected={props.game === name}><span>{name}</span>{props.game === name && <span className="select-check">✓</span>}</button>)}</div>}
      </div></label>
    </div>
  );
}
