import { DEFAULT_AI_MODEL, DEFAULT_DEEPSEEK_MODEL, type AiSettings } from "../api";
import { Icon, providerLabel, type Copy } from "../ui";

type Props = {
  text: Copy;
  settings: AiSettings;
  saved: boolean;
  error: string;
  providerOpen: boolean;
  onSettingsChange: (settings: AiSettings) => void;
  onProviderOpenChange: (open: boolean) => void;
  onProviderChange: (provider: AiSettings["provider"]) => void;
  onSave: () => void;
  onReset: () => void;
  onClearError: () => void;
};

export function ApiSettings(props: Props) {
  const { text, settings, error } = props;
  const update = (patch: Partial<AiSettings>) => props.onSettingsChange({ ...settings, ...patch });
  return (
    <section className="settings-section">
      <div className="section-heading"><h2>{text.apiSettings}</h2></div>
      <div className="field-stack provider-field">
        <span>{text.aiProvider}</span>
        <div className="custom-select" onBlur={(event) => { if (!event.currentTarget.contains(event.relatedTarget)) props.onProviderOpenChange(false); }}>
          <button type="button" className={props.providerOpen ? "custom-select-trigger open" : "custom-select-trigger"} onClick={() => props.onProviderOpenChange(!props.providerOpen)} aria-haspopup="listbox" aria-expanded={props.providerOpen}>
            <span>{providerLabel(settings.provider)}</span><Icon name="chevron" />
          </button>
          {props.providerOpen && (
            <div className="custom-select-menu" role="listbox">
              {(["anthropic", "deepseek"] as const).map((provider) => (
                <button key={provider} type="button" className={settings.provider === provider ? "selected" : ""} onClick={() => props.onProviderChange(provider)} role="option" aria-selected={settings.provider === provider}>
                  <span>{providerLabel(provider)}</span>{settings.provider === provider && <span className="select-check">✓</span>}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
      <label className="field-stack">
        <span>{text.aiApiKey} <em>({text.required})</em></span>
        <input className={error && !settings.apiKey.trim() ? "invalid" : ""} value={settings.apiKey} onChange={(event) => { update({ apiKey: event.target.value }); props.onClearError(); }} placeholder={settings.provider === "deepseek" ? "sk-..." : "sk-ant-..."} type="password" required spellCheck={false} />
      </label>
      <label className="field-stack">
        <span>{text.aiModel} <em>({text.required})</em></span>
        <input className={error && !settings.model.trim() ? "invalid" : ""} value={settings.model} onChange={(event) => { update({ model: event.target.value }); props.onClearError(); }} placeholder={settings.provider === "deepseek" ? DEFAULT_DEEPSEEK_MODEL : DEFAULT_AI_MODEL} required spellCheck={false} />
      </label>
      <label className="field-stack">
        <span>{text.aiBaseUrl}</span>
        <input value={settings.baseUrl} onChange={(event) => update({ baseUrl: event.target.value })} placeholder={settings.provider === "deepseek" ? "https://api.deepseek.com" : "https://api.anthropic.com"} inputMode="url" spellCheck={false} />
      </label>
      {error && <p className="field-error" role="alert">{error}</p>}
      <div className="action-row">
        <button type="button" className="secondary-action" onClick={props.onReset}>{text.resetApi}</button>
        <button type="button" className="primary-action" onClick={props.onSave}>{props.saved ? text.apiSaved : text.saveApi}</button>
      </div>
    </section>
  );
}
