import type { OverlayMode, OverlayPlacement } from "../tauri";
import type { Copy, Language } from "../ui";

type Props = {
  text: Copy;
  mode: OverlayMode;
  placement: OverlayPlacement;
  language: Language;
  onModeChange: (mode: Extract<OverlayMode, "popover" | "drawer">) => void;
  onPlacementChange: (placement: OverlayPlacement) => void;
  onLanguageChange: (language: Language) => void;
};

export function PreferencesSettings(props: Props) {
  const { text, mode, placement, language } = props;
  return (
    <section className="settings-section">
      <div className="section-heading"><h2>{text.preferences}</h2><p>{text.displayMode}</p></div>
      <div className="segmented-control">
        <button type="button" className={mode === "popover" ? "active" : ""} onClick={() => props.onModeChange("popover")} aria-pressed={mode === "popover"}>{text.compactMode}</button>
        <button type="button" className={mode === "drawer" ? "active" : ""} onClick={() => props.onModeChange("drawer")} aria-pressed={mode === "drawer"}>{text.drawerMode}</button>
      </div>
      <div className="section-heading placement-heading"><h3>{text.windowPosition}</h3></div>
      <div className="segmented-control placement-control" role="group" aria-label={text.windowPosition}>
        {(["bottom-right", "bottom-left", "center"] as const).map((value) => (
          <button key={value} type="button" className={placement === value ? "active" : ""} onClick={() => props.onPlacementChange(value)} aria-pressed={placement === value}>
            {value === "bottom-right" ? text.bottomRight : value === "bottom-left" ? text.bottomLeft : text.center}
          </button>
        ))}
      </div>
      <div className="setting-row">
        <div><h3>{text.language}</h3><p>{language === "zh" ? "中文" : "English"}</p></div>
        <label className="language-switch"><span>中文</span><input type="checkbox" checked={language === "en"} onChange={(event) => props.onLanguageChange(event.target.checked ? "en" : "zh")} /><i /><span>EN</span></label>
      </div>
    </section>
  );
}
