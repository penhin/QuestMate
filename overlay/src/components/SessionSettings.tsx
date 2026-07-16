import type { SessionSummary } from "../api";
import { formatMessageCount, Icon, type Copy, type Language } from "../ui";

type Props = {
  text: Copy;
  language: Language;
  sessions: SessionSummary[];
  activeSessionId?: string;
  draftSession: boolean;
  editingSessionId?: string;
  editingTitle: string;
  onNew: () => void;
  onOpen: (id: string) => void;
  onStartEdit: (id: string, title: string) => void;
  onEditingTitleChange: (title: string) => void;
  onSaveEdit: (id: string) => void;
  onCancelEdit: () => void;
  onDelete: (id: string) => void;
};

export function SessionSettings(props: Props) {
  const { text } = props;
  return (
    <section className="settings-section">
      <div className="section-heading"><h2>{text.sessionManagement}</h2></div>
      <div className="session-panel"><div className="session-list">
        <button type="button" className="session-row session-new-row" onClick={props.onNew} aria-label={text.newSession}><Icon name="plus" /></button>
        {props.draftSession && <div className="session-row session-draft active"><button type="button" className="session-open" onClick={props.onNew}><strong>{text.newSession}</strong><span>{formatMessageCount(0, props.language)}</span></button></div>}
        {props.sessions.map((session) => (
          <div key={session.session_id} className={session.session_id === props.activeSessionId ? "session-row active" : "session-row"} onBlur={(event) => {
            if (props.editingSessionId !== session.session_id) return;
            const nextTarget = event.relatedTarget;
            if (!(nextTarget instanceof Node) || !event.currentTarget.contains(nextTarget)) props.onCancelEdit();
          }}>
            {props.editingSessionId === session.session_id ? (
              <>
                <input className="session-title-input" value={props.editingTitle} onChange={(event) => props.onEditingTitleChange(event.target.value)} onKeyDown={(event) => {
                  if (event.key === "Enter") { event.preventDefault(); props.onSaveEdit(session.session_id); }
                  if (event.key === "Escape") props.onCancelEdit();
                }} autoFocus />
                <div className="session-actions"><button type="button" onClick={() => props.onSaveEdit(session.session_id)}>{text.saveSession}</button><button type="button" onClick={props.onCancelEdit}>{text.cancelEdit}</button></div>
              </>
            ) : (
              <>
                <button type="button" className="session-open" onClick={() => props.onOpen(session.session_id)}><strong>{session.title}</strong><span>{formatMessageCount(session.message_count, props.language)}</span></button>
                <div className="session-actions"><button type="button" onClick={() => props.onStartEdit(session.session_id, session.title)}>{text.editSession}</button><button type="button" onClick={() => props.onDelete(session.session_id)}>{text.deleteSession}</button></div>
              </>
            )}
          </div>
        ))}
      </div></div>
    </section>
  );
}
