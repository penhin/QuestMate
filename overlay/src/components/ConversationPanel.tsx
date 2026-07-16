import type { FormEvent } from "react";

import type { GameCandidate } from "../api";
import type { OverlayMode } from "../tauri";
import { formatCandidateHost, type Copy, type Message } from "../ui";

type Props = {
  messages: Message[];
  error: string;
  question: string;
  game: string;
  loading: boolean;
  mode: OverlayMode;
  text: Copy;
  onQuestionChange: (value: string) => void;
  onSubmit: (event: FormEvent) => void;
  onConfirmGame: (candidate: GameCandidate, pendingQuestion?: string) => void;
  onRejectGames: (pendingQuestion?: string) => void;
};

export function ConversationPanel(props: Props) {
  const { messages, error, question, game, loading, mode, text } = props;
  return (
    <>
      <section className="messages" aria-live="polite">
        {messages.map((message, index) => (
          <article key={`${message.role}-${index}`} className={`message ${message.role}${message.status ? " status" : ""}`}>
            {message.status && message.progress?.length ? (
              <section className="search-progress" role="status" aria-live="polite" aria-atomic="true">
                <div className="search-progress-heading"><span className="progress-spinner" aria-hidden="true" /><h2>{text.searchProgress}</h2></div>
                <ol>
                  {message.progress.map((step, stepIndex) => {
                    const isCurrent = stepIndex === message.progress!.length - 1;
                    return <li key={`${step}-${stepIndex}`} className={isCurrent ? "current" : "complete"}><span className="progress-marker" aria-hidden="true" /><span>{step}</span></li>;
                  })}
                </ol>
              </section>
            ) : <p>{message.content}</p>}
            {message.gameCandidates && message.gameCandidates.length > 0 && (
              <div className="game-candidates" aria-label={text.chooseGame}>
                <span>{text.chooseGame}</span>
                <div className="candidate-list">
                  {message.gameCandidates.map((candidate) => (
                    <button type="button" key={`${candidate.name}-${candidate.platform_urls[0] ?? candidate.official_urls[0] ?? candidate.identity_urls[0] ?? ""}`} className="candidate-card" onClick={() => props.onConfirmGame(candidate, message.pendingQuestion)}>
                      <strong>{candidate.name}</strong>
                      {candidate.tags.length > 0 && <small>{candidate.tags.join(" / ")}</small>}
                      {(candidate.platform_urls[0] || candidate.official_urls[0] || candidate.identity_urls[0]) && <em>{formatCandidateHost(candidate.platform_urls[0] || candidate.official_urls[0] || candidate.identity_urls[0])}</em>}
                    </button>
                  ))}
                  <button type="button" className="candidate-card none" onClick={() => props.onRejectGames(message.pendingQuestion)}><strong>{text.noneOfThese}</strong></button>
                </div>
              </div>
            )}
            {message.sources && message.sources.length > 0 && (
              <div className="sources"><span>{text.sources}</span><ul>{message.sources.map((source) => <li key={source.url}><a href={source.url} target="_blank" rel="noreferrer">{source.title}</a></li>)}</ul></div>
            )}
          </article>
        ))}
      </section>
      {error && <p className="error" role="alert">{error}</p>}
      <form className="ask-form" onSubmit={props.onSubmit}>
        <label><span>{text.question}</span><textarea value={question} onChange={(event) => props.onQuestionChange(event.target.value)} placeholder={text.questionPlaceholder} rows={mode === "drawer" ? 4 : 3} /></label>
        <button className="submit" disabled={loading || !game.trim() || !question.trim()} aria-busy={loading}>{loading ? text.loading : text.ask}</button>
      </form>
    </>
  );
}
