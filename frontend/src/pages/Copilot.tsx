/**
 * The copilot screen, and the demo's 0:55 → 1:10 beats.
 *
 * The "Explain How I Know This" panel is not a second inference pass — it is a projection of the
 * `reasoning_chain` the backend already returned. Every step shows what INDRA did, what it found,
 * how confident it was, and which documents it read. That is the whole trust argument, rendered.
 */

import { useState } from "react";
import type { Answer } from "../types";
import { ApiError, IndraApi } from "../api";
import {
  ConfidenceBar, ConfidenceDial, EmptyState, ErrorNote, Panel,
  SeverityBadge, Skeleton, SourceCard, UncertaintyBanner,
} from "../ui";
import { GraphCanvas } from "../graph";

const SUGGESTIONS = [
  "Why did P-101 fail last month?",
  "What is the OEM bearing wear limit for P-101?",
  "How do I replace the P-101 bearing?",
  "Will P-101 fail in the next 30 days?",
  "Are we compliant with Factory Act Section 41(b)?",
  "What don't we know about P-101?",
];

export function CopilotPage({ api }: { api: IndraApi }): JSX.Element {
  const [query, setQuery] = useState(SUGGESTIONS[0]);
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showChain, setShowChain] = useState(true);
  const [showCypher, setShowCypher] = useState(false);

  async function ask(q: string): Promise<void> {
    setBusy(true);
    setError(null);
    try {
      setAnswer(await api.ask(q));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
      setAnswer(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page">
      <Panel kicker="Explainable copilot" title="Grounded answers, not a chat-shaped guess.">
        <form
          className="askbar"
          onSubmit={(e) => {
            e.preventDefault();
            void ask(query);
          }}
        >
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask about any asset, procedure, failure or regulation…"
            aria-label="Question for INDRA"
          />
          <button type="submit" disabled={busy || !query.trim()}>
            {busy ? "Reasoning…" : "Ask INDRA"}
          </button>
        </form>
        <div className="chips">
          {SUGGESTIONS.map((s) => (
            <button key={s} type="button" className="chip"
                    onClick={() => { setQuery(s); void ask(s); }} disabled={busy}>
              {s}
            </button>
          ))}
        </div>
        {error ? <ErrorNote message={error} /> : null}
      </Panel>

      {busy ? <Panel><Skeleton rows={5} /></Panel> : null}

      {!busy && !answer && !error ? (
        <Panel>
          <EmptyState
            title="Ask something about the plant"
            hint="Every answer arrives with its sources, a confidence score per reasoning step, and the uncertainty it could not resolve."
          />
        </Panel>
      ) : null}

      {answer ? (
        <>
          <Panel
            kicker={`${answer.query_type.replace(/_/g, " ")} query · answered by ${answer.provider_used}`}
            title="Answer"
            actions={<ConfidenceDial value={answer.confidence.value} label="overall" />}
          >
            <p className="answer-text">{answer.answer_text}</p>
            <p className="rationale">{answer.confidence.rationale}</p>

            {answer.uncertainty_flags.length ? (
              <div className="uncert-stack">
                {answer.uncertainty_flags.map((f, i) => (
                  <UncertaintyBanner key={`${f.source}-${i}`} flag={f} />
                ))}
              </div>
            ) : null}

            {answer.recommended_actions.length ? (
              <div className="actions">
                <h3>Recommended actions</h3>
                {answer.recommended_actions.map((a, i) => (
                  <div key={i} className="action">
                    <SeverityBadge severity={a.urgency} />
                    <div>
                      <strong>{a.action}</strong>
                      {a.rationale ? <p>{a.rationale}</p> : null}
                      <span className="meta">
                        {a.owner_role ? `${a.owner_role} · ` : ""}
                        {a.due_within_days !== null ? `due in ${a.due_within_days}d` : "no deadline"}
                        {a.estimated_minutes ? ` · ~${a.estimated_minutes} min` : ""}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            ) : null}

            {answer.alternative_interpretations.length ? (
              <details className="alts">
                <summary>Alternative interpretations ({answer.alternative_interpretations.length})</summary>
                <ul>
                  {answer.alternative_interpretations.map((a, i) => <li key={i}>{a}</li>)}
                </ul>
              </details>
            ) : null}
          </Panel>

          <Panel
            kicker="Trust"
            title="Explain how I know this"
            actions={
              <div className="toggles">
                <button type="button" className="ghostbtn" onClick={() => setShowChain((v) => !v)}>
                  {showChain ? "Hide chain" : "Show chain"}
                </button>
                {answer.cypher_queries.length ? (
                  <button type="button" className="ghostbtn" onClick={() => setShowCypher((v) => !v)}>
                    {showCypher ? "Hide Cypher" : "Show Cypher"}
                  </button>
                ) : null}
              </div>
            }
          >
            {showChain ? (
              <ol className="chain">
                {answer.reasoning_chain.map((step) => (
                  <li key={step.order} className="chain-step">
                    <div className="chain-dot" />
                    <div className="chain-body">
                      <header>
                        <span className="chain-order">{step.order}</span>
                        <strong>{step.action}</strong>
                        <ConfidenceBar value={step.confidence.value} />
                      </header>
                      {step.finding ? <p className="finding">{step.finding}</p> : null}
                      <p className="rationale small">{step.confidence.rationale}</p>
                      {step.sources.length ? (
                        <div className="chain-sources">
                          {step.sources.slice(0, 4).map((s, i) => (
                            <span key={`${s.document_id}-${i}`} className="tag">
                              {s.document_title}{s.page ? ` p.${s.page}` : ""}
                            </span>
                          ))}
                          {step.sources.length > 4 ? (
                            <span className="tag ghost">+{step.sources.length - 4} more</span>
                          ) : null}
                        </div>
                      ) : null}
                      {step.graph_paths.length ? (
                        <div className="paths">
                          {step.graph_paths.slice(0, 3).map((p, i) => (
                            <code key={i} className="path">{p.narrative}</code>
                          ))}
                        </div>
                      ) : null}
                      {showCypher && step.cypher ? <pre className="cypher">{step.cypher}</pre> : null}
                    </div>
                  </li>
                ))}
              </ol>
            ) : null}
          </Panel>

          <Panel kicker="Evidence" title={`${answer.sources.length} cited sources`}>
            {answer.sources.length ? (
              <div className="sources">
                {answer.sources.map((s, i) => (
                  <SourceCard key={`${s.document_id}-${s.chunk_id ?? i}`} source={s} index={i} />
                ))}
              </div>
            ) : (
              <EmptyState
                title="No sources retrieved"
                hint="INDRA will say it does not know rather than fabricate an answer. That refusal is the feature."
              />
            )}
          </Panel>

          {answer.graph_preview && answer.graph_preview.nodes.length ? (
            <Panel kicker="Cross-document fusion" title="What INDRA connected">
              <GraphCanvas preview={answer.graph_preview} />
            </Panel>
          ) : null}
        </>
      ) : null}
    </div>
  );
}
