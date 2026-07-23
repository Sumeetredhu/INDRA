/**
 * Knowledge cliff dashboard — the demo's 2:30 beat, and the emotional core of the pitch.
 *
 * The score alone would be unfalsifiable, so the factor breakdown is rendered as a bar per
 * contributing factor. A plant manager has to be able to argue with the number; that is what makes
 * it credible enough to act on.
 */

import { useEffect, useState } from "react";
import type { KnowledgeCliffScore } from "../types";
import { ApiError, IndraApi } from "../api";
import { EmptyState, ErrorNote, Panel, SeverityBadge, Skeleton, StatTile } from "../ui";

const FACTOR_LABELS: Record<string, string> = {
  retirement_pressure: "Retirement pressure",
  documentation_deficit: "Documentation deficit",
  criticality: "Asset criticality",
  knowledge_concentration: "Knowledge concentration",
  critical_floor_applied: "Critical floor",
};

export function KnowledgeCliffPage({ api }: { api: IndraApi }): JSX.Element {
  const [scores, setScores] = useState<KnowledgeCliffScore[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState<string | null>(null);

  useEffect(() => {
    void (async () => {
      try {
        const data = await api.knowledgeCliff();
        data.sort((a, b) => b.score - a.score);
        setScores(data);
      } catch (err) {
        setError(err instanceof ApiError ? err.message : String(err));
        setScores([]);
      }
    })();
  }, [api]);

  const critical = (scores ?? []).filter((s) => s.severity === "CRITICAL" || s.score >= 75);

  return (
    <div className="page">
      <Panel
        kicker="Preserve expertise"
        title="When expertise walks out, it cannot be recovered."
      >
        <div className="stats">
          <StatTile label="assets at risk" value={scores?.length ?? 0} />
          <StatTile label="critical" value={critical.length} tone={critical.length ? "crit" : ""} />
          <StatTile
            label="experts retiring"
            value={new Set((scores ?? []).flatMap((s) => s.retiring_experts.map((p) => p.name))).size}
          />
        </div>
        {error ? <ErrorNote message={error} /> : null}
      </Panel>

      {scores === null ? (
        <Panel><Skeleton rows={4} /></Panel>
      ) : scores.length === 0 ? (
        <Panel>
          <EmptyState
            title="No knowledge-cliff risk detected"
            hint="This needs people with retirement dates and equipment expertise in the graph."
          />
        </Panel>
      ) : (
        <div className="cliffgrid">
          {scores.map((s) => {
            const expanded = open === s.equipment_tag;
            const factors = Object.entries(s.factors)
              .filter(([k]) => FACTOR_LABELS[k] !== undefined)
              .filter(([, v]) => v > 0);
            const peak = Math.max(1, ...factors.map(([, v]) => v));
            return (
              <article key={s.equipment_tag} className={`cliff ${s.severity.toLowerCase()}`}>
                <header>
                  <div className="cliff-score">
                    <span className="cliff-num">{Math.round(s.score)}</span>
                    <span className="cliff-of">/100</span>
                  </div>
                  <div>
                    <strong>{s.equipment_tag}</strong>
                    <div className="meta">
                      <SeverityBadge severity={s.severity} />
                      <span className="tag ghost">criticality {s.criticality}</span>
                      <span className="tag ghost">{s.document_count} documents</span>
                    </div>
                  </div>
                </header>

                {s.rationale ? <p className="rationale">{s.rationale}</p> : null}

                {s.retiring_experts.length ? (
                  <div className="experts">
                    {s.retiring_experts.map((p) => (
                      <div key={p.person_id} className="expert">
                        <strong>{p.name}</strong>
                        <span className="meta">
                          {p.years_experience ? `${p.years_experience} years` : "experience unknown"}
                          {p.retirement_date ? ` · retires ${p.retirement_date}` : ""}
                          {s.days_to_first_retirement !== null
                            ? ` · ${s.days_to_first_retirement} days`
                            : ""}
                        </span>
                      </div>
                    ))}
                  </div>
                ) : null}

                {factors.length ? (
                  <div className="factors">
                    <h4>Why this score</h4>
                    {factors.map(([key, value]) => (
                      <div key={key} className="factor">
                        <span className="factor-name">{FACTOR_LABELS[key] ?? key}</span>
                        <span className="factor-track">
                          <span className="factor-fill" style={{ width: `${(value / peak) * 100}%` }} />
                        </span>
                        <span className="factor-val">{value.toFixed(1)}</span>
                      </div>
                    ))}
                  </div>
                ) : null}

                {s.interview_questions.length ? (
                  <>
                    <button type="button" className="ghostbtn"
                            onClick={() => setOpen(expanded ? null : s.equipment_tag)}>
                      {expanded ? "Hide" : `Capture session — ${s.interview_questions.length} questions`}
                    </button>
                    {expanded ? (
                      <ol className="questions">
                        {s.interview_questions.map((q, i) => <li key={i}>{q}</li>)}
                      </ol>
                    ) : null}
                  </>
                ) : null}
              </article>
            );
          })}
        </div>
      )}
    </div>
  );
}
