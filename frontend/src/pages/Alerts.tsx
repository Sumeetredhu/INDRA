/**
 * Proactive alert feed — the demo's 2:00 → 2:15 beats.
 *
 * A compound signal is only interesting because of the conjunction, so each alert expands to show
 * its constituent signals and the documents behind each one. That expansion is the "cross-document
 * fusion" moment: four separate records, one conclusion.
 */

import { useEffect, useState } from "react";
import type { Alert } from "../types";
import { ApiError, IndraApi } from "../api";
import { ConfidenceBar, EmptyState, ErrorNote, Panel, SeverityBadge, Skeleton, StatTile } from "../ui";

const ORDER: Record<string, number> = { CRITICAL: 0, HIGH: 1, WARNING: 2, LOW: 3, INFO: 4 };

export function AlertsPage({ api }: { api: IndraApi }): JSX.Element {
  const [alerts, setAlerts] = useState<Alert[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [open, setOpen] = useState<string | null>(null);
  const [filter, setFilter] = useState<string>("ALL");

  async function load(): Promise<void> {
    setError(null);
    try {
      const data = await api.alerts(true);
      data.sort((a, b) => (ORDER[a.severity] ?? 9) - (ORDER[b.severity] ?? 9)
        || b.risk_percent - a.risk_percent);
      setAlerts(data);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
      setAlerts([]);
    }
  }

  async function rescan(): Promise<void> {
    setBusy(true);
    try {
      await api.scan();
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => { void load(); /* eslint-disable-next-line */ }, []);

  const shown = (alerts ?? []).filter((a) => filter === "ALL" || a.severity === filter);
  const counts = (alerts ?? []).reduce<Record<string, number>>((acc, a) => {
    acc[a.severity] = (acc[a.severity] ?? 0) + 1;
    return acc;
  }, {});

  return (
    <div className="page">
      <Panel
        kicker="Proactive intelligence"
        title="Detected without anyone asking"
        actions={
          <button type="button" className="ghostbtn" onClick={() => void rescan()} disabled={busy}>
            {busy ? "Scanning…" : "Run scan"}
          </button>
        }
      >
        <div className="stats">
          <StatTile label="critical" value={counts.CRITICAL ?? 0} tone="crit" />
          <StatTile label="high" value={counts.HIGH ?? 0} tone="high" />
          <StatTile label="warning" value={counts.WARNING ?? 0} tone="warn" />
          <StatTile label="total open" value={alerts?.length ?? 0} />
        </div>
        <div className="chips">
          {["ALL", "CRITICAL", "HIGH", "WARNING"].map((f) => (
            <button key={f} type="button"
                    className={`chip ${filter === f ? "on" : ""}`}
                    onClick={() => setFilter(f)}>
              {f}
            </button>
          ))}
        </div>
        {error ? <ErrorNote message={error} /> : null}
      </Panel>

      {alerts === null ? (
        <Panel><Skeleton rows={4} /></Panel>
      ) : shown.length === 0 ? (
        <Panel>
          <EmptyState
            title="No open alerts"
            hint="Run a scan after ingesting documents, or the fleet is genuinely quiet."
          />
        </Panel>
      ) : (
        <div className="alertfeed" aria-live="polite">
          {shown.map((alert) => {
            const expanded = open === alert.alert_id;
            return (
              <article key={alert.alert_id}
                       className={`alert ${alert.severity.toLowerCase()} ${expanded ? "open" : ""}`}>
                <header onClick={() => setOpen(expanded ? null : alert.alert_id)}
                        role="button" tabIndex={0}
                        onKeyDown={(e) => { if (e.key === "Enter") setOpen(expanded ? null : alert.alert_id); }}>
                  <SeverityBadge severity={alert.severity} />
                  <div className="alert-main">
                    <strong>{alert.title}</strong>
                    <span className="meta">
                      {alert.equipment_tag}
                      {alert.compound_signal
                        ? ` · ${alert.compound_signal.signals.length} compound signals`
                        : ""}
                      {alert.risk_percent ? ` · risk ${alert.risk_percent.toFixed(0)}%` : ""}
                    </span>
                  </div>
                  <span className="chev">{expanded ? "−" : "+"}</span>
                </header>

                {expanded ? (
                  <div className="alert-body">
                    <p>{alert.body}</p>

                    {alert.compound_signal ? (
                      <>
                        <div className="cs-head">
                          <span className="tag">{alert.compound_signal.rule_name}</span>
                          <ConfidenceBar value={alert.compound_signal.confidence.value} />
                        </div>
                        <p className="explanation">{alert.compound_signal.explanation}</p>
                        <h4>Signals combined</h4>
                        <ul className="signals">
                          {alert.compound_signal.signals.map((s) => (
                            <li key={s.signal_id}>
                              <span className="tag ghost">{s.kind.replace(/_/g, " ")}</span>
                              <span>{s.description}</span>
                              {s.sources.length ? (
                                <span className="meta">
                                  {s.sources.map((src) => src.document_title).slice(0, 3).join(" · ")}
                                </span>
                              ) : null}
                            </li>
                          ))}
                        </ul>
                      </>
                    ) : null}

                    {alert.recommended_actions.length ? (
                      <>
                        <h4>Recommended actions</h4>
                        {alert.recommended_actions.map((a, i) => (
                          <div key={i} className="action">
                            <SeverityBadge severity={a.urgency} />
                            <div>
                              <strong>{a.action}</strong>
                              {a.rationale ? <p>{a.rationale}</p> : null}
                            </div>
                          </div>
                        ))}
                      </>
                    ) : null}

                    {alert.sources.length ? (
                      <div className="chain-sources">
                        {alert.sources.map((s, i) => (
                          <span key={i} className="tag">
                            {s.document_title}{s.page ? ` p.${s.page}` : ""}
                          </span>
                        ))}
                      </div>
                    ) : null}
                  </div>
                ) : null}
              </article>
            );
          })}
        </div>
      )}
    </div>
  );
}
