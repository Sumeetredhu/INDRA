/**
 * Compliance gaps, matrix and the one-click audit package — the demo's 2:40 → 2:50 beats.
 *
 * Gap detection is deterministic on the backend, so this screen never editorialises: it shows the
 * requirement, the status, the evidence that was found, and what is missing.
 */

import { useEffect, useState } from "react";
import type { ComplianceGap, GapStatus } from "../types";
import { ApiError, IndraApi } from "../api";
import { EmptyState, ErrorNote, Panel, SeverityBadge, Skeleton, StatTile } from "../ui";

const STATUS_LABEL: Record<GapStatus, string> = {
  compliant: "Compliant",
  missing: "Missing",
  outdated: "Outdated",
  incomplete: "Incomplete",
};

export function CompliancePage({ api }: { api: IndraApi }): JSX.Element {
  const [gaps, setGaps] = useState<ComplianceGap[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pdf, setPdf] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [open, setOpen] = useState<string | null>(null);
  const [regFilter, setRegFilter] = useState("ALL");

  useEffect(() => {
    void (async () => {
      try {
        setGaps(await api.audit());
      } catch (err) {
        setError(err instanceof ApiError ? err.message : String(err));
        setGaps([]);
      }
    })();
  }, [api]);

  async function generate(): Promise<void> {
    setBusy(true);
    setError(null);
    try {
      const tags = Array.from(new Set((gaps ?? []).map((g) => g.equipment_tag)));
      const result = await api.auditPdf(tags);
      const path = (result.pdf_path ?? result.path ?? "") as string;
      setPdf(path || "generated");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  const regulations = Array.from(new Set((gaps ?? []).map((g) => g.requirement.regulation)));
  const shown = (gaps ?? []).filter((g) => regFilter === "ALL" || g.requirement.regulation === regFilter);
  const byStatus = (gaps ?? []).reduce<Record<string, number>>((acc, g) => {
    acc[g.status] = (acc[g.status] ?? 0) + 1;
    return acc;
  }, {});

  return (
    <div className="page">
      <Panel
        kicker="Continuous regulatory monitoring"
        title="Gaps, evidence and the audit package"
        actions={
          <button type="button" className="ghostbtn" onClick={() => void generate()} disabled={busy || !gaps?.length}>
            {busy ? "Building…" : "Generate audit package"}
          </button>
        }
      >
        <div className="stats">
          <StatTile label="missing" value={byStatus.missing ?? 0} tone="crit" />
          <StatTile label="outdated" value={byStatus.outdated ?? 0} tone="high" />
          <StatTile label="incomplete" value={byStatus.incomplete ?? 0} tone="warn" />
          <StatTile label="regulations" value={regulations.length} />
        </div>
        {pdf ? (
          <div className="okline">
            Audit package written to <code>{pdf}</code> — a formatted PDF with the compliance matrix,
            gap detail with citations, corrective actions, and an evidence appendix.
          </div>
        ) : null}
        {error ? <ErrorNote message={error} /> : null}
        {regulations.length > 1 ? (
          <div className="chips">
            {["ALL", ...regulations].map((r) => (
              <button key={r} type="button" className={`chip ${regFilter === r ? "on" : ""}`}
                      onClick={() => setRegFilter(r)}>
                {r}
              </button>
            ))}
          </div>
        ) : null}
      </Panel>

      {gaps === null ? (
        <Panel><Skeleton rows={4} /></Panel>
      ) : shown.length === 0 ? (
        <Panel>
          <EmptyState
            title="No compliance gaps found"
            hint="Either every applicable requirement has evidence inside its interval, or no assets are in scope yet."
          />
        </Panel>
      ) : (
        <Panel kicker="Compliance matrix" title={`${shown.length} findings`}>
          <div className="tablewrap">
            <table className="matrix">
              <thead>
                <tr>
                  <th>Regulation</th><th>Clause</th><th>Obligation</th>
                  <th>Asset</th><th>Status</th><th>Severity</th><th>Overdue</th>
                </tr>
              </thead>
              <tbody>
                {shown.map((g) => (
                  <>
                    <tr key={g.gap_id}
                        className={`row ${g.status}`}
                        onClick={() => setOpen(open === g.gap_id ? null : g.gap_id)}>
                      <td>{g.requirement.regulation}</td>
                      <td><code>{g.requirement.clause}</code></td>
                      <td className="obl">{g.requirement.obligation}</td>
                      <td>{g.equipment_tag}</td>
                      <td><span className={`status ${g.status}`}>{STATUS_LABEL[g.status]}</span></td>
                      <td><SeverityBadge severity={g.severity} /></td>
                      <td>{g.days_overdue !== null ? `${g.days_overdue}d` : "—"}</td>
                    </tr>
                    {open === g.gap_id ? (
                      <tr key={`${g.gap_id}-d`} className="detailrow">
                        <td colSpan={7}>
                          <p>{g.detail}</p>
                          <p className="reqtext">{g.requirement.text}</p>
                          <div className="meta">
                            {g.requirement.frequency_days
                              ? <span className="tag ghost">every {g.requirement.frequency_days} days</span>
                              : null}
                            {g.deadline ? <span className="tag warn">deadline {g.deadline}</span> : null}
                            {g.last_evidence_date
                              ? <span className="tag">last evidence {g.last_evidence_date}</span>
                              : <span className="tag crit">no evidence found</span>}
                          </div>
                          {g.requirement.penalty ? (
                            <p className="penalty"><strong>Penalty exposure:</strong> {g.requirement.penalty}</p>
                          ) : null}
                          {g.recommended_action ? (
                            <div className="action">
                              <SeverityBadge severity={g.recommended_action.urgency} />
                              <div>
                                <strong>{g.recommended_action.action}</strong>
                                {g.recommended_action.rationale ? <p>{g.recommended_action.rationale}</p> : null}
                              </div>
                            </div>
                          ) : null}
                          {g.evidence.length ? (
                            <div className="chain-sources">
                              {g.evidence.map((e, i) => (
                                <span key={i} className="tag">{e.document_title}</span>
                              ))}
                            </div>
                          ) : null}
                        </td>
                      </tr>
                    ) : null}
                  </>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      )}
    </div>
  );
}
