/**
 * Application shell.
 *
 * Navigation order deliberately follows the 3-minute demo script, and each tab carries its cue
 * timestamp, so the presenter moves left to right and never hunts for a screen mid-pitch.
 */

import { useEffect, useMemo, useState } from "react";
import { IndraApi, defaultApiBase, type Mode } from "./api";
import { CopilotPage } from "./pages/Copilot";
import { IngestPage } from "./pages/Ingest";
import { GraphExplorerPage } from "./pages/GraphExplorer";
import { AlertsPage } from "./pages/Alerts";
import { KnowledgeCliffPage } from "./pages/KnowledgeCliff";
import { CompliancePage } from "./pages/Compliance";
import { MobilePage } from "./pages/Mobile";
import { FleetPage, SystemPage } from "./pages/Fleet";
import "./styles.css";

type Tab =
  | "ingest" | "graph" | "copilot" | "alerts"
  | "cliff" | "compliance" | "mobile" | "fleet" | "system";

const TABS: { id: Tab; label: string; cue: string }[] = [
  { id: "ingest", label: "Ingest", cue: "0:00" },
  { id: "graph", label: "Graph", cue: "0:40" },
  { id: "copilot", label: "Copilot", cue: "0:55" },
  { id: "mobile", label: "Field", cue: "1:25" },
  { id: "alerts", label: "Alerts", cue: "2:00" },
  { id: "cliff", label: "Knowledge Cliff", cue: "2:30" },
  { id: "compliance", label: "Compliance", cue: "2:40" },
  { id: "fleet", label: "Fleet", cue: "" },
  { id: "system", label: "System", cue: "" },
];

export function App(): JSX.Element {
  const [tab, setTab] = useState<Tab>("copilot");
  const [base, setBase] = useState(defaultApiBase);
  const [draftBase, setDraftBase] = useState(defaultApiBase);
  const [online, setOnline] = useState<boolean | null>(null);
  const [mode, setMode] = useState<Mode>("unknown");
  const [nonce, setNonce] = useState(0);

  const api = useMemo(() => new IndraApi(base), [base]);

  useEffect(() => {
    let cancelled = false;
    api.onModeChange = (next) => { if (!cancelled) setMode(next); };
    void (async () => {
      try {
        await api.config();
        if (!cancelled) setOnline(true);
      } catch {
        if (!cancelled) setOnline(false);
      }
    })();
    return () => { cancelled = true; };
  }, [api, nonce]);

  const recorded = mode === "recorded";

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <span className="logo" aria-hidden="true" />
          <div>
            <h1>INDRA</h1>
            <p>Industrial Neural Data &amp; Reasoning Assistant</p>
          </div>
        </div>

        <div className="conn">
          <span className={`dot ${online === null ? "" : recorded ? "rec" : online ? "up" : "down"}`} />
          <span className="small">
            {online === null
              ? "connecting…"
              : recorded
                ? "recorded session · read-only"
                : online
                  ? "live API"
                  : "API unreachable"}
          </span>
          <form onSubmit={(e) => { e.preventDefault(); setBase(draftBase); setNonce((n) => n + 1); }}>
            <input
              value={draftBase}
              onChange={(e) => setDraftBase(e.target.value)}
              aria-label="API base URL"
              spellCheck={false}
            />
            <button type="submit" className="ghostbtn">Set</button>
          </form>
        </div>
      </header>

      <nav className="tabs" aria-label="Sections">
        {TABS.map((t) => (
          <button
            key={t.id}
            type="button"
            className={`tab ${tab === t.id ? "on" : ""}`}
            aria-current={tab === t.id ? "page" : undefined}
            onClick={() => setTab(t.id)}
          >
            {t.cue ? <span className="cue">{t.cue}</span> : null}
            {t.label}
          </button>
        ))}
      </nav>

      <main>
        {recorded ? (
          <div className="page">
            <div className="recnote" role="status">
              <strong>Recorded session</strong>
              <span>
                No live backend is reachable, so this is a captured run of the real pipeline —
                every answer, alert, gap and graph below came out of the actual system. Uploads,
                scans and PDF export need a running API: see the README to start one locally, then
                point the field above at it.
              </span>
            </div>
          </div>
        ) : online === false ? (
          <div className="page">
            <div className="errnote" role="alert">
              <strong>API unreachable</strong>
              <span>
                Start the backend with <code>python -m uvicorn indra.main:app --port 8000</code>,
                then press Set.
              </span>
            </div>
          </div>
        ) : null}

        {tab === "ingest" ? <IngestPage api={api} onIngested={() => setNonce((n) => n + 1)} /> : null}
        {tab === "graph" ? <GraphExplorerPage api={api} /> : null}
        {tab === "copilot" ? <CopilotPage api={api} /> : null}
        {tab === "mobile" ? <MobilePage api={api} /> : null}
        {tab === "alerts" ? <AlertsPage api={api} /> : null}
        {tab === "cliff" ? <KnowledgeCliffPage api={api} /> : null}
        {tab === "compliance" ? <CompliancePage api={api} /> : null}
        {tab === "fleet" ? <FleetPage api={api} /> : null}
        {tab === "system" ? <SystemPage api={api} /> : null}
      </main>

      <footer className="foot">
        <span>We are not building search. We are preserving the last engineer you will ever lose.</span>
      </footer>
    </div>
  );
}

export default App;
