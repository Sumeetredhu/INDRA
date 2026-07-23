/**
 * Application shell.
 *
 * Navigation order deliberately follows the 3-minute demo script, and each tab carries its cue
 * timestamp, so the presenter moves left to right and never hunts for a screen mid-pitch.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { IndraApi, type Mode } from "./api";
import { discoverBackend, rememberBackend } from "./backend";
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
  // Resolve once per page load: query param -> localStorage -> build env -> probed candidates.
  const discovery = useRef(discoverBackend()).current;

  const [tab, setTab] = useState<Tab>("copilot");
  const [base, setBase] = useState(discovery.initial);
  const [draftBase, setDraftBase] = useState(discovery.initial);
  const [online, setOnline] = useState<boolean | null>(null);
  const [mode, setMode] = useState<Mode>("unknown");
  const [searching, setSearching] = useState(!discovery.pinned);
  const [nonce, setNonce] = useState(0);

  const api = useMemo(() => new IndraApi(base), [base]);

  // Background discovery. The page has already rendered from the recording by the time this
  // resolves; when a live backend answers we switch over and everything re-fetches.
  useEffect(() => {
    let cancelled = false;
    void discovery.live.then((found) => {
      if (cancelled) return;
      setSearching(false);
      if (found && found !== base) {
        setBase(found);
        setDraftBase(found);
        setNonce((n) => n + 1);
      }
    });
    return () => { cancelled = true; };
    // Intentionally once per mount: `discovery` is a ref and never changes identity.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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

  function applyBase(url: string): void {
    const clean = url.trim().replace(/\/+$/, "");
    rememberBackend(clean);
    setBase(clean);
    setNonce((n) => n + 1);
  }

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
            {searching && recorded
              ? "recorded session · looking for a live backend…"
              : online === null
                ? "connecting…"
                : recorded
                  ? "recorded session · read-only"
                  : online
                    ? "live API"
                    : "API unreachable"}
          </span>
          <form onSubmit={(e) => { e.preventDefault(); applyBase(draftBase); }}>
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
                {searching
                  ? "Still probing for a live backend — a free-tier instance can take up to a minute to wake. Meanwhile this is a captured run of the real pipeline: every answer, alert, gap and graph below came out of the actual system."
                  : "No live backend answered, so this is a captured run of the real pipeline — every answer, alert, gap and graph below came out of the actual system. Uploads, scans and PDF export need a running API: deploy one with the button in the README, or start one locally and enter its URL above."}
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
