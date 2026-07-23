# INDRA — Industrial Neural Data & Reasoning Assistant

**A living industrial brain.** It reasons *across* plant documents, predicts failures before anyone
asks, explains how it knows, and works where technicians actually are — plant floor, offline, in
Hindi.

> **[▶ Open the live console](https://sumeetredhu.github.io/INDRA/)**
>
> The hosted build serves a **recorded session** — a captured run of the real pipeline. Every
> answer, alert, gap and graph in it came out of the actual system. Uploads, scans and PDF export
> need a running backend: see [Run it locally](#run-it-locally).

---

## The problem

- Workers in Indian heavy industry waste **35%** of their time searching for information (McKinsey)
- Plants run **7–12 disconnected** document systems (NASSCOM-EY)
- That fragmentation causes **18–22%** of unplanned downtime
- **25%** of experienced engineers retire within the decade, taking undocumented knowledge with them

The winner isn't the team with the best chatbot. It's the team that prevents the next failure and
preserves the last engineer you'll ever lose.

## Architecture

```
                    INDRA ORCHESTRATOR (FastAPI)
                             │
   ┌──────────┬──────────────┼──────────────┬────────────┬──────────┐
INGESTION  KNOWLEDGE     COPILOT       PROACTIVE      MOBILE   COMPLIANCE
  AGENT    GRAPH AGENT    AGENT       INTEL AGENT     AGENT      AGENT
```

Six agents, each a package behind a `Protocol` in [`indra/core/contracts.py`](indra/core/contracts.py).
They never import each other — cross-agent work goes through the orchestrator or a typed event bus.

| Agent | Owns |
|---|---|
| [Ingestion](indra/agents/ingestion_agent) | parsing, OCR, **P&ID computer vision**, chunking, embeddings |
| [Knowledge Graph](indra/agents/knowledge_graph_agent) | Neo4j schema, entity resolution, **hybrid GraphRAG** |
| [Copilot](indra/agents/copilot_agent) | query routing, 7 handlers, **explainable reasoning chains** |
| [Proactive Intelligence](indra/agents/proactive_intelligence_agent) | **compound signals**, knowledge cliff, prediction |
| [Mobile](indra/agents/mobile_agent) | voice, photo-to-query, offline sync |
| [Compliance](indra/agents/compliance_agent) | regulation parsing, gap detection, audit packages |

## Run it locally

**No Docker, no Neo4j, no API keys required.** Every store has an in-process fallback and the LLM
chain ends in a deterministic stub.

```bash
pip install -r requirements.txt
python -m scripts.generate_demo_data
```

**Terminal 1 — the API.** `BOOTSTRAP_DEMO` ingests the corpus in-process at startup:

```bash
INDRA_STORAGE_BACKEND=memory INDRA_BOOTSTRAP_DEMO=true python -m uvicorn indra.main:app --port 8000
```

PowerShell:

```powershell
$env:INDRA_STORAGE_BACKEND='memory'; $env:INDRA_BOOTSTRAP_DEMO='true'; python -m uvicorn indra.main:app --port 8000
```

**Terminal 2 — the console:**

```bash
cd frontend && npm install && npm run dev
```

Open `http://localhost:5173`. API docs at `http://localhost:8000/docs`.

> **In-memory means in-process.** `scripts/seed_demo_data.py` seeds its *own* orchestrator and then
> exits, so a separately-running API will not see that data. Use `INDRA_BOOTSTRAP_DEMO=true` as
> above, or upload through the running API.

### With real infrastructure

```bash
docker compose -f docker/docker-compose.yml up
```

Brings up Neo4j, Redis and Postgres. `INDRA_STORAGE_BACKEND=auto` probes each and binds the real
backend where reachable, falling back per-store otherwise. `/health` reports which is which.

## Deploy the backend (~3 minutes)

The console works for a stranger with no instructions once a backend exists. To create one:

**→ [Deploy on Render](https://dashboard.render.com/blueprint/new?repo=https://github.com/Sumeetredhu/INDRA)**
(the dashboard blueprint URL — more reliable than the `render.com/deploy` shortcut, which can fail
to load).

1. Open the link, sign in with GitHub, click **Deploy Blueprint** / **Apply**.
2. Render reads [`render.yaml`](render.yaml), installs [`requirements-deploy.txt`](requirements-deploy.txt)
   and starts the API. It ingests the demo corpus in-process at boot
   (`INDRA_BOOTSTRAP_DEMO=true`), so the instance is never an empty shell.
3. Copy the service URL, e.g. `https://indra-api.onrender.com`.

> Render's free web tier needs a payment method on file (it does not charge) and runs in
> **Oregon** — Singapore is not a free region. If the deploy page errors, it is almost always a
> free-plan/region mismatch; this blueprint already pins `region: oregon`.

Then make the public console use it — either way works:

- **Share a pinned link** (zero rebuild):
  `https://sumeetredhu.github.io/INDRA/?api=https://your-service.onrender.com`
  The URL is remembered in `localStorage`, so a visitor only needs it once.
- **Make it the default for everyone**: put the URL first in
  [`frontend/public/backends.json`](frontend/public/backends.json), then
  `cd frontend && npm run build` and push `dist/` to the `gh-pages` branch.

The console probes those candidates on every load in the background, so the moment a backend
answers `/health` the page upgrades itself from the recorded session to live. A free Render
instance sleeps after ~15 minutes idle and takes ~50s to wake — the probe waits, and the recording
holds the screen in the meantime rather than showing an error.

`Procfile`, `runtime.txt` and `railway.json` are also included, so Railway, Fly and Heroku-style
platforms work from the same repo.

Optional: set `GEMINI_API_KEY` ([free tier](https://aistudio.google.com/apikey)) to replace the
deterministic stub with real generation. Everything works without it.

### Point the hosted console at your own laptop

`https://sumeetredhu.github.io` is in the default CORS allowlist, so you can run the API locally
and drive the public UI against it — useful for a demo where you want live uploads:

```bash
INDRA_STORAGE_BACKEND=memory INDRA_BOOTSTRAP_DEMO=true python -m uvicorn indra.main:app --port 8000
```

then open `https://sumeetredhu.github.io/INDRA/?api=http://localhost:8000`.

## What makes this different

**P&ID computer vision.** Most systems handle text. INDRA parses engineering drawings — detects
pump, vessel and exchanger symbols, OCRs the tags inside them, traces pipe runs with Hough
transforms, reads flow direction from arrowheads, and writes `(Equipment)-[:CONNECTED_TO]->(Equipment)`
edges. OCR damage (`P-l0l`, `P-IOI`) is corrected against a glyph-confusion map and the live
registry — and **never silently**: ambiguity returns alternatives with lowered confidence, because a
confidently wrong tag on a plant floor is worse than an honest question.

**Hybrid GraphRAG.** Vector similarity finds candidates; graph traversal (1–3 hops) finds what no
single document contains. Both score families are min-max normalised *per query* before blending,
because a raw cosine sits in a narrow 0.7–0.9 band and would otherwise swamp an unbounded graph
boost. Reciprocal Rank Fusion is available as a scale-free alternative.

**Compound signals, not alerts.** One observation is noise. Six declarative rules fire on the
*conjunction* — "an operator bypassed the P-101 alarm twice on the night shift of 14 June while an
open work order records bearing wear at 78%; the 2022 seizure was preceded by the same combination."

**Explainability as data, not a second inference pass.** Every answer carries a reasoning chain with
per-step confidence and sources. Overall confidence is **weakest-link, not mean** — a 0.95 retrieval
step cannot rescue a 0.4 OCR read of the number the conclusion rests on.

**It cannot die on stage.** Every backend degrades to an in-process implementation, the LLM chain
ends in a seeded stub, and the hosted frontend falls back to a recorded session. Degradation is
always *reported*, never hidden: `/health` shows `graph: memory (fallback)`, not a green tick that
lies.

Full engineering rationale, including 11 deliberate deviations from the original spec, is in
[`docs/DECISIONS.md`](docs/DECISIONS.md). The integration contract is [`docs/WIRING.md`](docs/WIRING.md).

## Project status

Honest state of the build:

| Area | State |
|---|---|
| Core, storage, LLM router | Complete and verified |
| Six agents + orchestrator + API | Complete — 89 modules import, 25 endpoints live |
| Frontend console | Complete — 9 screens |
| Demo corpus | Complete — 12 documents including a real parseable P&ID and a degraded scan |
| **Entity extraction precision** | **Known defect.** The plant-tag grammar over-matches, so work-order numbers (`WO-2024`), incident IDs (`INC-2022`), employee IDs (`EMP-1533`) and part numbers (`SKF-6316`) are stored as equipment. ~40 of 47 "assets" are spurious, which inflates the alert feed with fleet-pattern noise. Fix is a prefix allowlist in the tag normaliser. |
| Test suite | **Thin** — 2 tests pass; the planned ~20 modules are not written |

## Stack

Python 3.11+ · FastAPI · Pydantic v2 · Neo4j · ChromaDB · SQLAlchemy · Redis ·
OpenCV · Tesseract · Gemini / Groq / Ollama · React 18 · TypeScript · Vite

## Licence

MIT
