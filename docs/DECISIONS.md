# Deviations from the original master prompt — and why

The master prompt (`INDRA_Master_Prompt_Claude_Code.pdf`) is the product spec. These are the
engineering decisions where the implementation intentionally differs. Each one is a *strengthening*
of the spec's own stated goals, not a reduction of scope.

---

## D1. Everything degrades to in-process fallbacks (`INDRA_STORAGE_BACKEND=memory`)

**Spec:** Neo4j + ChromaDB + PostgreSQL + Redis, via Docker.

**Change:** Every store sits behind a `Protocol` with two implementations — the real one and an
in-memory one. `auto` mode probes the real backend on startup and silently falls back.

**Why:** The spec's own demo script is a 3-minute live run in front of judges. A cold Neo4j
container or an exhausted Gemini quota kills it. This makes `git clone && pytest && uvicorn` work on
a laptop with zero infrastructure and zero API keys, while the Docker path stays fully real.
This is the single highest-leverage change in the build.

---

## D2. The LLM layer is provider-agnostic with a deterministic stub

**Spec:** Gemini Embedding + Gemini 2.5 Flash, Groq fallback, Ollama for offline.

**Change:** Kept exactly that routing (`gemini → groq → ollama`), but behind `ChatProvider` /
`EmbeddingProvider` protocols, plus:
- a `StubProvider` with **seeded deterministic** output for tests and demo-safe mode
- a hash-based deterministic embedder so vector search works with no API key
- an optional Anthropic adapter, since the interface is free

**Why:** The spec caps out at ~500 generations/day. A rehearsal loop burns that before the demo.
Provider abstraction also makes the "Fast fallback" requirement a config change, not a rewrite.

---

## D3. Retrieval fusion is configurable, and adds Reciprocal Rank Fusion

**Spec:** `vector_score * 0.6 + graph_boost * 0.4`.

**Change:** That formula is the default (`RETRIEVAL_VECTOR_WEIGHT=0.6`), but weights are settings and
an alternative `rrf` strategy is selectable.

**Why:** Linear blending of a cosine score (0–1, tightly clustered ~0.7–0.9) against a graph boost
built from unbounded centrality is numerically fragile — the vector term saturates and the graph
term dominates or vanishes depending on graph size. Scores are min-max normalised per query before
blending, and RRF is available because it is scale-free. The spec's headline claim — "finding
connections no single document contains" — depends entirely on this blend actually working.

---

## D4. P&ID vision: rule-based pipeline is the *primary* path, YOLO is the optional upgrade

**Spec:** "YOLOv8 (or rule-based fallback)".

**Change:** Inverted. The shipped default is template matching + Hough line detection + morphology +
region OCR, with a `Detector` protocol so a trained YOLO checkpoint drops in via config.

**Why:** There is no labelled P&ID dataset in this project and an untrained YOLOv8 detects nothing.
A rule-based pipeline that demonstrably finds pumps and traces pipes beats a model checkpoint that
does not exist. The spec itself says "But make it work."

---

## D5. OCR tag correction is a typed, testable component

**Spec:** Fuzzy-match `P-l0l` → `P-101`.

**Change:** A dedicated `TagNormalizer`: glyph-confusion map (`l/1/I`, `O/0`, `S/5`, `B/8`, `Z/2`),
structural regex for `<LETTER>-<DIGITS>` plant tag grammar, then `rapidfuzz` against the live
equipment registry, returning `(tag, confidence, alternatives)` — never a silent correction.

**Why:** A silent wrong correction on a plant floor is worse than an admitted unknown. The spec's own
explainability requirement demands the uncertainty be surfaced, not swallowed.

---

## D6. Ingestion is idempotent and content-addressed

**Spec:** Not addressed.

**Change:** SHA-256 of file bytes is the document identity. Re-uploading is a no-op that returns the
existing `document_id`; re-ingesting a *changed* file supersedes the old version rather than
duplicating nodes.

**Why:** Without this, a demo rehearsal duplicates every entity and the knowledge graph
visualisation turns into unreadable mush by the third run.

---

## D7. Compound signals are a declarative rule engine, not `if` statements

**Spec:** A table of 6 rules.

**Change:** All six are `SignalRule` objects (predicate + severity + evidence builder + explanation
template) evaluated by one engine, in `rules.py`. Adding a rule is adding a list entry.

**Why:** The spec asks for auditability of proactive alerts. A declarative rule carries its own
evidence and rationale; scattered conditionals cannot be explained to an auditor.

---

## D8. The event bus is Redis Streams, explicitly

**Spec:** "message broker or direct API calls".

**Change:** A single `EventBus` protocol; Redis Streams in Docker, in-memory pub/sub otherwise.
Agents publish typed events (`document.ingested`, `graph.updated`, `alert.raised`).

**Why:** "or direct API calls" leads to six agents importing each other. One typed bus keeps the
multi-agent claim architecturally true rather than cosmetic — this is 20% of the judging weight.

---

## D9. Security and observability are wired from commit one

**Spec:** Listed under "Execution Philosophy" but not designed.

**Change:** API-key auth middleware, per-IP rate limiting, upload magic-number + size validation,
request-id propagation, JSON structured logs, `/health` (liveness + per-dependency readiness) and
`/metrics`.

**Why:** "Production-ready" is a stated grading criterion, and these cost far less to build in than
to retrofit.

---

## D10. PostgreSQL is optional; SQLite is the default metadata store

**Spec:** PostgreSQL for metadata.

**Change:** SQLAlchemy async with SQLite as default, Postgres via `DATABASE_URL`.

**Why:** Same reasoning as D1 — nothing in the metadata layer needs Postgres at demo scale, and
requiring it adds a container to the critical path of a 3-minute demo.

---

## D11. Multi-language pipeline translates at the edges, reasons in English

**Spec:** Hindi/Tamil/Kannada/Marathi/English, Whisper → detect → translate → Copilot → translate back.

**Change:** Implemented as specified, with the addition that the **reasoning chain is retained in
English internally** and translated only at render time, and that equipment tags are masked from
translation.

**Why:** Round-tripping `P-101` through machine translation corrupts it (observed: `पी-१०१`).
Masking tags preserves the graph lookups that the whole answer depends on.
