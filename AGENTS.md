# INDRA — Industrial Neural Data & Reasoning Assistant

Production-grade multi-agent platform that turns scattered industrial plant documents into a
proactive intelligence layer. **This is real software, not a prototype.**

## The one-line thesis

Most teams build "upload PDFs → ask questions → get answers". INDRA is a *Living Industrial Brain*:
it reasons **across** documents, predicts failures **before anyone asks**, **explains how it knows**,
and works **where technicians actually are** — plant floor, offline, in Hindi.

## Architecture

```
                    INDRA ORCHESTRATOR (FastAPI)
                             │
   ┌──────────┬──────────────┼──────────────┬────────────┬──────────┐
   │          │              │              │            │          │
INGESTION  KNOWLEDGE     COPILOT       PROACTIVE      MOBILE   COMPLIANCE
  AGENT    GRAPH AGENT    AGENT       INTEL AGENT     AGENT      AGENT
```

Agents are **independent services behind Protocol interfaces** (`indra/core/contracts.py`).
They communicate through the orchestrator and a Redis-Streams event bus (`indra/core/events.py`),
never by importing each other's internals.

| Agent | Package | Owns |
|---|---|---|
| Ingestion | `indra/agents/ingestion_agent/` | parsing, OCR, P&ID vision, chunking, embeddings |
| Knowledge Graph | `indra/agents/knowledge_graph_agent/` | Neo4j schema, entity linking, GraphRAG retrieval |
| Copilot | `indra/agents/copilot_agent/` | query routing, handlers, explainable answers |
| Proactive Intelligence | `indra/agents/proactive_intelligence_agent/` | compound signals, knowledge cliff, prediction |
| Mobile | `indra/agents/mobile_agent/` | voice, photo-to-query, offline sync |
| Compliance | `indra/agents/compliance_agent/` | regulation parsing, gap detection, audit packages |

## Hard rules for all code in this repo

1. **Full type hints. No bare `Any`.** Use `Protocol`, `TypedDict`, generics, `Literal`.
2. **Every external call is wrapped** — network, DB, LLM, filesystem — and raises a typed
   `IndraError` subclass from `indra/core/exceptions.py` with an actionable message.
3. **Structured logging with correlation IDs.** Always `from indra.core.logging import get_logger`.
   Never `print()`. The correlation id flows across every agent hop.
4. **Async by default** for ingestion, LLM calls, graph queries, HTTP. Sync only for pure CPU work,
   and that work goes through `asyncio.to_thread`.
5. **No hardcoded secrets or paths.** Everything through `indra.core.config.get_settings()`.
6. **Never fail the demo.** Every external dependency has an in-process fallback
   (`memory` backends, deterministic stub LLM). Degrade loudly in logs, never crash the request.
7. **Confidence is a first-class value.** Anything derived from OCR, vision, or an LLM carries a
   `Confidence` and a `SourceRef`. An answer with no sources is a bug.
8. **Tests mock every external API.** `pytest`, fixtures in `tests/conftest.py`, no live network.

## Definition of done for a module

- Type-hinted, docstringed public surface
- Wired into the orchestrator/API where relevant
- Unit tests with mocked externals
- Works with `INDRA_STORAGE_BACKEND=memory` and no API keys set

## Commands

```bash
python -m scripts.seed_demo_data      # build synthetic plant corpus + ingest
uvicorn indra.main:app --reload       # API on :8000, docs at /docs
pytest -q                             # test suite
docker compose -f docker/docker-compose.yml up
```

## Deliberate deviations from the original spec

Recorded in `docs/DECISIONS.md`. Read it before "fixing" something that looks off-spec.
