# Canonical module and symbol names

Binding contract for every builder in this repo. The orchestrator imports exactly these names; a
module that renames one breaks wiring. If you believe a name is wrong, raise it — do not rename.

## Agents

| Package | Module | Class | Implements |
|---|---|---|---|
| `indra.agents.ingestion_agent` | `service` | `IngestionAgent` | `IngestionService` |
| `indra.agents.knowledge_graph_agent` | `service` | `KnowledgeGraphAgent` | `KnowledgeGraphService` |
| `indra.agents.copilot_agent` | `service` | `CopilotAgent` | `CopilotService` |
| `indra.agents.proactive_intelligence_agent` | `service` | `ProactiveIntelligenceAgent` | `ProactiveService` |
| `indra.agents.mobile_agent` | `service` | `MobileAgent` | `MobileService` |
| `indra.agents.compliance_agent` | `service` | `ComplianceAgent` | `ComplianceService` |

**Every agent has exactly this constructor:**

```python
def __init__(self, deps: AgentDeps) -> None: ...
```

Agents that need a sibling agent receive it through an explicit setter called by the orchestrator
after construction — never by importing it:

```python
# CopilotAgent
def bind(self, *, knowledge_graph: KnowledgeGraphService,
         proactive: ProactiveService, compliance: ComplianceService) -> None: ...

# ProactiveIntelligenceAgent
def bind(self, *, knowledge_graph: KnowledgeGraphService) -> None: ...

# MobileAgent
def bind(self, *, copilot: CopilotService, knowledge_graph: KnowledgeGraphService,
         proactive: ProactiveService) -> None: ...

# ComplianceAgent
def bind(self, *, knowledge_graph: KnowledgeGraphService) -> None: ...

# IngestionAgent
def bind(self, *, knowledge_graph: KnowledgeGraphService) -> None: ...
```

Each agent also exposes:

```python
async def startup(self) -> None: ...
async def shutdown(self) -> None: ...
async def health(self) -> dict[str, Any]:   # {"ok": bool, "backend": str, "detail": str}
```

## Platform builders

```python
from indra.llm.router import build_router          # (settings) -> LLMRouter
from indra.storage.factory import build_stores     # async (settings) -> StoreBundle
from indra.orchestrator.orchestrator import IndraOrchestrator, get_orchestrator
```

`StoreBundle` is a dataclass with fields `graph, vectors, metadata, blobs, events, cache,
bound_backends` — the same names as `AgentDeps`, so the orchestrator builds deps by unpacking it.

## API routers

Every router module exposes `router: APIRouter` and is mounted under `settings.api_prefix`:

| Module | Prefix | Owns |
|---|---|---|
| `indra.api.routes.system` | `/system` | health, metrics, config summary, backends |
| `indra.api.routes.ingestion` | `/ingest` | upload, batch, status, SSE progress |
| `indra.api.routes.query` | `/query` | ask, stream, explain, classify |
| `indra.api.routes.graph` | `/graph` | preview, stats, neighbours, cypher (read-only) |
| `indra.api.routes.equipment` | `/equipment` | list, detail, history, readings |
| `indra.api.routes.alerts` | `/alerts` | list, acknowledge, scan, predict, knowledge-cliff, SSE feed |
| `indra.api.routes.mobile` | `/mobile` | voice, photo, offline bundle, sync |
| `indra.api.routes.compliance` | `/compliance` | audit, gaps, matrix, package, export |

## FastAPI app

`indra/main.py` exposes `app: FastAPI`. Startup/shutdown go through the lifespan, which calls
`IndraOrchestrator.startup()` / `.shutdown()`. Route handlers get the orchestrator via the
`get_orchestrator` dependency — never a module-level global.
