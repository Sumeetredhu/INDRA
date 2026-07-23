"""Single owner of agent construction, lifecycle, and explicit cross-agent bindings."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from fastapi import Request

from indra.agents.compliance_agent.service import ComplianceAgent
from indra.agents.copilot_agent.service import CopilotAgent
from indra.agents.ingestion_agent.service import IngestionAgent
from indra.agents.knowledge_graph_agent.service import KnowledgeGraphAgent
from indra.agents.mobile_agent.service import MobileAgent
from indra.agents.proactive_intelligence_agent.service import ProactiveIntelligenceAgent
from indra.core.config import Settings, get_settings
from indra.core.deps import AgentDeps
from indra.core.logging import configure_logging, get_logger
from indra.llm.router import build_router
from indra.storage.factory import StoreBundle, build_stores

if TYPE_CHECKING:  # pragma: no cover - type aliases only
    pass

logger = get_logger(__name__)


class IndraOrchestrator:
    """Build each independent agent once and coordinate its lifecycle safely."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._stores: StoreBundle | None = None
        self._ingestion: IngestionAgent | None = None
        self._knowledge_graph: KnowledgeGraphAgent | None = None
        self._copilot: CopilotAgent | None = None
        self._proactive: ProactiveIntelligenceAgent | None = None
        self._mobile: MobileAgent | None = None
        self._compliance: ComplianceAgent | None = None
        self._started = False
        self._lock = asyncio.Lock()

    async def startup(self) -> None:
        """Bind stores, construct agents, wire protocols, then start all capabilities."""
        async with self._lock:
            if self._started:
                return
            configure_logging(level=self._settings.log_level, json_output=self._settings.log_json)
            stores = await build_stores(self._settings)
            router = build_router(self._settings)
            deps = AgentDeps(
                settings=self._settings,
                llm=router,
                graph=stores.graph,
                vectors=stores.vectors,
                metadata=stores.metadata,
                blobs=stores.blobs,
                events=stores.events,
                cache=stores.cache,
                bound_backends=stores.bound_backends,
            )
            self._stores = stores
            self._knowledge_graph = KnowledgeGraphAgent(deps)
            self._ingestion = IngestionAgent(deps)
            self._proactive = ProactiveIntelligenceAgent(deps)
            self._compliance = ComplianceAgent(deps)
            self._copilot = CopilotAgent(deps)
            self._mobile = MobileAgent(deps)

            self._ingestion.bind(knowledge_graph=self._knowledge_graph)
            self._proactive.bind(knowledge_graph=self._knowledge_graph)
            self._compliance.bind(knowledge_graph=self._knowledge_graph)
            self._copilot.bind(
                knowledge_graph=self._knowledge_graph,
                proactive=self._proactive,
                compliance=self._compliance,
            )
            self._mobile.bind(
                copilot=self._copilot,
                knowledge_graph=self._knowledge_graph,
                proactive=self._proactive,
            )
            agents = (
                self._knowledge_graph,
                self._compliance,
                self._proactive,
                self._copilot,
                self._ingestion,
                self._mobile,
            )
            try:
                for agent in agents:
                    await agent.startup()
            except Exception:
                await stores.close()
                self._stores = None
                raise
            self._started = True
            logger.info("INDRA orchestrator started", extra={"backends": stores.bound_backends})

    async def shutdown(self) -> None:
        """Stop agents in reverse dependency order and close every backend exactly once."""
        async with self._lock:
            if not self._started:
                return
            agents = (self._mobile, self._ingestion, self._copilot, self._proactive, self._compliance, self._knowledge_graph)
            for agent in agents:
                if agent is None:
                    continue
                try:
                    await agent.shutdown()
                except Exception as exc:
                    logger.warning("agent shutdown failed", extra={"agent": agent.name, "error": str(exc)})
            if self._stores is not None:
                await self._stores.close()
            self._started = False
            logger.info("INDRA orchestrator stopped")

    async def health(self) -> dict[str, object]:
        """Return liveness plus store and agent readiness without allowing health checks to raise."""
        if not self._started or self._stores is None:
            return {"ok": False, "detail": "orchestrator has not started"}
        agents = {
            "ingestion": self.ingestion,
            "knowledge_graph": self.knowledge_graph,
            "copilot": self.copilot,
            "proactive": self.proactive,
            "mobile": self.mobile,
            "compliance": self.compliance,
        }
        reports = await asyncio.gather(*(agent.health() for agent in agents.values()), return_exceptions=True)
        agent_health = {
            name: report if isinstance(report, dict) else {"ok": False, "detail": str(report)}
            for name, report in zip(agents, reports, strict=True)
        }
        stores = await self._stores.health()
        ok = all(bool(report.get("ok")) for report in agent_health.values()) and all(
            bool(report.get("ok")) for report in stores.values()
        )
        return {"ok": ok, "settings": {"environment": self._settings.environment.value, "deterministic": self._settings.deterministic}, "stores": stores, "agents": agent_health}

    @property
    def ingestion(self) -> IngestionAgent:
        return self._require(self._ingestion, "ingestion")

    @property
    def knowledge_graph(self) -> KnowledgeGraphAgent:
        return self._require(self._knowledge_graph, "knowledge graph")

    @property
    def copilot(self) -> CopilotAgent:
        return self._require(self._copilot, "copilot")

    @property
    def proactive(self) -> ProactiveIntelligenceAgent:
        return self._require(self._proactive, "proactive intelligence")

    @property
    def mobile(self) -> MobileAgent:
        return self._require(self._mobile, "mobile")

    @property
    def compliance(self) -> ComplianceAgent:
        return self._require(self._compliance, "compliance")

    @staticmethod
    def _require(value: object | None, label: str):  # type: ignore[no-untyped-def]
        if value is None:
            raise RuntimeError(f"INDRA {label} agent was requested before orchestrator startup.")
        return value


def get_orchestrator(request: Request) -> IndraOrchestrator:
    """FastAPI dependency that retrieves the lifespan-owned orchestrator instance."""
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if not isinstance(orchestrator, IndraOrchestrator):
        raise RuntimeError("INDRA orchestrator is not attached to this application.")
    return orchestrator


__all__ = ["IndraOrchestrator", "get_orchestrator"]
