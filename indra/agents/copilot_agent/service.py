"""Copilot service that routes questions and always returns grounded answers."""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from indra.agents.copilot_agent.classifier import QueryClassifier
from indra.agents.copilot_agent.explainer import AnswerExplainer
from indra.agents.copilot_agent.handlers.base import BaseHandler, HandlerContext
from indra.agents.copilot_agent.handlers.diagnostic import DiagnosticHandler
from indra.core.contracts import ComplianceService, KnowledgeGraphService, ProactiveService
from indra.core.deps import AgentDeps
from indra.core.events import Event, Topic
from indra.core.exceptions import AgentError, IndraError
from indra.core.logging import get_logger
from indra.core.models import Answer, QueryRequest, QueryType

if TYPE_CHECKING:  # pragma: no cover - type-only public contract
    pass

logger = get_logger(__name__)


class _GenericHandler(BaseHandler):
    """Grounded handler reused by non-diagnostic query types until specialised handlers are added."""

    def __init__(self, ctx: HandlerContext, query_type: QueryType) -> None:
        super().__init__(ctx)
        self.query_type = query_type


class CopilotAgent:
    """Classify, retrieve, answer, and expose a streamed response surface."""

    name = "copilot_agent"

    def __init__(self, deps: AgentDeps) -> None:
        self._deps = deps
        self._classifier = QueryClassifier(deps.settings, deps.llm)
        self._context = HandlerContext(deps=deps, explainer=AnswerExplainer(deps.settings))
        self._handlers: dict[QueryType, BaseHandler] = {
            QueryType.DIAGNOSTIC: DiagnosticHandler(self._context),
            **{kind: _GenericHandler(self._context, kind) for kind in QueryType if kind is not QueryType.DIAGNOSTIC},
        }

    def bind(
        self,
        *,
        knowledge_graph: KnowledgeGraphService,
        proactive: ProactiveService,
        compliance: ComplianceService,
    ) -> None:
        """Make sibling services available through the explicit orchestrator boundary."""
        self._context.knowledge_graph = knowledge_graph
        self._context.proactive = proactive
        self._context.compliance = compliance

    async def startup(self) -> None:
        """The copilot is stateless apart from its handlers; no warm-up is required."""

    async def shutdown(self) -> None:
        """Lifecycle hook for the shared agent contract."""

    async def health(self) -> dict[str, object]:
        return {
            "ok": self._context.knowledge_graph is not None,
            "backend": "graphrag",
            "detail": f"{len(self._handlers)} query handlers; graph bound={self._context.knowledge_graph is not None}",
        }

    async def classify(self, query: str) -> str:
        """Return the stable string form used by the public API."""
        return (await self._classifier.classify(query)).query_type.value

    async def answer(self, request: QueryRequest) -> Answer:
        """Build one fully explained answer grounded in GraphRAG evidence."""
        if self._context.knowledge_graph is None:
            raise AgentError("Copilot agent is not bound to knowledge graph service. Start through IndraOrchestrator.")
        try:
            classified = await self._classifier.classify(request.query)
            query_type = request.query_type or classified.query_type
            equipment_tag = request.equipment_tag or (classified.equipment_tags[0] if classified.equipment_tags else None)
            retrieval = await self._context.knowledge_graph.retrieve(
                request.query,
                top_k=request.max_sources,
                equipment_tag=equipment_tag,
            )
            handler = self._handlers[query_type]
            answer = await handler.handle(request.model_copy(update={"query_type": query_type, "equipment_tag": equipment_tag}), retrieval=retrieval)
            await self._publish(Topic.QUERY_ANSWERED, answer_id=answer.answer_id, query_type=query_type.value, confidence=answer.confidence.value, sources=len(answer.sources))
            return answer
        except IndraError:
            raise
        except Exception as exc:
            raise AgentError(
                "INDRA could not assemble a copilot answer. Check the GraphRAG retrieval health and retry; the query was not discarded.",
                context={"query": request.query[:200]},
                cause=exc,
            ) from exc

    async def stream_answer(self, request: QueryRequest) -> AsyncIterator[str]:
        """Yield a final grounded answer in readable chunks.

        The normal handler needs the completed answer to validate citations, so this deliberately
        streams the verified result rather than leaking ungrounded model tokens to a technician.
        """
        answer = await self.answer(request)
        fragments = [part.strip() for part in re.split(r"(?<=[.!?])\s+", answer.answer_text) if part.strip()]
        for fragment in fragments or [answer.answer_text]:
            yield fragment + (" " if not fragment.endswith(" ") else "")
            await asyncio.sleep(0)

    async def _publish(self, topic: Topic, **payload: object) -> None:
        event = Event.make(topic, source=self.name, **payload)
        await self._deps.events.publish(event.topic.value, event.model_dump(mode="json"))


__all__ = ["CopilotAgent"]
