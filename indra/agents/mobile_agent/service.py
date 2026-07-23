"""Field-facing mobile service composed from voice, photo, and offline modules."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from indra.agents.mobile_agent.offline_bundle import OfflineBundleBuilder
from indra.agents.mobile_agent.photo_query import PhotoQueryEngine
from indra.agents.mobile_agent.stt import build_stt
from indra.agents.mobile_agent.translation import ScriptTranslator
from indra.agents.mobile_agent.tts import build_tts
from indra.agents.mobile_agent.voice import VoicePipeline
from indra.core.contracts import CopilotService, KnowledgeGraphService, ProactiveService
from indra.core.deps import AgentDeps
from indra.core.events import Event, Topic
from indra.core.logging import get_logger
from indra.core.models import PhotoQueryResponse, VoiceQueryResponse

if TYPE_CHECKING:  # pragma: no cover - interface only
    pass

logger = get_logger(__name__)


class MobileAgent:
    """Provide hands-free, photo-to-query, and offline-ready technician workflows."""

    name = "mobile_agent"

    def __init__(self, deps: AgentDeps) -> None:
        self._deps = deps
        self._translator = ScriptTranslator(deps.settings, deps.llm, cache=deps.cache)
        self._voice = VoicePipeline(
            settings=deps.settings,
            stt=build_stt(deps.settings),
            tts=build_tts(deps.settings),
            translator=self._translator,
        )
        self._photo = PhotoQueryEngine(deps)
        self._offline = OfflineBundleBuilder(deps)

    def bind(
        self,
        *,
        copilot: CopilotService,
        knowledge_graph: KnowledgeGraphService,
        proactive: ProactiveService,
    ) -> None:
        """Connect sibling services through their protocols only."""
        self._voice.bind(copilot=copilot)
        self._photo.bind(knowledge_graph=knowledge_graph, proactive=proactive)

    async def startup(self) -> None:
        """Preload the small equipment registry used by photo queries."""
        try:
            await self._photo.warm()
        except Exception as exc:
            logger.warning("mobile photo registry warm-up degraded", extra={"error": str(exc)})

    async def shutdown(self) -> None:
        """All mobile collaborators are pure/in-memory and need no explicit close."""

    async def health(self) -> dict[str, object]:
        return {
            "ok": self._voice.has_copilot,
            "backend": "field_pipeline",
            "detail": f"photo_ocr={self._photo.ocr_backend}; copilot_bound={self._voice.has_copilot}",
        }

    async def voice_query(
        self,
        audio: bytes,
        *,
        language_hint: str | None = None,
        equipment_tag: str | None = None,
    ) -> VoiceQueryResponse:
        """Run speech recognition, tag-safe translation, copilot reasoning, and speech synthesis."""
        return await self._voice.run(audio, language_hint=language_hint, equipment_tag=equipment_tag)

    async def photo_query(self, image: bytes) -> PhotoQueryResponse:
        """Resolve a tag photograph into an AR-style operational overlay."""
        return await self._photo.run(image)

    async def build_offline_bundle(self, *, budget_bytes: int | None = None) -> dict[str, object]:
        """Create a priority-ordered offline payload and return its full serialisable form."""
        built = await self._offline.build(budget_bytes=budget_bytes)
        return built.payload()

    async def sync(self, items: Sequence[dict[str, object]]) -> dict[str, object]:
        """Persist offline actions then acknowledge their replay in the event stream."""
        accepted = 0
        for item in items:
            await self._deps.metadata.enqueue_sync(dict(item))
            accepted += 1
        drained = await self._deps.metadata.drain_sync(limit=max(accepted, 1))
        await self._publish(Topic.OFFLINE_SYNCED, accepted=accepted, synced=len(drained))
        return {"accepted": accepted, "synced": len(drained), "items": drained}

    async def _publish(self, topic: Topic, **payload: object) -> None:
        event = Event.make(topic, source=self.name, **payload)
        await self._deps.events.publish(event.topic.value, event.model_dump(mode="json"))


__all__ = ["MobileAgent"]
