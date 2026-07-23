"""The dependency container every agent receives.

One container, one constructor shape:

    class IngestionAgent:
        def __init__(self, deps: AgentDeps) -> None:
            self.deps = deps

Agents pull what they need off ``deps`` and nothing else. This is what makes the orchestrator's
wiring a single line per agent, keeps every agent independently testable against fakes, and means
swapping Neo4j for the in-memory graph store touches exactly one file.

**No agent constructs its own store or provider.** If it needs something that is not on this
container, that is a contract change, not a local workaround.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from indra.core.config import Settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from indra.core.contracts import (
        BlobStore,
        CacheStore,
        EventBus,
        GraphStore,
        LLMRouter,
        MetadataStore,
        VectorStore,
    )


@dataclass(frozen=True, slots=True)
class AgentDeps:
    """Everything an INDRA agent is allowed to depend on.

    Frozen: an agent may not swap a store out from under its siblings at runtime.
    """

    settings: Settings
    llm: LLMRouter
    graph: GraphStore
    vectors: VectorStore
    metadata: MetadataStore
    blobs: BlobStore
    events: EventBus
    cache: CacheStore

    #: Backend actually bound per store, e.g. ``{"graph": "memory", "vectors": "chroma"}``.
    #: Surfaced by ``/health`` so the ops panel never shows a green tick for a fallback.
    bound_backends: dict[str, str] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        """Summary for health checks and startup logs."""
        return {
            "environment": self.settings.environment.value,
            "backends": dict(self.bound_backends),
            "deterministic": self.settings.deterministic,
            "offline": self.settings.offline_mode,
        }


__all__ = ["AgentDeps"]
