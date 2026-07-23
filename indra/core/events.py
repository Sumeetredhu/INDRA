"""Typed event vocabulary for inter-agent communication (see ``docs/DECISIONS.md`` D8).

Agents never import each other. They publish facts and react to facts. This module defines the
topics and payloads; ``indra/storage/event_bus.py`` provides the Redis-Streams and in-memory
transports behind :class:`indra.core.contracts.EventBus`.

Choreography in the running system::

    ingestion  --document.ingested-->  knowledge_graph
    knowledge_graph  --graph.updated-->  proactive, compliance
    proactive  --alert.raised-->  api (SSE to UI), mobile (push)
    compliance --gap.detected-->  proactive (feeds the regulatory compound-signal rule)
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import Field

from indra.core.ids import get_correlation_id, new_id
from indra.core.models import IndraModel, Severity, utcnow


class Topic(str, Enum):
    """Every event channel in INDRA. Adding a topic means adding a member here first."""

    DOCUMENT_RECEIVED = "document.received"
    DOCUMENT_INGESTED = "document.ingested"
    DOCUMENT_FAILED = "document.failed"
    INGESTION_PROGRESS = "ingestion.progress"
    GRAPH_UPDATED = "graph.updated"
    QUERY_ANSWERED = "query.answered"
    ALERT_RAISED = "alert.raised"
    ALERT_ACKNOWLEDGED = "alert.acknowledged"
    GAP_DETECTED = "compliance.gap_detected"
    KNOWLEDGE_CLIFF_DETECTED = "knowledge.cliff_detected"
    OFFLINE_SYNCED = "mobile.offline_synced"
    AGENT_HEALTH = "agent.health"


class Event(IndraModel):
    """Envelope for everything on the bus.

    ``correlation_id`` is captured at publish time from the ambient context, so a document upload
    and the alert it eventually triggers share one traceable id across six agents.
    """

    event_id: str = Field(default_factory=lambda: new_id("event"))
    topic: Topic
    source_agent: str
    correlation_id: str = Field(default_factory=get_correlation_id)
    occurred_at: datetime = Field(default_factory=utcnow)
    payload: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def make(cls, topic: Topic, *, source: str, **payload: Any) -> Event:
        """Convenience constructor: ``Event.make(Topic.GRAPH_UPDATED, source="kg", nodes=12)``."""
        return cls(topic=topic, source_agent=source, payload=payload)


# --------------------------------------------------------------------------------------
# Payload shapes — documented contracts, validated where it matters
# --------------------------------------------------------------------------------------


class DocumentIngestedPayload(IndraModel):
    """Emitted once a document is parsed, chunked, embedded and queued for the graph."""

    document_id: str
    title: str
    document_type: str
    chunks: int
    entities: int
    relationships: int
    is_pid: bool = False
    duplicate_of: str | None = None


class GraphUpdatedPayload(IndraModel):
    """Emitted after graph writes land. Triggers proactive rescans of the touched assets."""

    document_id: str | None = None
    nodes_written: int = 0
    relationships_written: int = 0
    affected_tags: list[str] = Field(default_factory=list)


class AlertRaisedPayload(IndraModel):
    """Emitted when a compound signal crosses the surfacing threshold."""

    alert_id: str
    equipment_tag: str
    severity: Severity
    title: str
    risk_percent: float = 0.0
    signal_count: int = 0
    rule_id: str | None = None


class GapDetectedPayload(IndraModel):
    """Emitted per compliance gap. Feeds the regulatory-deadline compound-signal rule."""

    gap_id: str
    equipment_tag: str
    regulation: str
    clause: str
    status: str
    severity: Severity
    deadline: str | None = None


class KnowledgeCliffPayload(IndraModel):
    """Emitted when an asset's knowledge-risk score crosses the critical threshold."""

    equipment_tag: str
    score: float
    experts: list[str] = Field(default_factory=list)
    document_count: int = 0


class AgentHealthPayload(IndraModel):
    """Periodic heartbeat, surfaced on the ops panel of the dashboard."""

    agent: str
    status: Literal["ok", "degraded", "down"]
    backend: str = ""
    detail: str = ""


__all__ = [
    "AgentHealthPayload",
    "AlertRaisedPayload",
    "DocumentIngestedPayload",
    "Event",
    "GapDetectedPayload",
    "GraphUpdatedPayload",
    "KnowledgeCliffPayload",
    "Topic",
]
