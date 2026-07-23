"""Proactive intelligence service: signal scans, durable alerts, and knowledge-risk scoring."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING

from indra.agents.proactive_intelligence_agent.knowledge_cliff import KnowledgeCliffAnalyzer
from indra.agents.proactive_intelligence_agent.rules import evaluate_rules, make_context
from indra.agents.proactive_intelligence_agent.scoring import (
    build_alert,
    build_compound_signal,
    collect_sources,
    headline_signal,
)
from indra.agents.proactive_intelligence_agent.signals import GapLedger, SignalCollector, detect_all
from indra.core.contracts import KnowledgeGraphService
from indra.core.deps import AgentDeps
from indra.core.events import Event, Topic
from indra.core.exceptions import AgentError, IndraError
from indra.core.logging import get_logger
from indra.core.models import (
    Alert,
    Confidence,
    FailurePrediction,
    KnowledgeCliffScore,
    Person,
    RecommendedAction,
    Severity,
)

if TYPE_CHECKING:  # pragma: no cover - interface only
    pass

logger = get_logger(__name__)


class ProactiveIntelligenceAgent:
    """Surface compound risks without requiring a user to formulate a query."""

    name = "proactive_intelligence_agent"

    def __init__(self, deps: AgentDeps) -> None:
        self._deps = deps
        self._gaps = GapLedger()
        self._collector = SignalCollector(graph=deps.graph, vectors=deps.vectors, settings=deps.settings, gaps=self._gaps)
        self._cliffs = KnowledgeCliffAnalyzer(graph=deps.graph, settings=deps.settings, llm=deps.llm)
        self._knowledge_graph: KnowledgeGraphService | None = None

    def bind(self, *, knowledge_graph: KnowledgeGraphService) -> None:
        """Keep the cross-agent reference explicit even though scans read storage directly."""
        self._knowledge_graph = knowledge_graph

    async def startup(self) -> None:
        """Subscribe to graph and compliance events that make cached proactive state stale."""
        await self._deps.events.subscribe(Topic.GRAPH_UPDATED.value, self._on_graph_updated)
        await self._deps.events.subscribe(Topic.GAP_DETECTED.value, self._on_gap_detected)

    async def shutdown(self) -> None:
        """Event transport owns consumer shutdown; this service keeps only memory state."""

    async def health(self) -> dict[str, object]:
        return {
            "ok": self._knowledge_graph is not None,
            "backend": "rule_engine",
            "detail": f"gap ledger holds {self._gaps.size()} regulatory gap(s)",
        }

    async def scan(self, *, equipment_tags: Sequence[str] | None = None) -> list[object]:
        """Evaluate every declarative compound-signal rule for the selected fleet."""
        try:
            fleet = await self._collector.fleet_index(force_refresh=True)
            wanted = {tag.strip().upper() for tag in equipment_tags or ()}
            selected = [asset for asset in fleet.equipment if not wanted or asset.tag.upper() in wanted]
            compounds: list[object] = []
            for asset in selected:
                snapshot = await self._collector.snapshot(asset)
                signals = await asyncio.to_thread(detect_all, snapshot, fleet, settings=self._deps.settings)
                for match in evaluate_rules(make_context(asset, signals, settings=self._deps.settings)):
                    compound, _ = build_compound_signal(match, asset)
                    compounds.append(compound)
                    alert = build_alert(compound, match, asset, settings=self._deps.settings)
                    previous = await self._deps.metadata.find_alert_by_dedupe_key(
                        alert.dedupe_key, within_seconds=self._deps.settings.alert_dedupe_window_s
                    )
                    if previous is None:
                        await self._deps.metadata.save_alert(alert)
                        await self._publish(
                            Topic.ALERT_RAISED,
                            alert_id=alert.alert_id,
                            equipment_tag=alert.equipment_tag,
                            severity=alert.severity.value,
                            title=alert.title,
                            risk_percent=alert.risk_percent,
                            signal_count=len(compound.signals),
                            rule_id=compound.rule_id,
                        )
            logger.info("proactive scan complete", extra={"assets": len(selected), "compound_signals": len(compounds)})
            return compounds
        except IndraError:
            raise
        except Exception as exc:
            raise AgentError(
                "Proactive scan could not complete. Existing alerts remain available; check graph and vector health before retrying.",
                cause=exc,
            ) from exc

    async def alerts(self, *, unresolved_only: bool = True) -> list[Alert]:
        """Return persisted alerts in severity-first order."""
        alerts = await self._deps.metadata.list_alerts(unresolved_only=unresolved_only)
        return sorted(alerts, key=lambda item: (-item.severity.rank, -item.risk_percent, item.raised_at), reverse=False)

    async def predict(self, tag: str, *, horizon_days: int = 30) -> FailurePrediction:
        """Project current compound signals into a transparent near-term risk forecast."""
        compounds = await self.scan(equipment_tags=[tag])
        if not compounds:
            return FailurePrediction(
                equipment_tag=tag.upper(),
                failure_mode="no current compound pattern detected",
                probability=0.0,
                horizon_days=horizon_days,
                confidence=Confidence(value=0.35, rationale="No compound signals matched the available evidence", method="aggregate"),
            )
        ranked = sorted(compounds, key=lambda item: float(getattr(item, "risk_score", 0.0)), reverse=True)
        top = ranked[0]
        signals = list(getattr(top, "signals", []))
        lead = headline_signal(signals)
        actions = [RecommendedAction(
            action=f"Inspect {tag.upper()} against the evidence behind its active compound signal.",
            urgency=getattr(top, "severity", Severity.WARNING),
            owner_role="Reliability Engineer",
            due_within_days=1,
            rationale="Current compound-risk prediction",
        )]
        return FailurePrediction(
            equipment_tag=tag.upper(),
            failure_mode=lead.description if lead is not None else "compound equipment risk",
            probability=float(getattr(top, "risk_score", 0.0)),
            horizon_days=horizon_days,
            drivers=[signal.description for signal in signals],
            confidence=getattr(top, "confidence", Confidence(value=0.5, rationale="Rule-based risk projection", method="heuristic")),
            recommended_actions=actions,
            sources=collect_sources(signals),
        )

    async def knowledge_cliff(self, *, tags: Sequence[str] | None = None) -> list[KnowledgeCliffScore]:
        """Score equipment knowledge risk and publish critical findings."""
        equipment = await self._deps.graph.list_equipment()
        wanted = {tag.strip().upper() for tag in tags or ()}
        selected = [item for item in equipment if not wanted or item.tag.upper() in wanted]
        scores = await self._cliffs.score_many(selected)
        for score in scores:
            if score.severity is Severity.CRITICAL:
                await self._publish(
                    Topic.KNOWLEDGE_CLIFF_DETECTED,
                    equipment_tag=score.equipment_tag,
                    score=score.score,
                    experts=[person.name for person in score.retiring_experts],
                    document_count=score.document_count,
                )
        return scores

    async def interview_questions(self, tag: str, *, person: Person | None = None) -> list[str]:
        """Create asset-specific knowledge-capture questions for a departing expert."""
        equipment = await self._deps.graph.get_equipment(tag)
        if equipment is None:
            raise AgentError(f"Cannot draft interview questions: equipment {tag.upper()} is not in the graph.")
        return await self._cliffs.interview_questions(equipment, person=person)

    async def _on_graph_updated(self, record: dict[str, object]) -> None:
        body = record.get("payload")
        tags = body.get("affected_tags") if isinstance(body, dict) else None
        selected = [str(item) for item in tags] if isinstance(tags, list) else None
        self._collector.invalidate_fleet()
        try:
            await self.scan(equipment_tags=selected)
        except Exception as exc:
            logger.warning("event-triggered proactive scan failed", extra={"error": str(exc)})

    async def _on_gap_detected(self, record: dict[str, object]) -> None:
        self._gaps.record(record)

    async def _publish(self, topic: Topic, **payload: object) -> None:
        event = Event.make(topic, source=self.name, **payload)
        await self._deps.events.publish(event.topic.value, event.model_dump(mode="json"))


__all__ = ["ProactiveIntelligenceAgent"]
