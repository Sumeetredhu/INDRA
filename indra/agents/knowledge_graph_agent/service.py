"""Knowledge graph service: durable indexing, hybrid retrieval, and graph previews."""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

from indra.agents.knowledge_graph_agent.entity_linking import EntityResolver
from indra.agents.knowledge_graph_agent.graphrag import GraphRAGRetriever
from indra.core.deps import AgentDeps
from indra.core.events import Event, Topic
from indra.core.exceptions import AgentError, IndraError
from indra.core.logging import get_logger
from indra.core.models import (
    ConditionReading,
    Confidence,
    Criticality,
    DocumentType,
    Equipment,
    FailureEvent,
    IngestionStage,
    MaintenanceRecord,
    ParsedDocument,
    Person,
    Procedure,
    RetrievalResult,
    SourceRef,
)

if TYPE_CHECKING:  # pragma: no cover - static contract declaration
    pass

logger = get_logger(__name__)

_PERCENT_RE = re.compile(r"\b(?:bearing\s+wear|wear)\D{0,20}(\d{1,3}(?:\.\d+)?)\s*%", re.IGNORECASE)
_VIBRATION_RE = re.compile(r"\bvibration\D{0,20}(\d+(?:\.\d+)?)\s*(?:mm/?s|mm/s)?", re.IGNORECASE)
_THRESHOLD_RE = re.compile(r"\b(?:replace(?:ment)?\s+(?:at|above)|threshold(?:\s+of)?|limit(?:\s+of)?)\D{0,30}(\d{1,3}(?:\.\d+)?)\s*%", re.IGNORECASE)
_RETIREMENT_RE = re.compile(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+).*?\bretir(?:e|ing|ement).*?(20\d{2})", re.IGNORECASE | re.DOTALL)
_DOWNTIME_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)\s+(?:of\s+)?downtime", re.IGNORECASE)
_COST_RE = re.compile(r"(?:₹|rs\.?|inr)\s*([\d,.]+)\s*(?:lakh|l|lac)?", re.IGNORECASE)


class KnowledgeGraphAgent:
    """Own graph writes and GraphRAG without importing any sibling agent internals."""

    name = "knowledge_graph_agent"

    def __init__(self, deps: AgentDeps) -> None:
        self._deps = deps
        self._resolver = EntityResolver(deps.settings)
        self._retriever = GraphRAGRetriever(deps, self._resolver)

    async def startup(self) -> None:
        """Ensure graph constraints and prime entity resolution from persistent graph state."""
        await self._deps.graph.ensure_schema()
        try:
            equipment = await self._deps.graph.list_equipment()
            self._resolver.seed_registry(item.tag for item in equipment)
        except Exception as exc:
            logger.warning("graph registry warm-up degraded", extra={"error": str(exc)})

    async def shutdown(self) -> None:
        """No owned connection; stores are closed by the orchestrator."""

    async def health(self) -> dict[str, object]:
        stats = await self._deps.graph.stats()
        return {
            "ok": True,
            "backend": self._deps.bound_backends.get("graph", "unknown"),
            "detail": f"resolved equipment registry: {len(self._resolver.registry)} tags",
            "stats": stats,
        }

    async def index(self, parsed: ParsedDocument) -> dict[str, int]:
        """Index parsed content in graph and vector stores, retaining source provenance."""
        meta = parsed.meta
        try:
            await self._deps.graph.upsert_document(meta)
            await self._deps.metadata.save_document(meta)
            link = self._resolver.resolve_batch(parsed.entities)
            remapped = self._resolver.remap_relationships(parsed.relationships, link.key_map)

            chunks = list(parsed.chunks)
            embeddings: list[list[float]] = []
            if chunks:
                embeddings = await self._deps.llm.embed([chunk.text for chunk in chunks], task="document")
                for chunk, vector in zip(chunks, embeddings, strict=True):
                    chunk.embedding = vector
                chunks_written = await self._deps.vectors.upsert(chunks, embeddings=embeddings)
            else:
                chunks_written = 0
            entities_written = await self._deps.graph.upsert_entities(link.entities)
            relationships_written = await self._deps.graph.upsert_relationships(remapped)

            equipment, people, maintenance, failures, procedures, readings = self._derive_records(parsed, link.entities)
            if equipment:
                await self._deps.graph.upsert_equipment(equipment)
                self._resolver.seed_registry(item.tag for item in equipment)
            if people:
                await self._deps.graph.upsert_people(people)
            if maintenance:
                await self._deps.graph.upsert_maintenance(maintenance)
            if failures:
                await self._deps.graph.upsert_failures(failures)
            if procedures:
                await self._deps.graph.upsert_procedures(procedures)
            if readings:
                await self._deps.graph.upsert_readings(readings)

            parsed.stage = IngestionStage.GRAPH_WRITTEN
            counts = {
                "chunks": chunks_written,
                "entities": entities_written,
                "relationships": relationships_written,
                "equipment": len(equipment),
                "people": len(people),
                "maintenance": len(maintenance),
                "failures": len(failures),
                "procedures": len(procedures),
                "readings": len(readings),
            }
            await self._publish(Topic.GRAPH_UPDATED, document_id=meta.document_id, nodes_written=entities_written, relationships_written=relationships_written, affected_tags=[item.tag for item in equipment])
            logger.info("document indexed", extra={"document_id": meta.document_id, **counts})
            return counts
        except IndraError:
            raise
        except Exception as exc:
            raise AgentError(
                f"Could not index {meta.filename} into the knowledge graph. The raw document remains stored; retry ingestion after checking graph and vector backends.",
                context={"document_id": meta.document_id},
                cause=exc,
            ) from exc

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        equipment_tag: str | None = None,
        max_hops: int | None = None,
        filters: dict[str, object] | None = None,
    ) -> RetrievalResult:
        """Return evidence ranked by vector similarity and graph proximity."""
        return await self._retriever.retrieve(
            query,
            top_k=top_k,
            equipment_tag=equipment_tag,
            max_hops=max_hops,
            filters=filters,
        )

    async def resolve_entities(self, text: str) -> list[str]:
        """Resolve plant tags and previously learned aliases in free text."""
        return self._resolver.resolve_text(text)

    async def graph_preview(
        self, entity_keys: Sequence[str], *, hops: int = 2, limit: int = 60
    ) -> dict[str, object]:
        """Shape a bounded graph neighbourhood into a React-Flow-compatible projection."""
        nodes: dict[str, dict[str, object]] = {}
        edges: dict[str, dict[str, object]] = {}
        for key in entity_keys:
            paths = await self._deps.graph.neighbours(key, hops=max(1, hops), limit=limit)
            for path in paths:
                for node_key in path.nodes:
                    nodes.setdefault(node_key, {
                        "id": node_key,
                        "label": node_key.split(":", 1)[-1],
                        "type": node_key.split(":", 1)[0].lower(),
                    })
                for index, relation in enumerate(path.relations):
                    if index + 1 >= len(path.nodes):
                        continue
                    source, target = path.nodes[index], path.nodes[index + 1]
                    edge_id = f"{source}|{relation.value}|{target}"
                    edges.setdefault(edge_id, {"id": edge_id, "source": source, "target": target, "label": relation.value, "confidence": path.confidence})
        return {"nodes": list(nodes.values())[:limit], "edges": list(edges.values())[:limit]}

    def _derive_records(
        self, parsed: ParsedDocument, entities: Sequence[object]
    ) -> tuple[list[Equipment], list[Person], list[MaintenanceRecord], list[FailureEvent], list[Procedure], list[ConditionReading]]:
        """Project grounded text facts into the structured records used by other agents.

        This is deliberately conservative: text-derived records retain their source citation and
        no value is created unless a recognisable document type plus an equipment tag exists.
        """
        tags = sorted({str(getattr(entity, "canonical_name", None) or getattr(entity, "name", "")).upper() for entity in entities if getattr(getattr(entity, "type", None), "value", None) == "Equipment"})
        source = self._source(parsed)
        text = parsed.text
        equipment: list[Equipment] = []
        for tag in tags:
            thresholds: dict[str, float] = {}
            threshold = _THRESHOLD_RE.search(text)
            if threshold:
                thresholds["bearing_wear_pct"] = float(threshold.group(1))
            criticality = Criticality.A if re.search(rf"{re.escape(tag)}[^.\n]{{0,120}}criticality\s*[:=-]?\s*A\b", text, re.IGNORECASE) else Criticality.C
            equipment.append(Equipment(tag=tag, name=tag, equipment_type=self._equipment_type(tag), criticality=criticality, oem_thresholds=thresholds))

        people: list[Person] = []
        for name, year in _RETIREMENT_RE.findall(text):
            retirement = date(int(year), 3, 31)
            people.append(Person(name=name, retirement_date=retirement, expertise_tags=tags, years_experience=self._years_experience(text, name)))

        readings = self._readings(tags, text, source)
        maintenance: list[MaintenanceRecord] = []
        failures: list[FailureEvent] = []
        procedures: list[Procedure] = []
        record_date = parsed.meta.document_date or datetime.now(UTC).date()
        if parsed.meta.document_type in {DocumentType.WORK_ORDER, DocumentType.INSPECTION_REPORT}:
            record_type = "inspection" if parsed.meta.document_type is DocumentType.INSPECTION_REPORT else "work_order"
            for tag in tags:
                tagged_readings = [item for item in readings if item.equipment_tag == tag]
                maintenance.append(MaintenanceRecord(
                    equipment_tag=tag,
                    record_type=record_type,
                    performed_on=record_date,
                    findings=text[:3000],
                    recommendations="Inspect and correct the documented condition before return to service.",
                    readings=tagged_readings,
                    status="open" if "open" in text.lower() or "recommend" in text.lower() else "closed",
                    sources=[source],
                ))
        if parsed.meta.document_type in {DocumentType.INCIDENT_REPORT, DocumentType.ROOT_CAUSE_ANALYSIS}:
            mode = self._failure_mode(text)
            for tag in tags:
                failures.append(FailureEvent(
                    equipment_tag=tag,
                    failure_mode=mode,
                    occurred_on=record_date,
                    root_cause=self._root_cause(text),
                    downtime_hours=self._number(_DOWNTIME_RE, text),
                    cost_inr=self._cost(text),
                    precursor_text=text[:2500],
                    sources=[source],
                ))
        if parsed.meta.document_type is DocumentType.SOP:
            steps = [sentence.strip(" -•\t") for sentence in re.split(r"\n+|(?<=[.;])\s+(?=\d+[.)])", text) if len(sentence.strip()) > 12][:20]
            for tag in tags:
                procedures.append(Procedure(title=parsed.meta.title, applies_to=[tag], steps=steps, sources=[source]))
        return equipment, people, maintenance, failures, procedures, readings

    @staticmethod
    def _equipment_type(tag: str) -> str:
        from indra.agents.ingestion_agent.tag_normalizer import equipment_type_for

        return equipment_type_for(tag)

    @staticmethod
    def _source(parsed: ParsedDocument) -> SourceRef:
        if parsed.chunks:
            return parsed.chunks[0].to_source_ref(parsed.meta, relevance=1.0, retrieved_via="direct")
        return SourceRef(document_id=parsed.meta.document_id, document_title=parsed.meta.title, document_type=parsed.meta.document_type, relevance=1.0, retrieved_via="direct", document_date=parsed.meta.document_date)

    @staticmethod
    def _readings(tags: Sequence[str], text: str, source: SourceRef) -> list[ConditionReading]:
        readings: list[ConditionReading] = []
        moment = datetime.combine(source.document_date or datetime.now(UTC).date(), datetime.min.time(), tzinfo=UTC)
        for tag in tags:
            for match in _PERCENT_RE.finditer(text):
                readings.append(ConditionReading(equipment_tag=tag, parameter="bearing_wear_pct", value=float(match.group(1)), unit="%", measured_at=moment, source=source, confidence=Confidence(value=0.86, rationale="Structured percentage pattern in source document", method="heuristic")))
            for match in _VIBRATION_RE.finditer(text):
                readings.append(ConditionReading(equipment_tag=tag, parameter="vibration_mm_s", value=float(match.group(1)), unit="mm/s", measured_at=moment, source=source, confidence=Confidence(value=0.78, rationale="Vibration reading pattern in source document", method="heuristic")))
        return readings

    @staticmethod
    def _failure_mode(text: str) -> str:
        match = re.search(r"\b(?:bearing\s+seizure|bearing\s+failure|lubrication\s+failure|cavitation|overheating)\b", text, re.IGNORECASE)
        return match.group(0).lower() if match else "unspecified failure"

    @staticmethod
    def _root_cause(text: str) -> str | None:
        match = re.search(r"\broot\s+cause\s*[:\-]?\s*([^\.\n]+)", text, re.IGNORECASE)
        return match.group(1).strip() if match else None

    @staticmethod
    def _number(pattern: re.Pattern[str], text: str) -> float | None:
        match = pattern.search(text)
        return float(match.group(1).replace(",", "")) if match else None

    @staticmethod
    def _cost(text: str) -> float | None:
        match = _COST_RE.search(text)
        if not match:
            return None
        amount = float(match.group(1).replace(",", ""))
        return amount * 100_000.0 if re.search(r"(?:lakh|\blac\b|\bL\b)", match.group(0), re.IGNORECASE) else amount

    @staticmethod
    def _years_experience(text: str, name: str) -> float | None:
        nearby = re.search(rf"{re.escape(name)}.{{0,160}}?(\d{{1,2}})\s+years?", text, re.IGNORECASE | re.DOTALL)
        return float(nearby.group(1)) if nearby else None

    async def _publish(self, topic: Topic, **payload: object) -> None:
        event = Event.make(topic, source=self.name, **payload)
        await self._deps.events.publish(event.topic.value, event.model_dump(mode="json"))


__all__ = ["KnowledgeGraphAgent"]
