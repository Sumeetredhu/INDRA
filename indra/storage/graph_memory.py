"""In-process knowledge graph.

**This is not a stub.** It is the backend INDRA actually runs on whenever Neo4j is absent
(``docs/DECISIONS.md`` D1), which includes the test suite, a cold laptop, and a demo where the
container did not come up. Every traversal, centrality and scoring behaviour the product depends on
is implemented here for real, with the same semantics as the Cypher path.

Concurrency: all mutations are synchronous between ``await`` points, so no lock is required under a
single asyncio loop. Do not introduce an ``await`` inside a mutation without adding one.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from datetime import date, datetime, timezone
from typing import Any, Final, Iterable, Sequence

from indra.core.exceptions import GraphStoreError
from indra.core.logging import get_logger
from indra.core.models import (
    ConditionReading,
    Criticality,
    DocumentMeta,
    Equipment,
    ExtractedEntity,
    ExtractedRelationship,
    FailureEvent,
    GraphPath,
    MaintenanceRecord,
    Person,
    Procedure,
    RelationType,
)

logger = get_logger(__name__)

#: Confidence assumed for an edge that arrived without one.
_DEFAULT_EDGE_CONFIDENCE: Final[float] = 0.85

#: Per-hop decay applied to path confidence, so a 3-hop link is weaker than a direct one.
_HOP_DECAY: Final[float] = 0.9

#: Cap on frontier width during BFS, to keep a hub node from exploding the search.
_FRONTIER_LIMIT: Final[int] = 512


def _as_date(value: object) -> date | None:
    """Coerce whatever a caller passed into a ``date``, or ``None``."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


class _Edge:
    """One directed relationship. Slots because a plant graph gets wide fast."""

    __slots__ = ("source", "target", "type", "confidence", "properties", "document_id", "chunk_id")

    def __init__(
        self,
        source: str,
        target: str,
        rel_type: RelationType,
        *,
        confidence: float,
        properties: dict[str, Any],
        document_id: str | None,
        chunk_id: str | None,
    ) -> None:
        self.source = source
        self.target = target
        self.type = rel_type
        self.confidence = confidence
        self.properties = properties
        self.document_id = document_id
        self.chunk_id = chunk_id

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.source, self.type.value, self.target)


class MemoryGraphStore:
    """A real knowledge graph in plain Python, implementing ``contracts.GraphStore``."""

    name = "graph:memory"
    backend = "memory"

    def __init__(self) -> None:
        # key -> {"label": str, "name": str, "properties": dict}
        self._nodes: dict[str, dict[str, Any]] = {}
        self._edges: dict[tuple[str, str, str], _Edge] = {}
        self._out: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
        self._in: dict[str, set[tuple[str, str, str]]] = defaultdict(set)

        self._documents: dict[str, DocumentMeta] = {}
        self._equipment: dict[str, Equipment] = {}
        self._people: dict[str, Person] = {}
        self._maintenance: dict[str, list[MaintenanceRecord]] = defaultdict(list)
        self._failures: dict[str, list[FailureEvent]] = defaultdict(list)
        self._procedures: dict[str, Procedure] = {}
        self._readings: dict[str, list[ConditionReading]] = defaultdict(list)

        # entity key -> chunk id -> mention confidence
        self._mentions: dict[str, dict[str, float]] = defaultdict(dict)
        self._chunk_document: dict[str, str] = {}
        self._document_chunks: dict[str, set[str]] = defaultdict(set)
        self._document_entities: dict[str, set[str]] = defaultdict(set)

    # ------------------------------------------------------------------ schema

    async def ensure_schema(self) -> None:
        """No-op: an in-memory graph has no DDL. Present so startup is backend-agnostic."""
        logger.debug("memory graph store ready; no schema to apply")

    # ------------------------------------------------------------------ writes

    def _touch_node(self, key: str, *, label: str, name: str = "", **properties: Any) -> dict[str, Any]:
        node = self._nodes.get(key)
        if node is None:
            node = {"key": key, "label": label, "name": name or key.split(":", 1)[-1], "properties": {}}
            self._nodes[key] = node
        if properties:
            node["properties"].update({k: v for k, v in properties.items() if v is not None})
        return node

    async def upsert_document(self, meta: DocumentMeta) -> None:
        self._documents[meta.document_id] = meta
        self._touch_node(
            f"Document:{meta.document_id}",
            label="Document",
            name=meta.title,
            document_type=meta.document_type.value,
            document_date=meta.document_date.isoformat() if meta.document_date else None,
            content_hash=meta.content_hash,
        )

    async def upsert_entities(self, entities: Sequence[ExtractedEntity]) -> int:
        written = 0
        for entity in entities:
            key = entity.key
            # Entity resolution retains the canonical graph key in attributes for auditability.
            # ``_touch_node`` already receives that identity positionally, so pass it as a normal
            # stored property only under a non-conflicting name.
            properties = dict(entity.attributes)
            resolved_key = properties.pop("key", None)
            if resolved_key is not None:
                properties["resolved_key"] = resolved_key
            node = self._touch_node(
                key,
                label=entity.type.value,
                name=entity.canonical_name or entity.name,
                **properties,
            )
            # Union provenance rather than overwrite — losing a source loses an answer's citation.
            surfaces: set[str] = set(node["properties"].get("surface_forms", []))
            surfaces.add(entity.name)
            node["properties"]["surface_forms"] = sorted(surfaces)

            if entity.chunk_id:
                previous = self._mentions[key].get(entity.chunk_id, 0.0)
                self._mentions[key][entity.chunk_id] = max(previous, entity.confidence.value)
                if entity.document_id:
                    self._chunk_document.setdefault(entity.chunk_id, entity.document_id)
                    self._document_chunks[entity.document_id].add(entity.chunk_id)
            if entity.document_id:
                self._document_entities[entity.document_id].add(key)
                doc_key = f"Document:{entity.document_id}"
                if doc_key in self._nodes:
                    self._add_edge(
                        doc_key, key, RelationType.MENTIONS,
                        confidence=entity.confidence.value,
                        properties={"chunk_id": entity.chunk_id},
                        document_id=entity.document_id,
                        chunk_id=entity.chunk_id,
                    )
            written += 1
        return written

    def _add_edge(
        self,
        source: str,
        target: str,
        rel_type: RelationType,
        *,
        confidence: float,
        properties: dict[str, Any] | None = None,
        document_id: str | None = None,
        chunk_id: str | None = None,
    ) -> None:
        if source == target:
            return
        key = (source, rel_type.value, target)
        existing = self._edges.get(key)
        if existing is not None:
            # Repeated observation of the same edge is corroboration, not duplication.
            existing.confidence = max(existing.confidence, confidence)
            if properties:
                existing.properties.update({k: v for k, v in properties.items() if v is not None})
            return
        edge = _Edge(
            source, target, rel_type,
            confidence=confidence,
            properties=dict(properties or {}),
            document_id=document_id,
            chunk_id=chunk_id,
        )
        self._edges[key] = edge
        self._out[source].add(key)
        self._in[target].add(key)

    async def upsert_relationships(self, relationships: Sequence[ExtractedRelationship]) -> int:
        written = 0
        for rel in relationships:
            if rel.source_key not in self._nodes or rel.target_key not in self._nodes:
                # Tolerate dangling edges by materialising a thin node; dropping them would
                # silently lose cross-document links, which is the whole product.
                for key in (rel.source_key, rel.target_key):
                    if key not in self._nodes:
                        label, _, name = key.partition(":")
                        self._touch_node(key, label=label or "Entity", name=name or key)
            self._add_edge(
                rel.source_key, rel.target_key, rel.type,
                confidence=rel.confidence.value,
                properties=dict(rel.properties),
                document_id=rel.document_id,
                chunk_id=rel.chunk_id,
            )
            written += 1
        return written

    async def upsert_equipment(self, equipment: Sequence[Equipment]) -> int:
        for item in equipment:
            existing = self._equipment.get(item.tag)
            if existing is not None:
                # Merge: a later document may carry thresholds the first one lacked.
                merged = existing.model_copy(deep=True)
                data = item.model_dump(exclude_defaults=True)
                for field, value in data.items():
                    if field == "oem_thresholds":
                        merged.oem_thresholds = {**merged.oem_thresholds, **value}
                    elif field == "specifications":
                        merged.specifications = {**merged.specifications, **value}
                    elif value not in (None, "", [], {}):
                        setattr(merged, field, value)
                self._equipment[item.tag] = merged
            else:
                self._equipment[item.tag] = item
            final = self._equipment[item.tag]
            self._touch_node(
                f"Equipment:{item.tag}",
                label="Equipment",
                name=final.name or item.tag,
                criticality=final.criticality.value,
                equipment_type=final.equipment_type,
                location=final.location,
            )
        return len(equipment)

    async def upsert_people(self, people: Sequence[Person]) -> int:
        for person in people:
            self._people[person.person_id] = person
            key = f"Person:{person.name.strip().upper()}"
            self._touch_node(
                key, label="Person", name=person.name,
                role=person.role,
                retirement_date=person.retirement_date.isoformat() if person.retirement_date else None,
                years_experience=person.years_experience,
            )
            for tag in person.expertise_tags:
                equip_key = f"Equipment:{tag.strip().upper()}"
                self._touch_node(equip_key, label="Equipment", name=tag)
                self._add_edge(
                    key, equip_key, RelationType.HAS_EXPERTISE,
                    confidence=0.9,
                    properties={
                        "years": person.years_experience,
                        "retirement_date": person.retirement_date.isoformat() if person.retirement_date else None,
                        "documented_count": person.documented_contributions,
                    },
                )
        return len(people)

    async def upsert_maintenance(self, records: Sequence[MaintenanceRecord]) -> int:
        for record in records:
            tag = record.equipment_tag.strip().upper()
            bucket = self._maintenance[tag]
            if not any(r.record_id == record.record_id for r in bucket):
                bucket.append(record)
            equip_key = f"Equipment:{tag}"
            self._touch_node(equip_key, label="Equipment", name=tag)
            for source in record.sources:
                doc_key = f"Document:{source.document_id}"
                self._touch_node(doc_key, label="Document", name=source.document_title)
                self._add_edge(
                    doc_key, equip_key, RelationType.MAINTAINED,
                    confidence=0.95,
                    properties={
                        "date": record.performed_on.isoformat(),
                        "findings": record.findings,
                        "recommendations": record.recommendations,
                        "performed_by": record.performed_by,
                        "record_type": record.record_type,
                    },
                    document_id=source.document_id,
                )
            for reading in record.readings:
                self._readings[tag].append(reading)
        return len(records)

    async def upsert_failures(self, events: Sequence[FailureEvent]) -> int:
        for event in events:
            tag = event.equipment_tag.strip().upper()
            bucket = self._failures[tag]
            if not any(e.event_id == event.event_id for e in bucket):
                bucket.append(event)
            equip_key = f"Equipment:{tag}"
            mode_key = f"FailureMode:{event.failure_mode.strip().upper()}"
            self._touch_node(equip_key, label="Equipment", name=tag)
            self._touch_node(mode_key, label="FailureMode", name=event.failure_mode)
            self._add_edge(
                equip_key, mode_key, RelationType.FAILED_WITH_MODE,
                confidence=0.95,
                properties={
                    "date": event.occurred_on.isoformat(),
                    "root_cause": event.root_cause,
                    "downtime_hours": event.downtime_hours,
                    "cost_inr": event.cost_inr,
                    "precursor_text": event.precursor_text,
                },
                document_id=event.sources[0].document_id if event.sources else None,
            )
        return len(events)

    async def upsert_procedures(self, procedures: Sequence[Procedure]) -> int:
        for procedure in procedures:
            self._procedures[procedure.procedure_id] = procedure
            proc_key = f"Procedure:{procedure.title.strip().upper()}"
            self._touch_node(
                proc_key, label="Procedure", name=procedure.title,
                estimated_minutes=procedure.estimated_minutes,
                revision=procedure.revision,
                procedure_id=procedure.procedure_id,
            )
            for tag in procedure.applies_to:
                equip_key = f"Equipment:{tag.strip().upper()}"
                self._touch_node(equip_key, label="Equipment", name=tag)
                self._add_edge(proc_key, equip_key, RelationType.APPLIES_TO, confidence=0.95)
        return len(procedures)

    async def upsert_readings(self, readings: Sequence[ConditionReading]) -> int:
        for reading in readings:
            self._readings[reading.equipment_tag.strip().upper()].append(reading)
        return len(readings)

    async def delete_document(self, document_id: str) -> None:
        self._documents.pop(document_id, None)
        doc_key = f"Document:{document_id}"
        for edge_key in list(self._out.get(doc_key, set()) | self._in.get(doc_key, set())):
            edge = self._edges.pop(edge_key, None)
            if edge is not None:
                self._out[edge.source].discard(edge_key)
                self._in[edge.target].discard(edge_key)
        self._nodes.pop(doc_key, None)
        for chunk_id in self._document_chunks.pop(document_id, set()):
            self._chunk_document.pop(chunk_id, None)
            for mentions in self._mentions.values():
                mentions.pop(chunk_id, None)
        self._document_entities.pop(document_id, None)

    # ------------------------------------------------------------------ reads

    async def get_equipment(self, tag: str) -> Equipment | None:
        return self._equipment.get(tag.strip().upper())

    async def list_equipment(self, *, criticality: str | None = None) -> list[Equipment]:
        items = list(self._equipment.values())
        if criticality:
            wanted = Criticality(criticality.upper())
            items = [e for e in items if e.criticality is wanted]
        return sorted(items, key=lambda e: e.tag)

    async def get_people(self, *, retiring_within_days: int | None = None) -> list[Person]:
        people = list(self._people.values())
        if retiring_within_days is None:
            return sorted(people, key=lambda p: p.name)
        today = datetime.now(timezone.utc).date()
        horizon = retiring_within_days
        selected = [
            p for p in people
            if p.retirement_date is not None and 0 <= (p.retirement_date - today).days <= horizon
        ]
        return sorted(selected, key=lambda p: p.retirement_date or today)

    async def neighbours(
        self,
        entity_key: str,
        *,
        hops: int = 1,
        relation_types: Sequence[str] | None = None,
        limit: int = 50,
    ) -> list[GraphPath]:
        """Breadth-first expansion producing real ``GraphPath`` objects with narratives."""
        if entity_key not in self._nodes:
            return []
        wanted = {t.upper() for t in relation_types} if relation_types else None
        hops = max(1, min(hops, 4))

        paths: list[GraphPath] = []
        seen: set[str] = {entity_key}
        # queue entries: (node, node_chain, relation_chain, confidence)
        queue: deque[tuple[str, list[str], list[RelationType], float]] = deque(
            [(entity_key, [entity_key], [], 1.0)]
        )

        while queue and len(paths) < limit:
            node, chain, relations, confidence = queue.popleft()
            if len(relations) >= hops:
                continue
            edge_keys = list(self._out.get(node, set())) + list(self._in.get(node, set()))
            for edge_key in edge_keys[:_FRONTIER_LIMIT]:
                edge = self._edges.get(edge_key)
                if edge is None:
                    continue
                if wanted is not None and edge.type.value not in wanted:
                    continue
                other = edge.target if edge.source == node else edge.source
                if other in chain:
                    continue  # cycle guard
                next_chain = [*chain, other]
                next_relations = [*relations, edge.type]
                next_confidence = confidence * edge.confidence * _HOP_DECAY

                if other not in seen or len(next_relations) == 1:
                    seen.add(other)
                    paths.append(
                        GraphPath(
                            nodes=next_chain,
                            relations=next_relations,
                            hops=len(next_relations),
                            confidence=max(0.0, min(1.0, next_confidence)),
                            narrative=self._narrate(next_chain, next_relations),
                        )
                    )
                    if len(paths) >= limit:
                        break
                if len(next_relations) < hops:
                    queue.append((other, next_chain, next_relations, next_confidence))

        paths.sort(key=lambda p: (p.hops, -p.confidence))
        return paths[:limit]

    def _narrate(self, nodes: Sequence[str], relations: Sequence[RelationType]) -> str:
        """Render a path the way a human reads it: ``P-101 —CONNECTED_TO→ V-201``."""
        parts: list[str] = [self._display(nodes[0])]
        for relation, node in zip(relations, nodes[1:]):
            parts.append(f" —{relation.value}→ ")
            parts.append(self._display(node))
        return "".join(parts)

    def _display(self, key: str) -> str:
        node = self._nodes.get(key)
        if node is not None and node.get("name"):
            return str(node["name"])
        return key.split(":", 1)[-1]

    async def chunks_for_entities(
        self, entity_keys: Sequence[str], *, limit: int = 50
    ) -> list[tuple[str, float]]:
        """Score chunks by how many query entities they mention, and how confidently."""
        if not entity_keys:
            return []
        scores: dict[str, float] = defaultdict(float)
        matched: dict[str, int] = defaultdict(int)
        for key in entity_keys:
            for chunk_id, confidence in self._mentions.get(key, {}).items():
                scores[chunk_id] += confidence
                matched[chunk_id] += 1

        if not scores:
            return []
        # Reward chunks touching MORE of the query's entities — that is the cross-document signal.
        total_entities = len(entity_keys)
        ranked = [
            (chunk_id, (score / total_entities) * (0.6 + 0.4 * (matched[chunk_id] / total_entities)))
            for chunk_id, score in scores.items()
        ]
        peak = max(value for _, value in ranked) or 1.0
        ranked = [(chunk_id, value / peak) for chunk_id, value in ranked]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:limit]

    async def maintenance_history(self, tag: str, *, since: date | None = None) -> list[MaintenanceRecord]:
        records = list(self._maintenance.get(tag.strip().upper(), []))
        if since is not None:
            records = [r for r in records if r.performed_on >= since]
        return sorted(records, key=lambda r: r.performed_on, reverse=True)

    async def failure_history(self, tag: str, *, since: date | None = None) -> list[FailureEvent]:
        events = list(self._failures.get(tag.strip().upper(), []))
        if since is not None:
            events = [e for e in events if e.occurred_on >= since]
        return sorted(events, key=lambda e: e.occurred_on, reverse=True)

    async def procedures_for(self, tag: str) -> list[Procedure]:
        wanted = tag.strip().upper()
        return [
            procedure for procedure in self._procedures.values()
            if any(t.strip().upper() == wanted for t in procedure.applies_to)
        ]

    async def readings_for(
        self, tag: str, *, parameter: str | None = None, since: date | None = None
    ) -> list[ConditionReading]:
        readings = list(self._readings.get(tag.strip().upper(), []))
        if parameter:
            readings = [r for r in readings if r.parameter == parameter]
        if since is not None:
            readings = [r for r in readings if r.measured_at.date() >= since]
        return sorted(readings, key=lambda r: r.measured_at)

    async def documents_for_tag(self, tag: str, *, limit: int = 50) -> list[DocumentMeta]:
        key = f"Equipment:{tag.strip().upper()}"
        document_ids: set[str] = set()
        for edge_key in self._in.get(key, set()) | self._out.get(key, set()):
            edge = self._edges.get(edge_key)
            if edge is not None and edge.document_id:
                document_ids.add(edge.document_id)
        for document_id, entities in self._document_entities.items():
            if key in entities:
                document_ids.add(document_id)
        metas = [self._documents[d] for d in document_ids if d in self._documents]
        metas.sort(key=lambda m: m.document_date or date.min, reverse=True)
        return metas[:limit]

    async def centrality(self, entity_keys: Sequence[str]) -> dict[str, float]:
        """Degree centrality normalised against the busiest node in the graph."""
        if not self._nodes:
            return {key: 0.0 for key in entity_keys}
        degrees = {
            key: len(self._out.get(key, set())) + len(self._in.get(key, set()))
            for key in self._nodes
        }
        peak = max(degrees.values()) if degrees else 0
        if peak <= 0:
            return {key: 0.0 for key in entity_keys}
        # Log scaling: a hub with 200 edges should not be 100x a node with 2.
        log_peak = math.log1p(peak)
        return {
            key: round(math.log1p(degrees.get(key, 0)) / log_peak, 6) if log_peak else 0.0
            for key in entity_keys
        }

    async def document_meta(self, document_ids: Sequence[str]) -> dict[str, DocumentMeta]:
        return {d: self._documents[d] for d in document_ids if d in self._documents}

    async def query(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        raise GraphStoreError(
            "Raw Cypher is unavailable on the in-memory graph backend. "
            "Start Neo4j and set INDRA_STORAGE_BACKEND=external, or use the typed graph methods.",
            context={"backend": "memory"},
        )

    async def stats(self) -> dict[str, int]:
        by_label: dict[str, int] = defaultdict(int)
        for node in self._nodes.values():
            by_label[str(node["label"])] += 1
        by_relation: dict[str, int] = defaultdict(int)
        for edge in self._edges.values():
            by_relation[edge.type.value] += 1
        return {
            "nodes": len(self._nodes),
            "relationships": len(self._edges),
            "documents": len(self._documents),
            "equipment": len(self._equipment),
            "people": len(self._people),
            **{f"node:{label}": count for label, count in sorted(by_label.items())},
            **{f"rel:{rel}": count for rel, count in sorted(by_relation.items())},
        }

    async def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "backend": "memory",
            "detail": f"{len(self._nodes)} nodes, {len(self._edges)} relationships (in-process fallback)",
        }

    async def close(self) -> None:
        return None

    # ------------------------------------------------------------------ preview support

    def iter_nodes(self) -> Iterable[dict[str, Any]]:
        """Used by the graph preview builder. Read-only view."""
        return iter(self._nodes.values())

    def edges_for(self, key: str) -> list[_Edge]:
        return [
            self._edges[edge_key]
            for edge_key in (self._out.get(key, set()) | self._in.get(key, set()))
            if edge_key in self._edges
        ]

    def node(self, key: str) -> dict[str, Any] | None:
        return self._nodes.get(key)


__all__ = ["MemoryGraphStore"]
