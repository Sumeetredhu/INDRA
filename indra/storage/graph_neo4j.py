"""Neo4j knowledge graph backend.

Mirrors :class:`~indra.storage.graph_memory.MemoryGraphStore` semantics exactly, so switching
backends changes performance and durability but never answers.

**Modelling note.** Structure lives in the graph (nodes and typed edges, which is what traversal,
centrality and the React-Flow preview read). Full domain records — ``MaintenanceRecord``,
``FailureEvent``, ``Procedure``, ``ConditionReading`` — are *additionally* serialised as JSON on a
``:RecordStore`` node per asset. That dual-write is deliberate: the graph gives us traversal, the
JSON gives us lossless reconstruction of a Pydantic model without shredding twenty optional fields
into properties and re-hydrating them by hand. The JSON is never traversed; the graph is never
parsed.

The ``neo4j`` driver is imported at module scope (it is a light pure-Python package and is listed
in ``requirements.txt``), but a connection is never opened at import time.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any, Final, Sequence

from indra.core.config import Settings
from indra.core.exceptions import GraphStoreError
from indra.core.logging import get_logger
from indra.core.models import (
    ConditionReading,
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

_CONNECT_TIMEOUT_S: Final[float] = 5.0

#: Relationship types the traversal helper will interpolate. Bound to the enum so a caller can
#: never inject a type name — Cypher cannot parameterise relationship types.
_ALLOWED_RELATIONS: Final[frozenset[str]] = frozenset(member.value for member in RelationType)


def _scrub(properties: dict[str, Any]) -> dict[str, Any]:
    """Reduce a property dict to Neo4j-storable scalars, JSON-encoding anything else."""
    clean: dict[str, Any] = {}
    for key, value in properties.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        elif isinstance(value, (datetime, date)):
            clean[key] = value.isoformat()
        elif isinstance(value, (list, tuple)) and all(
            isinstance(v, (str, int, float, bool)) for v in value
        ):
            clean[key] = list(value)
        else:
            clean[key] = json.dumps(value, default=str)
    return clean


class Neo4jGraphStore:
    """Implements ``contracts.GraphStore`` against Neo4j Community."""

    name = "graph:neo4j"
    backend = "neo4j"

    def __init__(self, driver: Any, *, database: str) -> None:
        self._driver = driver
        self._database = database

    # ------------------------------------------------------------------ lifecycle

    @classmethod
    async def connect(cls, settings: Settings) -> Neo4jGraphStore:
        """Open a driver and verify connectivity. Raises if Neo4j is not reachable."""
        try:
            from neo4j import AsyncGraphDatabase
        except ImportError as exc:  # pragma: no cover - listed in requirements
            raise GraphStoreError(
                "The neo4j driver is not installed. `pip install neo4j`, or set "
                "INDRA_STORAGE_BACKEND=memory to run without a graph database.",
                cause=exc,
            ) from exc

        password = settings.secret("neo4j_password") or ""
        try:
            driver = AsyncGraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, password),
                connection_timeout=_CONNECT_TIMEOUT_S,
                max_connection_lifetime=300,
            )
            await driver.verify_connectivity()
        except Exception as exc:  # noqa: BLE001 - driver raises a wide family
            raise GraphStoreError(
                f"Neo4j is not reachable at {settings.neo4j_uri}. Start it with "
                "`docker compose -f docker/docker-compose.yml up neo4j`, or set "
                "INDRA_STORAGE_BACKEND=memory.",
                context={"uri": settings.neo4j_uri, "user": settings.neo4j_user},
                cause=exc,
            ) from exc
        return cls(driver, database=settings.neo4j_database)

    async def close(self) -> None:
        try:
            await self._driver.close()
        except Exception:  # noqa: BLE001 - shutdown is best effort
            logger.debug("neo4j driver close failed")

    # ------------------------------------------------------------------ execution

    async def _run(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        try:
            async with self._driver.session(database=self._database) as session:
                result = await session.run(cypher, params or {})
                return [record.data() async for record in result]
        except Exception as exc:  # noqa: BLE001
            raise GraphStoreError(
                "Neo4j query failed.",
                context={"cypher": cypher.strip().splitlines()[0][:160]},
                cause=exc,
            ) from exc

    # ------------------------------------------------------------------ schema

    async def ensure_schema(self) -> None:
        """Create constraints and indexes. Idempotent — safe on every startup."""
        statements = [
            "CREATE CONSTRAINT indra_entity_key IF NOT EXISTS "
            "FOR (n:Entity) REQUIRE n.key IS UNIQUE",
            "CREATE CONSTRAINT indra_equipment_tag IF NOT EXISTS "
            "FOR (n:Equipment) REQUIRE n.tag IS UNIQUE",
            "CREATE CONSTRAINT indra_document_id IF NOT EXISTS "
            "FOR (n:Document) REQUIRE n.document_id IS UNIQUE",
            "CREATE CONSTRAINT indra_person_id IF NOT EXISTS "
            "FOR (n:Person) REQUIRE n.person_id IS UNIQUE",
            "CREATE INDEX indra_document_hash IF NOT EXISTS FOR (n:Document) ON (n.content_hash)",
            "CREATE INDEX indra_document_date IF NOT EXISTS FOR (n:Document) ON (n.document_date)",
            "CREATE INDEX indra_equipment_crit IF NOT EXISTS FOR (n:Equipment) ON (n.criticality)",
            "CREATE INDEX indra_person_retire IF NOT EXISTS FOR (n:Person) ON (n.retirement_date)",
            "CREATE INDEX indra_entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name)",
        ]
        for statement in statements:
            try:
                await self._run(statement)
            except GraphStoreError as exc:
                # An existing equivalent constraint under a different name is not fatal.
                logger.warning("schema statement skipped", extra={"detail": str(exc)})

    # ------------------------------------------------------------------ writes

    async def upsert_document(self, meta: DocumentMeta) -> None:
        await self._run(
            """
            MERGE (d:Entity:Document {key: $key})
            SET d.document_id = $document_id,
                d.name        = $title,
                d += $props
            """,
            {
                "key": f"Document:{meta.document_id}",
                "document_id": meta.document_id,
                "title": meta.title,
                "props": _scrub(
                    {
                        "content_hash": meta.content_hash,
                        "document_type": meta.document_type.value,
                        "document_date": meta.document_date,
                        "ingested_at": meta.ingested_at,
                        "filename": meta.filename,
                        "language": meta.language,
                        "page_count": meta.page_count,
                        "payload": meta.model_dump(mode="json"),
                    }
                ),
            },
        )

    async def upsert_entities(self, entities: Sequence[ExtractedEntity]) -> int:
        if not entities:
            return 0
        rows = [
            {
                "key": entity.key,
                "label": entity.type.value,
                "name": entity.canonical_name or entity.name,
                "surface": entity.name,
                "confidence": entity.confidence.value,
                "chunk_id": entity.chunk_id,
                "document_id": entity.document_id,
                "page": entity.page,
                "props": _scrub(entity.attributes),
            }
            for entity in entities
        ]
        # apoc is not assumed; a per-label MERGE keeps this portable across Neo4j editions.
        await self._run(
            """
            UNWIND $rows AS row
            MERGE (e:Entity {key: row.key})
            SET e.name  = coalesce(e.name, row.name),
                e.label = row.label,
                e += row.props,
                e.surface_forms =
                    CASE WHEN row.surface IN coalesce(e.surface_forms, [])
                         THEN e.surface_forms
                         ELSE coalesce(e.surface_forms, []) + row.surface END
            WITH e, row WHERE row.document_id IS NOT NULL
            MERGE (d:Entity:Document {key: 'Document:' + row.document_id})
              ON CREATE SET d.document_id = row.document_id, d.label = 'Document'
            MERGE (d)-[m:MENTIONS {chunk_id: coalesce(row.chunk_id, '')}]->(e)
            SET m.confidence = CASE WHEN m.confidence IS NULL OR m.confidence < row.confidence
                                    THEN row.confidence ELSE m.confidence END,
                m.page = row.page
            """,
            {"rows": rows},
        )
        # Apply the concrete label so schema constraints and label-scoped indexes engage.
        for label in {entity.type.value for entity in entities}:
            if label not in {"Document"}:
                await self._run(
                    f"MATCH (e:Entity) WHERE e.label = $label AND NOT e:{label} SET e:{label}",
                    {"label": label},
                )
        return len(entities)

    async def upsert_relationships(self, relationships: Sequence[ExtractedRelationship]) -> int:
        written = 0
        # Relationship type cannot be parameterised in Cypher, so group by type and validate
        # each against the enum before interpolation.
        by_type: dict[str, list[dict[str, Any]]] = {}
        for rel in relationships:
            if rel.type.value not in _ALLOWED_RELATIONS:  # pragma: no cover - enum guarantees this
                continue
            by_type.setdefault(rel.type.value, []).append(
                {
                    "source": rel.source_key,
                    "target": rel.target_key,
                    "confidence": rel.confidence.value,
                    "props": _scrub(
                        {
                            **rel.properties,
                            "document_id": rel.document_id,
                            "chunk_id": rel.chunk_id,
                            "method": rel.method,
                            "evidence": rel.evidence_text[:400] if rel.evidence_text else None,
                        }
                    ),
                }
            )
        for rel_type, rows in by_type.items():
            await self._run(
                f"""
                UNWIND $rows AS row
                MERGE (a:Entity {{key: row.source}})
                MERGE (b:Entity {{key: row.target}})
                MERGE (a)-[r:{rel_type}]->(b)
                SET r += row.props,
                    r.confidence = CASE WHEN r.confidence IS NULL OR r.confidence < row.confidence
                                        THEN row.confidence ELSE r.confidence END
                """,
                {"rows": rows},
            )
            written += len(rows)
        return written

    async def upsert_equipment(self, equipment: Sequence[Equipment]) -> int:
        if not equipment:
            return 0
        await self._run(
            """
            UNWIND $rows AS row
            MERGE (e:Entity:Equipment {key: row.key})
            SET e.tag = row.tag, e.name = row.name, e.label = 'Equipment', e += row.props
            """,
            {
                "rows": [
                    {
                        "key": f"Equipment:{item.tag}",
                        "tag": item.tag,
                        "name": item.name or item.tag,
                        "props": _scrub(
                            {
                                "criticality": item.criticality.value,
                                "equipment_type": item.equipment_type,
                                "manufacturer": item.manufacturer,
                                "model": item.model,
                                "location": item.location,
                                "unit": item.unit,
                                "installed_on": item.installed_on,
                                "payload": item.model_dump(mode="json"),
                            }
                        ),
                    }
                    for item in equipment
                ]
            },
        )
        return len(equipment)

    async def upsert_people(self, people: Sequence[Person]) -> int:
        if not people:
            return 0
        await self._run(
            """
            UNWIND $rows AS row
            MERGE (p:Entity:Person {key: row.key})
            SET p.person_id = row.person_id, p.name = row.name, p.label = 'Person', p += row.props
            WITH p, row
            UNWIND row.tags AS tag
            MERGE (e:Entity:Equipment {key: 'Equipment:' + tag})
              ON CREATE SET e.tag = tag, e.name = tag, e.label = 'Equipment'
            MERGE (p)-[h:HAS_EXPERTISE]->(e)
            SET h += row.expertise
            """,
            {
                "rows": [
                    {
                        "key": f"Person:{person.name.strip().upper()}",
                        "person_id": person.person_id,
                        "name": person.name,
                        "tags": [t.strip().upper() for t in person.expertise_tags],
                        "props": _scrub(
                            {
                                "role": person.role,
                                "years_experience": person.years_experience,
                                "retirement_date": person.retirement_date,
                                "documented_contributions": person.documented_contributions,
                                "payload": person.model_dump(mode="json"),
                            }
                        ),
                        "expertise": _scrub(
                            {
                                "years": person.years_experience,
                                "retirement_date": person.retirement_date,
                                "documented_count": person.documented_contributions,
                                "confidence": 0.9,
                            }
                        ),
                    }
                    for person in people
                ]
            },
        )
        return len(people)

    async def _append_records(self, tag: str, kind: str, payloads: list[dict[str, Any]]) -> None:
        """Append JSON-serialised domain records to the per-asset record store."""
        await self._run(
            """
            MERGE (e:Entity:Equipment {key: 'Equipment:' + $tag})
              ON CREATE SET e.tag = $tag, e.name = $tag, e.label = 'Equipment'
            MERGE (s:RecordStore {key: 'RecordStore:' + $tag + ':' + $kind})
            SET s.tag = $tag, s.kind = $kind,
                s.records = coalesce(s.records, []) + $payloads
            MERGE (e)-[:DOCUMENTED_BY]->(s)
            """,
            {"tag": tag, "kind": kind, "payloads": payloads},
        )

    async def _read_records(self, tag: str, kind: str) -> list[dict[str, Any]]:
        rows = await self._run(
            "MATCH (s:RecordStore {key: 'RecordStore:' + $tag + ':' + $kind}) RETURN s.records AS records",
            {"tag": tag, "kind": kind},
        )
        if not rows or not rows[0].get("records"):
            return []
        out: list[dict[str, Any]] = []
        for raw in rows[0]["records"]:
            try:
                out.append(json.loads(raw))
            except (TypeError, ValueError):
                continue
        return out

    async def upsert_maintenance(self, records: Sequence[MaintenanceRecord]) -> int:
        by_tag: dict[str, list[MaintenanceRecord]] = {}
        for record in records:
            by_tag.setdefault(record.equipment_tag.strip().upper(), []).append(record)
        for tag, items in by_tag.items():
            existing = {r.get("record_id") for r in await self._read_records(tag, "maintenance")}
            fresh = [r for r in items if r.record_id not in existing]
            if fresh:
                await self._append_records(
                    tag, "maintenance", [r.model_dump_json() for r in fresh]
                )
            for record in items:
                for source in record.sources:
                    await self._run(
                        """
                        MERGE (d:Entity:Document {key: 'Document:' + $document_id})
                          ON CREATE SET d.document_id = $document_id, d.label = 'Document'
                        MERGE (e:Entity:Equipment {key: 'Equipment:' + $tag})
                          ON CREATE SET e.tag = $tag, e.name = $tag, e.label = 'Equipment'
                        MERGE (d)-[m:MAINTAINED]->(e)
                        SET m += $props
                        """,
                        {
                            "document_id": source.document_id,
                            "tag": tag,
                            "props": _scrub(
                                {
                                    "date": record.performed_on,
                                    "findings": record.findings[:600],
                                    "recommendations": record.recommendations[:600],
                                    "performed_by": record.performed_by,
                                    "work_order": record.record_id,
                                    "confidence": 0.95,
                                }
                            ),
                        },
                    )
        return len(records)

    async def upsert_failures(self, events: Sequence[FailureEvent]) -> int:
        by_tag: dict[str, list[FailureEvent]] = {}
        for event in events:
            by_tag.setdefault(event.equipment_tag.strip().upper(), []).append(event)
        for tag, items in by_tag.items():
            existing = {e.get("event_id") for e in await self._read_records(tag, "failures")}
            fresh = [e for e in items if e.event_id not in existing]
            if fresh:
                await self._append_records(tag, "failures", [e.model_dump_json() for e in fresh])
            for event in items:
                await self._run(
                    """
                    MERGE (e:Entity:Equipment {key: 'Equipment:' + $tag})
                      ON CREATE SET e.tag = $tag, e.name = $tag, e.label = 'Equipment'
                    MERGE (f:Entity:FailureMode {key: 'FailureMode:' + $mode_key})
                      ON CREATE SET f.name = $mode, f.label = 'FailureMode'
                    MERGE (e)-[r:FAILED_WITH_MODE {date: $date}]->(f)
                    SET r += $props
                    """,
                    {
                        "tag": tag,
                        "mode": event.failure_mode,
                        "mode_key": event.failure_mode.strip().upper(),
                        "date": event.occurred_on.isoformat(),
                        "props": _scrub(
                            {
                                "root_cause": event.root_cause,
                                "downtime_hours": event.downtime_hours,
                                "cost_inr": event.cost_inr,
                                "precursor_text": event.precursor_text[:600],
                                "confidence": 0.95,
                            }
                        ),
                    },
                )
        return len(events)

    async def upsert_procedures(self, procedures: Sequence[Procedure]) -> int:
        for procedure in procedures:
            await self._run(
                """
                MERGE (p:Entity:Procedure {key: 'Procedure:' + $key})
                SET p.name = $title, p.label = 'Procedure', p += $props
                WITH p
                UNWIND $tags AS tag
                MERGE (e:Entity:Equipment {key: 'Equipment:' + tag})
                  ON CREATE SET e.tag = tag, e.name = tag, e.label = 'Equipment'
                MERGE (p)-[:APPLIES_TO]->(e)
                """,
                {
                    "key": procedure.title.strip().upper(),
                    "title": procedure.title,
                    "tags": [t.strip().upper() for t in procedure.applies_to],
                    "props": _scrub(
                        {
                            "procedure_id": procedure.procedure_id,
                            "estimated_minutes": procedure.estimated_minutes,
                            "revision": procedure.revision,
                            "payload": procedure.model_dump(mode="json"),
                        }
                    ),
                },
            )
        return len(procedures)

    async def upsert_readings(self, readings: Sequence[ConditionReading]) -> int:
        by_tag: dict[str, list[ConditionReading]] = {}
        for reading in readings:
            by_tag.setdefault(reading.equipment_tag.strip().upper(), []).append(reading)
        for tag, items in by_tag.items():
            await self._append_records(tag, "readings", [r.model_dump_json() for r in items])
        return len(readings)

    async def delete_document(self, document_id: str) -> None:
        await self._run(
            "MATCH (d:Document {document_id: $document_id}) DETACH DELETE d",
            {"document_id": document_id},
        )

    # ------------------------------------------------------------------ reads

    async def get_equipment(self, tag: str) -> Equipment | None:
        rows = await self._run(
            "MATCH (e:Equipment {tag: $tag}) RETURN e.payload AS payload",
            {"tag": tag.strip().upper()},
        )
        if not rows or not rows[0].get("payload"):
            return None
        return Equipment.model_validate(json.loads(rows[0]["payload"]))

    async def list_equipment(self, *, criticality: str | None = None) -> list[Equipment]:
        cypher = "MATCH (e:Equipment) WHERE e.payload IS NOT NULL"
        params: dict[str, Any] = {}
        if criticality:
            cypher += " AND e.criticality = $criticality"
            params["criticality"] = criticality.upper()
        cypher += " RETURN e.payload AS payload ORDER BY e.tag"
        rows = await self._run(cypher, params)
        return [Equipment.model_validate(json.loads(r["payload"])) for r in rows if r.get("payload")]

    async def get_people(self, *, retiring_within_days: int | None = None) -> list[Person]:
        rows = await self._run(
            "MATCH (p:Person) WHERE p.payload IS NOT NULL RETURN p.payload AS payload"
        )
        people = [Person.model_validate(json.loads(r["payload"])) for r in rows if r.get("payload")]
        if retiring_within_days is None:
            return sorted(people, key=lambda p: p.name)
        today = datetime.now().date()
        return sorted(
            [
                p for p in people
                if p.retirement_date and 0 <= (p.retirement_date - today).days <= retiring_within_days
            ],
            key=lambda p: p.retirement_date or today,
        )

    async def neighbours(
        self,
        entity_key: str,
        *,
        hops: int = 1,
        relation_types: Sequence[str] | None = None,
        limit: int = 50,
    ) -> list[GraphPath]:
        hops = max(1, min(hops, 4))
        limit = max(1, min(limit, 500))
        # Relationship types cannot be parameterised; validate against the enum then interpolate.
        filter_fragment = ""
        if relation_types:
            valid = [t.upper() for t in relation_types if t.upper() in _ALLOWED_RELATIONS]
            if not valid:
                return []
            filter_fragment = ":" + "|".join(valid)

        rows = await self._run(
            f"""
            MATCH path = (start:Entity {{key: $key}})-[r{filter_fragment}*1..{hops}]-(other:Entity)
            WHERE other.key <> $key
            RETURN [n IN nodes(path) | n.key]  AS node_keys,
                   [n IN nodes(path) | coalesce(n.name, n.key)] AS node_names,
                   [rel IN relationships(path) | type(rel)] AS rel_types,
                   reduce(c = 1.0, rel IN relationships(path) |
                          c * coalesce(rel.confidence, 0.85) * 0.9) AS confidence,
                   length(path) AS hops
            ORDER BY hops ASC, confidence DESC
            LIMIT {limit}
            """,
            {"key": entity_key},
        )

        paths: list[GraphPath] = []
        for row in rows:
            rel_types = [t for t in row.get("rel_types", []) if t in _ALLOWED_RELATIONS]
            if len(rel_types) != len(row.get("rel_types", [])):
                continue
            names = row.get("node_names") or []
            narrative_parts: list[str] = [str(names[0])] if names else []
            for rel, name in zip(rel_types, names[1:]):
                narrative_parts.append(f" —{rel}→ ")
                narrative_parts.append(str(name))
            paths.append(
                GraphPath(
                    nodes=list(row.get("node_keys") or []),
                    relations=[RelationType(t) for t in rel_types],
                    hops=int(row.get("hops") or len(rel_types)),
                    confidence=max(0.0, min(1.0, float(row.get("confidence") or 0.0))),
                    narrative="".join(narrative_parts),
                )
            )
        return paths

    async def chunks_for_entities(
        self, entity_keys: Sequence[str], *, limit: int = 50
    ) -> list[tuple[str, float]]:
        if not entity_keys:
            return []
        rows = await self._run(
            """
            MATCH (d:Document)-[m:MENTIONS]->(e:Entity)
            WHERE e.key IN $keys AND m.chunk_id IS NOT NULL AND m.chunk_id <> ''
            WITH m.chunk_id AS chunk_id,
                 sum(coalesce(m.confidence, 0.7)) AS total,
                 count(DISTINCT e.key) AS matched
            RETURN chunk_id, total, matched
            ORDER BY total DESC
            LIMIT $limit
            """,
            {"keys": list(entity_keys), "limit": max(1, min(limit, 500))},
        )
        if not rows:
            return []
        total_entities = len(entity_keys)
        scored = [
            (
                str(row["chunk_id"]),
                (float(row["total"]) / total_entities)
                * (0.6 + 0.4 * (float(row["matched"]) / total_entities)),
            )
            for row in rows
        ]
        peak = max(value for _, value in scored) or 1.0
        return [(cid, value / peak) for cid, value in scored]

    async def maintenance_history(self, tag: str, *, since: date | None = None) -> list[MaintenanceRecord]:
        raw = await self._read_records(tag.strip().upper(), "maintenance")
        records = [MaintenanceRecord.model_validate(item) for item in raw]
        if since is not None:
            records = [r for r in records if r.performed_on >= since]
        return sorted(records, key=lambda r: r.performed_on, reverse=True)

    async def failure_history(self, tag: str, *, since: date | None = None) -> list[FailureEvent]:
        raw = await self._read_records(tag.strip().upper(), "failures")
        events = [FailureEvent.model_validate(item) for item in raw]
        if since is not None:
            events = [e for e in events if e.occurred_on >= since]
        return sorted(events, key=lambda e: e.occurred_on, reverse=True)

    async def procedures_for(self, tag: str) -> list[Procedure]:
        rows = await self._run(
            """
            MATCH (p:Procedure)-[:APPLIES_TO]->(e:Equipment {tag: $tag})
            WHERE p.payload IS NOT NULL RETURN p.payload AS payload
            """,
            {"tag": tag.strip().upper()},
        )
        return [Procedure.model_validate(json.loads(r["payload"])) for r in rows if r.get("payload")]

    async def readings_for(
        self, tag: str, *, parameter: str | None = None, since: date | None = None
    ) -> list[ConditionReading]:
        raw = await self._read_records(tag.strip().upper(), "readings")
        readings = [ConditionReading.model_validate(item) for item in raw]
        if parameter:
            readings = [r for r in readings if r.parameter == parameter]
        if since is not None:
            readings = [r for r in readings if r.measured_at.date() >= since]
        return sorted(readings, key=lambda r: r.measured_at)

    async def documents_for_tag(self, tag: str, *, limit: int = 50) -> list[DocumentMeta]:
        rows = await self._run(
            """
            MATCH (d:Document)-[]-(e:Equipment {tag: $tag})
            WHERE d.payload IS NOT NULL
            RETURN DISTINCT d.payload AS payload
            LIMIT $limit
            """,
            {"tag": tag.strip().upper(), "limit": max(1, min(limit, 500))},
        )
        return [DocumentMeta.model_validate(json.loads(r["payload"])) for r in rows if r.get("payload")]

    async def centrality(self, entity_keys: Sequence[str]) -> dict[str, float]:
        if not entity_keys:
            return {}
        rows = await self._run(
            """
            MATCH (n:Entity) WHERE n.key IN $keys
            OPTIONAL MATCH (n)-[r]-()
            WITH n.key AS key, count(r) AS degree
            RETURN key, degree
            """,
            {"keys": list(entity_keys)},
        )
        peak_rows = await self._run(
            "MATCH (n:Entity) OPTIONAL MATCH (n)-[r]-() "
            "WITH n, count(r) AS d RETURN max(d) AS peak"
        )
        peak = float((peak_rows[0].get("peak") if peak_rows else 0) or 0)
        if peak <= 0:
            return {key: 0.0 for key in entity_keys}
        import math

        log_peak = math.log1p(peak)
        degrees = {str(row["key"]): float(row["degree"]) for row in rows}
        return {
            key: round(math.log1p(degrees.get(key, 0.0)) / log_peak, 6) if log_peak else 0.0
            for key in entity_keys
        }

    async def document_meta(self, document_ids: Sequence[str]) -> dict[str, DocumentMeta]:
        if not document_ids:
            return {}
        rows = await self._run(
            "MATCH (d:Document) WHERE d.document_id IN $ids AND d.payload IS NOT NULL "
            "RETURN d.document_id AS id, d.payload AS payload",
            {"ids": list(document_ids)},
        )
        return {
            str(row["id"]): DocumentMeta.model_validate(json.loads(row["payload"]))
            for row in rows if row.get("payload")
        }

    async def query(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Read-only escape hatch. Write clauses are rejected before execution."""
        lowered = cypher.lower()
        forbidden = ("create", "merge", "delete", "set ", "remove", "drop", "detach", "call db.")
        if any(token in lowered for token in forbidden):
            raise GraphStoreError(
                "Only read-only Cypher is permitted through this endpoint.",
                context={"rejected": next(t for t in forbidden if t in lowered)},
            )
        return await self._run(cypher, params)

    async def stats(self) -> dict[str, int]:
        totals = await self._run(
            "MATCH (n) WITH count(n) AS nodes "
            "MATCH ()-[r]->() RETURN nodes, count(r) AS relationships"
        )
        by_label = await self._run(
            "MATCH (n:Entity) WHERE n.label IS NOT NULL "
            "RETURN n.label AS label, count(*) AS count ORDER BY label"
        )
        by_rel = await self._run(
            "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count ORDER BY type"
        )
        head = totals[0] if totals else {}
        out: dict[str, int] = {
            "nodes": int(head.get("nodes") or 0),
            "relationships": int(head.get("relationships") or 0),
        }
        for row in by_label:
            out[f"node:{row['label']}"] = int(row["count"])
        for row in by_rel:
            out[f"rel:{row['type']}"] = int(row["count"])
        return out

    async def health(self) -> dict[str, Any]:
        try:
            rows = await self._run("MATCH (n) RETURN count(n) AS nodes")
            return {
                "ok": True,
                "backend": "neo4j",
                "detail": f"{rows[0]['nodes'] if rows else 0} nodes",
            }
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "backend": "neo4j", "detail": str(exc)}


__all__ = ["Neo4jGraphStore"]
