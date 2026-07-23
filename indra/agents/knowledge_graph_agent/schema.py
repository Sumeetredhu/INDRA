"""Knowledge-graph schema: node/relationship definitions and idempotent DDL.

The graph is the product's memory. Its shape is declared here once, as data, so that:

* the Neo4j DDL, the in-memory store's expectations and the documentation cannot drift apart;
* ``ensure_schema()`` is a loop over a list rather than a wall of string literals;
* every property that appears in a ``WHERE`` has a declared index, which is the difference between
  a 3-hop traversal returning in 40 ms and in 4 s.

**Backend neutrality.** ``GraphStore.query`` is documented as *"rejected when the backend is
in-memory"*. So :func:`ensure_schema` probes with the first statement: if the backend refuses raw
Cypher it logs one informational line and skips the rest. The in-memory store enforces the same
uniqueness semantics in Python, so the graph behaves identically either way (D1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

from indra.core.exceptions import GraphStoreError, IndraError
from indra.core.logging import get_logger
from indra.core.models import EntityType, RelationType

if TYPE_CHECKING:  # pragma: no cover - typing only
    from indra.core.contracts import GraphStore

logger = get_logger(__name__)


# ======================================================================================
# Declarative schema
# ======================================================================================


@dataclass(frozen=True, slots=True)
class NodeSpec:
    """One node label, its identity property, and the properties worth indexing.

    Args:
        label: Neo4j label, identical to an :class:`~indra.core.models.EntityType` value.
        key_property: The property carrying node identity. A uniqueness constraint is created on it.
        indexed: Properties that appear in a ``WHERE`` somewhere in :mod:`.cypher`.
        description: Why this label exists, for the schema endpoint and for humans.
    """

    label: str
    key_property: str
    indexed: tuple[str, ...] = ()
    description: str = ""

    @property
    def constraint_name(self) -> str:
        return f"indra_{self.label.lower()}_{self.key_property.lower()}_unique"


@dataclass(frozen=True, slots=True)
class RelationshipSpec:
    """One relationship type and the properties that carry its meaning.

    The properties *are* the point: an edge with no ``date``, ``confidence`` or ``source_document``
    cannot support an explanation, and an unexplainable edge is worse than no edge.
    """

    type: RelationType
    properties: tuple[str, ...] = ()
    indexed: tuple[str, ...] = ()
    description: str = ""


#: Every node label INDRA writes. ``key`` is the universal identity property
#: (:attr:`indra.core.models.ExtractedEntity.key`); ``Equipment``, ``Document`` and ``Person`` also
#: carry a domain-natural key because the rest of the platform addresses them by it.
NODE_SPECS: Final[tuple[NodeSpec, ...]] = (
    NodeSpec(
        label=EntityType.EQUIPMENT.value,
        key_property="tag",
        indexed=("key", "criticality", "equipment_type", "unit", "location", "manufacturer"),
        description="A physical asset. The plant-wide join key across every document type.",
    ),
    NodeSpec(
        label=EntityType.DOCUMENT.value,
        key_property="document_id",
        indexed=("key", "content_hash", "document_type", "document_date", "ingested_at", "title"),
        description="A source document. Every claim in INDRA traces back to one of these.",
    ),
    NodeSpec(
        label=EntityType.PERSON.value,
        key_property="person_id",
        indexed=("key", "name", "retirement_date", "role"),
        description="A plant person. The unit of the knowledge-cliff calculation.",
    ),
    NodeSpec(
        label=EntityType.FAILURE_MODE.value,
        key_property="key",
        indexed=("name",),
        description="A named way an asset fails, e.g. 'bearing seizure'.",
    ),
    NodeSpec(
        label=EntityType.PROCEDURE.value,
        key_property="key",
        indexed=("name", "title", "revision"),
        description="An SOP or maintenance procedure.",
    ),
    NodeSpec(
        label=EntityType.REGULATORY_CLAUSE.value,
        key_property="key",
        indexed=("name", "regulation", "clause", "frequency_days"),
        description="One atomic obligation parsed out of a regulation.",
    ),
    NodeSpec(
        label=EntityType.CONDITION_READING.value,
        key_property="key",
        indexed=("parameter", "measured_at", "equipment_tag"),
        description="A measured value with a timestamp.",
    ),
    NodeSpec(
        label=EntityType.MEASUREMENT.value,
        key_property="key",
        indexed=("name", "unit"),
        description="A quantity mentioned in text, normalised where possible.",
    ),
    NodeSpec(
        label=EntityType.DATE.value,
        key_property="key",
        indexed=("name",),
        description="A date mentioned in text; anchors PRECEDED_BY temporal chains.",
    ),
    NodeSpec(
        label=EntityType.LOCATION.value,
        key_property="key",
        indexed=("name",),
        description="A plant area, unit or physical location.",
    ),
    NodeSpec(
        label=EntityType.MATERIAL.value,
        key_property="key",
        indexed=("name",),
        description="A process fluid, lubricant or material of construction.",
    ),
    NodeSpec(
        label=EntityType.ORGANISATION.value,
        key_property="key",
        indexed=("name",),
        description="An OEM, contractor or regulator.",
    ),
)

#: Every relationship type, with the properties that make it explainable.
RELATIONSHIP_SPECS: Final[tuple[RelationshipSpec, ...]] = (
    RelationshipSpec(
        type=RelationType.CONNECTED_TO,
        properties=("pipe_spec", "flow_direction", "confidence", "source_document"),
        indexed=("confidence",),
        description="Process connectivity traced from a P&ID. Powers 'what is downstream of P-101'.",
    ),
    RelationshipSpec(
        type=RelationType.MAINTAINED,
        properties=("date", "findings", "recommendations", "performed_by", "work_order", "confidence"),
        indexed=("date",),
        description="A work order or planned maintenance touch on an asset.",
    ),
    RelationshipSpec(
        type=RelationType.FAILED_WITH_MODE,
        properties=("date", "root_cause", "downtime_hours", "cost_inr", "confidence"),
        indexed=("date",),
        description="A historical failure, with the cost that makes the business case concrete.",
    ),
    RelationshipSpec(
        type=RelationType.MENTIONS,
        properties=("chunk_id", "char_start", "char_end", "confidence", "page"),
        indexed=("chunk_id",),
        description="Document → entity provenance. The backbone of chunk-level graph retrieval.",
    ),
    RelationshipSpec(
        type=RelationType.HAS_EXPERTISE,
        properties=("years", "retirement_date", "documented_count", "confidence"),
        indexed=("retirement_date",),
        description="Person → equipment tacit knowledge. Drives the knowledge cliff.",
    ),
    RelationshipSpec(
        type=RelationType.REQUIRES,
        properties=("clause", "frequency_days", "deadline", "confidence"),
        indexed=("deadline",),
        description="An obligation an asset must satisfy.",
    ),
    RelationshipSpec(
        type=RelationType.APPLIES_TO,
        properties=("clause", "frequency_days", "deadline", "confidence"),
        indexed=("deadline",),
        description="Regulatory clause → asset or asset class.",
    ),
    RelationshipSpec(
        type=RelationType.PRECEDED_BY,
        properties=("gap_days", "confidence"),
        indexed=("gap_days",),
        description="Temporal sequencing. This is what makes 3-hop causal chains readable.",
    ),
    RelationshipSpec(
        type=RelationType.DOCUMENTED_BY,
        properties=("date", "confidence"),
        description="Entity → the document that records it.",
    ),
    RelationshipSpec(
        type=RelationType.INSPECTED_BY,
        properties=("date", "findings", "confidence"),
        description="Asset → inspector or inspection body.",
    ),
    RelationshipSpec(
        type=RelationType.PART_OF,
        properties=("confidence",),
        description="Component → parent asset or unit.",
    ),
    RelationshipSpec(
        type=RelationType.SIMILAR_TO,
        properties=("similarity", "basis", "confidence"),
        indexed=("similarity",),
        description="Fleet-level similarity, used to transfer a failure precursor between assets.",
    ),
    RelationshipSpec(
        type=RelationType.CAUSED_BY,
        properties=("confidence", "evidence"),
        description="Failure → root cause.",
    ),
    RelationshipSpec(
        type=RelationType.RESOLVED_BY,
        properties=("date", "confidence"),
        description="Failure or gap → the action that closed it.",
    ),
    RelationshipSpec(
        type=RelationType.SUPERSEDES,
        properties=("date",),
        description="Document revision chain (D6). Keeps a stale manual from outranking its update.",
    ),
)

#: Labels the Cypher builders will accept. Anything else is a programming error, not user input.
ALLOWED_LABELS: Final[frozenset[str]] = frozenset(spec.label for spec in NODE_SPECS)

#: Relationship types the Cypher builders will accept.
ALLOWED_RELATION_TYPES: Final[frozenset[str]] = frozenset(spec.type.value for spec in RELATIONSHIP_SPECS)


# ======================================================================================
# DDL rendering
# ======================================================================================


def constraint_statements() -> list[str]:
    """Return idempotent uniqueness-constraint DDL for every node label.

    Uses Neo4j 5.x ``IF NOT EXISTS`` syntax, so re-running on every startup is free.
    """
    statements: list[str] = []
    for spec in NODE_SPECS:
        statements.append(
            f"CREATE CONSTRAINT {spec.constraint_name} IF NOT EXISTS "
            f"FOR (n:{spec.label}) REQUIRE n.{spec.key_property} IS UNIQUE"
        )
    return statements


def index_statements() -> list[str]:
    """Return idempotent index DDL for every property used in a ``WHERE`` or an ``ORDER BY``."""
    statements: list[str] = []
    for spec in NODE_SPECS:
        for prop in spec.indexed:
            if prop == spec.key_property:
                continue  # the uniqueness constraint already backs this with an index
            statements.append(
                f"CREATE INDEX indra_{spec.label.lower()}_{prop.lower()}_idx IF NOT EXISTS "
                f"FOR (n:{spec.label}) ON (n.{prop})"
            )
    for rel in RELATIONSHIP_SPECS:
        for prop in rel.indexed:
            statements.append(
                f"CREATE INDEX indra_rel_{rel.type.value.lower()}_{prop.lower()}_idx IF NOT EXISTS "
                f"FOR ()-[r:{rel.type.value}]-() ON (r.{prop})"
            )
    return statements


def schema_statements() -> list[str]:
    """Constraints first, then indexes. Order matters: a constraint creates its backing index."""
    return [*constraint_statements(), *index_statements()]


def describe_schema() -> dict[str, object]:
    """Serialisable schema description, surfaced by ``/graph`` and by :meth:`health`."""
    return {
        "nodes": [
            {
                "label": spec.label,
                "key": spec.key_property,
                "indexed": list(spec.indexed),
                "description": spec.description,
            }
            for spec in NODE_SPECS
        ],
        "relationships": [
            {
                "type": spec.type.value,
                "properties": list(spec.properties),
                "indexed": list(spec.indexed),
                "description": spec.description,
            }
            for spec in RELATIONSHIP_SPECS
        ],
        "constraints": len(constraint_statements()),
        "indexes": len(index_statements()),
    }


# ======================================================================================
# Application
# ======================================================================================


@dataclass(slots=True)
class SchemaReport:
    """Outcome of :func:`ensure_schema`, logged at startup and shown by ``health()``."""

    applied: int = 0
    skipped: int = 0
    ddl_supported: bool = True
    base_ensured: bool = False
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "applied": self.applied,
            "skipped": self.skipped,
            "ddl_supported": self.ddl_supported,
            "base_ensured": self.base_ensured,
            "errors": list(self.errors),
        }


async def ensure_schema(graph: GraphStore) -> SchemaReport:
    """Create constraints and indexes. Idempotent; safe on every startup.

    Two layers, in order:

    1. ``graph.ensure_schema()`` — whatever the bound backend needs for its own bookkeeping.
    2. The domain DDL declared in this module, issued through ``graph.query``.

    Step 2 is *best effort by design*. The in-memory backend rejects raw Cypher, and that is a
    supported configuration, not a failure. The first rejection flips ``ddl_supported`` and the
    remaining statements are skipped without noise.

    Args:
        graph: The bound graph store.

    Returns:
        A :class:`SchemaReport` describing what was applied.

    Raises:
        GraphStoreError: Only if the backend's own ``ensure_schema`` fails, which means the graph is
            unusable and startup should say so loudly.
    """
    report = SchemaReport()

    try:
        await graph.ensure_schema()
        report.base_ensured = True
    except IndraError as exc:
        raise GraphStoreError(
            "Graph backend rejected ensure_schema(); the knowledge graph cannot be written. "
            "Check the Neo4j connection settings or set INDRA_STORAGE_BACKEND=memory.",
            context={"error": exc.message},
            cause=exc,
        ) from exc
    except Exception as exc:  # noqa: BLE001 - boundary: unknown backend failure becomes typed
        raise GraphStoreError(
            "Graph backend raised an unexpected error during ensure_schema(). "
            "Check the Neo4j connection settings or set INDRA_STORAGE_BACKEND=memory.",
            context={"error": repr(exc)},
            cause=exc,
        ) from exc

    for statement in schema_statements():
        if not report.ddl_supported:
            report.skipped += 1
            continue
        try:
            await graph.query(statement)
            report.applied += 1
        except Exception as exc:  # noqa: BLE001 - boundary: a DDL refusal must never fail startup
            report.ddl_supported = False
            report.skipped += 1
            report.errors.append(repr(exc))
            logger.info(
                "graph backend does not accept schema DDL; relying on backend-native constraints",
                extra={"reason": repr(exc), "statements_skipped": len(schema_statements()) - report.applied},
            )

    logger.info(
        "knowledge graph schema ensured",
        extra={
            "constraints": len(constraint_statements()),
            "indexes": len(index_statements()),
            "applied": report.applied,
            "skipped": report.skipped,
            "ddl_supported": report.ddl_supported,
        },
    )
    return report


__all__ = [
    "ALLOWED_LABELS",
    "ALLOWED_RELATION_TYPES",
    "NODE_SPECS",
    "RELATIONSHIP_SPECS",
    "NodeSpec",
    "RelationshipSpec",
    "SchemaReport",
    "constraint_statements",
    "describe_schema",
    "ensure_schema",
    "index_statements",
    "schema_statements",
]
