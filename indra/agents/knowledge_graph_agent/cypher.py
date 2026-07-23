"""Parameterised Cypher builders.

**No user input is ever formatted into a query string.** Every value travels in the ``params`` dict.

Labels and relationship types are the one thing Cypher cannot parameterise — ``MATCH (n:$label)``
is not valid syntax. So they are never taken from the caller verbatim either: they are validated
against the allowlists in :mod:`.schema`, which are derived from the ``EntityType`` and
``RelationType`` enums. An unknown label is a programming error and raises rather than being
interpolated. That closes the only injection vector the language leaves open.

Every builder returns a :class:`CypherQuery` so the query text and its parameters travel together
and can be attached verbatim to a :class:`~indra.core.models.ReasoningStep` for the
"Explain How I Know This" panel.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Final, Iterable, Sequence

from indra.core.exceptions import GraphStoreError
from indra.core.models import EntityType, RelationType
from indra.agents.knowledge_graph_agent.schema import ALLOWED_LABELS, ALLOWED_RELATION_TYPES

#: Clauses that mutate. Used by :func:`assert_read_only` to gate the ``/graph/cypher`` escape hatch.
_WRITE_CLAUSES: Final[frozenset[str]] = frozenset(
    {
        "create", "merge", "delete", "detach", "set", "remove", "drop",
        "foreach", "load", "call", "using", "grant", "revoke", "deny", "start",
    }
)

#: Tokenises a Cypher statement into bare words, ignoring string literals so that a *value*
#: containing the word "delete" cannot trip the read-only guard.
_STRING_LITERAL_RE: Final[re.Pattern[str]] = re.compile(r"'[^']*'|\"[^\"]*\"")
_COMMENT_RE: Final[re.Pattern[str]] = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
_WORD_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z_]+")

#: Hard ceiling on any ``LIMIT`` this module emits. A caller asking for 10 million rows is a bug or
#: an attack; either way the graph should not try.
MAX_LIMIT: Final[int] = 1000


@dataclass(frozen=True, slots=True)
class CypherQuery:
    """A query and its parameters, kept together.

    Attributes:
        text: The statement. Contains only literals authored in this module plus ``$param`` markers.
        params: Every caller-supplied value.
        purpose: Short human description, rendered in the reasoning chain.
    """

    text: str
    params: dict[str, Any] = field(default_factory=dict)
    purpose: str = ""

    def as_tuple(self) -> tuple[str, dict[str, Any]]:
        """Convenience for ``await graph.query(*q.as_tuple())``."""
        return self.text, dict(self.params)

    def render_for_display(self) -> str:
        """The statement with a parameter summary appended, for the explainability panel.

        Values are *not* substituted — this is documentation, not an executable string.
        """
        if not self.params:
            return self.text
        keys = ", ".join(f"${k}" for k in sorted(self.params))
        return f"{self.text}\n// parameters: {keys}"


# ======================================================================================
# Identifier validation
# ======================================================================================


def validate_label(label: str | EntityType) -> str:
    """Return ``label`` if it is a known node label, else raise.

    Raises:
        GraphStoreError: The label is not declared in :mod:`.schema`. This is always a code defect —
            user input never reaches this function.
    """
    value = label.value if isinstance(label, EntityType) else str(label)
    if value not in ALLOWED_LABELS:
        raise GraphStoreError(
            f"Refusing to build Cypher for unknown node label {value!r}. "
            f"Add a NodeSpec to indra.agents.knowledge_graph_agent.schema before using it.",
            context={"label": value, "allowed": sorted(ALLOWED_LABELS)},
        )
    return value


def validate_relation_types(types: Iterable[str | RelationType] | None) -> tuple[str, ...]:
    """Return a validated tuple of relationship type names, or ``()`` for "any type".

    Raises:
        GraphStoreError: One of the types is not declared in :mod:`.schema`.
    """
    if types is None:
        return ()
    validated: list[str] = []
    for item in types:
        value = item.value if isinstance(item, RelationType) else str(item)
        if value not in ALLOWED_RELATION_TYPES:
            raise GraphStoreError(
                f"Refusing to build Cypher for unknown relationship type {value!r}. "
                f"Add a RelationshipSpec to indra.agents.knowledge_graph_agent.schema first.",
                context={"relation_type": value, "allowed": sorted(ALLOWED_RELATION_TYPES)},
            )
        if value not in validated:
            validated.append(value)
    return tuple(validated)


def clamp_limit(limit: int) -> int:
    """Clamp a caller-supplied limit into ``[1, MAX_LIMIT]``.

    The clamped value is still passed as a parameter, never formatted into the text.
    """
    return max(1, min(int(limit), MAX_LIMIT))


def relation_filter_fragment(types: Sequence[str]) -> str:
    """Render a validated relationship-type filter, e.g. ``:CONNECTED_TO|MAINTAINED``.

    Every element must already have passed :func:`validate_relation_types`.
    """
    return f":{'|'.join(types)}" if types else ""


def assert_read_only(statement: str) -> None:
    """Raise unless ``statement`` is a pure read.

    Backs the read-only Cypher endpoint. String literals and comments are stripped first, so a
    query searching for the *text* ``"delete"`` is not rejected while ``DETACH DELETE`` is.

    Raises:
        GraphStoreError: The statement contains a mutating clause or a procedure call.
    """
    stripped = _COMMENT_RE.sub(" ", statement)
    stripped = _STRING_LITERAL_RE.sub(" ", stripped)
    words = {word.lower() for word in _WORD_RE.findall(stripped)}
    offending = sorted(words & _WRITE_CLAUSES)
    if offending:
        raise GraphStoreError(
            f"Refusing to run a Cypher statement containing {offending!r}: this endpoint is "
            f"read-only. Use the agent's index() path for writes.",
            context={"clauses": offending},
        )
    if "match" not in words and "return" not in words:
        raise GraphStoreError(
            "Refusing to run a Cypher statement with neither MATCH nor RETURN. "
            "Read-only queries must start from a MATCH.",
            context={},
        )


# ======================================================================================
# Write builders
# ======================================================================================


def upsert_document(properties: dict[str, Any]) -> CypherQuery:
    """MERGE a ``Document`` node on ``document_id`` and overwrite its properties."""
    return CypherQuery(
        text=(
            "MERGE (d:Document {document_id: $document_id})\n"
            "SET d += $properties, d.updated_at = timestamp()\n"
            "RETURN d.document_id AS document_id"
        ),
        params={"document_id": properties.get("document_id"), "properties": properties},
        purpose="Upsert the source document node",
    )


def upsert_entity(label: str | EntityType, key: str, properties: dict[str, Any]) -> CypherQuery:
    """MERGE one entity node on its universal ``key`` property.

    ``key`` is :attr:`indra.core.models.ExtractedEntity.key` — the same string the entity resolver
    merges on — so a node written from a work order and a node written from a P&ID collapse.
    """
    validated = validate_label(label)
    return CypherQuery(
        text=(
            f"MERGE (n:{validated} {{key: $key}})\n"
            "SET n += $properties, n.updated_at = timestamp()\n"
            "RETURN n.key AS key"
        ),
        params={"key": key, "properties": properties},
        purpose=f"Upsert a {validated} node",
    )


def upsert_relationship(
    relation_type: str | RelationType,
    *,
    source_key: str,
    target_key: str,
    properties: dict[str, Any],
) -> CypherQuery:
    """MERGE an edge between two nodes addressed by their universal ``key``.

    The nodes are matched by ``key`` regardless of label, so an edge can join an ``Equipment`` to a
    ``FailureMode`` without the builder needing to know either label.
    """
    types = validate_relation_types([relation_type])
    return CypherQuery(
        text=(
            "MATCH (a {key: $source_key})\n"
            "MATCH (b {key: $target_key})\n"
            f"MERGE (a)-[r:{types[0]}]->(b)\n"
            "SET r += $properties, r.updated_at = timestamp()\n"
            "RETURN type(r) AS type"
        ),
        params={"source_key": source_key, "target_key": target_key, "properties": properties},
        purpose=f"Upsert a {types[0]} relationship",
    )


def upsert_equipment(properties: dict[str, Any]) -> CypherQuery:
    """MERGE an ``Equipment`` node on its plant tag — the plant-wide primary key."""
    tag = str(properties.get("tag", "")).strip().upper()
    return CypherQuery(
        text=(
            "MERGE (e:Equipment {tag: $tag})\n"
            "SET e += $properties, e.key = $key, e.updated_at = timestamp()\n"
            "RETURN e.tag AS tag"
        ),
        params={"tag": tag, "key": f"{EntityType.EQUIPMENT.value}:{tag}", "properties": properties},
        purpose="Upsert an equipment node",
    )


def upsert_person(properties: dict[str, Any]) -> CypherQuery:
    """MERGE a ``Person`` node on ``person_id``."""
    return CypherQuery(
        text=(
            "MERGE (p:Person {person_id: $person_id})\n"
            "SET p += $properties, p.updated_at = timestamp()\n"
            "RETURN p.person_id AS person_id"
        ),
        params={"person_id": properties.get("person_id"), "properties": properties},
        purpose="Upsert a person node",
    )


def delete_document(document_id: str) -> CypherQuery:
    """Detach-delete a document and the ``MENTIONS`` edges it owns.

    Entity nodes survive: they are shared across documents, and deleting one because a single
    source went away would silently destroy provenance from the others.
    """
    return CypherQuery(
        text=(
            "MATCH (d:Document {document_id: $document_id})\n"
            "OPTIONAL MATCH (d)-[r:MENTIONS]->()\n"
            "DELETE r\n"
            "WITH d\n"
            "DETACH DELETE d"
        ),
        params={"document_id": document_id},
        purpose="Remove a document and its provenance edges",
    )


# ======================================================================================
# Read builders
# ======================================================================================


def get_equipment(tag: str) -> CypherQuery:
    """Fetch one equipment node by tag."""
    return CypherQuery(
        text="MATCH (e:Equipment {tag: $tag}) RETURN e LIMIT 1",
        params={"tag": tag.strip().upper()},
        purpose="Look up equipment by tag",
    )


def list_equipment(*, criticality: str | None = None, limit: int = 500) -> CypherQuery:
    """List equipment, optionally filtered by criticality class."""
    if criticality is None:
        text = "MATCH (e:Equipment) RETURN e ORDER BY e.tag LIMIT $limit"
        params: dict[str, Any] = {"limit": clamp_limit(limit)}
    else:
        text = (
            "MATCH (e:Equipment) WHERE e.criticality = $criticality "
            "RETURN e ORDER BY e.tag LIMIT $limit"
        )
        params = {"criticality": criticality, "limit": clamp_limit(limit)}
    return CypherQuery(text=text, params=params, purpose="List equipment")


def neighbours(
    entity_key: str,
    *,
    hops: int = 1,
    relation_types: Sequence[str] | None = None,
    limit: int = 50,
) -> CypherQuery:
    """Variable-length expansion from one node.

    ``hops`` bounds the pattern length. It is validated to ``1..4`` and rendered into the pattern
    because Cypher does not accept a parameter inside ``*1..n``; the value never originates from
    user text, only from ``settings.max_hops`` and the API's validated query model.
    """
    bounded = max(1, min(int(hops), 4))
    types = validate_relation_types(relation_types)
    return CypherQuery(
        text=(
            "MATCH (start {key: $entity_key})\n"
            f"MATCH path = (start)-[{relation_filter_fragment(types)}*1..{bounded}]-(other)\n"
            "WHERE other.key <> $entity_key\n"
            "WITH path, other,\n"
            "     reduce(c = 1.0, r IN relationships(path) | c * coalesce(r.confidence, 0.9)) AS confidence\n"
            "RETURN [n IN nodes(path) | n.key] AS nodes,\n"
            "       [r IN relationships(path) | type(r)] AS relations,\n"
            "       length(path) AS hops,\n"
            "       confidence\n"
            "ORDER BY hops ASC, confidence DESC\n"
            "LIMIT $limit"
        ),
        params={"entity_key": entity_key, "limit": clamp_limit(limit)},
        purpose=f"Expand up to {bounded} hop(s) from {entity_key}",
    )


def chunks_for_entities(entity_keys: Sequence[str], *, limit: int = 50) -> CypherQuery:
    """Chunks that mention any of these entities, scored by mention confidence and multiplicity.

    ``graph_relevance`` rewards a chunk that mentions *several* of the query's entities, which is
    exactly the cross-document passage the fusion stage is trying to surface.
    """
    return CypherQuery(
        text=(
            "MATCH (d:Document)-[m:MENTIONS]->(e)\n"
            "WHERE e.key IN $entity_keys AND m.chunk_id IS NOT NULL\n"
            "WITH m.chunk_id AS chunk_id,\n"
            "     count(DISTINCT e.key) AS matched,\n"
            "     avg(coalesce(m.confidence, 0.8)) AS mention_confidence\n"
            "RETURN chunk_id,\n"
            "       (toFloat(matched) / toFloat($entity_count)) * mention_confidence AS graph_relevance\n"
            "ORDER BY graph_relevance DESC\n"
            "LIMIT $limit"
        ),
        params={
            "entity_keys": list(entity_keys),
            "entity_count": max(1, len(entity_keys)),
            "limit": clamp_limit(limit),
        },
        purpose="Find chunks mentioning the query's entities",
    )


def centrality(entity_keys: Sequence[str]) -> CypherQuery:
    """Degree centrality for the given keys, normalised against the busiest node in the result.

    Degree — not PageRank — on purpose: it is cheap, it is stable under incremental writes, and on a
    plant graph "how many things touch this asset" is the quantity an engineer would actually name.
    """
    return CypherQuery(
        text=(
            "MATCH (n)\n"
            "WHERE n.key IN $entity_keys\n"
            "WITH n, size([(n)--() | 1]) AS degree\n"
            "WITH collect({key: n.key, degree: degree}) AS rows,\n"
            "     max(degree) AS peak\n"
            "UNWIND rows AS row\n"
            "RETURN row.key AS key,\n"
            "       CASE WHEN peak > 0 THEN toFloat(row.degree) / toFloat(peak) ELSE 0.0 END AS centrality"
        ),
        params={"entity_keys": list(entity_keys)},
        purpose="Score how connected the query's entities are",
    )


def maintenance_history(tag: str, *, since: date | None = None, limit: int = 200) -> CypherQuery:
    """Maintenance edges on one asset, newest first."""
    text = (
        "MATCH (e:Equipment {tag: $tag})-[r:MAINTAINED]-(other)\n"
        "WHERE ($since IS NULL OR r.date >= $since)\n"
        "RETURN r AS record, other AS counterpart\n"
        "ORDER BY r.date DESC\n"
        "LIMIT $limit"
    )
    return CypherQuery(
        text=text,
        params={
            "tag": tag.strip().upper(),
            "since": since.isoformat() if since else None,
            "limit": clamp_limit(limit),
        },
        purpose="Read the maintenance history of one asset",
    )


def failure_history(tag: str, *, since: date | None = None, limit: int = 200) -> CypherQuery:
    """Failure edges on one asset, newest first."""
    return CypherQuery(
        text=(
            "MATCH (e:Equipment {tag: $tag})-[r:FAILED_WITH_MODE]->(mode:FailureMode)\n"
            "WHERE ($since IS NULL OR r.date >= $since)\n"
            "RETURN r AS event, mode.name AS failure_mode\n"
            "ORDER BY r.date DESC\n"
            "LIMIT $limit"
        ),
        params={
            "tag": tag.strip().upper(),
            "since": since.isoformat() if since else None,
            "limit": clamp_limit(limit),
        },
        purpose="Read the failure history of one asset",
    )


def procedures_for(tag: str, *, limit: int = 100) -> CypherQuery:
    """Procedures that apply to one asset, directly or through its parent."""
    return CypherQuery(
        text=(
            "MATCH (e:Equipment {tag: $tag})\n"
            "OPTIONAL MATCH (e)-[:PART_OF*0..2]->(parent:Equipment)\n"
            "WITH collect(DISTINCT e) + collect(DISTINCT parent) AS assets\n"
            "UNWIND assets AS asset\n"
            "MATCH (p:Procedure)-[:APPLIES_TO]->(asset)\n"
            "RETURN DISTINCT p AS procedure\n"
            "LIMIT $limit"
        ),
        params={"tag": tag.strip().upper(), "limit": clamp_limit(limit)},
        purpose="Find procedures covering one asset",
    )


def temporal_chain(entity_key: str, *, limit: int = 50) -> CypherQuery:
    """Follow ``PRECEDED_BY`` edges to reconstruct a 3-hop temporal sequence.

    This is the query behind "these three unrelated-looking events are the same story".
    """
    return CypherQuery(
        text=(
            "MATCH path = (start {key: $entity_key})-[:PRECEDED_BY*1..3]->(earlier)\n"
            "WITH path, reduce(g = 0, r IN relationships(path) | g + coalesce(r.gap_days, 0)) AS span_days\n"
            "RETURN [n IN nodes(path) | n.key] AS nodes,\n"
            "       [r IN relationships(path) | type(r)] AS relations,\n"
            "       length(path) AS hops,\n"
            "       span_days\n"
            "ORDER BY span_days ASC\n"
            "LIMIT $limit"
        ),
        params={"entity_key": entity_key, "limit": clamp_limit(limit)},
        purpose="Reconstruct a temporal chain",
    )


def subgraph_for_preview(entity_keys: Sequence[str], *, hops: int = 2, limit: int = 60) -> CypherQuery:
    """The bounded neighbourhood the React-Flow preview renders."""
    bounded = max(1, min(int(hops), 4))
    return CypherQuery(
        text=(
            "MATCH (seed)\n"
            "WHERE seed.key IN $entity_keys\n"
            f"OPTIONAL MATCH path = (seed)-[*1..{bounded}]-(other)\n"
            "WITH seed, path, other\n"
            "LIMIT $limit\n"
            "RETURN seed.key AS seed_key,\n"
            "       [n IN coalesce(nodes(path), []) | {key: n.key, labels: labels(n)}] AS nodes,\n"
            "       [r IN coalesce(relationships(path), []) | \n"
            "           {type: type(r), start: startNode(r).key, end: endNode(r).key, \n"
            "            confidence: coalesce(r.confidence, 0.9)}] AS edges"
        ),
        params={"entity_keys": list(entity_keys), "limit": clamp_limit(limit)},
        purpose="Fetch the neighbourhood for the graph preview",
    )


def document_meta(document_ids: Sequence[str]) -> CypherQuery:
    """Fetch document nodes in one round trip."""
    return CypherQuery(
        text="MATCH (d:Document) WHERE d.document_id IN $document_ids RETURN d",
        params={"document_ids": list(document_ids)},
        purpose="Load document metadata for citations",
    )


def graph_stats() -> CypherQuery:
    """Node and relationship counts by label, for the visualisation header."""
    return CypherQuery(
        text=(
            "MATCH (n)\n"
            "WITH labels(n) AS labels\n"
            "UNWIND labels AS label\n"
            "RETURN label AS name, count(*) AS count\n"
            "ORDER BY count DESC"
        ),
        params={},
        purpose="Count nodes by label",
    )


__all__ = [
    "MAX_LIMIT",
    "CypherQuery",
    "assert_read_only",
    "centrality",
    "chunks_for_entities",
    "clamp_limit",
    "delete_document",
    "document_meta",
    "failure_history",
    "get_equipment",
    "graph_stats",
    "list_equipment",
    "maintenance_history",
    "neighbours",
    "procedures_for",
    "relation_filter_fragment",
    "subgraph_for_preview",
    "temporal_chain",
    "upsert_document",
    "upsert_entity",
    "upsert_equipment",
    "upsert_person",
    "upsert_relationship",
    "validate_label",
    "validate_relation_types",
]
