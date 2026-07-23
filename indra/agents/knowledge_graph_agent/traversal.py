"""Bounded graph expansion with cycle guards, producing readable :class:`GraphPath` objects.

Three depths, three different questions:

======  =========================================================================================
1 hop   *Direct* facts. "What is P-101 connected to? Who maintained it?"
2 hops  *Indirect* facts — the ones no single document contains. "What failure mode has been seen
        on equipment that shares a discharge header with P-101?"
3 hops  *Temporal chains.* "Which sequence of events preceded the last failure of this type?"
======  =========================================================================================

Two things make this module more than a wrapper over ``GraphStore.neighbours``:

* **Cycle guards.** ``A—B—A`` is a real path in a graph and a useless one in an explanation. Any
  path revisiting a node is discarded before it reaches the fusion stage.
* **Backend levelling.** ``GraphStore.neighbours`` accepts a ``hops`` argument, but a backend is
  free to honour only the first hop. When the returned paths are shallower than requested, this
  module stitches the remaining levels itself from 1-hop calls. Retrieval quality therefore does
  not depend on which store happens to be bound (D1).

The ``narrative`` string is not decoration. It is the sentence the Copilot's "Explain How I Know
This" panel shows, so it is generated here where the path is still structured, not reconstructed
later from prose.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Iterable, Sequence

from indra.core.config import Settings
from indra.core.exceptions import GraphStoreError
from indra.core.logging import get_logger
from indra.core.models import GraphPath, RelationType
from indra.agents.knowledge_graph_agent._guards import degraded
from indra.agents.knowledge_graph_agent.entity_linking import display_name

if TYPE_CHECKING:  # pragma: no cover - typing only
    from indra.core.contracts import GraphStore

logger = get_logger(__name__)

#: Rendered between two nodes, e.g. ``P-101 —CONNECTED_TO→ V-201``.
_ARROW: Final[str] = " —{relation}→ "

#: How many nodes from one BFS level are expanded into the next. A plant graph has hub nodes (a
#: utility header touches everything); without a cap, hop three of a 3-hop expansion enumerates the
#: plant. Ranked by path confidence, so the cap keeps the *best* frontier, not an arbitrary one.
_FRONTIER_LIMIT: Final[int] = 24

#: Confidence assumed for an edge whose store did not record one. Deliberately below 1.0: an edge
#: with no recorded confidence is an edge nobody vouched for.
_DEFAULT_EDGE_CONFIDENCE: Final[float] = 0.9

#: Multiplicative penalty per extra hop, applied when this module stitches a path itself. Distance
#: costs certainty: a 3-hop inference is genuinely weaker than the three facts it chains.
_HOP_DECAY: Final[float] = 0.9


def render_narrative(nodes: Sequence[str], relations: Sequence[RelationType]) -> str:
    """Render a path as a sentence: ``P-101 —CONNECTED_TO→ V-201 —MAINTAINED→ WO-2024-0342``.

    Node keys are stripped of their ``<Type>:`` prefix, because an operator reads asset tags, not
    graph identifiers.
    """
    if not nodes:
        return ""
    parts: list[str] = [display_name(nodes[0])]
    for index, node in enumerate(nodes[1:]):
        relation = relations[index].value if index < len(relations) else "RELATED_TO"
        parts.append(_ARROW.format(relation=relation))
        parts.append(display_name(node))
    return "".join(parts)


@dataclass(frozen=True, slots=True)
class PathHit:
    """The best way the traversal found to reach one entity from the query's seeds."""

    key: str
    hops: int
    confidence: float
    seed: str
    path: GraphPath

    def is_seed(self) -> bool:
        return self.hops == 0


@dataclass(slots=True)
class ExpansionResult:
    """Everything one expansion produced."""

    seeds: list[str] = field(default_factory=list)
    paths: list[GraphPath] = field(default_factory=list)
    #: Best hit per reachable entity key, including the seeds themselves at 0 hops.
    reachable: dict[str, PathHit] = field(default_factory=dict)
    max_depth_reached: int = 0
    stitched: bool = False
    truncated: bool = False

    @property
    def keys(self) -> list[str]:
        """Every entity key reachable from the seeds, seeds first, then by increasing distance."""
        return [
            hit.key
            for hit in sorted(
                self.reachable.values(), key=lambda h: (h.hops, -h.confidence, h.key)
            )
        ]

    def confidence_for(self, key: str) -> float:
        """Confidence of the best path to ``key``; ``1.0`` for a seed, ``0.0`` if unreachable."""
        hit = self.reachable.get(key)
        return hit.confidence if hit else 0.0

    def hops_for(self, key: str) -> int:
        """Distance to ``key`` from the nearest seed; ``0`` for a seed itself."""
        hit = self.reachable.get(key)
        return hit.hops if hit else 0

    def summary(self) -> dict[str, int]:
        return {
            "seeds": len(self.seeds),
            "paths": len(self.paths),
            "reachable": len(self.reachable),
            "max_depth": self.max_depth_reached,
        }


class GraphTraverser:
    """Expands a set of seed entities outward through the knowledge graph."""

    __slots__ = ("_graph", "_settings")

    def __init__(self, graph: GraphStore, settings: Settings) -> None:
        self._graph = graph
        self._settings = settings

    async def expand(
        self,
        seeds: Sequence[str],
        *,
        hops: int | None = None,
        relation_types: Sequence[RelationType | str] | None = None,
        limit: int | None = None,
    ) -> ExpansionResult:
        """Expand outward from ``seeds``, returning deduplicated acyclic paths.

        Args:
            seeds: Entity keys to start from, e.g. ``["Equipment:P-101"]``.
            hops: Maximum path length. Clamped to ``settings.max_hops``.
            relation_types: Restrict traversal to these edge types. ``None`` means any.
            limit: Maximum paths to return per seed. Defaults to ``settings.graph_top_k``.

        Returns:
            An :class:`ExpansionResult`. Never raises for an unreachable seed — an entity with no
            edges yet is a normal state, especially on the first document.

        Raises:
            GraphStoreError: Only if *every* seed query failed, which means the graph is down
                rather than empty.
        """
        bounded_hops = max(1, min(int(hops) if hops is not None else self._settings.max_hops,
                                  self._settings.max_hops))
        per_seed_limit = max(1, int(limit) if limit is not None else self._settings.graph_top_k)
        wanted = self._relation_names(relation_types)

        result = ExpansionResult(seeds=list(dict.fromkeys(seeds)))
        if not result.seeds:
            return result

        for seed in result.seeds:
            result.reachable[seed] = PathHit(
                key=seed,
                hops=0,
                confidence=1.0,
                seed=seed,
                path=GraphPath(nodes=[seed], relations=[], hops=0, confidence=1.0,
                               narrative=display_name(seed)),
            )

        raw_batches = await asyncio.gather(
            *(
                self._neighbours(seed, hops=bounded_hops, relation_types=relation_types,
                                 limit=per_seed_limit)
                for seed in result.seeds
            )
        )
        if all(batch is None for batch in raw_batches):
            raise GraphStoreError(
                "Graph expansion failed for every query entity; the knowledge graph is "
                "unreachable. Check the Neo4j connection or set INDRA_STORAGE_BACKEND=memory.",
                context={"seeds": result.seeds[:5], "hops": bounded_hops},
            )

        seen: set[tuple[str, ...]] = set()
        for seed, batch in zip(result.seeds, raw_batches, strict=True):
            for path in batch or ():
                self._accept(result, seed, path, wanted, seen)

        if bounded_hops > 1 and result.max_depth_reached < bounded_hops:
            # The backend answered only the shallow part of the request. Finish the job here so
            # that 2- and 3-hop reasoning works identically on every store.
            await self._stitch(result, wanted, bounded_hops, per_seed_limit, seen)
            result.stitched = True

        result.paths.sort(key=lambda p: (p.hops, -p.confidence, p.narrative))
        total_cap = per_seed_limit * max(1, len(result.seeds))
        if len(result.paths) > total_cap:
            result.paths = result.paths[:total_cap]
            result.truncated = True

        logger.debug("graph expansion complete", extra=result.summary())
        return result

    # ---------------------------------------------------------------- internals

    @staticmethod
    def _relation_names(relation_types: Sequence[RelationType | str] | None) -> frozenset[str] | None:
        """Normalise the relation filter to a set of names, or ``None`` for "any type"."""
        if not relation_types:
            return None
        return frozenset(
            item.value if isinstance(item, RelationType) else str(item) for item in relation_types
        )

    async def _neighbours(
        self,
        seed: str,
        *,
        hops: int,
        relation_types: Sequence[RelationType | str] | None,
        limit: int,
    ) -> list[GraphPath] | None:
        """One guarded ``neighbours`` call. Returns ``None`` when the store failed."""
        names: list[str] | None = None
        if relation_types:
            names = [item.value if isinstance(item, RelationType) else str(item) for item in relation_types]

        async def call() -> list[GraphPath] | None:
            return await self._graph.neighbours(
                seed, hops=hops, relation_types=names, limit=limit
            )

        return await degraded(
            f"graph.neighbours({seed})",
            call,
            fallback=None,
            capability="graph expansion from this entity",
            seed=seed,
            hops=hops,
        )

    def _accept(
        self,
        result: ExpansionResult,
        seed: str,
        path: GraphPath,
        wanted: frozenset[str] | None,
        seen: set[tuple[str, ...]],
    ) -> None:
        """Validate, deduplicate and record one path returned by the store."""
        if not path.nodes:
            return
        if len(set(path.nodes)) != len(path.nodes):
            return  # cycle guard: a path that revisits a node explains nothing
        if wanted is not None and any(rel.value not in wanted for rel in path.relations):
            return
        if path.hops > self._settings.max_hops:
            return

        identity = (*path.nodes, *(rel.value for rel in path.relations))
        if identity in seen:
            return
        seen.add(identity)

        narrative = path.narrative or render_narrative(path.nodes, path.relations)
        stored = path if path.narrative else path.model_copy(update={"narrative": narrative})
        result.paths.append(stored)
        result.max_depth_reached = max(result.max_depth_reached, stored.hops)

        endpoint = stored.nodes[-1]
        self._record(result, endpoint, stored.hops, float(stored.confidence), seed, stored)

    @staticmethod
    def _record(
        result: ExpansionResult,
        key: str,
        hops: int,
        confidence: float,
        seed: str,
        path: GraphPath,
    ) -> None:
        """Keep the shortest, then most confident, route to ``key``."""
        existing = result.reachable.get(key)
        if existing is not None and (existing.hops, -existing.confidence) <= (hops, -confidence):
            return
        result.reachable[key] = PathHit(
            key=key, hops=hops, confidence=confidence, seed=seed, path=path
        )

    async def _stitch(
        self,
        result: ExpansionResult,
        wanted: frozenset[str] | None,
        target_hops: int,
        limit: int,
        seen: set[tuple[str, ...]],
    ) -> None:
        """Extend shallow results to ``target_hops`` with breadth-first 1-hop calls.

        Runs only when the backend under-delivered. Each level expands at most
        :data:`_FRONTIER_LIMIT` nodes, chosen by path confidence, so a hub node cannot make the
        third hop enumerate the plant.
        """
        depth = max(1, result.max_depth_reached)
        while depth < target_hops:
            frontier = [
                hit for hit in result.reachable.values() if hit.hops == depth
            ]
            if not frontier:
                return
            frontier.sort(key=lambda h: -h.confidence)
            if len(frontier) > _FRONTIER_LIMIT:
                frontier = frontier[:_FRONTIER_LIMIT]
                result.truncated = True

            batches = await asyncio.gather(
                *(
                    self._neighbours(hit.key, hops=1, relation_types=None, limit=limit)
                    for hit in frontier
                )
            )
            grew = False
            for hit, batch in zip(frontier, batches, strict=True):
                for extension in batch or ():
                    joined = self._join(hit, extension, wanted)
                    if joined is None:
                        continue
                    identity = (*joined.nodes, *(rel.value for rel in joined.relations))
                    if identity in seen:
                        continue
                    seen.add(identity)
                    result.paths.append(joined)
                    result.max_depth_reached = max(result.max_depth_reached, joined.hops)
                    self._record(result, joined.nodes[-1], joined.hops,
                                 float(joined.confidence), hit.seed, joined)
                    grew = True
            if not grew:
                return
            depth += 1

    def _join(
        self,
        prefix: PathHit,
        extension: GraphPath,
        wanted: frozenset[str] | None,
    ) -> GraphPath | None:
        """Concatenate a 1-hop extension onto an existing path, or reject it.

        Rejects when the extension does not start where the prefix ends, when the join would create
        a cycle, when a relation type is filtered out, or when the result exceeds ``max_hops``.
        """
        if not extension.nodes or len(extension.nodes) < 2:
            return None
        if extension.nodes[0] != prefix.key:
            return None
        if wanted is not None and any(rel.value not in wanted for rel in extension.relations):
            return None

        tail = extension.nodes[1:]
        nodes = [*prefix.path.nodes, *tail]
        if len(set(nodes)) != len(nodes):
            return None  # cycle guard
        relations = [*prefix.path.relations, *extension.relations]
        hops = len(nodes) - 1
        if hops > self._settings.max_hops:
            return None

        extension_confidence = float(extension.confidence) or _DEFAULT_EDGE_CONFIDENCE
        confidence = max(0.0, min(1.0, prefix.confidence * extension_confidence * _HOP_DECAY))
        return GraphPath(
            nodes=nodes,
            relations=relations,
            hops=hops,
            confidence=round(confidence, 4),
            narrative=render_narrative(nodes, relations),
        )


def paths_touching(paths: Iterable[GraphPath], keys: Iterable[str]) -> list[GraphPath]:
    """Filter ``paths`` down to those that pass through any of ``keys``.

    Used to attach exactly the paths that justify a passage to that passage's explanation, rather
    than dumping the whole expansion into every citation.
    """
    wanted = set(keys)
    return [path for path in paths if wanted.intersection(path.nodes)]


__all__ = [
    "ExpansionResult",
    "GraphTraverser",
    "PathHit",
    "paths_touching",
    "render_narrative",
]
