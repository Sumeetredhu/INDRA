"""Score fusion — the part of INDRA that decides what the answer is made of (D3).

The spec's formula is ``vector_score * 0.6 + graph_boost * 0.4``. Applied to raw numbers it does
not work, and the reason is worth stating precisely because everything downstream depends on it:

* A cosine similarity from a modern embedding model does not use its range. Real passages for a
  real query land in a **0.70–0.90 band**. The gap between the best and the worst passage is
  ~0.15, which after multiplying by 0.6 becomes a ~0.09 swing.
* A graph boost built from entity overlap, centrality, path confidence and recency has no such
  clustering. It legitimately spans 0.0 to 1.0, so after multiplying by 0.4 it swings ~0.40.

Blended raw, the graph term therefore decides the ranking roughly four times out of five, and the
vector term is decoration. Change the corpus size and the balance silently inverts. So:

**Both score families are min-max normalised across the candidate pool for this query, before
blending.** Then the configured weights mean what they say. Reciprocal Rank Fusion is available as
an alternative (``settings.fusion_strategy``) because it is scale-free by construction and does not
need the normalisation step at all.

The graph boost itself is a weighted sum over ``settings.graph_boost_weights`` of four factors:

======================  ====================================================================
``entity_overlap``      Jaccard of the query's resolved entities against the passage's.
``centrality``          How connected the matched entities are (from ``GraphStore.centrality``,
                        already 0–1).
``confidence``          Confidence of the graph path that connects the passage to the query.
``recency``             Exponential decay on document age, ``settings.recency_half_life_days``.
======================  ====================================================================

Every passage leaves this module with an ``explanation`` naming the numbers that put it where it
is. An unexplainable ranking is not auditable, and an unauditable ranking is not usable on a plant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Mapping, Sequence

from indra.core.config import FusionStrategy, Settings
from indra.core.logging import get_logger
from indra.core.models import Chunk, DocumentMeta, RetrievedPassage
from indra.agents.knowledge_graph_agent.entity_linking import display_name

logger = get_logger(__name__)

#: Below this, two floats are the same number as far as ranking is concerned.
_EPSILON: Final[float] = 1e-9

#: Names of the four graph-boost factors, validated by ``Settings.graph_boost_weights``.
_FACTOR_NAMES: Final[tuple[str, ...]] = ("entity_overlap", "centrality", "confidence", "recency")

#: Recency assumed when a document carries no date at all. Mid-scale on purpose: an undated
#: document should neither be promoted as fresh nor buried as ancient.
_UNDATED_RECENCY: Final[float] = 0.5

#: How many matched entities are named in an explanation before it elides.
_EXPLAIN_ENTITY_LIMIT: Final[int] = 4


# ======================================================================================
# Primitives
# ======================================================================================


def minmax_normalise(values: Sequence[float]) -> list[float]:
    """Min-max normalise a score family to ``[0, 1]``, per query.

    Degenerate spread — every candidate scoring the same — carries no ranking information. It maps
    to ``1.0`` when the family fired at all and ``0.0`` when it did not, which is rank-preserving
    either way and keeps a uniformly-strong family from being erased.
    """
    if not values:
        return []
    low = min(values)
    high = max(values)
    span = high - low
    if span <= _EPSILON:
        return [1.0 if high > _EPSILON else 0.0 for _ in values]
    return [(value - low) / span for value in values]


def jaccard(matched: int, query_count: int, passage_count: int) -> float:
    """Jaccard similarity of the query's entity set against the passage's.

    ``matched`` is the size of the intersection; the union is reconstructed from the two set sizes.
    Jaccard rather than raw overlap count on purpose: a passage that mentions both entities the
    question is about beats a passage that mentions one of them plus forty others.
    """
    if query_count <= 0 or matched <= 0:
        return 0.0
    passage = max(passage_count, matched)
    union = query_count + passage - matched
    if union <= 0:
        return 0.0
    return min(1.0, matched / union)


def recency_decay(age_days: float | None, half_life_days: float) -> float:
    """Exponential decay: a document exactly ``half_life_days`` old scores ``0.5``.

    Half-life rather than a cliff because plant knowledge decays smoothly — last month's shift log
    is more relevant than last year's, which is still more relevant than the 2009 commissioning
    report, and no threshold captures that.
    """
    if age_days is None:
        return _UNDATED_RECENCY
    if half_life_days <= 0:
        return _UNDATED_RECENCY
    clamped_age = max(0.0, float(age_days))
    return float(2.0 ** (-clamped_age / float(half_life_days)))


# ======================================================================================
# Candidates and boosts
# ======================================================================================


@dataclass(frozen=True, slots=True)
class Candidate:
    """One passage in contention, with every raw input the fusion needs.

    Assembled by :mod:`.graphrag`; this module never touches a store.
    """

    chunk: Chunk
    document: DocumentMeta
    #: Raw cosine similarity from the vector store, or ``0.0`` if it was found only by the graph.
    vector_score: float
    #: Raw relevance from ``GraphStore.chunks_for_entities``. Used for candidate selection and for
    #: the explanation; it is *not* a fifth boost factor, because the four weights sum to 1.0.
    graph_relevance: float
    #: Query entity keys this passage actually mentions.
    matched_entities: tuple[str, ...]
    #: Number of distinct entities the passage mentions in total, for the Jaccard denominator.
    passage_entity_count: int
    #: Mean 0–1 centrality of the matched entities.
    centrality: float
    #: Confidence of the best graph path connecting this passage to a query entity.
    path_confidence: float
    #: Graph distance from the nearest query entity. ``0`` means the passage names it directly.
    hops: int
    #: Document age in days, or ``None`` when undated.
    age_days: float | None

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def document_id(self) -> str:
        return self.document.document_id


@dataclass(frozen=True, slots=True)
class GraphBoost:
    """The four factors and their weighted total, kept decomposed so the score is auditable."""

    entity_overlap: float
    centrality: float
    confidence: float
    recency: float
    total: float

    def as_dict(self) -> dict[str, float]:
        return {
            "entity_overlap": round(self.entity_overlap, 4),
            "centrality": round(self.centrality, 4),
            "confidence": round(self.confidence, 4),
            "recency": round(self.recency, 4),
            "total": round(self.total, 4),
        }

    def dominant_factor(self) -> str:
        """The factor contributing most to the total — the headline of the explanation."""
        return max(
            zip(_FACTOR_NAMES, (self.entity_overlap, self.centrality, self.confidence, self.recency),
                strict=True),
            key=lambda item: item[1],
        )[0]


def compute_graph_boost(
    candidate: Candidate,
    *,
    query_entity_count: int,
    weights: Mapping[str, float],
    half_life_days: float,
) -> GraphBoost:
    """Compute the graph boost for one candidate.

    Args:
        candidate: The passage under consideration.
        query_entity_count: How many entities the query resolved to. Zero means the query named no
            entity, in which case entity overlap cannot discriminate and contributes nothing.
        weights: ``settings.graph_boost_weights``; validated to sum to 1.0 by ``Settings``.
        half_life_days: ``settings.recency_half_life_days``.

    Returns:
        A :class:`GraphBoost` in ``[0, 1]``.
    """
    overlap = jaccard(
        matched=len(candidate.matched_entities),
        query_count=query_entity_count,
        passage_count=candidate.passage_entity_count,
    )
    centrality = max(0.0, min(1.0, candidate.centrality))
    confidence = max(0.0, min(1.0, candidate.path_confidence))
    recency = recency_decay(candidate.age_days, half_life_days)

    total = (
        weights.get("entity_overlap", 0.0) * overlap
        + weights.get("centrality", 0.0) * centrality
        + weights.get("confidence", 0.0) * confidence
        + weights.get("recency", 0.0) * recency
    )
    return GraphBoost(
        entity_overlap=overlap,
        centrality=centrality,
        confidence=confidence,
        recency=recency,
        total=max(0.0, min(1.0, total)),
    )


# ======================================================================================
# The fuser
# ======================================================================================


@dataclass(frozen=True, slots=True)
class FusedScore:
    """Everything that went into one passage's final position, retained for the explanation."""

    candidate: Candidate
    boost: GraphBoost
    vector_normalised: float
    graph_normalised: float
    vector_rank: int | None
    graph_rank: int | None
    fused: float


class ScoreFuser:
    """Blends the vector and graph score families into one ranking.

    Stateless apart from settings, so it is trivially testable: hand it candidates, read the order.
    """

    __slots__ = ("_settings",)

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def strategy(self) -> FusionStrategy:
        return self._settings.fusion_strategy

    def fuse(
        self,
        candidates: Sequence[Candidate],
        *,
        query_entities: Sequence[str],
    ) -> list[RetrievedPassage]:
        """Rank ``candidates`` and return them as explained :class:`RetrievedPassage` objects.

        Args:
            candidates: The full per-query pool. Normalisation is computed across exactly this set,
                which is what "per-query normalisation" means — a candidate's score depends on its
                competition, not on a global calibration that does not exist.
            query_entities: Resolved entity keys for the query, used for the Jaccard denominator
                and named in the explanations.

        Returns:
            Passages sorted by fused score, descending.
        """
        if not candidates:
            return []

        weights = self._settings.graph_boost_weights
        boosts = [
            compute_graph_boost(
                candidate,
                query_entity_count=len(query_entities),
                weights=weights,
                half_life_days=self._settings.recency_half_life_days,
            )
            for candidate in candidates
        ]

        vector_normalised = minmax_normalise([c.vector_score for c in candidates])
        graph_normalised = minmax_normalise([b.total for b in boosts])

        vector_ranks = self._ranks([c.vector_score for c in candidates])
        graph_ranks = self._ranks([b.total for b in boosts])

        if self._settings.fusion_strategy is FusionStrategy.RRF:
            fused_values = self._reciprocal_rank_fusion(vector_ranks, graph_ranks)
        else:
            fused_values = [
                self._settings.retrieval_vector_weight * vector_normalised[i]
                + self._settings.retrieval_graph_weight * graph_normalised[i]
                for i in range(len(candidates))
            ]

        scored = [
            FusedScore(
                candidate=candidates[i],
                boost=boosts[i],
                vector_normalised=vector_normalised[i],
                graph_normalised=graph_normalised[i],
                vector_rank=vector_ranks[i],
                graph_rank=graph_ranks[i],
                fused=fused_values[i],
            )
            for i in range(len(candidates))
        ]
        scored.sort(key=lambda s: (-s.fused, -s.candidate.vector_score, s.candidate.chunk_id))

        passages = [
            RetrievedPassage(
                chunk=score.candidate.chunk,
                document=score.candidate.document,
                vector_score=round(score.candidate.vector_score, 6),
                graph_score=round(score.boost.total, 6),
                fused_score=round(score.fused, 6),
                hops=score.candidate.hops,
                matched_entities=list(score.candidate.matched_entities),
                explanation=self._explain(score, position=position, pool=len(scored),
                                          query_entities=query_entities),
            )
            for position, score in enumerate(scored, start=1)
        ]

        logger.debug(
            "fusion complete",
            extra={
                "strategy": self._settings.fusion_strategy.value,
                "candidates": len(candidates),
                "top_fused": passages[0].fused_score if passages else 0.0,
                "top_chunk": passages[0].chunk.chunk_id if passages else None,
            },
        )
        return passages

    # ---------------------------------------------------------------- internals

    @staticmethod
    def _ranks(values: Sequence[float]) -> list[int | None]:
        """1-based rank within a family, or ``None`` when the family did not return the candidate.

        A zero raw score means the family never saw the candidate — a chunk found only through the
        graph has no cosine, and giving it the worst vector rank would penalise it for something it
        was never entered in. Genuine RRF ranks only members of each list.
        """
        indexed = [(value, index) for index, value in enumerate(values) if value > _EPSILON]
        indexed.sort(key=lambda item: (-item[0], item[1]))
        ranks: list[int | None] = [None] * len(values)
        for position, (_, index) in enumerate(indexed, start=1):
            ranks[index] = position
        return ranks

    def _reciprocal_rank_fusion(
        self,
        vector_ranks: Sequence[int | None],
        graph_ranks: Sequence[int | None],
    ) -> list[float]:
        """Weighted RRF, rescaled to ``[0, 1]``.

        Raw RRF scores are tiny (``1/(60+1) ≈ 0.016``) and their absolute magnitude is meaningless,
        but ``SourceRef.relevance`` is a 0–1 quantity the UI renders. Dividing by the maximum —
        rather than min-max normalising — preserves the ratios between passages and keeps the
        weakest survivor above zero instead of pretending it is irrelevant.
        """
        k = float(self._settings.retrieval_rrf_k)
        vector_weight = self._settings.retrieval_vector_weight
        graph_weight = self._settings.retrieval_graph_weight

        raw: list[float] = []
        for vector_rank, graph_rank in zip(vector_ranks, graph_ranks, strict=True):
            score = 0.0
            if vector_rank is not None:
                score += vector_weight / (k + vector_rank)
            if graph_rank is not None:
                score += graph_weight / (k + graph_rank)
            raw.append(score)

        peak = max(raw, default=0.0)
        if peak <= _EPSILON:
            return raw
        return [value / peak for value in raw]

    def _explain(
        self,
        score: FusedScore,
        *,
        position: int,
        pool: int,
        query_entities: Sequence[str],
    ) -> str:
        """Write the sentence that says why this passage is where it is.

        Structure: position, the two families with raw *and* normalised values, the boost
        decomposition, and a plain-language reason naming the factor that actually decided it.
        """
        candidate = score.candidate
        boost = score.boost
        matched = [display_name(key) for key in candidate.matched_entities]
        matched_text = ", ".join(matched[:_EXPLAIN_ENTITY_LIMIT]) or "none"
        if len(matched) > _EXPLAIN_ENTITY_LIMIT:
            matched_text += f" (+{len(matched) - _EXPLAIN_ENTITY_LIMIT} more)"

        if self._settings.fusion_strategy is FusionStrategy.RRF:
            blend = (
                f"Reciprocal rank fusion (k={self._settings.retrieval_rrf_k}): "
                f"vector rank {score.vector_rank or '—'}, graph rank {score.graph_rank or '—'} "
                f"→ {score.fused:.3f}."
            )
        else:
            blend = (
                f"Weighted blend {self._settings.retrieval_vector_weight:.2f}·vector + "
                f"{self._settings.retrieval_graph_weight:.2f}·graph on per-query normalised scores "
                f"→ {score.fused:.3f}."
            )

        age = "undated" if candidate.age_days is None else f"{candidate.age_days:.0f} days old"
        reach = "names a query entity directly" if candidate.hops == 0 else f"reached in {candidate.hops} hop(s)"

        return (
            f"Ranked {position} of {pool}. "
            f"Vector similarity {candidate.vector_score:.3f} (normalised {score.vector_normalised:.2f}); "
            f"graph boost {boost.total:.3f} (normalised {score.graph_normalised:.2f}) from "
            f"entity overlap {boost.entity_overlap:.2f} on [{matched_text}], "
            f"centrality {boost.centrality:.2f}, path confidence {boost.confidence:.2f}, "
            f"recency {boost.recency:.2f} ({age}). {blend} "
            f"Selected because it {self._reason(score, query_entities)}; {reach}."
        )

    def _reason(self, score: FusedScore, query_entities: Sequence[str]) -> str:
        """One clause naming why this passage beat the others.

        Deliberately not generated by an LLM: the reason is a fact about the arithmetic that just
        ran, and inventing prose for it would be exactly the unsupported assertion INDRA exists to
        avoid.
        """
        candidate = score.candidate
        matched = len(candidate.matched_entities)

        if matched >= 2 and matched >= len(query_entities) and len(query_entities) >= 2:
            names = ", ".join(display_name(k) for k in candidate.matched_entities[:_EXPLAIN_ENTITY_LIMIT])
            return (
                f"ties together every entity the question is about ({names}) — a link no "
                f"single-subject passage provides"
            )
        if score.graph_normalised > score.vector_normalised + 0.1:
            factor = score.boost.dominant_factor().replace("_", " ")
            return f"the graph evidence carried it: strongest contribution from {factor}"
        if score.vector_normalised > score.graph_normalised + 0.1:
            return "its wording is the closest match to the question in this candidate pool"
        if candidate.hops > 0:
            return "the graph connected it to the question even though its wording did not"
        return "vector similarity and graph evidence agreed on it"


def summarise_fusion(passages: Sequence[RetrievedPassage]) -> dict[str, float]:
    """Compact metrics for logging and ``/metrics``."""
    if not passages:
        return {"count": 0.0, "top_score": 0.0, "mean_score": 0.0, "graph_only": 0.0}
    scores = [p.fused_score for p in passages]
    graph_only = sum(1 for p in passages if p.vector_score <= _EPSILON)
    return {
        "count": float(len(passages)),
        "top_score": round(max(scores), 4),
        "mean_score": round(math.fsum(scores) / len(scores), 4),
        "graph_only": float(graph_only),
    }


__all__ = [
    "Candidate",
    "FusedScore",
    "GraphBoost",
    "ScoreFuser",
    "compute_graph_boost",
    "jaccard",
    "minmax_normalise",
    "recency_decay",
    "summarise_fusion",
]
