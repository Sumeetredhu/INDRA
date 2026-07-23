"""The GraphRAG retrieval pipeline — where vector recall and graph reasoning become one ranking.

Everything else in this package is a component. This module is the sequence that turns a question
into evidence:

============  =================================================================================
1. resolve    The query's own entities, through :class:`~...entity_linking.EntityResolver`.
2. embed      One embedding of the query text (``task="query"``).
3. recall     Dense search for ``settings.vector_top_k`` chunks.
4. expand     Bounded traversal from the query's entities to ``settings.graph_top_k`` neighbours,
              and the chunks that mention them. This is the half a pure-vector RAG cannot do.
5. fuse       :class:`~...fusion.ScoreFuser` — weighted blend on per-query min-max normalised
              families, or Reciprocal Rank Fusion (D3).
6. diversify  MMR over the fused ranking, so one verbose OEM manual cannot occupy every slot and
              crowd out the two-line shift log that actually holds the answer.
7. assemble   Pack the winners into ``settings.context_window_tokens``.
============  =================================================================================

**Nothing here fails the request.** Each external family — embeddings, the vector store, the graph
— degrades independently (CLAUDE.md rule 6). Losing the embedder leaves graph-only retrieval;
losing the graph leaves dense-only retrieval; losing *both* is the one case that raises
:class:`~indra.core.exceptions.RetrievalError`, because then there is genuinely nothing to say.

**Recency is anchored on the corpus, not on the clock.** ``scripts/demo_facts.py`` is explicit that
consumers windowing on evidence recency must anchor on the newest document rather than
``datetime.now()``; a corpus generated last year would otherwise have every passage decay to the
same near-zero recency and the factor would stop discriminating. Age is therefore measured against
the newest dated document in the candidate pool.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any, Final, Iterable, Sequence

from indra.core.config import Settings
from indra.core.deps import AgentDeps
from indra.core.exceptions import RetrievalError
from indra.core.logging import get_logger
from indra.core.models import (
    Chunk,
    DocumentMeta,
    EntityType,
    GraphPath,
    RetrievalResult,
    RetrievedPassage,
)
from indra.agents.knowledge_graph_agent._guards import degraded
from indra.agents.knowledge_graph_agent.entity_linking import (
    EntityResolver,
    canonical_tag,
    entity_key,
    key_type,
)
from indra.agents.knowledge_graph_agent.fusion import Candidate, ScoreFuser, summarise_fusion
from indra.agents.knowledge_graph_agent.traversal import (
    ExpansionResult,
    GraphTraverser,
    paths_touching,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from indra.core.contracts import GraphStore, MetadataStore, VectorStore

logger = get_logger(__name__)

#: MMR trade-off. ``1.0`` is pure relevance (no diversity), ``0.0`` is pure novelty. 0.72 keeps
#: relevance clearly dominant — the top passage is always the top-fused one — while still letting a
#: second, differently-sourced passage outrank a near-duplicate of the winner.
_MMR_LAMBDA: Final[float] = 0.72

#: Redundancy floor between two passages from the *same document*. Two paragraphs of one manual are
#: substitutable evidence even when their wording differs, and the whole point of the diversity pass
#: is that a long document must not spend every slot in the context window.
_SAME_DOCUMENT_REDUNDANCY: Final[float] = 0.5

#: Characters per token when no tokenizer is available. GPT-family English averages ~4.
_CHARS_PER_TOKEN: Final[float] = 4.0

#: Words shorter than this carry no topical signal in the lexical similarity fallback.
_MIN_CONTENT_WORD: Final[int] = 4

#: Never return an empty ranking purely because of the relevance floor: the best of a weak pool is
#: still the honest answer to "what do you have", and the Copilot decides whether it is enough.
_MIN_KEPT_PASSAGES: Final[int] = 1

#: Lazily-initialised ``tiktoken`` encoding. ``None`` means "not tried yet", ``False`` means
#: "tried and unavailable", so the import is attempted exactly once per process.
_ENCODING: Any = None


# ======================================================================================
# Token accounting
# ======================================================================================


def _encoding() -> Any:
    """Return a ``tiktoken`` encoding, or ``None`` if the package is unusable.

    Imported lazily and exactly once: ``tiktoken`` costs ~100 ms to initialise and retrieval must
    not pay that on a cold path that may never need it.
    """
    global _ENCODING
    if _ENCODING is None:
        try:
            import tiktoken

            _ENCODING = tiktoken.get_encoding("cl100k_base")
        except Exception as exc:  # noqa: BLE001 - optional dependency, any failure is the same
            logger.debug("tiktoken unavailable; using character heuristic", extra={"error": repr(exc)})
            _ENCODING = False
    return _ENCODING or None


def estimate_tokens(text: str) -> int:
    """Estimate the token cost of ``text``.

    Exact when ``tiktoken`` is importable, a character heuristic otherwise. The heuristic errs
    slightly high, which is the safe direction: overestimating trims one passage, underestimating
    overflows the model's context.
    """
    if not text:
        return 0
    encoder = _encoding()
    if encoder is not None:
        try:
            return len(encoder.encode(text))
        except Exception:  # noqa: BLE001 - a tokenizer failure must not break retrieval
            pass
    return max(1, int(math.ceil(len(text) / _CHARS_PER_TOKEN)))


def chunk_tokens(chunk: Chunk) -> int:
    """Token cost of a chunk, trusting the ingestion agent's count when it recorded one."""
    if chunk.token_count > 0:
        return chunk.token_count
    return estimate_tokens(chunk.text)


# ======================================================================================
# Diversity
# ======================================================================================


def _content_words(text: str) -> frozenset[str]:
    """Lowercased content words, used by the lexical similarity fallback."""
    return frozenset(
        word for word in "".join(c if c.isalnum() else " " for c in text.lower()).split()
        if len(word) >= _MIN_CONTENT_WORD
    )


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity mapped to ``[0, 1]``. Returns ``0.0`` for degenerate vectors."""
    width = min(len(left), len(right))
    if width == 0:
        return 0.0
    dot = math.fsum(left[i] * right[i] for i in range(width))
    left_norm = math.sqrt(math.fsum(value * value for value in left[:width]))
    right_norm = math.sqrt(math.fsum(value * value for value in right[:width]))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return max(0.0, min(1.0, (dot / (left_norm * right_norm) + 1.0) / 2.0))


def passage_redundancy(left: RetrievedPassage, right: RetrievedPassage) -> float:
    """How much of ``left`` is already covered by ``right``, in ``[0, 1]``.

    Two signals, combined with ``max`` because either one alone is sufficient grounds to call a
    passage redundant:

    * **Same document.** Bounded below by :data:`_SAME_DOCUMENT_REDUNDANCY` — the diversity pass
      exists precisely to stop one document owning the whole context window.
    * **Content overlap.** Embedding cosine when both chunks carry vectors (the vector store
      normalises and returns them), Jaccard over content words otherwise.
    """
    same_document = _SAME_DOCUMENT_REDUNDANCY if left.chunk.document_id == right.chunk.document_id else 0.0

    if left.chunk.embedding and right.chunk.embedding:
        content = _cosine(left.chunk.embedding, right.chunk.embedding)
    else:
        left_words = _content_words(left.chunk.text)
        right_words = _content_words(right.chunk.text)
        union = left_words | right_words
        content = len(left_words & right_words) / len(union) if union else 0.0

    return max(same_document, content)


def mmr_rerank(passages: Sequence[RetrievedPassage], *, k: int) -> list[RetrievedPassage]:
    """Maximal Marginal Relevance over an already-fused ranking.

    Greedy: repeatedly take the passage maximising
    ``λ · fused_score − (1 − λ) · max redundancy against everything already chosen``.

    The first pick is always the top-fused passage, so diversity never costs the best answer. Every
    later pick records the penalty it survived in its ``explanation``, because a ranking the
    operator cannot account for is not auditable.

    Args:
        passages: Fused passages, best first.
        k: How many to keep.

    Returns:
        Up to ``k`` passages in presentation order.
    """
    if k <= 0 or not passages:
        return []
    if len(passages) <= 1:
        return list(passages[:k])

    remaining = list(passages)
    selected: list[RetrievedPassage] = [remaining.pop(0)]

    while remaining and len(selected) < k:
        best_index = 0
        best_value = -math.inf
        best_penalty = 0.0
        for index, candidate in enumerate(remaining):
            penalty = max(passage_redundancy(candidate, chosen) for chosen in selected)
            value = _MMR_LAMBDA * candidate.fused_score - (1.0 - _MMR_LAMBDA) * penalty
            if value > best_value:
                best_value, best_index, best_penalty = value, index, penalty
        winner = remaining.pop(best_index)
        if best_penalty > 0.0:
            winner = winner.model_copy(
                update={
                    "explanation": (
                        f"{winner.explanation} Diversity pass: {best_penalty:.2f} redundancy "
                        f"against already-selected evidence, MMR value {best_value:.3f} "
                        f"(λ={_MMR_LAMBDA:.2f})."
                    )
                }
            )
        selected.append(winner)

    return selected


# ======================================================================================
# Context window
# ======================================================================================


@dataclass(frozen=True, slots=True)
class ContextWindow:
    """The passages that fit the model's budget, and what it cost to get there."""

    passages: list[RetrievedPassage]
    tokens: int
    dropped: int
    budget: int

    def summary(self) -> dict[str, int]:
        return {
            "passages": len(self.passages),
            "tokens": self.tokens,
            "dropped": self.dropped,
            "budget": self.budget,
        }


def assemble_context(passages: Sequence[RetrievedPassage], *, budget_tokens: int) -> ContextWindow:
    """Pack ``passages`` into a token budget, best first.

    A passage that does not fit is skipped rather than ending the loop: a 900-token manual extract
    in slot three must not evict the 40-token shift-log line in slot four, which is exactly the
    failure mode this whole module is built to avoid.
    """
    kept: list[RetrievedPassage] = []
    used = 0
    dropped = 0
    budget = max(1, int(budget_tokens))

    for passage in passages:
        cost = chunk_tokens(passage.chunk)
        if kept and used + cost > budget:
            dropped += 1
            continue
        kept.append(passage)
        used += cost

    return ContextWindow(passages=kept, tokens=used, dropped=dropped, budget=budget)


# ======================================================================================
# Candidate assembly
# ======================================================================================


@dataclass(slots=True)
class _Pool:
    """Raw material for one query, gathered before any scoring happens."""

    chunks: list[Chunk] = field(default_factory=list)
    documents: dict[str, DocumentMeta] = field(default_factory=dict)
    vector_hits: dict[str, float] = field(default_factory=dict)
    graph_hits: dict[str, float] = field(default_factory=dict)
    centrality: dict[str, float] = field(default_factory=dict)
    vector_ok: bool = True
    graph_ok: bool = True


def _document_anchor(documents: Iterable[DocumentMeta]) -> date | None:
    """The date every age in this pool is measured against: the newest document present."""
    dates = [meta.document_date for meta in documents if meta.document_date is not None]
    return max(dates) if dates else None


def _passage_entity_keys(chunk: Chunk, resolver: EntityResolver) -> list[str]:
    """Every entity key this passage mentions.

    Three sources, unioned, because different ingestion paths populate different ones:

    1. ``chunk.entity_ids`` entries that are already node keys.
    2. ``chunk.metadata["entity_keys"]``, which a parser may attach directly.
    3. The resolver's own scan of the text — always available, and the only source that works for a
       chunk written before the entity extractor ran.
    """
    keys: list[str] = []
    seen: set[str] = set()

    def _admit(value: object) -> None:
        if not isinstance(value, str) or ":" not in value:
            return
        if key_type(value) is None:
            return
        if value not in seen:
            seen.add(value)
            keys.append(value)

    for entity_id in chunk.entity_ids:
        _admit(entity_id)
    metadata_keys = chunk.metadata.get("entity_keys")
    if isinstance(metadata_keys, (list, tuple)):
        for value in metadata_keys:
            _admit(value)
    for value in resolver.resolve_text(chunk.text):
        _admit(value)

    return keys


# ======================================================================================
# The retriever
# ======================================================================================


class GraphRAGRetriever:
    """Hybrid vector + graph retrieval with score fusion, diversity and a token budget.

    Owned by :class:`~indra.agents.knowledge_graph_agent.service.KnowledgeGraphAgent`, which shares
    its long-lived :class:`EntityResolver` so that query-time entity resolution benefits from every
    document indexed so far.
    """

    __slots__ = ("_cache", "_deps", "_fuser", "_graph", "_metadata", "_resolver", "_settings",
                 "_traverser", "_vectors")

    def __init__(self, deps: AgentDeps, resolver: EntityResolver) -> None:
        self._deps = deps
        self._settings: Settings = deps.settings
        self._resolver = resolver
        self._graph: GraphStore = deps.graph
        self._vectors: VectorStore = deps.vectors
        self._metadata: MetadataStore = deps.metadata
        self._cache = deps.cache
        self._fuser = ScoreFuser(deps.settings)
        self._traverser = GraphTraverser(deps.graph, deps.settings)

    # ---------------------------------------------------------------- public API

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        equipment_tag: str | None = None,
        max_hops: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """Run the full hybrid pipeline for one question.

        Args:
            query: The user's question, verbatim.
            top_k: Passages to return. Defaults to ``settings.final_top_k``.
            equipment_tag: Force this asset into the seed set even if the text did not name it —
                used by the mobile agent, where the tag came from a photo rather than from words.
            max_hops: Traversal depth, clamped to ``settings.max_hops``.
            filters: Passed through to the vector store, e.g. ``{"document_id": [...]}``.

        Returns:
            A :class:`RetrievalResult` carrying passages *and* the graph paths that justify them.
            Empty is a valid result: an un-ingested corpus has no evidence, and saying so is
            correct behaviour.

        Raises:
            RetrievalError: Only when dense recall and graph expansion both failed — that is a dead
                platform, not an empty one.
        """
        started = time.perf_counter()
        settings = self._settings
        final_k = max(1, int(top_k) if top_k is not None else settings.final_top_k)

        seeds = self._seed_entities(query, equipment_tag)
        embedding, expansion = await asyncio.gather(
            self._embed(query),
            self._expand(seeds, max_hops),
        )

        pool = _Pool(vector_ok=embedding is not None, graph_ok=expansion is not None)
        expanded = expansion if expansion is not None else ExpansionResult(seeds=list(seeds))

        vector_ranked = await self._vector_search(embedding, filters) if embedding is not None else None
        if vector_ranked is None:
            pool.vector_ok = False
        pool.vector_hits = dict(vector_ranked or ())

        graph_ranked = await self._graph_chunks(expanded)
        if graph_ranked is None:
            pool.graph_ok = False
        pool.graph_hits = dict(graph_ranked or ())

        if not pool.vector_ok and not pool.graph_ok:
            raise RetrievalError(
                "Both dense recall and graph expansion failed, so no evidence could be assembled. "
                "Check the vector and graph backends, or restart with INDRA_STORAGE_BACKEND=memory "
                "to run fully in-process.",
                context={"query": query[:200], "seeds": seeds[:5]},
            )

        chunk_ids = self._merge_candidate_ids(pool)
        pool.chunks = await self._load_chunks(chunk_ids)
        pool.documents = await self._load_documents(pool.chunks)
        pool.centrality = await self._centrality(expanded, pool)

        candidates = await asyncio.to_thread(self._build_candidates, pool, expanded)
        passages = self._fuser.fuse(candidates, query_entities=seeds)
        passages = self._apply_relevance_floor(passages)
        passages = mmr_rerank(passages, k=final_k)
        window = assemble_context(passages, budget_tokens=settings.context_window_tokens)

        paths = self._paths_for(expanded, window.passages, seeds)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        result = RetrievalResult(
            query=query,
            query_entities=seeds,
            passages=window.passages,
            paths=paths,
            strategy=settings.fusion_strategy.value,  # type: ignore[arg-type]
            total_candidates=len(candidates),
            retrieval_ms=round(elapsed_ms, 3),
        )
        logger.info(
            "graphrag retrieval complete",
            extra={
                "query_entities": len(seeds),
                "vector_hits": len(pool.vector_hits),
                "graph_hits": len(pool.graph_hits),
                "candidates": len(candidates),
                "returned": len(result.passages),
                "paths": len(result.paths),
                "vector_ok": pool.vector_ok,
                "graph_ok": pool.graph_ok,
                "strategy": settings.fusion_strategy.value,
                "elapsed_ms": round(elapsed_ms, 1),
                **window.summary(),
                **summarise_fusion(result.passages),
            },
        )
        return result

    # ---------------------------------------------------------------- stages

    def _seed_entities(self, query: str, equipment_tag: str | None) -> list[str]:
        """Resolve the entities the question is about, an explicit tag first if one was supplied."""
        keys: list[str] = []
        if equipment_tag:
            tag = canonical_tag(equipment_tag) or equipment_tag.strip().upper()
            keys.append(entity_key(EntityType.EQUIPMENT, tag))
        for key in self._resolver.resolve_text(query):
            if key not in keys:
                keys.append(key)
        return keys

    async def _embed(self, query: str) -> list[float] | None:
        """Embed the query. ``None`` means dense recall is unavailable for this request."""

        async def call() -> list[float] | None:
            vectors = await self._deps.llm.embed([query], task="query")
            return list(vectors[0]) if vectors else None

        return await degraded(
            "llm.embed(query)",
            call,
            fallback=None,
            capability="dense vector recall (falling back to graph-only retrieval)",
            query_length=len(query),
        )

    async def _vector_search(
        self, embedding: Sequence[float], filters: dict[str, Any] | None
    ) -> list[tuple[str, float]] | None:
        """Dense top-k. ``None`` means the vector store failed."""

        async def call() -> list[tuple[str, float]]:
            return await self._vectors.search(
                embedding, top_k=self._settings.vector_top_k, filters=filters
            )

        return await degraded(
            "vectors.search",
            call,
            fallback=None,
            capability="dense vector recall (falling back to graph-only retrieval)",
            top_k=self._settings.vector_top_k,
        )

    async def _expand(self, seeds: Sequence[str], max_hops: int | None) -> ExpansionResult | None:
        """Traverse outward from the query's entities. ``None`` means the graph failed."""
        if not seeds:
            return ExpansionResult()

        async def call() -> ExpansionResult:
            return await self._traverser.expand(
                seeds, hops=max_hops, limit=self._settings.graph_top_k
            )

        return await degraded(
            "graph expansion",
            call,
            fallback=None,
            capability="graph reasoning (falling back to vector-only retrieval)",
            seeds=list(seeds)[:5],
        )

    async def _graph_chunks(self, expansion: ExpansionResult) -> list[tuple[str, float]] | None:
        """Chunks mentioning the expanded entity set. ``None`` means the graph failed."""
        keys = expansion.keys[: self._settings.graph_top_k]
        if not keys:
            return []

        async def call() -> list[tuple[str, float]]:
            return await self._graph.chunks_for_entities(keys, limit=self._settings.graph_top_k)

        return await degraded(
            "graph.chunks_for_entities",
            call,
            fallback=None,
            capability="graph-sourced passages",
            entities=len(keys),
        )

    def _merge_candidate_ids(self, pool: _Pool) -> list[str]:
        """Union of both families' chunk ids, strongest first within each, vector family first."""
        ordered: list[str] = []
        seen: set[str] = set()
        for source in (
            sorted(pool.vector_hits.items(), key=lambda item: -item[1]),
            sorted(pool.graph_hits.items(), key=lambda item: -item[1]),
        ):
            for chunk_id, _score in source:
                if chunk_id not in seen:
                    seen.add(chunk_id)
                    ordered.append(chunk_id)
        ceiling = self._settings.vector_top_k + self._settings.graph_top_k
        return ordered[:ceiling]

    async def _load_chunks(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        """Hydrate candidate chunks. An empty list degrades to a paths-only answer."""
        if not chunk_ids:
            return []

        async def call() -> list[Chunk]:
            return await self._vectors.get_chunks(chunk_ids)

        return await degraded(
            "vectors.get_chunks",
            call,
            fallback=[],
            capability="passage text (graph paths are still returned)",
            requested=len(chunk_ids),
        )

    async def _load_documents(self, chunks: Sequence[Chunk]) -> dict[str, DocumentMeta]:
        """Resolve every candidate's document metadata: graph first, metadata store for the rest.

        A chunk whose document cannot be resolved is *not* dropped — losing evidence is worse than
        citing it under a placeholder title — but the gap is logged so it can be fixed upstream.
        """
        wanted = list(dict.fromkeys(chunk.document_id for chunk in chunks))
        if not wanted:
            return {}

        async def from_graph() -> dict[str, DocumentMeta]:
            return await self._graph.document_meta(wanted)

        found = await degraded(
            "graph.document_meta",
            from_graph,
            fallback={},
            capability="document metadata from the graph",
            requested=len(wanted),
        )
        resolved: dict[str, DocumentMeta] = dict(found)

        missing = [document_id for document_id in wanted if document_id not in resolved]
        if missing:
            recovered = await asyncio.gather(
                *(self._document_from_metadata(document_id) for document_id in missing)
            )
            for document_id, meta in zip(missing, recovered, strict=True):
                resolved[document_id] = meta or self._placeholder_document(document_id)

        return resolved

    async def _document_from_metadata(self, document_id: str) -> DocumentMeta | None:
        async def call() -> DocumentMeta | None:
            return await self._metadata.get_document(document_id)

        return await degraded(
            "metadata.get_document",
            call,
            fallback=None,
            capability="document metadata from the relational store",
            document_id=document_id,
        )

    @staticmethod
    def _placeholder_document(document_id: str) -> DocumentMeta:
        """Stand-in metadata for a chunk whose document node is missing."""
        logger.warning(
            "citing a chunk whose document metadata is missing; re-index this document",
            extra={"document_id": document_id},
        )
        return DocumentMeta(
            document_id=document_id,
            title=f"Unresolved document {document_id}",
            filename=document_id,
            content_hash="",
            size_bytes=0,
        )

    async def _centrality(self, expansion: ExpansionResult, pool: _Pool) -> dict[str, float]:
        """Degree centrality for every entity that could be matched to a passage."""
        keys = expansion.keys
        if not keys:
            return {}

        async def call() -> dict[str, float]:
            return await self._graph.centrality(keys)

        return await degraded(
            "graph.centrality",
            call,
            fallback={},
            capability="centrality weighting in the graph boost",
            entities=len(keys),
        )

    def _build_candidates(self, pool: _Pool, expansion: ExpansionResult) -> list[Candidate]:
        """Assemble scored inputs for the fuser. Pure CPU — called through ``asyncio.to_thread``."""
        anchor = _document_anchor(pool.documents.values())
        reachable = set(expansion.reachable)
        ordered_keys = expansion.keys
        candidates: list[Candidate] = []

        for chunk in pool.chunks:
            document = pool.documents.get(chunk.document_id)
            if document is None:  # pragma: no cover - _load_documents always fills this
                document = self._placeholder_document(chunk.document_id)

            passage_keys = _passage_entity_keys(chunk, self._resolver)
            passage_key_set = set(passage_keys)
            matched = tuple(key for key in ordered_keys if key in passage_key_set)

            if matched:
                centrality = math.fsum(pool.centrality.get(key, 0.0) for key in matched) / len(matched)
                path_confidence = max(expansion.confidence_for(key) for key in matched)
                hops = min(expansion.hops_for(key) for key in matched)
            else:
                centrality = 0.0
                path_confidence = 0.0
                hops = 0

            age_days: float | None = None
            if anchor is not None and document.document_date is not None:
                age_days = float(max(0, (anchor - document.document_date).days))

            candidates.append(
                Candidate(
                    chunk=chunk,
                    document=document,
                    vector_score=pool.vector_hits.get(chunk.chunk_id, 0.0),
                    graph_relevance=pool.graph_hits.get(chunk.chunk_id, 0.0),
                    matched_entities=matched,
                    passage_entity_count=len(passage_key_set | (passage_key_set & reachable)),
                    centrality=centrality,
                    path_confidence=path_confidence,
                    hops=hops,
                    age_days=age_days,
                )
            )
        return candidates

    def _apply_relevance_floor(self, passages: Sequence[RetrievedPassage]) -> list[RetrievedPassage]:
        """Drop passages below ``settings.min_relevance_score``, keeping at least one.

        The floor is applied to the *fused* score, which is per-query normalised, so it means
        "clearly worse than the best thing I found" rather than an absolute quality bar.
        """
        floor = self._settings.min_relevance_score
        kept = [passage for passage in passages if passage.fused_score >= floor]
        if not kept and passages:
            kept = list(passages[:_MIN_KEPT_PASSAGES])
        return kept

    def _paths_for(
        self,
        expansion: ExpansionResult,
        passages: Sequence[RetrievedPassage],
        seeds: Sequence[str],
    ) -> list[GraphPath]:
        """The traversal paths that justify the returned passages.

        Restricted to paths actually touching a returned passage's entities (or a seed), so the
        "Explain How I Know This" panel shows the reasoning behind *this* answer rather than the
        whole neighbourhood.
        """
        wanted: set[str] = set(seeds)
        for passage in passages:
            wanted.update(passage.matched_entities)
        if not wanted:
            return []
        relevant = paths_touching(expansion.paths, wanted)
        return relevant[: self._settings.graph_top_k]


__all__ = [
    "ContextWindow",
    "GraphRAGRetriever",
    "assemble_context",
    "chunk_tokens",
    "estimate_tokens",
    "mmr_rerank",
    "passage_redundancy",
]
