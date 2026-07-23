"""Shared scaffolding for every Copilot query handler.

The pipeline is always the same shape — **retrieve → reason → generate → ground** — and this module
owns it so that seven handlers differ only in the reasoning they add and the prompt they compose.

Two rules are enforced here rather than left to each handler:

**Citations that do not resolve are dropped, not guessed at.** A model asked to cite passage
numbers will occasionally emit ``[9]`` when eight passages were supplied. The tempting repair is to
clamp it to the nearest real passage. That fabricates provenance — it attaches a real document to a
claim that document never made, which is worse than no citation at all because it survives review.
So an unresolvable index is deleted from the text and counted; the sentence stands uncited and the
grounding ratio it damages flows into the answer's confidence.

**An answer with no evidence is a refusal, not an invention.** When retrieval comes back empty the
handler returns an explicit account of what was searched and what would let INDRA answer. That
path is written to read like a product feature because on a plant floor it is one.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar, Final, Mapping, Sequence

from indra.agents.copilot_agent import prompts
from indra.agents.copilot_agent.explainer import AnswerExplainer, summarise_paths
from indra.core.config import Settings
from indra.core.contracts import (
    CacheStore,
    ComplianceService,
    EventBus,
    GraphStore,
    KnowledgeGraphService,
    LLMRouter,
    ProactiveService,
)
from indra.core.deps import AgentDeps
from indra.core.exceptions import IndraError
from indra.core.logging import get_logger
from indra.core.models import (
    Answer,
    Confidence,
    QueryRequest,
    QueryType,
    ReasoningStep,
    RecommendedAction,
    RetrievalResult,
    RetrievedPassage,
    Severity,
    SourceRef,
)

logger = get_logger(__name__)


# --------------------------------------------------------------------------------------
# Structural constants (algorithmic, not deployment tunables)
# --------------------------------------------------------------------------------------

#: Share of the context window given to retrieved passages. The remainder holds the system
#: instruction, the structured evidence digest, and room for the model's own answer.
CONTEXT_BUDGET_FRACTION: Final[float] = 0.72

#: Characters per token used to size the context window. Estimating rather than tokenising is
#: deliberate: ``tiktoken`` fetches its BPE table over the network on first use, and a query-time
#: network call is exactly the failure this system is built to avoid. A ten-percent sizing error
#: costs nothing; a hung request during a demo costs everything.
CHARS_PER_TOKEN: Final[int] = 4

#: Maximum graph paths narrated into a reasoning step before the finding text stops being readable.
MAX_NARRATED_PATHS: Final[int] = 6

#: Maximum recommended actions requested from a model. More than this is a backlog, not a plan.
MAX_RECOMMENDED_ACTIONS: Final[int] = 5

#: Sentences kept per passage when composing the extractive fallback answer.
EXTRACTIVE_SENTENCES_PER_PASSAGE: Final[int] = 2

#: Passages quoted in the extractive fallback.
EXTRACTIVE_PASSAGE_LIMIT: Final[int] = 3

_CITATION_RE: Final[re.Pattern[str]] = re.compile(
    r"\[\s*(?:passage[s]?\s*)?(\d+(?:\s*(?:,|;|and|&)\s*(?:passage[s]?\s*)?\d+)*)\s*\]",
    re.IGNORECASE,
)

_CITATION_NUMBER_RE: Final[re.Pattern[str]] = re.compile(r"\d+")

_SENTENCE_RE: Final[re.Pattern[str]] = re.compile(r"(?<=[.!?])\s+")

_SEVERITY_BY_NAME: Final[dict[str, Severity]] = {member.value: member for member in Severity}


def estimate_tokens(text: str) -> int:
    """Approximate the token cost of ``text``.

    Word count is the floor and characters/4 the estimate; technical plant prose with tags and
    units sits near that ratio. See :data:`CHARS_PER_TOKEN` for why this is not a real tokeniser.
    """
    if not text:
        return 0
    return max(len(text) // CHARS_PER_TOKEN, len(text.split()))


# --------------------------------------------------------------------------------------
# Handler context
# --------------------------------------------------------------------------------------


@dataclass
class HandlerContext:
    """Everything a handler may touch.

    Mutable by design: the orchestrator constructs the Copilot, then calls ``bind()``. Handlers
    hold a reference to this one object, so sibling services appear on them the moment they are
    bound without any handler being rebuilt. The sibling fields stay ``None`` until then, and every
    handler must cope with that — an unbound Proactive Agent degrades the predictive answer, it
    does not fail the request.
    """

    deps: AgentDeps
    explainer: AnswerExplainer
    knowledge_graph: KnowledgeGraphService | None = None
    proactive: ProactiveService | None = None
    compliance: ComplianceService | None = None

    @property
    def settings(self) -> Settings:
        return self.deps.settings

    @property
    def llm(self) -> LLMRouter:
        return self.deps.llm

    @property
    def graph(self) -> GraphStore:
        return self.deps.graph

    @property
    def cache(self) -> CacheStore:
        return self.deps.cache

    @property
    def events(self) -> EventBus:
        return self.deps.events


# --------------------------------------------------------------------------------------
# Generation and citation results
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Outcome of one model call, including the failure case as data rather than an exception."""

    text: str
    provider: str
    ok: bool
    detail: str = ""


@dataclass(frozen=True, slots=True)
class CitationMapping:
    """Answer text with every citation index resolved to a real passage.

    ``dropped`` holds the indices the model emitted that pointed at nothing. It is not diagnostic
    noise — it is the numerator of the grounding ratio that discounts the answer's confidence.
    """

    text: str
    sources: list[SourceRef] = field(default_factory=list)
    cited: list[int] = field(default_factory=list)
    dropped: list[int] = field(default_factory=list)

    @property
    def grounding_ratio(self) -> float:
        """Fraction of emitted citations that resolved to real evidence."""
        total = len(self.cited) + len(self.dropped)
        return 1.0 if total == 0 else len(self.cited) / total


# --------------------------------------------------------------------------------------
# Base handler
# --------------------------------------------------------------------------------------


class BaseHandler:
    """Retrieve → reason → generate → ground.

    Subclasses implement :meth:`compose` (the prompt) and optionally :meth:`extra_steps` (reasoning
    performed before generation) and :meth:`recommend` (actions derived after it).
    """

    query_type: ClassVar[QueryType] = QueryType.FACTUAL

    def __init__(self, ctx: HandlerContext) -> None:
        self.ctx = ctx

    @property
    def settings(self) -> Settings:
        return self.ctx.settings

    # ==================================================================================
    # Entry point
    # ==================================================================================

    async def handle(self, request: QueryRequest, *, retrieval: RetrievalResult) -> Answer:
        """Answer ``request`` from ``retrieval``. Conforms to ``contracts.QueryHandler``."""
        started = time.perf_counter()
        steps: list[ReasoningStep] = []
        return await self.grounded_answer(request, retrieval, steps, started=started)

    # ==================================================================================
    # Hooks for subclasses
    # ==================================================================================

    async def compose(
        self,
        request: QueryRequest,
        retrieval: RetrievalResult,
        context_block: str,
        extras: Mapping[str, str],
    ) -> tuple[str, str]:
        """Return ``(system, prompt)`` for the generation call.

        The default is a plain grounded lookup; every handler but the simplest overrides it.
        """
        return (
            prompts.GROUNDED_SYSTEM_V1,
            prompts.render(
                prompts.FACTUAL_PROMPT_V1,
                context=context_block,
                records=extras.get("records", prompts.DIGEST_EMPTY_V1),
                query=request.query,
            ),
        )

    async def recommend(
        self,
        request: QueryRequest,
        *,
        answer_text: str,
        sources: Sequence[SourceRef],
    ) -> list[RecommendedAction]:
        """Actions derived from the answer. Most handlers have none; diagnostics have several."""
        return []

    # ==================================================================================
    # The pipeline
    # ==================================================================================

    async def grounded_answer(
        self,
        request: QueryRequest,
        retrieval: RetrievalResult,
        steps: list[ReasoningStep],
        *,
        started: float,
        extras: Mapping[str, str] | None = None,
        extra_actions: Sequence[RecommendedAction] | None = None,
        extra_alternatives: Sequence[str] | None = None,
        related_alerts: Sequence[str] | None = None,
        cypher_queries: Sequence[str] | None = None,
        pinned_sources: Sequence[SourceRef] | None = None,
    ) -> Answer:
        """Run the shared retrieve → generate → ground → explain pipeline.

        Args:
            steps: Reasoning steps already recorded by the caller. Handlers that do their own
                evidence gathering pass them in so the chain reads in execution order.
            extras: Named blocks the handler's prompt template expects — the structured evidence
                digest, a prediction summary, a comparison table.
            pinned_sources: Evidence that must appear on the answer even if the model never cited
                it: structured plant records, procedure documents, compliance evidence.
        """
        passages = self.select_passages(retrieval, request)
        steps.append(self.retrieval_step(retrieval, passages, order=len(steps) + 1))

        pinned = list(pinned_sources or [])
        if not passages and not retrieval.paths and not pinned:
            return self.no_evidence_answer(request, retrieval, steps, started=started)

        context_block = self.render_context(passages, retrieval)
        system, prompt = await self.compose(request, retrieval, context_block, extras or {})

        gen_started = time.perf_counter()
        generation = await self.generate(prompt, system=system)
        gen_ms = (time.perf_counter() - gen_started) * 1000.0

        if generation.ok:
            mapping = self.map_citations(generation.text, passages, request=request)
        else:
            mapping = self.extractive_fallback(passages, request=request)

        sources = self.merge_sources(mapping.sources, pinned, passages, request=request)
        steps.append(
            self.generation_step(
                generation,
                mapping,
                order=len(steps) + 1,
                sources=sources,
                duration_ms=gen_ms,
            )
        )

        actions = list(extra_actions or [])
        actions.extend(await self.recommend(request, answer_text=mapping.text, sources=sources))

        graph_preview = await self.graph_preview(request, retrieval)

        return self.ctx.explainer.build_answer(
            request=request,
            query_type=self.query_type,
            answer_text=mapping.text,
            steps=steps,
            sources=sources,
            provider_used=generation.provider,
            graph_preview=graph_preview,
            cypher_queries=cypher_queries or [s.cypher for s in steps if s.cypher],
            recommended_actions=actions,
            extra_alternatives=extra_alternatives,
            related_alerts=related_alerts,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    # ==================================================================================
    # Retrieval handling
    # ==================================================================================

    def select_passages(
        self, retrieval: RetrievalResult, request: QueryRequest
    ) -> list[RetrievedPassage]:
        """Filter to passages worth showing a model, best first.

        Passages below ``settings.min_relevance_score`` are dropped rather than padded in. A weak
        passage in the context window is not neutral — it gives the model something plausible to
        cite, which is how ungrounded answers acquire footnotes.
        """
        floor = self.settings.min_relevance_score
        kept = [p for p in retrieval.passages if p.fused_score >= floor]
        kept.sort(key=lambda p: p.fused_score, reverse=True)
        limit = min(request.max_sources, self.settings.final_top_k)
        if len(kept) < len(retrieval.passages):
            logger.debug(
                "dropped low-relevance passages",
                extra={"dropped": len(retrieval.passages) - len(kept), "floor": floor},
            )
        return kept[:limit]

    def render_context(
        self, passages: Sequence[RetrievedPassage], retrieval: RetrievalResult
    ) -> str:
        """Build the numbered CONTEXT block, respecting the context-window budget."""
        budget = int(self.settings.context_window_tokens * CONTEXT_BUDGET_FRACTION)
        min_useful = self.settings.chunk_min_tokens
        lines: list[str] = [prompts.CONTEXT_HEADER_V1]
        used = estimate_tokens(prompts.CONTEXT_HEADER_V1)

        for index, passage in enumerate(passages, start=1):
            remaining = budget - used
            if remaining < min_useful:
                logger.debug(
                    "context window exhausted",
                    extra={"included": index - 1, "available": len(passages), "budget": budget},
                )
                break
            text = passage.chunk.text.strip()
            if estimate_tokens(text) > remaining:
                text = text[: remaining * CHARS_PER_TOKEN].rstrip() + " …[passage truncated]"
            block = prompts.render(
                prompts.CONTEXT_PASSAGE_V1,
                index=index,
                title=passage.document.title,
                page=f" p.{passage.chunk.page}" if passage.chunk.page else "",
                doc_type=passage.document.document_type.value,
                doc_date=(
                    passage.document.document_date.isoformat()
                    if passage.document.document_date
                    else "undated"
                ),
                relevance=f"{passage.fused_score:.2f}",
                text=text,
            )
            lines.append(block)
            used += estimate_tokens(block)

        if retrieval.paths:
            lines.append(prompts.CONTEXT_PATHS_HEADER_V1)
            for path in retrieval.paths[:MAX_NARRATED_PATHS]:
                lines.append(
                    prompts.render(
                        prompts.CONTEXT_PATH_V1,
                        narrative=path.narrative.strip() or " → ".join(path.nodes),
                        hops=path.hops,
                        confidence=f"{path.confidence:.2f}",
                    )
                )
        return "\n".join(lines)

    # ==================================================================================
    # Generation
    # ==================================================================================

    async def generate(self, prompt: str, *, system: str) -> GenerationResult:
        """Call the router. Never raises — a dead provider chain becomes ``ok=False``."""
        try:
            text, provider = await self.ctx.llm.generate(
                prompt,
                system=system,
                temperature=self.settings.llm_temperature,
                max_tokens=self.settings.llm_max_output_tokens,
            )
        except IndraError as exc:
            logger.warning(
                "generation failed; falling back to extractive answer",
                extra={"query_type": self.query_type.value, "error": exc.error_code, "detail": exc.message},
            )
            return GenerationResult(text="", provider="unavailable", ok=False, detail=exc.message)
        except Exception as exc:  # pragma: no cover - defensive; router raises IndraError
            logger.error(
                "generation raised an untyped error; falling back to extractive answer",
                extra={"query_type": self.query_type.value, "detail": str(exc)},
                exc_info=True,
            )
            return GenerationResult(text="", provider="unavailable", ok=False, detail=str(exc))

        cleaned = text.strip()
        if not cleaned:
            return GenerationResult(
                text="", provider=provider, ok=False, detail="Provider returned an empty completion."
            )
        return GenerationResult(text=cleaned, provider=provider, ok=True)

    async def generate_json(
        self, prompt: str, *, schema: dict[str, Any], system: str
    ) -> tuple[dict[str, Any] | None, str]:
        """Structured extraction. Returns ``(None, provider)`` rather than raising."""
        try:
            payload, provider = await self.ctx.llm.generate_json(
                prompt, schema=schema, system=system, temperature=self.settings.llm_temperature
            )
            return payload, provider
        except IndraError as exc:
            logger.warning(
                "structured extraction unavailable",
                extra={"query_type": self.query_type.value, "error": exc.error_code},
            )
            return None, "unavailable"
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(
                "structured extraction raised an untyped error",
                extra={"query_type": self.query_type.value, "detail": str(exc)},
                exc_info=True,
            )
            return None, "unavailable"

    # ==================================================================================
    # Citation grounding
    # ==================================================================================

    def map_citations(
        self,
        text: str,
        passages: Sequence[RetrievedPassage],
        *,
        request: QueryRequest,
    ) -> CitationMapping:
        """Resolve every ``[n]`` in ``text`` to a real passage; delete the ones that do not resolve.

        Citations are renumbered in order of first appearance, so the answer reads ``[1]``, ``[2]``
        top to bottom and the source list matches that order exactly.
        """
        order: list[int] = []
        position: dict[int, int] = {}
        dropped: list[int] = []
        cited: list[int] = []

        def replace(match: re.Match[str]) -> str:
            emitted = [int(n) for n in _CITATION_NUMBER_RE.findall(match.group(1))]
            rendered: list[int] = []
            for number in emitted:
                index = number - 1
                if index < 0 or index >= len(passages):
                    dropped.append(number)
                    continue
                if index not in position:
                    order.append(index)
                    position[index] = len(order)
                cited.append(number)
                if position[index] not in rendered:
                    rendered.append(position[index])
            if not rendered:
                return ""
            return "[" + ",".join(str(n) for n in rendered) + "]"

        rewritten = _CITATION_RE.sub(replace, text)
        rewritten = _tidy_whitespace(rewritten)

        if dropped:
            logger.warning(
                "dropped unresolvable citations from generated answer",
                extra={
                    "query_type": self.query_type.value,
                    "dropped_indices": sorted(set(dropped)),
                    "passages_supplied": len(passages),
                },
            )

        sources = [
            passages[index].as_source() for index in order[: request.max_sources]
        ]
        return CitationMapping(text=rewritten, sources=sources, cited=cited, dropped=dropped)

    def merge_sources(
        self,
        cited: Sequence[SourceRef],
        pinned: Sequence[SourceRef],
        passages: Sequence[RetrievedPassage],
        *,
        request: QueryRequest,
    ) -> list[SourceRef]:
        """Cited evidence first, then everything else that was considered.

        Citation index ``n`` in the answer text always addresses ``sources[n-1]`` because the cited
        refs keep their positions; the remainder is appended so the UI can show what INDRA read and
        chose not to use, which is as informative as what it did use.
        """
        out: list[SourceRef] = list(cited)
        seen = {(s.document_id, s.chunk_id) for s in out}

        for src in pinned:
            key = (src.document_id, src.chunk_id)
            if key not in seen:
                seen.add(key)
                out.append(src)

        for passage in passages:
            src = passage.as_source()
            key = (src.document_id, src.chunk_id)
            if key not in seen:
                seen.add(key)
                out.append(src)

        return out[: request.max_sources]

    def extractive_fallback(
        self, passages: Sequence[RetrievedPassage], *, request: QueryRequest
    ) -> CitationMapping:
        """Compose an answer with no model at all.

        Quotes the top passages verbatim and labels itself as an extract. The citations are exact —
        it is only the connective reasoning that is missing, and saying so is more useful than
        silently returning a worse-composed answer.
        """
        chosen = list(passages[:EXTRACTIVE_PASSAGE_LIMIT])
        if not chosen:
            return CitationMapping(text="", sources=[], cited=[], dropped=[])

        extracts: list[str] = []
        for index, passage in enumerate(chosen, start=1):
            sentences = [s.strip() for s in _SENTENCE_RE.split(passage.chunk.text.strip()) if s.strip()]
            quoted = " ".join(sentences[:EXTRACTIVE_SENTENCES_PER_PASSAGE]) or passage.chunk.text.strip()
            extracts.append(prompts.render(prompts.EXTRACTIVE_PASSAGE_V1, text=quoted, index=index))

        text = prompts.render(prompts.EXTRACTIVE_FALLBACK_V1, extracts="\n\n".join(extracts))
        return CitationMapping(
            text=text,
            sources=[p.as_source() for p in chosen][: request.max_sources],
            cited=list(range(1, len(chosen) + 1)),
            dropped=[],
        )

    # ==================================================================================
    # Reasoning steps
    # ==================================================================================

    def retrieval_step(
        self,
        retrieval: RetrievalResult,
        passages: Sequence[RetrievedPassage],
        *,
        order: int,
    ) -> ReasoningStep:
        """Record what retrieval found, and how strongly."""
        if passages:
            scores = [max(0.0, min(1.0, p.fused_score)) for p in passages]
            value = round(sum(scores) / len(scores), 4)
            documents = {p.document.document_id for p in passages}
            finding = (
                f"Retrieved {len(passages)} passage(s) across {len(documents)} document(s) from "
                f"{retrieval.total_candidates} candidate(s) using the "
                f"'{retrieval.strategy}' fusion strategy."
            )
            rationale = (
                f"Mean fused relevance {value:.2f} across {len(passages)} passage(s) "
                f"from {len(documents)} independent document(s)."
            )
        else:
            value = 0.0
            finding = (
                f"Retrieval returned no passage above the {self.settings.min_relevance_score:.2f} "
                f"relevance floor, from {retrieval.total_candidates} candidate(s)."
            )
            rationale = "Nothing cleared the relevance floor, so there is no textual evidence."

        if retrieval.paths:
            narrated = summarise_paths(retrieval.paths, limit=MAX_NARRATED_PATHS)
            finding += " Graph connections: " + "; ".join(narrated)

        return ReasoningStep(
            order=order,
            action="Retrieved evidence with hybrid vector + graph search",
            finding=finding,
            confidence=Confidence(value=value, rationale=rationale, method="semantic"),
            sources=[p.as_source() for p in passages],
            graph_paths=list(retrieval.paths[:MAX_NARRATED_PATHS]),
            duration_ms=retrieval.retrieval_ms,
        )

    def generation_step(
        self,
        generation: GenerationResult,
        mapping: CitationMapping,
        *,
        order: int,
        sources: Sequence[SourceRef],
        duration_ms: float,
    ) -> ReasoningStep:
        """Record the generation, discounted by how well the model actually cited its evidence."""
        if not generation.ok:
            return ReasoningStep(
                order=order,
                action="Composed the answer without a language model",
                finding=(
                    "No provider in the chain was reachable, so the answer is a verbatim extract of "
                    f"the top passages rather than a composed narrative. Detail: {generation.detail}"
                ),
                confidence=Confidence(
                    value=round(max((s.relevance for s in sources), default=0.0), 4),
                    rationale=(
                        "Extractive fallback: citations are exact but no reasoning was performed "
                        "over them, so confidence is capped at the best passage's relevance."
                    ),
                    method="heuristic",
                ),
                sources=list(sources),
                duration_ms=duration_ms,
            )

        ratio = mapping.grounding_ratio
        mean_relevance = (
            sum(s.relevance for s in mapping.sources) / len(mapping.sources)
            if mapping.sources
            else 0.0
        )
        value = round(max(0.0, min(1.0, ratio * mean_relevance)), 4)
        detail = (
            f"{len(mapping.cited)} citation(s) resolved"
            + (f", {len(set(mapping.dropped))} dropped as unresolvable" if mapping.dropped else "")
            + f"; mean cited relevance {mean_relevance:.2f}."
        )
        return ReasoningStep(
            order=order,
            action=f"Generated the answer with provider '{generation.provider}'",
            finding=(
                f"Composed a grounded answer from {len(mapping.sources)} cited passage(s). {detail}"
                + (
                    " Citation indices that pointed at no retrieved passage were removed rather "
                    "than reassigned, so no claim carries borrowed provenance."
                    if mapping.dropped
                    else ""
                )
            ),
            confidence=Confidence(
                value=value,
                rationale=(
                    f"Grounding ratio {ratio:.2f} × mean cited relevance {mean_relevance:.2f}. "
                    "An answer is worth what its citations are worth."
                ),
                method="llm",
            ),
            sources=list(mapping.sources),
            duration_ms=duration_ms,
        )

    # ==================================================================================
    # Empty-retrieval path
    # ==================================================================================

    def no_evidence_answer(
        self,
        request: QueryRequest,
        retrieval: RetrievalResult,
        steps: list[ReasoningStep],
        *,
        started: float,
        extra_searched: Sequence[tuple[str, str]] = (),
        suggestions: Sequence[str] = (),
    ) -> Answer:
        """The honest refusal: "I don't know, and here is exactly what I looked at"."""
        backends = self.ctx.deps.bound_backends
        searched: list[tuple[str, str]] = [
            ("Question as searched", request.query),
            (
                "Equipment filter",
                request.equipment_tag or "none — the question named no tag INDRA could resolve",
            ),
            (
                "Entities resolved from the question",
                ", ".join(retrieval.query_entities) if retrieval.query_entities else "none",
            ),
            (
                "Vector index",
                f"{backends.get('vectors', 'unknown')} backend, top {self.settings.vector_top_k}, "
                f"{retrieval.total_candidates} candidate(s) scored",
            ),
            (
                "Knowledge graph",
                f"{backends.get('graph', 'unknown')} backend, up to "
                f"{self.settings.max_hops} hop(s) from the query entities, "
                f"{len(retrieval.paths)} path(s) found",
            ),
            (
                "Relevance floor",
                f"{self.settings.min_relevance_score:.2f} — nothing retrieved cleared it",
            ),
            ("Retrieval time", f"{retrieval.retrieval_ms:.0f} ms"),
        ]
        searched.extend(extra_searched)

        default_suggestions = [
            "Ingest the OEM manual, work orders or inspection reports covering this asset — "
            "INDRA can only reason over documents it has been given.",
            "Check the equipment tag: a tag that is not in the graph cannot anchor a search "
            "(try the plain asset name instead).",
            "Ask a narrower question naming the specific parameter, document or date range.",
        ]
        chosen_suggestions = list(suggestions) or default_suggestions

        answer_text = prompts.render(
            prompts.NO_EVIDENCE_ANSWER_V1,
            search_report="\n".join(
                prompts.render(prompts.NO_EVIDENCE_SEARCH_LINE_V1, label=label, detail=detail)
                for label, detail in searched
            ),
            suggestions="\n".join(
                prompts.render(prompts.NO_EVIDENCE_SUGGESTION_V1, suggestion=s)
                for s in chosen_suggestions
            ),
        )

        steps.append(
            ReasoningStep(
                order=len(steps) + 1,
                action="Declined to answer",
                finding=(
                    "No evidence cleared the relevance floor. INDRA is returning an account of the "
                    "search instead of an answer, because a composed answer here would be "
                    "unsourced by construction."
                ),
                confidence=Confidence(
                    value=1.0,
                    rationale="The absence of retrieved evidence is itself a certain observation.",
                    method="exact",
                ),
                duration_ms=0.0,
            )
        )

        return self.ctx.explainer.build_answer(
            request=request,
            query_type=self.query_type,
            answer_text=answer_text,
            steps=steps,
            sources=[],
            provider_used="none",
            confidence=Confidence(
                value=0.0,
                rationale=(
                    "No supporting evidence was retrieved. INDRA is declining to answer rather "
                    "than composing something unsourced."
                ),
                method="exact",
            ),
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    # ==================================================================================
    # Shared helpers for subclasses
    # ==================================================================================

    async def graph_preview(
        self, request: QueryRequest, retrieval: RetrievalResult
    ) -> dict[str, Any] | None:
        """Nodes/edges for the React Flow panel. Never fails the answer."""
        if not request.include_graph_preview or self.ctx.knowledge_graph is None:
            return None
        keys = list(retrieval.query_entities)
        if not keys:
            return None
        try:
            return await self.ctx.knowledge_graph.graph_preview(keys, hops=self.settings.max_hops)
        except IndraError as exc:
            logger.warning(
                "graph preview unavailable", extra={"error": exc.error_code, "detail": exc.message}
            )
            return None
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("graph preview raised an untyped error", extra={"detail": str(exc)})
            return None

    async def retrieve(
        self,
        query: str,
        *,
        equipment_tag: str | None = None,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """Additional retrieval from inside a handler. Returns an empty result on failure."""
        if self.ctx.knowledge_graph is None:
            logger.warning("knowledge graph service is not bound; retrieval skipped")
            return RetrievalResult(query=query)
        try:
            return await self.ctx.knowledge_graph.retrieve(
                query, top_k=top_k, equipment_tag=equipment_tag, filters=filters
            )
        except IndraError as exc:
            logger.warning(
                "supplementary retrieval failed",
                extra={"query": query, "error": exc.error_code, "detail": exc.message},
            )
            return RetrievalResult(query=query)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(
                "supplementary retrieval raised an untyped error",
                extra={"query": query, "detail": str(exc)},
                exc_info=True,
            )
            return RetrievalResult(query=query)

    async def llm_actions(
        self,
        *,
        findings: str,
        tag: str,
        criticality: str,
        limit: int = MAX_RECOMMENDED_ACTIONS,
    ) -> list[RecommendedAction]:
        """Ask the model for concrete next actions. Returns ``[]`` when unavailable."""
        payload, _provider = await self.generate_json(
            prompts.render(
                prompts.RECOMMENDED_ACTIONS_PROMPT_V1,
                findings=findings,
                tag=tag,
                criticality=criticality,
                limit=limit,
            ),
            schema=prompts.RECOMMENDED_ACTIONS_SCHEMA_V1,
            system=prompts.EXTRACTION_SYSTEM_V1,
        )
        if not payload:
            return []
        raw_actions = payload.get("actions")
        if not isinstance(raw_actions, list):
            return []

        out: list[RecommendedAction] = []
        for item in raw_actions[:limit]:
            if not isinstance(item, dict):
                continue
            action_text = item.get("action")
            if not isinstance(action_text, str) or not action_text.strip():
                continue
            urgency = _SEVERITY_BY_NAME.get(str(item.get("urgency", "")).upper(), Severity.WARNING)
            due = item.get("due_within_days")
            owner = item.get("owner_role")
            rationale = item.get("rationale")
            out.append(
                RecommendedAction(
                    action=action_text.strip(),
                    urgency=urgency,
                    owner_role=owner.strip() if isinstance(owner, str) and owner.strip() else None,
                    due_within_days=int(due) if isinstance(due, int) and due >= 0 else None,
                    rationale=rationale.strip() if isinstance(rationale, str) else "",
                )
            )
        return out

    async def gather(self, *awaitables: Any) -> list[Any]:
        """``asyncio.gather`` that returns exceptions as values instead of cancelling siblings.

        Handlers fan out across several stores; one dead store must degrade one branch of the
        evidence, not the whole answer.
        """
        return list(await asyncio.gather(*awaitables, return_exceptions=True))


def _tidy_whitespace(text: str) -> str:
    """Clean up the holes left where unresolvable citations were removed."""
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


__all__ = [
    "BaseHandler",
    "CHARS_PER_TOKEN",
    "CONTEXT_BUDGET_FRACTION",
    "CitationMapping",
    "GenerationResult",
    "HandlerContext",
    "MAX_RECOMMENDED_ACTIONS",
    "estimate_tokens",
]
