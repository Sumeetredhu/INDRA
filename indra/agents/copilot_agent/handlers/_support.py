"""Evidence helpers shared by the Copilot's structured handlers.

Six of the seven handlers begin the same way — work out which asset the question is about, read
structured plant records around it, and render those records into the digest block their prompt
expects. That shared shape lives here so each handler contains only the reasoning that is specific
to its query type.

Nothing in this module talks to a model. Everything it returns is either read straight out of the
graph or derived arithmetically from what was read, which is what lets the handlers label these
findings ``Confidence.exact`` and mean it.

Two conventions are enforced here rather than repeated in every handler:

**A graph call never raises into a handler.** :func:`graph_call` turns any failure into the caller's
declared default plus a warning, because one dead store must degrade one branch of the evidence
rather than the whole answer.

**Absence is a finding.** Helpers that come back empty say so in the text they produce. "No
inspection record in 180 days" is evidence about the asset; rendering it as a blank section throws
that evidence away.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Awaitable, Final, Iterable, Sequence, TypeVar

from indra.agents.copilot_agent import prompts
from indra.agents.copilot_agent.classifier import extract_equipment_tags
from indra.agents.copilot_agent.handlers.base import HandlerContext
from indra.core.exceptions import IndraError
from indra.core.logging import get_logger
from indra.core.models import (
    Confidence,
    DocumentMeta,
    Equipment,
    GraphPath,
    QueryRequest,
    ReasoningStep,
    RetrievalResult,
    RetrievedPassage,
    SourceRef,
)

logger = get_logger(__name__)

T = TypeVar("T")


# --------------------------------------------------------------------------------------
# Structural constants — properties of the algorithm, not deployment tunables.
# --------------------------------------------------------------------------------------

#: Tag-shaped tokens probed against the registry before the search is abandoned. A question naming
#: more assets than this is a fleet query, not a lookup, and probing every candidate turns one
#: answer into dozens of round trips.
MAX_TAG_CANDIDATES: Final[int] = 6

#: Characters kept when quoting evidence into a finding. Long enough to carry the observation,
#: short enough that a reasoning chain stays readable on a phone.
QUOTE_LIMIT: Final[int] = 220

#: Items rendered per digest section, so a prompt cannot be crowded out by one verbose section.
MAX_DIGEST_ITEMS: Final[int] = 8


CYPHER_EQUIPMENT: Final[str] = (
    "MATCH (e:Equipment {tag: $tag}) RETURN e.tag, e.name, e.equipment_type, e.manufacturer, "
    "e.model, e.criticality, e.location, e.installed_on, e.specifications, e.oem_thresholds"
)


def clip(text: str, limit: int = QUOTE_LIMIT) -> str:
    """Collapse whitespace and truncate on a character budget."""
    cleaned = " ".join(text.split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1].rstrip() + "…"


async def graph_call(label: str, awaitable: Awaitable[T], *, default: T, **context: Any) -> T:
    """Await a store call, returning ``default`` instead of propagating any failure.

    Args:
        label: What was being read, used verbatim in the warning so a log line is self-explanatory.
        awaitable: The already-constructed coroutine.
        default: What the caller treats as "nothing found" — usually ``[]`` or ``None``.
        context: Extra structured fields for the log record, e.g. ``tag="P-101"``.
    """
    try:
        return await awaitable
    except IndraError as exc:
        logger.warning(
            f"{label} unavailable",
            extra={**context, "error": exc.error_code, "detail": exc.message},
        )
        return default
    except Exception as exc:  # pragma: no cover - defensive; stores raise IndraError
        logger.error(
            f"{label} raised an untyped error",
            extra={**context, "detail": str(exc)},
            exc_info=True,
        )
        return default


# --------------------------------------------------------------------------------------
# Equipment resolution
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class EquipmentResolution:
    """Which assets a question is about, and which of them the registry actually knows.

    The distinction matters: a tag that parses but is absent from the registry is not a failed
    lookup, it is the finding that INDRA has never been given documents for that asset.
    """

    candidates: list[str] = field(default_factory=list)
    resolved: dict[str, Equipment] = field(default_factory=dict)
    unresolved: list[str] = field(default_factory=list)

    @property
    def resolved_tags(self) -> list[str]:
        """Registry-known tags, in the order the question named them."""
        return [tag for tag in self.candidates if tag in self.resolved]

    @property
    def primary_tag(self) -> str:
        """The asset the answer is anchored on: first resolved, else first named, else empty."""
        resolved = self.resolved_tags
        if resolved:
            return resolved[0]
        return self.candidates[0] if self.candidates else ""

    @property
    def primary(self) -> Equipment | None:
        return self.resolved.get(self.primary_tag)

    @property
    def subjects(self) -> list[str]:
        """Every named asset, resolved or not — the comparison handler compares all of them."""
        return list(self.candidates)

    def equipment_for(self, tag: str) -> Equipment | None:
        return self.resolved.get(tag)


async def resolve_equipment(
    ctx: HandlerContext,
    request: QueryRequest,
    retrieval: RetrievalResult,
    *,
    limit: int = MAX_TAG_CANDIDATES,
) -> EquipmentResolution:
    """Collect every tag the question could be about and probe each against the registry.

    Candidates come from three places in priority order: the explicit ``equipment_tag`` on the
    request (a mobile client scanning a nameplate), tag-shaped tokens in the question text, and the
    entity keys retrieval resolved. Order is preserved throughout, because "compare P-101 and
    P-102" must keep the operator's subject order in the answer.
    """
    candidates: list[str] = []
    if request.equipment_tag:
        candidates.append(request.equipment_tag.strip().upper())
    candidates.extend(extract_equipment_tags(request.query))
    for key in retrieval.query_entities:
        _, _, value = key.partition(":")
        candidates.extend(extract_equipment_tags(value or key))

    seen: set[str] = set()
    ordered = [c for c in candidates if not (c in seen or seen.add(c))][:limit]

    resolution = EquipmentResolution(candidates=ordered)
    for tag in ordered:
        found = await graph_call(
            "equipment lookup", ctx.graph.get_equipment(tag), default=None, tag=tag
        )
        if found is not None:
            resolution.resolved[tag] = found
        else:
            resolution.unresolved.append(tag)
    return resolution


def resolution_step(
    resolution: EquipmentResolution, *, order: int, duration_ms: float = 0.0
) -> ReasoningStep:
    """Record what the registry knew about the assets the question named."""
    resolved = resolution.resolved_tags
    if resolved:
        described = "; ".join(
            f"{tag} — {eq.name or eq.equipment_type}, criticality {eq.criticality.value}"
            + (f", {eq.manufacturer}" if eq.manufacturer else "")
            + (f" {eq.model}" if eq.model else "")
            + (f", located {eq.location}" if eq.location else "")
            for tag, eq in ((t, resolution.resolved[t]) for t in resolved)
        )
        finding = f"Resolved {len(resolved)} asset(s) in the equipment registry: {described}."
        if resolution.unresolved:
            finding += (
                f" {', '.join(resolution.unresolved)} parse(s) as a plant tag but has no registry "
                "node, so no structured history exists for it."
            )
        confidence = Confidence.exact(
            f"Exact tag match on the equipment registry primary key ({', '.join(resolved)})."
        )
    elif resolution.unresolved:
        finding = (
            f"{', '.join(resolution.unresolved)} parse(s) as a plant tag but has no node in the "
            "equipment registry. The answer will rest on retrieved documents alone, with no "
            "structured specification, maintenance or failure history behind it."
        )
        confidence = Confidence(
            value=0.4,
            rationale=(
                "Tag grammar matched but the registry lookup missed; the tag may be mistyped, "
                "superseded, or its documents may never have been ingested."
            ),
            method="heuristic",
        )
    else:
        finding = (
            "The question names no plant tag INDRA can resolve, so no asset anchors the structured "
            "record lookup. Only document retrieval contributes to this answer."
        )
        confidence = Confidence(
            value=0.2,
            rationale="No tag-shaped token in the question matched the registry.",
            method="heuristic",
        )

    return ReasoningStep(
        order=order,
        action="Resolved the equipment against the plant registry",
        finding=finding,
        confidence=confidence,
        cypher=CYPHER_EQUIPMENT,
        duration_ms=duration_ms,
    )


# --------------------------------------------------------------------------------------
# Source helpers
# --------------------------------------------------------------------------------------


def document_source(
    meta: DocumentMeta, *, relevance: float = 0.0, snippet: str = ""
) -> SourceRef:
    """Cite a document that was found structurally rather than retrieved by similarity.

    ``retrieved_via="direct"`` is the honest label: the graph says this document mentions the
    asset, which is a different claim from "a vector search scored it relevant to the question".
    """
    return SourceRef(
        document_id=meta.document_id,
        document_title=meta.title,
        document_type=meta.document_type,
        snippet=clip(snippet, 600) if snippet else "",
        relevance=max(0.0, min(1.0, relevance)),
        retrieved_via="direct",
        document_date=meta.document_date,
    )


def dedupe_sources(sources: Iterable[SourceRef]) -> list[SourceRef]:
    """Drop repeats, keeping the first occurrence and therefore the caller's priority order."""
    seen: set[tuple[str, str | None]] = set()
    out: list[SourceRef] = []
    for src in sources:
        key = (src.document_id, src.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(src)
    return out


def sources_of(items: Iterable[Any]) -> list[SourceRef]:
    """Flatten the ``sources`` list off any record model that carries one."""
    out: list[SourceRef] = []
    for item in items:
        refs = getattr(item, "sources", None)
        if refs:
            out.extend(refs)
    return out


# --------------------------------------------------------------------------------------
# Digest rendering
# --------------------------------------------------------------------------------------


def render_digest(sections: Sequence[tuple[str, Sequence[str]]]) -> str:
    """Render titled evidence sections for a generation prompt.

    An empty section renders as ``(searched, nothing found)`` rather than being omitted, so the
    model can see the difference between "not looked for" and "looked for and absent" — the second
    is a finding it is expected to report.
    """
    rendered: list[str] = []
    for title, items in sections:
        body = (
            "\n".join(
                prompts.render(prompts.DIGEST_BULLET_V1, text=item)
                for item in list(items)[:MAX_DIGEST_ITEMS]
            )
            if items
            else prompts.DIGEST_EMPTY_V1
        )
        rendered.append(prompts.render(prompts.DIGEST_SECTION_V1, title=title, body=body))
    return "\n".join(rendered)


# --------------------------------------------------------------------------------------
# Retrieval merging
# --------------------------------------------------------------------------------------


def merge_retrievals(query: str, results: Sequence[RetrievalResult]) -> RetrievalResult:
    """Fuse several retrievals into one context set.

    Used where a handler retrieves once per subject: the generation prompt needs a single numbered
    CONTEXT block, and a passage retrieved for two subjects must appear once, keeping its best
    score. Retrieval time is reported as the slowest branch rather than the sum, because the
    branches ran concurrently.
    """
    best: dict[str, RetrievedPassage] = {}
    entities: list[str] = []
    paths: list[GraphPath] = []
    candidates = 0
    slowest = 0.0
    strategy = "weighted"

    for result in results:
        candidates += result.total_candidates
        slowest = max(slowest, result.retrieval_ms)
        strategy = result.strategy
        for key in result.query_entities:
            if key not in entities:
                entities.append(key)
        paths.extend(result.paths)
        for passage in result.passages:
            existing = best.get(passage.chunk.chunk_id)
            if existing is None or passage.fused_score > existing.fused_score:
                best[passage.chunk.chunk_id] = passage

    merged = sorted(best.values(), key=lambda p: p.fused_score, reverse=True)

    seen_paths: set[tuple[str, ...]] = set()
    unique_paths: list[GraphPath] = []
    for path in paths:
        key = tuple(path.nodes)
        if key in seen_paths:
            continue
        seen_paths.add(key)
        unique_paths.append(path)

    return RetrievalResult(
        query=query,
        query_entities=entities,
        passages=merged,
        paths=unique_paths,
        strategy=strategy,  # type: ignore[arg-type]
        total_candidates=candidates,
        retrieval_ms=slowest,
    )


# --------------------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------------------


def format_date(value: date | None) -> str:
    """ISO date, or the word that says the record carried none."""
    return value.isoformat() if value is not None else "undated"


def format_number(value: float, unit: str = "") -> str:
    """Render a measurement without trailing zeros, with its unit attached."""
    return f"{value:g}{unit}" if unit else f"{value:g}"


__all__ = [
    "CYPHER_EQUIPMENT",
    "EquipmentResolution",
    "MAX_DIGEST_ITEMS",
    "MAX_TAG_CANDIDATES",
    "QUOTE_LIMIT",
    "clip",
    "dedupe_sources",
    "document_source",
    "format_date",
    "format_number",
    "graph_call",
    "merge_retrievals",
    "render_digest",
    "resolution_step",
    "resolve_equipment",
    "sources_of",
]
