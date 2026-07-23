"""Assembly of the explainability payload that rides on every :class:`Answer`.

"Explain How I Know This" is a *projection of data INDRA already holds*, never a second inference
pass asking a model to justify itself. That distinction is the whole point: a model asked to
explain its own answer will produce a plausible explanation whether or not it is the real one.
Everything here is computed from the retrieved evidence and the recorded reasoning steps.

What this module derives:

* **Overall confidence** — :meth:`Confidence.aggregate`, which is weakest-link, not mean. A 0.95
  retrieval step cannot rescue a 0.42 OCR read of the number the conclusion rests on.
* **Uncertainty flags** — from real signals only. Each flag below is triggered by a measurable
  property of the evidence, never by a model's self-report:
  ``LOW_OCR_CONFIDENCE`` (extraction confidence under ``settings.ocr_min_confidence``),
  ``STALE_DOCUMENT`` (evidence older than the interval that document class is relevant for),
  ``CONFLICTING_SOURCES`` (two documents asserting different values for the same measurand),
  ``SPARSE_EVIDENCE`` (fewer than two independent sources), and
  ``VISION_INFERENCE`` (a fact read off a drawing rather than out of text).
* **Alternative interpretations** — generated from those same signals, so a conflict in the
  evidence always surfaces as a competing reading rather than being averaged away.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Final, Iterable, Sequence

from indra.core.config import Settings
from indra.core.logging import get_logger
from indra.core.models import (
    Answer,
    Confidence,
    DocumentType,
    GraphPath,
    QueryRequest,
    QueryType,
    ReasoningStep,
    RecommendedAction,
    Severity,
    SourceRef,
    UncertaintyFlag,
    UncertaintySource,
)

logger = get_logger(__name__)


# --------------------------------------------------------------------------------------
# Structural constants
#
# These are invariants of the explanation model, not deployment tunables — which is why they are
# named here rather than added to Settings. Changing one changes what "corroborated" means.
# --------------------------------------------------------------------------------------

#: Corroboration requires at least two independent sources. One document agreeing with itself is
#: not evidence, so a single-source answer is always flagged sparse regardless of its confidence.
MIN_CORROBORATING_SOURCES: Final[int] = 2

#: Two measurements of the same quantity are treated as contradictory when they differ by more than
#: this fraction of the larger value. Below it, the gap is measurement noise or rounding between
#: two readings, and flagging it would train operators to ignore the flag.
NUMERIC_CONFLICT_TOLERANCE: Final[float] = 0.10

#: Cap on generated alternative interpretations. Past a handful they stop being alternatives and
#: start being noise that buries the real one.
MAX_ALTERNATIVES: Final[int] = 5

#: Physical units worth comparing across documents. A number without a unit is a count, a date
#: fragment or a document reference, and comparing those manufactures conflicts that do not exist.
_UNIT_ALIASES: Final[dict[str, str]] = {
    "%": "%", "pct": "%", "percent": "%",
    "mm/s": "mm/s", "mm/sec": "mm/s",
    "mm": "mm", "micron": "um", "um": "um",
    "°c": "degC", "degc": "degC", "c": "degC", "celsius": "degC",
    "bar": "bar", "barg": "bar", "kpa": "kPa", "mpa": "MPa", "psi": "psi",
    "rpm": "rpm", "hz": "Hz",
    "kw": "kW", "hp": "hp", "amps": "A", "amp": "A", "a": "A", "v": "V",
    "hour": "h", "hours": "h", "hrs": "h", "hr": "h",
}

_MEASUREMENT_RE: Final[re.Pattern[str]] = re.compile(
    r"(?P<label>(?:[A-Za-z][A-Za-z_-]*\s+){0,3}[A-Za-z][A-Za-z_-]*)"
    r"\s*(?:is|was|at|of|to|reads?|reading|measured(?:\s+at)?|recorded(?:\s+at)?|[:=])?\s*"
    r"(?P<value>\d{1,7}(?:\.\d+)?)\s*"
    r"(?P<unit>%|mm/sec|mm/s|mm|micron|um|°C|degC|celsius|bar|barg|kPa|MPa|psi|rpm|Hz|kW|hp|amps?|hours?|hrs?)\b",
    re.IGNORECASE,
)

#: Words that carry no meaning as a measurand label. Stripped before grouping so "the bearing wear"
#: and "bearing wear was" collapse to the same key.
_LABEL_STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "the", "a", "an", "of", "for", "at", "on", "in", "to", "and", "is", "was", "were", "are",
        "with", "its", "this", "that", "has", "had", "have", "been", "measured", "recorded",
        "reading", "reads", "observed", "found", "noted", "reported", "shows", "showed", "value",
        "current", "latest", "last", "approximately", "approx", "about", "around", "over", "under",
    }
)


def _staleness_interval_days(document_type: DocumentType, settings: Settings) -> int | None:
    """How long a document of this class stays *current* evidence.

    Derived from the same lookback windows the diagnostic chain queries with, so "stale" means
    exactly "outside the window INDRA considers relevant for this class of record". Reference
    documents — OEM manuals, SOPs, regulations, P&IDs — do not expire on this clock; they are
    superseded by revision, which ``DocumentMeta.supersedes`` tracks separately (D6).
    """
    return {
        DocumentType.WORK_ORDER: settings.maintenance_lookback_days,
        DocumentType.INSPECTION_REPORT: settings.inspection_lookback_days,
        DocumentType.INCIDENT_REPORT: settings.inspection_lookback_days,
        DocumentType.ROOT_CAUSE_ANALYSIS: settings.inspection_lookback_days,
        DocumentType.SHIFT_LOG: settings.shift_log_lookback_days,
        DocumentType.EMAIL: settings.shift_log_lookback_days,
        DocumentType.SPREADSHEET: settings.maintenance_lookback_days,
    }.get(document_type)


@dataclass(frozen=True, slots=True)
class _Measurement:
    """One numeric claim lifted out of a source snippet."""

    label: str
    value: float
    unit: str
    source: SourceRef

    @property
    def rendered(self) -> str:
        return f"{self.value:g} {self.unit}".strip()


class AnswerExplainer:
    """Turns reasoning steps and evidence into a fully explained :class:`Answer`.

    Stateless apart from settings, so one instance is shared by every handler.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    # ==================================================================================
    # Public surface
    # ==================================================================================

    def build_answer(
        self,
        *,
        request: QueryRequest,
        query_type: QueryType,
        answer_text: str,
        steps: Sequence[ReasoningStep],
        sources: Sequence[SourceRef],
        provider_used: str,
        graph_preview: dict[str, Any] | None = None,
        cypher_queries: Sequence[str] | None = None,
        recommended_actions: Sequence[RecommendedAction] | None = None,
        extra_alternatives: Sequence[str] | None = None,
        related_alerts: Sequence[str] | None = None,
        confidence: Confidence | None = None,
        latency_ms: float = 0.0,
    ) -> Answer:
        """Assemble the final answer with its full explainability payload.

        ``confidence`` is normally left ``None`` so it is aggregated from the steps. Pass it only
        where the chain's confidence is not the answer's confidence — the honest "I don't know"
        response being the one real case.
        """
        source_list = list(sources)
        step_list = list(steps)

        flags = self.derive_uncertainty_flags(source_list, steps=step_list)
        overall = confidence or self.aggregate_confidence(step_list)
        alternatives = self.alternative_interpretations(
            source_list, flags=flags, steps=step_list, extra=extra_alternatives
        )

        answer = Answer(
            query=request.query,
            query_type=query_type,
            answer_text=answer_text,
            confidence=overall,
            reasoning_chain=step_list,
            sources=source_list,
            uncertainty_flags=flags,
            alternative_interpretations=alternatives,
            recommended_actions=list(recommended_actions or []),
            graph_preview=graph_preview if request.include_graph_preview else None,
            cypher_queries=list(cypher_queries or []) if request.include_cypher else [],
            related_alerts=list(related_alerts or []),
            latency_ms=round(latency_ms, 2),
            provider_used=provider_used,
        )
        logger.info(
            "answer assembled",
            extra={
                "query_type": query_type.value,
                "sources": len(answer.sources),
                "steps": len(answer.reasoning_chain),
                "flags": [f.source.value for f in answer.uncertainty_flags],
                "confidence": answer.confidence.value,
                "provider_used": answer.provider_used,
            },
        )
        return answer

    def aggregate_confidence(self, steps: Sequence[ReasoningStep]) -> Confidence:
        """Weakest-link aggregation across the chain.

        Delegates to :meth:`Confidence.aggregate` rather than re-deriving it, so the whole system
        has exactly one definition of what a chain of confidences is worth.
        """
        if not steps:
            return Confidence(
                value=0.0,
                rationale="No reasoning steps were recorded, so nothing supports this answer.",
                method="aggregate",
            )
        parts = [step.confidence for step in steps]
        weakest = min(parts, key=lambda c: c.value)
        weakest_step = next(s for s in steps if s.confidence is weakest)
        return Confidence.aggregate(
            parts,
            rationale=(
                f"Weakest link is step {weakest_step.order} ({weakest_step.action}) at "
                f"{weakest.value:.2f}: {weakest.rationale}"
            ),
        )

    def derive_uncertainty_flags(
        self,
        sources: Sequence[SourceRef],
        *,
        steps: Sequence[ReasoningStep] = (),
    ) -> list[UncertaintyFlag]:
        """Derive every caveat the operator must see, from measurable properties of the evidence."""
        flags: list[UncertaintyFlag] = []
        flags.extend(self._flag_low_ocr(sources))
        flags.extend(self._flag_stale(sources))
        flags.extend(self._flag_conflicts(sources))
        flags.extend(self._flag_vision(sources))
        sparse = self._flag_sparse(sources, steps=steps)
        if sparse is not None:
            flags.append(sparse)
        flags.sort(key=lambda f: f.severity.rank, reverse=True)
        return flags

    def alternative_interpretations(
        self,
        sources: Sequence[SourceRef],
        *,
        flags: Sequence[UncertaintyFlag],
        steps: Sequence[ReasoningStep] = (),
        extra: Sequence[str] | None = None,
    ) -> list[str]:
        """Competing readings of the same evidence.

        Derived from the uncertainty signals rather than invented: where the evidence conflicts,
        each side of the conflict is a real alternative, and saying so is more useful than
        silently preferring the higher-scoring passage.
        """
        out: list[str] = [text.strip() for text in (extra or []) if text.strip()]

        for flag in flags:
            if flag.source is UncertaintySource.CONFLICTING_SOURCES:
                out.append(
                    f"The sources disagree: {flag.message} Either reading changes the conclusion, "
                    "so the conflict should be resolved by direct measurement before acting."
                )
            elif flag.source is UncertaintySource.SPARSE_EVIDENCE and len(sources) == 1:
                out.append(
                    f"This rests on a single document ({sources[0].citation}). An equally consistent "
                    "reading is that the document is unrepresentative and the pattern does not "
                    "generalise."
                )
            elif flag.source is UncertaintySource.LOW_OCR_CONFIDENCE:
                out.append(
                    f"If the OCR read is wrong ({flag.message}), the numeric basis of this answer "
                    "changes and the conclusion may not hold."
                )
            elif flag.source is UncertaintySource.STALE_DOCUMENT:
                out.append(
                    f"The evidence may be superseded ({flag.message}); work done since it was "
                    "written would change the picture."
                )

        weak_steps = [s for s in steps if s.confidence.band == "low"]
        for step in weak_steps[:MAX_ALTERNATIVES]:
            out.append(
                f"Step {step.order} ({step.action}) is weakly supported — {step.confidence.rationale} "
                "Remove it and the chain no longer reaches this conclusion."
            )

        seen: set[str] = set()
        unique: list[str] = []
        for text in out:
            key = text.lower()
            if key not in seen:
                seen.add(key)
                unique.append(text)
        return unique[:MAX_ALTERNATIVES]

    def to_payload(self, answer: Answer) -> dict[str, Any]:
        """Project an answer into the "Explain How I Know This" panel's shape.

        The API's ``/query/explain`` route renders this. It contains nothing that is not already on
        the answer — it is a reshaping, which is precisely why it can be trusted.
        """
        return {
            "answer_id": answer.answer_id,
            "query": answer.query,
            "query_type": answer.query_type.value,
            "confidence": {
                "value": answer.confidence.value,
                "band": answer.confidence.band,
                "rationale": answer.confidence.rationale,
                "method": answer.confidence.method,
            },
            "reasoning_chain": [
                {
                    "order": step.order,
                    "action": step.action,
                    "finding": step.finding,
                    "confidence": {
                        "value": step.confidence.value,
                        "band": step.confidence.band,
                        "rationale": step.confidence.rationale,
                    },
                    "sources": [self._source_payload(src) for src in step.sources],
                    "graph_paths": [
                        {"narrative": path.narrative, "hops": path.hops, "nodes": path.nodes,
                         "relations": [r.value for r in path.relations], "confidence": path.confidence}
                        for path in step.graph_paths
                    ],
                    "cypher": step.cypher,
                    "duration_ms": step.duration_ms,
                }
                for step in answer.reasoning_chain
            ],
            "sources": [self._source_payload(src) for src in answer.sources],
            "uncertainty_flags": [
                {
                    "source": flag.source.value,
                    "message": flag.message,
                    "severity": flag.severity.value,
                    "affected_claim": flag.affected_claim,
                    "suggested_action": flag.suggested_action,
                    "citation": flag.affected_source.citation if flag.affected_source else None,
                }
                for flag in answer.uncertainty_flags
            ],
            "alternative_interpretations": answer.alternative_interpretations,
            "graph_preview": answer.graph_preview,
            "cypher_queries": answer.cypher_queries,
            "provider_used": answer.provider_used,
            "latency_ms": answer.latency_ms,
        }

    # ==================================================================================
    # Flag derivation
    # ==================================================================================

    def _flag_low_ocr(self, sources: Sequence[SourceRef]) -> list[UncertaintyFlag]:
        """Evidence whose extraction confidence is below the configured OCR floor."""
        flags: list[UncertaintyFlag] = []
        threshold = self._settings.ocr_min_confidence
        for src in sources:
            if src.extraction_confidence >= threshold:
                continue
            flags.append(
                UncertaintyFlag(
                    source=UncertaintySource.LOW_OCR_CONFIDENCE,
                    message=(
                        f"{src.citation} was read by OCR at confidence "
                        f"{src.extraction_confidence:.2f}, below the {threshold:.2f} floor. Any "
                        "figure taken from it may be misread."
                    ),
                    severity=Severity.HIGH if src.relevance >= 0.5 else Severity.WARNING,
                    affected_claim=src.snippet[:200] or None,
                    affected_source=src,
                    suggested_action="Verify the value against the original page before acting.",
                )
            )
        return flags

    def _flag_stale(self, sources: Sequence[SourceRef]) -> list[UncertaintyFlag]:
        """Evidence outside the window its document class is considered current for."""
        flags: list[UncertaintyFlag] = []
        today = date.today()
        for src in sources:
            if src.document_date is None:
                continue
            interval = _staleness_interval_days(src.document_type, self._settings)
            if interval is None:
                continue
            age_days = (today - src.document_date).days
            if age_days <= interval:
                continue
            flags.append(
                UncertaintyFlag(
                    source=UncertaintySource.STALE_DOCUMENT,
                    message=(
                        f"{src.citation} is dated {src.document_date.isoformat()} — {age_days} days "
                        f"old, against a {interval}-day relevance window for a "
                        f"{src.document_type.value.replace('_', ' ')}. Later work may supersede it."
                    ),
                    severity=Severity.WARNING,
                    affected_source=src,
                    suggested_action="Check for a more recent record before relying on this.",
                )
            )
        return flags

    def _flag_conflicts(self, sources: Sequence[SourceRef]) -> list[UncertaintyFlag]:
        """Two documents asserting materially different values for the same measurand."""
        by_key: dict[tuple[str, str], list[_Measurement]] = defaultdict(list)
        for measurement in self._extract_measurements(sources):
            by_key[(measurement.label, measurement.unit)].append(measurement)

        flags: list[UncertaintyFlag] = []
        for (label, unit), measurements in sorted(by_key.items()):
            if len(measurements) < 2:
                continue
            lowest = min(measurements, key=lambda m: m.value)
            highest = max(measurements, key=lambda m: m.value)
            if lowest.source.document_id == highest.source.document_id:
                continue  # One document restating itself is not a conflict.
            span = highest.value - lowest.value
            scale = abs(highest.value) or 1.0
            if span / scale <= NUMERIC_CONFLICT_TOLERANCE:
                continue
            flags.append(
                UncertaintyFlag(
                    source=UncertaintySource.CONFLICTING_SOURCES,
                    message=(
                        f"'{label}' is given as {lowest.rendered} in {lowest.source.citation} but "
                        f"{highest.rendered} in {highest.source.citation} — a "
                        f"{100 * span / scale:.0f}% difference in {unit}."
                    ),
                    severity=Severity.HIGH,
                    affected_claim=f"{label} ({unit})",
                    affected_source=highest.source,
                    suggested_action="Take a fresh measurement; do not average the two records.",
                )
            )
        return flags

    def _flag_vision(self, sources: Sequence[SourceRef]) -> list[UncertaintyFlag]:
        """Facts read off a drawing rather than out of text."""
        flags: list[UncertaintyFlag] = []
        for src in sources:
            is_vision = src.retrieved_via == "vision" or src.document_type is DocumentType.PID_DRAWING
            if not is_vision:
                continue
            flags.append(
                UncertaintyFlag(
                    source=UncertaintySource.VISION_INFERENCE,
                    message=(
                        f"{src.citation} is a drawing. The symbol classes, tags and pipe runs behind "
                        "this claim were inferred by the vision pipeline at "
                        f"{src.extraction_confidence:.2f} confidence, not read from text."
                    ),
                    severity=Severity.WARNING,
                    affected_source=src,
                    suggested_action="Confirm against the controlled drawing before acting on connectivity.",
                )
            )
        return flags

    def _flag_sparse(
        self,
        sources: Sequence[SourceRef],
        *,
        steps: Sequence[ReasoningStep] = (),
    ) -> UncertaintyFlag | None:
        """Fewer than two *independent* documents behind the answer.

        Independence is counted by document, not by passage: three chunks of one manual are one
        source, and treating them as three is how a system talks itself into false confidence.
        """
        documents = {src.document_id for src in sources}
        graph_only = any(step.graph_paths and not step.sources for step in steps)

        if len(documents) >= MIN_CORROBORATING_SOURCES:
            return None
        if not documents:
            return UncertaintyFlag(
                source=UncertaintySource.SPARSE_EVIDENCE,
                message=(
                    "No document supports this answer."
                    + (" It rests entirely on knowledge-graph structure." if graph_only else "")
                ),
                severity=Severity.CRITICAL,
                suggested_action="Treat as unverified; consult plant records directly.",
            )
        only = next(iter(documents))
        citation = next((s.citation for s in sources if s.document_id == only), only)
        return UncertaintyFlag(
            source=UncertaintySource.SPARSE_EVIDENCE,
            message=(
                f"Only one document ({citation}) supports this answer, so nothing corroborates it. "
                f"Corroboration needs at least {MIN_CORROBORATING_SOURCES} independent sources."
            ),
            severity=Severity.WARNING,
            suggested_action="Look for a second, independent record before acting.",
        )

    # ==================================================================================
    # Measurement extraction
    # ==================================================================================

    def _extract_measurements(self, sources: Sequence[SourceRef]) -> list[_Measurement]:
        """Lift ``(label, value, unit)`` triples out of source snippets.

        Deliberately conservative: a number is only considered when it carries a recognised
        physical unit. Bare numbers are dates, counts, work-order ids and revision numbers, and
        comparing those across documents produces conflicts that are not real — which trains
        operators to dismiss the flag that matters.
        """
        out: list[_Measurement] = []
        for src in sources:
            if not src.snippet:
                continue
            for match in _MEASUREMENT_RE.finditer(src.snippet):
                label = _normalise_label(match.group("label"))
                if not label:
                    continue
                unit = _UNIT_ALIASES.get(match.group("unit").lower())
                if unit is None:
                    continue
                try:
                    value = float(match.group("value"))
                except ValueError:  # pragma: no cover - regex guarantees a numeric literal
                    continue
                out.append(_Measurement(label=label, value=value, unit=unit, source=src))
        return out

    # ==================================================================================
    # Rendering helpers
    # ==================================================================================

    @staticmethod
    def _source_payload(src: SourceRef) -> dict[str, Any]:
        return {
            "document_id": src.document_id,
            "document_title": src.document_title,
            "document_type": src.document_type.value,
            "citation": src.citation,
            "page": src.page,
            "snippet": src.snippet,
            "relevance": round(src.relevance, 4),
            "extraction_confidence": round(src.extraction_confidence, 4),
            "retrieved_via": src.retrieved_via,
            "document_date": src.document_date.isoformat() if src.document_date else None,
            "chunk_id": src.chunk_id,
            "bbox": list(src.bbox) if src.bbox else None,
        }


def _normalise_label(raw: str) -> str:
    """Reduce a measurand label to its meaningful words.

    ``"the bearing wear was"`` and ``"Bearing Wear"`` must collapse to the same key, or a genuine
    conflict between two documents is invisible to the grouping.
    """
    words = [w for w in re.split(r"[\s_-]+", raw.strip().lower()) if w and w not in _LABEL_STOPWORDS]
    # The last two words carry the measurand; earlier ones are usually sentence scaffolding.
    return " ".join(words[-2:]) if words else ""


def summarise_paths(paths: Iterable[GraphPath], *, limit: int) -> list[str]:
    """Render graph paths as narrative lines for a reasoning step's finding text."""
    lines: list[str] = []
    for path in paths:
        if len(lines) >= limit:
            break
        narrative = path.narrative.strip() or " → ".join(path.nodes)
        lines.append(f"{narrative} ({path.hops} hop(s), confidence {path.confidence:.2f})")
    return lines


__all__ = [
    "AnswerExplainer",
    "MAX_ALTERNATIVES",
    "MIN_CORROBORATING_SOURCES",
    "NUMERIC_CONFLICT_TOLERANCE",
    "summarise_paths",
]
