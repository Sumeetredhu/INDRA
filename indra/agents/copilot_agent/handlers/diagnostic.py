"""The diagnostic handler: eleven recorded steps from a question to a root cause.

This is the handler the whole system exists to support. "Why did P-101 fail last month?" is not a
retrieval problem — the answer is not written down in any single document. It lives in the join
between a work order from March, a shift log entry from April, a 2022 incident report, and the
OEM's bearing limit. Step 7 is where that join happens.

The chain, each link recorded as a :class:`ReasoningStep` carrying its own sources, confidence and
optional Cypher:

1.  Resolve the equipment against the graph registry.
2.  Maintenance records within ``settings.maintenance_lookback_days``.
3.  Inspection history within ``settings.inspection_lookback_days``.
4.  Every historical failure, all time — old failures are the most valuable evidence here.
5.  Shift logs within ``settings.shift_log_lookback_days``.
6.  OEM specifications and threshold limits.
7.  **Semantic precursor match.** Current findings are embedded and compared against the precursor
    text of historical failures. A cosine above ``settings.precursor_similarity_threshold`` means
    today's symptoms read like the run-up to a failure that already happened once. No document
    contains this fact; it only exists in the comparison.
8.  Current condition readings against OEM thresholds.
9.  Alarm-bypass events coinciding with the anomaly window.
10. Chain assembly — the causal ordering, with the weakest link named.
11. The natural-language answer plus concrete recommended actions.

Every step degrades independently. A dead embedding provider drops step 7 to lexical similarity
and says so in its rationale; it does not fail the answer.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Final, Mapping, Sequence

import numpy as np
from rapidfuzz import fuzz

from indra.agents.copilot_agent import prompts
from indra.agents.copilot_agent.classifier import extract_equipment_tags
from indra.agents.copilot_agent.handlers.base import (
    MAX_RECOMMENDED_ACTIONS,
    BaseHandler,
)
from indra.core.exceptions import IndraError
from indra.core.logging import get_logger
from indra.core.models import (
    Answer,
    Confidence,
    ConditionReading,
    DocumentType,
    Equipment,
    FailureEvent,
    MaintenanceRecord,
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
# Read-only Cypher recorded on each step. Displayed to technical users, never executed here —
# the in-memory graph backend rejects raw Cypher by contract, and a diagnostic chain that only
# works against Neo4j is not a diagnostic chain.
# --------------------------------------------------------------------------------------

CYPHER_RESOLVE_EQUIPMENT: Final[str] = (
    "MATCH (e:Equipment {tag: $tag}) RETURN e.tag, e.name, e.equipment_type, e.manufacturer, "
    "e.model, e.criticality, e.location, e.oem_thresholds"
)
CYPHER_MAINTENANCE: Final[str] = (
    "MATCH (e:Equipment {tag: $tag})<-[:APPLIES_TO]-(m:MaintenanceRecord) "
    "WHERE m.performed_on >= date($since) "
    "RETURN m ORDER BY m.performed_on DESC"
)
CYPHER_INSPECTIONS: Final[str] = (
    "MATCH (e:Equipment {tag: $tag})<-[:APPLIES_TO]-(m:MaintenanceRecord) "
    "WHERE m.record_type = 'inspection' AND m.performed_on >= date($since) "
    "RETURN m ORDER BY m.performed_on DESC"
)
CYPHER_FAILURES: Final[str] = (
    "MATCH (e:Equipment {tag: $tag})-[:FAILED_WITH_MODE]->(f:FailureMode) "
    "RETURN f.failure_mode, f.occurred_on, f.root_cause, f.precursor_text, f.downtime_hours, "
    "f.cost_inr ORDER BY f.occurred_on DESC"
)
CYPHER_SHIFT_LOGS: Final[str] = (
    "MATCH (d:Document {document_type: 'shift_log'})-[:MENTIONS]->(e:Equipment {tag: $tag}) "
    "WHERE d.document_date >= date($since) RETURN d ORDER BY d.document_date DESC"
)
CYPHER_OEM: Final[str] = (
    "MATCH (d:Document {document_type: 'oem_manual'})-[:APPLIES_TO]->(e:Equipment {tag: $tag}) "
    "RETURN d.title, e.oem_thresholds, e.specifications"
)
CYPHER_PRECURSOR: Final[str] = (
    "MATCH (e:Equipment {tag: $tag})-[:FAILED_WITH_MODE]->(f:FailureMode) "
    "RETURN f.precursor_text, f.occurred_on, f.failure_mode "
    "// cosine similarity against current findings is computed in-process, not in Cypher"
)
CYPHER_THRESHOLDS: Final[str] = (
    "MATCH (e:Equipment {tag: $tag})<-[:APPLIES_TO]-(m:MaintenanceRecord)-[:HAS_READING]->(r:ConditionReading) "
    "RETURN r.parameter, r.value, r.unit, r.measured_at, e.oem_thresholds "
    "ORDER BY r.measured_at DESC"
)
CYPHER_BYPASS: Final[str] = (
    "MATCH (d:Document)-[:MENTIONS]->(e:Equipment {tag: $tag}) "
    "WHERE d.document_date >= date($window_start) AND d.document_date <= date($window_end) "
    "AND (d.text CONTAINS 'bypass' OR d.text CONTAINS 'inhibit' OR d.text CONTAINS 'override') "
    "RETURN d.title, d.document_date"
)


# --------------------------------------------------------------------------------------
# Domain vocabulary. Taxonomy, not tunables: these are the words plant staff actually write.
# --------------------------------------------------------------------------------------

BYPASS_PATTERNS: Final[tuple[re.Pattern[str], ...]] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\balarm[s]?\s+(?:was|were|is|are|been)?\s*bypass(?:ed|ing)?\b",
        r"\bbypass(?:ed|ing)?\s+(?:the\s+)?(?:alarm|trip|interlock|protection|shutdown)\b",
        r"\b(?:trip|interlock|shutdown|protection)\s+(?:was|were|is|are)?\s*(?:inhibit|defeat|disabl|overrid|forc)\w*\b",
        r"\b(?:inhibit|defeat|disabl|overrid|forc)\w*\s+(?:the\s+)?(?:alarm|trip|interlock|protection)\b",
        r"\balarm\s+(?:suppress|silenc|mask)\w*\b",
        r"\b(?:mos|maintenance\s+override\s+switch)\b",
    )
)

ANOMALY_PATTERNS: Final[tuple[re.Pattern[str], ...]] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bvibrat\w*\b", r"\bnois[ey]\w*\b", r"\brattl\w*\b", r"\bknock\w*\b",
        r"\bleak\w*\b", r"\bseep\w*\b", r"\bweep\w*\b",
        r"\boverheat\w*\b", r"\bhigh\s+temperature\b", r"\btemperature\s+ris\w*\b",
        r"\bcavitat\w*\b", r"\bmisalign\w*\b", r"\bimbalance\b|\bunbalanc\w*\b",
        r"\bwear\b|\bworn\b|\bscor\w*\b|\bpitt\w*\b",
        r"\bseal\s+(?:failure|leak|damage)\b", r"\bbearing\s+(?:noise|wear|damage|failure)\b",
        r"\btrip(?:ped|ping)?\b", r"\bsurg\w*\b", r"\bfluctuat\w*\b", r"\bdegrad\w*\b",
    )
)

#: Document classes that count as shift-floor narrative for step 5.
SHIFT_LOG_TYPES: Final[frozenset[DocumentType]] = frozenset(
    {DocumentType.SHIFT_LOG, DocumentType.EMAIL, DocumentType.INCIDENT_REPORT}
)

#: Document classes that count as OEM reference for step 6.
OEM_TYPES: Final[frozenset[DocumentType]] = frozenset(
    {DocumentType.OEM_MANUAL, DocumentType.SOP}
)

#: Maximum precursor matches narrated. Beyond the strongest few the signal is diluted.
MAX_PRECURSOR_MATCHES: Final[int] = 4

#: Maximum evidence items rendered per digest section, to keep the prompt inside its budget.
MAX_DIGEST_ITEMS: Final[int] = 8

#: Lexical fallback for step 7 when no embedding provider is reachable. rapidfuzz returns 0-100;
#: scaled to 0-1 and compared against the same configured threshold, then discounted in the
#: step's confidence because token overlap is a weaker claim than semantic proximity.
LEXICAL_MATCH_DISCOUNT: Final[float] = 0.75


# --------------------------------------------------------------------------------------
# Evidence containers
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class PrecursorMatch:
    """One historical failure whose run-up reads like today's findings."""

    failure: FailureEvent
    finding_text: str
    finding_source: SourceRef | None
    similarity: float
    method: str

    @property
    def narrative(self) -> str:
        return (
            f"Current finding \"{_clip(self.finding_text)}\" matches the precursor recorded before "
            f"the {self.failure.failure_mode} failure of "
            f"{self.failure.occurred_on.isoformat()} (\"{_clip(self.failure.precursor_text)}\") "
            f"at {self.similarity:.2f} {self.method} similarity"
            + (f"; that failure's root cause was recorded as {self.failure.root_cause}"
               if self.failure.root_cause else "")
            + "."
        )


@dataclass(slots=True)
class ThresholdBreach:
    """A condition reading measured against its OEM limit."""

    reading: ConditionReading
    limit: float
    ratio: float
    exceeded: bool

    @property
    def narrative(self) -> str:
        verdict = "EXCEEDS" if self.exceeded else "approaching"
        return (
            f"{self.reading.parameter} measured {self.reading.value:g}{self.reading.unit} on "
            f"{self.reading.measured_at.date().isoformat()} {verdict} the OEM limit of "
            f"{self.limit:g} ({self.ratio * 100:.0f}% of limit)."
        )


@dataclass(slots=True)
class BypassEvent:
    """An alarm-bypass mention, with the date that decides whether it coincides."""

    text: str
    occurred_on: date | None
    source: SourceRef | None
    within_window: bool


@dataclass(slots=True)
class DiagnosticEvidence:
    """Everything the chain gathered, in one place, so step 10 can order it causally."""

    tag: str
    equipment: Equipment | None = None
    maintenance: list[MaintenanceRecord] = field(default_factory=list)
    inspections: list[MaintenanceRecord] = field(default_factory=list)
    failures: list[FailureEvent] = field(default_factory=list)
    shift_logs: list[RetrievedPassage] = field(default_factory=list)
    oem_passages: list[RetrievedPassage] = field(default_factory=list)
    precursor_matches: list[PrecursorMatch] = field(default_factory=list)
    breaches: list[ThresholdBreach] = field(default_factory=list)
    bypasses: list[BypassEvent] = field(default_factory=list)
    window_start: date | None = None
    window_end: date | None = None


class DiagnosticHandler(BaseHandler):
    """Root-cause analysis across the whole plant record set."""

    query_type = QueryType.DIAGNOSTIC

    async def handle(self, request: QueryRequest, *, retrieval: RetrievalResult) -> Answer:
        started = time.perf_counter()
        steps: list[ReasoningStep] = []

        # ---- Step 1 -----------------------------------------------------------------
        evidence, step = await self._step_resolve(request, retrieval)
        steps.append(step)

        if evidence.equipment is None and evidence.tag == "":
            # No asset to anchor on. Fall through to a grounded answer over whatever retrieval
            # found — still a real answer, just without the structured chain.
            logger.info("diagnostic query names no resolvable equipment; using grounded path")
            return await self.grounded_answer(
                request,
                retrieval,
                steps,
                started=started,
                extras={"tag": "(unresolved)", "evidence": prompts.DIGEST_EMPTY_V1},
            )

        # ---- Steps 2-6: evidence gathering ------------------------------------------
        steps.append(await self._step_maintenance(evidence))
        steps.append(await self._step_inspections(evidence))
        steps.append(await self._step_failures(evidence))
        steps.append(await self._step_shift_logs(evidence, request))
        steps.append(await self._step_oem(evidence, request))

        # ---- Step 7: the cross-document insight --------------------------------------
        steps.append(await self._step_precursor_match(evidence))

        # ---- Steps 8-9 ----------------------------------------------------------------
        steps.append(self._step_thresholds(evidence))
        steps.append(self._step_bypass(evidence, retrieval))

        # ---- Step 10 -------------------------------------------------------------------
        steps.append(self._step_assemble_chain(evidence, steps))

        # ---- Step 11: answer + actions (composed by grounded_answer) -------------------
        digest = self._render_digest(evidence)
        pinned = self._pinned_sources(evidence)
        actions = self._deterministic_actions(evidence)

        return await self.grounded_answer(
            request,
            retrieval,
            steps,
            started=started,
            extras={"tag": evidence.tag, "evidence": digest},
            extra_actions=actions,
            pinned_sources=pinned,
        )

    # ==================================================================================
    # Prompt composition (step 11)
    # ==================================================================================

    async def compose(
        self,
        request: QueryRequest,
        retrieval: RetrievalResult,
        context_block: str,
        extras: Mapping[str, str],
    ) -> tuple[str, str]:
        return (
            prompts.GROUNDED_SYSTEM_V1,
            prompts.render(
                prompts.DIAGNOSTIC_PROMPT_V1,
                context=context_block,
                tag=extras.get("tag", "the asset"),
                evidence=extras.get("evidence", prompts.DIGEST_EMPTY_V1),
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
        """Model-proposed actions on top of the deterministic ones. Empty when unavailable."""
        tags = extract_equipment_tags(request.query)
        tag = request.equipment_tag or (tags[0] if tags else "")
        if not tag or not answer_text.strip():
            return []
        return await self.llm_actions(
            findings=answer_text,
            tag=tag,
            criticality="unknown",
            limit=MAX_RECOMMENDED_ACTIONS,
        )

    # ==================================================================================
    # Step 1 — resolve the equipment
    # ==================================================================================

    async def _step_resolve(
        self, request: QueryRequest, retrieval: RetrievalResult
    ) -> tuple[DiagnosticEvidence, ReasoningStep]:
        started = time.perf_counter()
        candidates: list[str] = []
        if request.equipment_tag:
            candidates.append(request.equipment_tag.strip().upper())
        candidates.extend(extract_equipment_tags(request.query))
        for key in retrieval.query_entities:
            _, _, value = key.partition(":")
            candidates.extend(extract_equipment_tags(value or key))

        seen: set[str] = set()
        ordered = [c for c in candidates if not (c in seen or seen.add(c))]

        equipment: Equipment | None = None
        resolved_tag = ""
        for candidate in ordered:
            found = await self._get_equipment(candidate)
            if found is not None:
                equipment = found
                resolved_tag = found.tag
                break

        if equipment is None and ordered:
            # A tag was written but the graph does not know it. That is a real finding — the
            # question is about an asset INDRA has never been given documents for.
            resolved_tag = ordered[0]

        evidence = DiagnosticEvidence(tag=resolved_tag, equipment=equipment)

        if equipment is not None:
            finding = (
                f"Resolved {equipment.tag} in the equipment registry: "
                f"{equipment.name or equipment.equipment_type}, criticality "
                f"{equipment.criticality.value}"
                + (f", manufacturer {equipment.manufacturer}" if equipment.manufacturer else "")
                + (f", model {equipment.model}" if equipment.model else "")
                + (f", located {equipment.location}" if equipment.location else "")
                + "."
            )
            confidence = Confidence.exact(
                f"Exact tag match on the equipment registry primary key ({equipment.tag})."
            )
        elif resolved_tag:
            finding = (
                f"'{resolved_tag}' parses as a plant tag but has no node in the equipment "
                "registry. Diagnosis will proceed on retrieved documents alone, with no "
                "structured maintenance, failure or specification history to draw on."
            )
            confidence = Confidence(
                value=0.4,
                rationale=(
                    "Tag grammar matched but the registry lookup missed; the tag may be "
                    "mistyped, superseded, or its documents may never have been ingested."
                ),
                method="heuristic",
            )
        else:
            finding = (
                "The question names no plant tag INDRA can resolve, so there is no asset to "
                "anchor a structured diagnosis on."
            )
            confidence = Confidence(
                value=0.2,
                rationale="No tag-shaped token in the question matched the registry.",
                method="heuristic",
            )

        return evidence, ReasoningStep(
            order=1,
            action="Resolved the equipment against the plant registry",
            finding=finding,
            confidence=confidence,
            cypher=CYPHER_RESOLVE_EQUIPMENT,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    # ==================================================================================
    # Step 2 — maintenance history
    # ==================================================================================

    async def _step_maintenance(self, evidence: DiagnosticEvidence) -> ReasoningStep:
        started = time.perf_counter()
        window = self.settings.maintenance_lookback_days
        since = date.today() - timedelta(days=window)
        records = await self._maintenance_history(evidence.tag, since=since)
        evidence.maintenance = [r for r in records if r.record_type != "inspection"]

        sources = _sources_of(evidence.maintenance)
        if evidence.maintenance:
            latest = max(evidence.maintenance, key=lambda r: r.performed_on)
            anomalies = [r for r in evidence.maintenance if _matches_any(r.findings, ANOMALY_PATTERNS)]
            finding = (
                f"{len(evidence.maintenance)} maintenance record(s) in the last {window} days. "
                f"Most recent: {latest.performed_on.isoformat()}"
                + (f" by {latest.performed_by}" if latest.performed_by else "")
                + (f" — {_clip(latest.findings)}" if latest.findings else "")
                + "."
                + (
                    f" {len(anomalies)} record(s) describe abnormal condition: "
                    + "; ".join(_clip(r.findings) for r in anomalies[:MAX_DIGEST_ITEMS])
                    if anomalies
                    else " No record in the window describes an abnormal condition."
                )
            )
            confidence = Confidence.exact(
                f"{len(evidence.maintenance)} structured maintenance record(s) read directly from "
                "the graph; no inference involved."
            )
        else:
            finding = (
                f"No maintenance record for {evidence.tag} in the last {window} days. For a "
                "diagnostic question this is itself evidence: either the asset was untouched, or "
                "work was done and never documented."
            )
            confidence = Confidence.exact(
                "Absence confirmed against the structured maintenance store, which is authoritative "
                "for what it contains."
            )

        return ReasoningStep(
            order=2,
            action=f"Read maintenance history ({window}-day window)",
            finding=finding,
            confidence=confidence,
            sources=sources,
            cypher=CYPHER_MAINTENANCE,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    # ==================================================================================
    # Step 3 — inspection history
    # ==================================================================================

    async def _step_inspections(self, evidence: DiagnosticEvidence) -> ReasoningStep:
        started = time.perf_counter()
        window = self.settings.inspection_lookback_days
        since = date.today() - timedelta(days=window)
        records = await self._maintenance_history(evidence.tag, since=since)
        evidence.inspections = [r for r in records if r.record_type in ("inspection", "calibration")]

        sources = _sources_of(evidence.inspections)
        if evidence.inspections:
            latest = max(evidence.inspections, key=lambda r: r.performed_on)
            reading_count = sum(len(r.readings) for r in evidence.inspections)
            finding = (
                f"{len(evidence.inspections)} inspection/calibration record(s) in the last {window} "
                f"days carrying {reading_count} condition reading(s). Latest "
                f"{latest.performed_on.isoformat()}"
                + (f": {_clip(latest.findings)}" if latest.findings else "")
                + (f" Recommendation on file: {_clip(latest.recommendations)}"
                   if latest.recommendations else "")
            )
            confidence = Confidence.exact(
                f"{len(evidence.inspections)} structured inspection record(s) with "
                f"{reading_count} measured value(s)."
            )
        else:
            finding = (
                f"No inspection or calibration record for {evidence.tag} in the last {window} days. "
                "Without inspection data there are no measured trends to compare against OEM limits."
            )
            confidence = Confidence.exact("Absence confirmed against the structured inspection store.")

        return ReasoningStep(
            order=3,
            action=f"Read inspection history ({window}-day window)",
            finding=finding,
            confidence=confidence,
            sources=sources,
            cypher=CYPHER_INSPECTIONS,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    # ==================================================================================
    # Step 4 — all historical failures
    # ==================================================================================

    async def _step_failures(self, evidence: DiagnosticEvidence) -> ReasoningStep:
        started = time.perf_counter()
        evidence.failures = await self._failure_history(evidence.tag)

        sources: list[SourceRef] = []
        for failure in evidence.failures:
            sources.extend(failure.sources)

        if evidence.failures:
            modes: dict[str, int] = {}
            for failure in evidence.failures:
                modes[failure.failure_mode] = modes.get(failure.failure_mode, 0) + 1
            repeat = [mode for mode, count in modes.items() if count >= self.settings.fleet_failure_min_count]
            total_cost = sum(f.cost_inr or 0.0 for f in evidence.failures)
            total_downtime = sum(f.downtime_hours or 0.0 for f in evidence.failures)
            listing = "; ".join(
                f"{f.occurred_on.isoformat()} {f.failure_mode}"
                + (f" (root cause: {f.root_cause})" if f.root_cause else "")
                for f in sorted(evidence.failures, key=lambda f: f.occurred_on, reverse=True)[
                    :MAX_DIGEST_ITEMS
                ]
            )
            finding = (
                f"{len(evidence.failures)} historical failure(s), all time: {listing}."
                + (
                    f" Recurring mode(s): {', '.join(repeat)} — a repeat failure means the previous "
                    "root cause was either wrong or never corrected."
                    if repeat
                    else ""
                )
                + (f" Cumulative recorded downtime {total_downtime:.0f} h." if total_downtime else "")
                + (f" Cumulative recorded cost ₹{total_cost:,.0f}." if total_cost else "")
            )
            confidence = Confidence.exact(
                f"{len(evidence.failures)} structured failure event(s) from the graph, unfiltered "
                "by date so nothing old is lost."
            )
        else:
            finding = (
                f"No failure has ever been recorded against {evidence.tag}. Any root cause "
                "proposed below therefore has no precedent on this asset to corroborate it."
            )
            confidence = Confidence.exact("Absence confirmed against the structured failure store.")

        return ReasoningStep(
            order=4,
            action="Read every historical failure (no date limit)",
            finding=finding,
            confidence=confidence,
            sources=sources,
            cypher=CYPHER_FAILURES,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    # ==================================================================================
    # Step 5 — shift logs
    # ==================================================================================

    async def _step_shift_logs(
        self, evidence: DiagnosticEvidence, request: QueryRequest
    ) -> ReasoningStep:
        started = time.perf_counter()
        window = self.settings.shift_log_lookback_days
        cutoff = date.today() - timedelta(days=window)

        result = await self.retrieve(
            request.query,
            equipment_tag=evidence.tag or None,
            filters={"document_type": DocumentType.SHIFT_LOG.value},
        )
        candidates = list(result.passages)
        if not candidates:
            # A store that cannot filter by document type still gives us passages to sift.
            fallback = await self.retrieve(request.query, equipment_tag=evidence.tag or None)
            candidates = list(fallback.passages)

        evidence.shift_logs = [
            p
            for p in candidates
            if p.document.document_type in SHIFT_LOG_TYPES
            and (p.document.document_date is None or p.document.document_date >= cutoff)
        ]

        sources = [p.as_source() for p in evidence.shift_logs]
        anomalous = [p for p in evidence.shift_logs if _matches_any(p.chunk.text, ANOMALY_PATTERNS)]

        if evidence.shift_logs:
            finding = (
                f"{len(evidence.shift_logs)} shift-floor entry(ies) mentioning {evidence.tag} in the "
                f"last {window} days."
                + (
                    " Entries describing abnormal behaviour: "
                    + "; ".join(_clip(p.chunk.text) for p in anomalous[:MAX_DIGEST_ITEMS])
                    if anomalous
                    else " None describes abnormal behaviour."
                )
            )
            relevances = [max(0.0, min(1.0, p.fused_score)) for p in evidence.shift_logs]
            confidence = Confidence(
                value=round(sum(relevances) / len(relevances), 4),
                rationale=(
                    f"Retrieved {len(evidence.shift_logs)} shift-log passage(s) at mean relevance "
                    f"{sum(relevances) / len(relevances):.2f}; shift logs are free text, so this is "
                    "a retrieval judgement rather than a structured fact."
                ),
                method="semantic",
            )
        else:
            finding = (
                f"No shift log, email or incident entry mentioning {evidence.tag} in the last "
                f"{window} days. Operator narrative is where symptoms appear before instruments "
                "catch them, so its absence narrows what can be reconstructed."
            )
            confidence = Confidence(
                value=0.5,
                rationale=(
                    "Absence is from a retrieval search over free text, not a structured store; "
                    "an entry that never named the tag would not be found."
                ),
                method="semantic",
            )

        return ReasoningStep(
            order=5,
            action=f"Searched shift logs and operator narrative ({window}-day window)",
            finding=finding,
            confidence=confidence,
            sources=sources,
            cypher=CYPHER_SHIFT_LOGS,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    # ==================================================================================
    # Step 6 — OEM specifications
    # ==================================================================================

    async def _step_oem(self, evidence: DiagnosticEvidence, request: QueryRequest) -> ReasoningStep:
        started = time.perf_counter()
        thresholds = dict(evidence.equipment.oem_thresholds) if evidence.equipment else {}
        specifications = dict(evidence.equipment.specifications) if evidence.equipment else {}

        result = await self.retrieve(
            request.query,
            equipment_tag=evidence.tag or None,
            filters={"document_type": DocumentType.OEM_MANUAL.value},
        )
        evidence.oem_passages = [p for p in result.passages if p.document.document_type in OEM_TYPES]
        sources = [p.as_source() for p in evidence.oem_passages]

        if thresholds:
            rendered = ", ".join(f"{name}={limit:g}" for name, limit in sorted(thresholds.items()))
            finding = f"OEM limits on file for {evidence.tag}: {rendered}."
            if specifications:
                finding += " Specifications: " + ", ".join(
                    f"{k}={v}" for k, v in sorted(specifications.items())[:MAX_DIGEST_ITEMS]
                ) + "."
            if evidence.oem_passages:
                finding += (
                    f" {len(evidence.oem_passages)} OEM manual passage(s) retrieved as provenance."
                )
            confidence = Confidence.exact(
                f"{len(thresholds)} threshold(s) held as structured properties on the equipment node."
            )
        elif evidence.oem_passages:
            finding = (
                f"No structured OEM threshold is stored against {evidence.tag}, but "
                f"{len(evidence.oem_passages)} manual passage(s) were retrieved. Limits quoted "
                "below come from manual text, not from a parsed specification table."
            )
            confidence = Confidence(
                value=round(
                    max((min(1.0, p.fused_score) for p in evidence.oem_passages), default=0.0), 4
                ),
                rationale=(
                    "Limits read from prose rather than a structured field; a misparsed unit here "
                    "would change every comparison that follows."
                ),
                method="semantic",
            )
        else:
            finding = (
                f"No OEM manual and no stored threshold for {evidence.tag}. Nothing can be compared "
                "against a manufacturer limit, so step 8 has no reference to work from."
            )
            confidence = Confidence.exact(
                "Absence confirmed against both the equipment node's threshold map and OEM document "
                "retrieval."
            )

        return ReasoningStep(
            order=6,
            action="Read OEM specifications and threshold limits",
            finding=finding,
            confidence=confidence,
            sources=sources,
            cypher=CYPHER_OEM,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    # ==================================================================================
    # Step 7 — semantic precursor match  (the insight no single document contains)
    # ==================================================================================

    async def _step_precursor_match(self, evidence: DiagnosticEvidence) -> ReasoningStep:
        started = time.perf_counter()
        threshold = self.settings.precursor_similarity_threshold

        findings = self._current_findings(evidence)
        historical = [f for f in evidence.failures if f.precursor_text.strip()]

        if not findings or not historical:
            missing = []
            if not findings:
                missing.append("no current symptom text (no recent findings, inspections or logs)")
            if not historical:
                missing.append("no historical failure carries recorded precursor text")
            return ReasoningStep(
                order=7,
                action="Matched current findings against historical failure precursors",
                finding=(
                    "The cross-document precursor comparison could not run: "
                    + " and ".join(missing)
                    + ". This is the step that would surface a pattern no single document records, "
                    "so its absence is a real limitation of this diagnosis, not a formality."
                ),
                confidence=Confidence(
                    value=0.0,
                    rationale="Comparison had nothing to compare; it contributes no evidence.",
                    method="semantic",
                ),
                cypher=CYPHER_PRECURSOR,
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )

        matches, method = await self._similarity_matches(findings, historical, threshold)
        evidence.precursor_matches = matches[:MAX_PRECURSOR_MATCHES]

        sources: list[SourceRef] = []
        for match in evidence.precursor_matches:
            if match.finding_source is not None:
                sources.append(match.finding_source)
            sources.extend(match.failure.sources)

        if evidence.precursor_matches:
            best = evidence.precursor_matches[0]
            finding = (
                f"{len(evidence.precursor_matches)} of {len(findings)} current finding(s) match a "
                f"recorded failure precursor above the {threshold:.2f} threshold. "
                + " ".join(m.narrative for m in evidence.precursor_matches)
                + " This link exists in no single document — it is the comparison between today's "
                "observations and the run-up to a failure that already happened."
            )
            confidence = Confidence(
                value=round(best.similarity if method == "cosine"
                            else best.similarity * LEXICAL_MATCH_DISCOUNT, 4),
                rationale=(
                    f"Strongest match {best.similarity:.2f} by {method} similarity against the "
                    f"{best.failure.occurred_on.isoformat()} {best.failure.failure_mode} event"
                    + (
                        "; computed on token overlap rather than embeddings because no embedding "
                        "provider was reachable, so the claim is weaker than a semantic match."
                        if method == "lexical"
                        else "."
                    )
                ),
                method="semantic",
            )
        else:
            finding = (
                f"None of the {len(findings)} current finding(s) resembles any of the "
                f"{len(historical)} recorded failure precursor(s) above the {threshold:.2f} "
                f"{method} similarity threshold. Today's symptoms do not look like the run-up to "
                "any failure this asset has had before, which argues against a repeat of a known "
                "mode."
            )
            confidence = Confidence(
                value=round(threshold, 4),
                rationale=(
                    f"A negative result from a complete comparison of {len(findings)} finding(s) "
                    f"against {len(historical)} precursor(s) by {method} similarity."
                ),
                method="semantic",
            )

        return ReasoningStep(
            order=7,
            action="Matched current findings against historical failure precursors",
            finding=finding,
            confidence=confidence,
            sources=sources,
            cypher=CYPHER_PRECURSOR,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    # ==================================================================================
    # Step 8 — OEM threshold comparison
    # ==================================================================================

    def _step_thresholds(self, evidence: DiagnosticEvidence) -> ReasoningStep:
        started = time.perf_counter()
        warn_ratio = self.settings.oem_threshold_warning_ratio
        thresholds = dict(evidence.equipment.oem_thresholds) if evidence.equipment else {}

        readings: list[ConditionReading] = []
        for record in (*evidence.inspections, *evidence.maintenance):
            readings.extend(record.readings)

        breaches: list[ThresholdBreach] = []
        compared = 0
        for reading in readings:
            limit = _match_threshold(reading.parameter, thresholds)
            if limit is None or limit == 0:
                continue
            compared += 1
            ratio = reading.value / limit
            if ratio >= warn_ratio:
                breaches.append(
                    ThresholdBreach(
                        reading=reading, limit=limit, ratio=ratio, exceeded=ratio >= 1.0
                    )
                )
        breaches.sort(key=lambda b: b.ratio, reverse=True)
        evidence.breaches = breaches

        sources = [r.source for r in readings if r.source is not None]

        if breaches:
            exceeded = [b for b in breaches if b.exceeded]
            finding = (
                f"{len(breaches)} of {compared} reading(s) compared against an OEM limit are at or "
                f"past {warn_ratio * 100:.0f}% of it. "
                + " ".join(b.narrative for b in breaches[:MAX_DIGEST_ITEMS])
                + (
                    f" {len(exceeded)} reading(s) exceed the limit outright."
                    if exceeded
                    else " None has crossed the limit yet."
                )
            )
            confidences = [r.confidence for b in breaches for r in [b.reading]]
            weakest = min(confidences, key=lambda c: c.value)
            confidence = Confidence(
                value=weakest.value,
                rationale=(
                    f"Arithmetic against stored limits is exact, so this is capped by the weakest "
                    f"reading it used: {weakest.rationale} ({weakest.value:.2f})."
                ),
                method="aggregate",
            )
        elif compared:
            finding = (
                f"All {compared} reading(s) with a matching OEM limit sit below "
                f"{warn_ratio * 100:.0f}% of it. No measured parameter supports a "
                "threshold-exceedance explanation."
            )
            confidence = Confidence.exact(
                f"{compared} reading(s) compared arithmetically against stored OEM limits."
            )
        else:
            reason = (
                "no condition readings were found"
                if not readings
                else f"none of the {len(readings)} reading(s) has a matching OEM limit"
                if thresholds
                else f"{len(readings)} reading(s) exist but no OEM limit is stored to compare them to"
            )
            finding = (
                f"No comparison against OEM limits was possible: {reason}. A conclusion about "
                "condition therefore rests on narrative evidence rather than measurement."
            )
            confidence = Confidence(
                value=0.0,
                rationale="No reading/limit pair existed to compare; this step contributes nothing.",
                method="exact",
            )

        return ReasoningStep(
            order=8,
            action="Compared condition readings against OEM thresholds",
            finding=finding,
            confidence=confidence,
            sources=sources,
            cypher=CYPHER_THRESHOLDS,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    # ==================================================================================
    # Step 9 — alarm-bypass coincidence
    # ==================================================================================

    def _step_bypass(
        self, evidence: DiagnosticEvidence, retrieval: RetrievalResult
    ) -> ReasoningStep:
        started = time.perf_counter()
        window_start, window_end = self._anomaly_window(evidence)
        evidence.window_start, evidence.window_end = window_start, window_end

        bypasses: list[BypassEvent] = []

        for record in (*evidence.maintenance, *evidence.inspections):
            text = f"{record.findings} {record.recommendations}".strip()
            if _matches_any(text, BYPASS_PATTERNS):
                bypasses.append(
                    BypassEvent(
                        text=_clip(text),
                        occurred_on=record.performed_on,
                        source=record.sources[0] if record.sources else None,
                        within_window=window_start <= record.performed_on <= window_end,
                    )
                )

        seen_chunks: set[str] = set()
        for passage in (*evidence.shift_logs, *retrieval.passages):
            if passage.chunk.chunk_id in seen_chunks:
                continue
            seen_chunks.add(passage.chunk.chunk_id)
            if not _matches_any(passage.chunk.text, BYPASS_PATTERNS):
                continue
            doc_date = passage.document.document_date
            bypasses.append(
                BypassEvent(
                    text=_extract_matching_sentence(passage.chunk.text, BYPASS_PATTERNS),
                    occurred_on=doc_date,
                    source=passage.as_source(),
                    # An undated document cannot be excluded from the window, so it is kept and
                    # labelled — dropping it would hide the single most incriminating class of
                    # evidence on a technicality.
                    within_window=doc_date is None or window_start <= doc_date <= window_end,
                )
            )

        evidence.bypasses = bypasses
        coincident = [b for b in bypasses if b.within_window]
        sources = [b.source for b in bypasses if b.source is not None]

        if coincident:
            finding = (
                f"{len(coincident)} alarm-bypass or protection-override mention(s) fall inside the "
                f"anomaly window {window_start.isoformat()} to {window_end.isoformat()}: "
                + " ".join(
                    f"[{b.occurred_on.isoformat() if b.occurred_on else 'undated'}] {b.text}"
                    for b in coincident[:MAX_DIGEST_ITEMS]
                )
                + " A bypassed alarm during a developing anomaly removes the protection that would "
                "have stopped the machine, and turns a detectable deviation into a failure."
            )
            dated = [b for b in coincident if b.occurred_on is not None]
            confidence = Confidence(
                value=1.0 if dated else 0.5,
                rationale=(
                    f"{len(dated)} of {len(coincident)} bypass mention(s) carry a date placing them "
                    "inside the window; undated mentions are included but cannot be time-correlated."
                )
                if dated
                else "Every bypass mention found is undated, so coincidence with the window is assumed, not shown.",
                method="heuristic",
            )
        elif bypasses:
            finding = (
                f"{len(bypasses)} alarm-bypass mention(s) exist for {evidence.tag} but all fall "
                f"outside the anomaly window {window_start.isoformat()} to "
                f"{window_end.isoformat()}, so none is contemporaneous with this event."
            )
            confidence = Confidence.exact(
                "Bypass mentions found and date-compared against the anomaly window."
            )
        else:
            finding = (
                f"No alarm bypass, interlock inhibit or protection override is recorded for "
                f"{evidence.tag} in the anomaly window {window_start.isoformat()} to "
                f"{window_end.isoformat()}. The protective systems were not documented as defeated."
            )
            confidence = Confidence(
                value=0.6,
                rationale=(
                    "A keyword search over the available narrative found nothing; bypasses are "
                    "often performed verbally and never written down, so absence here is weak "
                    "evidence of absence."
                ),
                method="heuristic",
            )

        return ReasoningStep(
            order=9,
            action="Checked for alarm-bypass events coinciding with the anomaly window",
            finding=finding,
            confidence=confidence,
            sources=sources,
            cypher=CYPHER_BYPASS,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    # ==================================================================================
    # Step 10 — chain assembly
    # ==================================================================================

    def _step_assemble_chain(
        self, evidence: DiagnosticEvidence, steps: Sequence[ReasoningStep]
    ) -> ReasoningStep:
        started = time.perf_counter()
        links: list[str] = []

        for match in evidence.precursor_matches:
            links.append(
                f"A pattern seen before the {match.failure.occurred_on.isoformat()} "
                f"{match.failure.failure_mode} failure is present again "
                f"({match.similarity:.2f} similarity)"
            )
        for breach in evidence.breaches:
            links.append(
                f"{breach.reading.parameter} at {breach.ratio * 100:.0f}% of its OEM limit on "
                f"{breach.reading.measured_at.date().isoformat()}"
            )
        for bypass in evidence.bypasses:
            if bypass.within_window:
                links.append(
                    "Alarm protection was defeated during the same window"
                    + (f" ({bypass.occurred_on.isoformat()})" if bypass.occurred_on else "")
                )
        if evidence.failures:
            modes = {f.failure_mode for f in evidence.failures}
            links.append(f"This asset has failed before by: {', '.join(sorted(modes))}")

        contributing = [s for s in steps if s.confidence.value > 0.0]
        weakest = min(contributing, key=lambda s: s.confidence.value) if contributing else None

        if links:
            finding = (
                "Causal chain assembled from "
                f"{len(contributing)} contributing step(s): "
                + " → ".join(links)
                + "."
                + (
                    f" The chain is only as strong as step {weakest.order} ({weakest.action}) at "
                    f"{weakest.confidence.value:.2f}."
                    if weakest
                    else ""
                )
            )
        else:
            finding = (
                "No causal chain could be assembled: no precursor match, no threshold exceedance "
                "and no alarm bypass was found. Any cause stated below rests on the retrieved "
                "narrative alone and should be treated as a hypothesis."
            )

        confidence = (
            Confidence.aggregate(
                [s.confidence for s in contributing],
                rationale=(
                    f"Weakest contributing link is step {weakest.order} ({weakest.action}) at "
                    f"{weakest.confidence.value:.2f}: {weakest.confidence.rationale}"
                    if weakest
                    else "No step contributed evidence."
                ),
            )
            if contributing
            else Confidence(
                value=0.0,
                rationale="No step in the chain produced usable evidence.",
                method="aggregate",
            )
        )

        return ReasoningStep(
            order=10,
            action="Assembled the causal chain across all evidence",
            finding=finding,
            confidence=confidence,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    # ==================================================================================
    # Evidence helpers
    # ==================================================================================

    def _current_findings(self, evidence: DiagnosticEvidence) -> list[tuple[str, SourceRef | None]]:
        """The symptom texts that describe the asset's present condition.

        Drawn from maintenance findings, inspection findings and shift-log narrative — the three
        places a technician writes down what they saw. Only texts carrying an anomaly cue are kept:
        matching "routine lubrication completed" against a failure precursor produces noise that
        drowns the real signal.
        """
        out: list[tuple[str, SourceRef | None]] = []
        for record in (*evidence.maintenance, *evidence.inspections):
            for text in (record.findings, record.recommendations):
                cleaned = text.strip()
                if cleaned and _matches_any(cleaned, ANOMALY_PATTERNS):
                    out.append((cleaned, record.sources[0] if record.sources else None))
        for passage in evidence.shift_logs:
            cleaned = passage.chunk.text.strip()
            if cleaned and _matches_any(cleaned, ANOMALY_PATTERNS):
                out.append((_extract_matching_sentence(cleaned, ANOMALY_PATTERNS), passage.as_source()))
        return out

    async def _similarity_matches(
        self,
        findings: Sequence[tuple[str, SourceRef | None]],
        historical: Sequence[FailureEvent],
        threshold: float,
    ) -> tuple[list[PrecursorMatch], str]:
        """Cosine over embeddings, falling back to token overlap when no embedder is reachable."""
        finding_texts = [text for text, _ in findings]
        precursor_texts = [f.precursor_text.strip() for f in historical]

        vectors = await self._embed(finding_texts + precursor_texts)
        if vectors is not None:
            matrix = np.asarray(vectors, dtype=np.float64)
            similarity = await self._cosine_matrix(matrix, len(finding_texts))
            method = "cosine"
        else:
            similarity = await self._lexical_matrix(finding_texts, precursor_texts)
            method = "lexical"

        matches: list[PrecursorMatch] = []
        for i, (text, source) in enumerate(findings):
            for j, failure in enumerate(historical):
                score = float(similarity[i][j])
                if score >= threshold:
                    matches.append(
                        PrecursorMatch(
                            failure=failure,
                            finding_text=text,
                            finding_source=source,
                            similarity=round(score, 4),
                            method=method,
                        )
                    )
        matches.sort(key=lambda m: m.similarity, reverse=True)

        # One line per historical failure: the same event matching five findings is one insight.
        deduped: list[PrecursorMatch] = []
        seen_events: set[str] = set()
        for match in matches:
            if match.failure.event_id in seen_events:
                continue
            seen_events.add(match.failure.event_id)
            deduped.append(match)
        return deduped, method

    async def _embed(self, texts: Sequence[str]) -> list[list[float]] | None:
        """Embed a batch. ``None`` means no provider was reachable — never an exception."""
        if not texts:
            return None
        try:
            vectors = await self.ctx.llm.embed(list(texts), task="document")
        except IndraError as exc:
            logger.warning(
                "embedding unavailable for precursor match; falling back to lexical similarity",
                extra={"error": exc.error_code, "detail": exc.message},
            )
            return None
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(
                "embedding raised an untyped error; falling back to lexical similarity",
                extra={"detail": str(exc)},
                exc_info=True,
            )
            return None

        if len(vectors) != len(texts) or not vectors or not vectors[0]:
            logger.warning(
                "embedding provider returned an unusable batch; falling back to lexical similarity",
                extra={"expected": len(texts), "received": len(vectors)},
            )
            return None
        return vectors

    @staticmethod
    async def _cosine_matrix(matrix: np.ndarray, split: int) -> list[list[float]]:
        """Cosine similarity of the first ``split`` rows against the rest.

        CPU-bound array work, so it runs off the event loop per CLAUDE.md rule 4.
        """

        def compute() -> list[list[float]]:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            unit = matrix / norms
            left, right = unit[:split], unit[split:]
            if left.size == 0 or right.size == 0:
                return [[0.0] * right.shape[0] for _ in range(left.shape[0])]
            # Cosine of two unit vectors is their dot product; clip absorbs float error at ±1.
            return np.clip(left @ right.T, -1.0, 1.0).tolist()

        import asyncio

        return await asyncio.to_thread(compute)

    @staticmethod
    async def _lexical_matrix(left: Sequence[str], right: Sequence[str]) -> list[list[float]]:
        """Token-overlap similarity, scaled to 0-1 to share the configured threshold."""

        def compute() -> list[list[float]]:
            return [
                [fuzz.token_set_ratio(a, b) / 100.0 for b in right]
                for a in left
            ]

        import asyncio

        return await asyncio.to_thread(compute)

    def _anomaly_window(self, evidence: DiagnosticEvidence) -> tuple[date, date]:
        """The period the anomaly developed over.

        Starts at the earliest dated sign of trouble — an anomalous work order, a reading past the
        warning ratio, or a matched precursor — and ends at the failure being diagnosed, or today
        if the asset has not failed. Floored at the maintenance lookback so a decade-old record
        cannot stretch the window until coincidence becomes meaningless.
        """
        today = date.today()
        floor = today - timedelta(days=self.settings.maintenance_lookback_days)

        end = today
        recent_failures = [f for f in evidence.failures if f.occurred_on >= floor]
        if recent_failures:
            end = max(f.occurred_on for f in recent_failures)

        starts: list[date] = []
        for record in (*evidence.maintenance, *evidence.inspections):
            text = f"{record.findings} {record.recommendations}"
            if _matches_any(text, ANOMALY_PATTERNS) and record.performed_on >= floor:
                starts.append(record.performed_on)
        for breach in evidence.breaches:
            measured = breach.reading.measured_at.date()
            if measured >= floor:
                starts.append(measured)
        for passage in evidence.shift_logs:
            doc_date = passage.document.document_date
            if doc_date is not None and doc_date >= floor and _matches_any(
                passage.chunk.text, ANOMALY_PATTERNS
            ):
                starts.append(doc_date)

        start = min(starts) if starts else floor
        if start > end:
            start, end = end, start
        return start, end

    def _pinned_sources(self, evidence: DiagnosticEvidence) -> list[SourceRef]:
        """Structured evidence that must appear on the answer whether or not the model cited it."""
        pinned: list[SourceRef] = []
        for record in (*evidence.maintenance, *evidence.inspections):
            pinned.extend(record.sources)
        for failure in evidence.failures:
            pinned.extend(failure.sources)
        for match in evidence.precursor_matches:
            if match.finding_source is not None:
                pinned.append(match.finding_source)
        for bypass in evidence.bypasses:
            if bypass.source is not None and bypass.within_window:
                pinned.append(bypass.source)
        return pinned

    def _render_digest(self, evidence: DiagnosticEvidence) -> str:
        """Render the structured evidence for the generation prompt."""
        sections: list[tuple[str, list[str]]] = [
            (
                "Equipment",
                [
                    f"{evidence.equipment.tag}: {evidence.equipment.name or evidence.equipment.equipment_type}, "
                    f"criticality {evidence.equipment.criticality.value}"
                    + (f", {evidence.equipment.manufacturer}" if evidence.equipment.manufacturer else "")
                    + (f" {evidence.equipment.model}" if evidence.equipment.model else "")
                ]
                if evidence.equipment
                else [],
            ),
            (
                f"Maintenance, last {self.settings.maintenance_lookback_days} days",
                [
                    f"{r.performed_on.isoformat()} ({r.record_type}"
                    + (f", {r.performed_by}" if r.performed_by else "")
                    + f"): {_clip(r.findings) or 'no findings recorded'}"
                    for r in sorted(evidence.maintenance, key=lambda r: r.performed_on, reverse=True)[
                        :MAX_DIGEST_ITEMS
                    ]
                ],
            ),
            (
                f"Inspections, last {self.settings.inspection_lookback_days} days",
                [
                    f"{r.performed_on.isoformat()}: {_clip(r.findings) or 'no findings recorded'}"
                    + (
                        " | readings: "
                        + ", ".join(f"{rd.parameter}={rd.value:g}{rd.unit}" for rd in r.readings)
                        if r.readings
                        else ""
                    )
                    for r in sorted(evidence.inspections, key=lambda r: r.performed_on, reverse=True)[
                        :MAX_DIGEST_ITEMS
                    ]
                ],
            ),
            (
                "Historical failures, all time",
                [
                    f"{f.occurred_on.isoformat()} {f.failure_mode}"
                    + (f" | root cause: {f.root_cause}" if f.root_cause else "")
                    + (f" | precursor: {_clip(f.precursor_text)}" if f.precursor_text else "")
                    + (f" | downtime {f.downtime_hours:g} h" if f.downtime_hours else "")
                    for f in sorted(evidence.failures, key=lambda f: f.occurred_on, reverse=True)[
                        :MAX_DIGEST_ITEMS
                    ]
                ],
            ),
            (
                f"Shift-floor narrative, last {self.settings.shift_log_lookback_days} days",
                [
                    (
                        f"{p.document.document_date.isoformat()}: "
                        if p.document.document_date
                        else ""
                    )
                    + _clip(p.chunk.text)
                    for p in evidence.shift_logs[:MAX_DIGEST_ITEMS]
                ],
            ),
            (
                "OEM limits",
                [
                    f"{name} limit {limit:g}"
                    for name, limit in sorted(
                        (evidence.equipment.oem_thresholds if evidence.equipment else {}).items()
                    )
                ],
            ),
            (
                "PRECURSOR MATCHES (cross-document — present in no single source)",
                [m.narrative for m in evidence.precursor_matches],
            ),
            (
                "Threshold comparison",
                [b.narrative for b in evidence.breaches[:MAX_DIGEST_ITEMS]],
            ),
            (
                "Alarm bypass within the anomaly window "
                + (
                    f"({evidence.window_start.isoformat()} to {evidence.window_end.isoformat()})"
                    if evidence.window_start and evidence.window_end
                    else ""
                ),
                [
                    f"{b.occurred_on.isoformat() if b.occurred_on else 'undated'}: {b.text}"
                    for b in evidence.bypasses
                    if b.within_window
                ],
            ),
        ]

        rendered: list[str] = []
        for title, items in sections:
            body = (
                "\n".join(prompts.render(prompts.DIGEST_BULLET_V1, text=item) for item in items)
                if items
                else prompts.DIGEST_EMPTY_V1
            )
            rendered.append(prompts.render(prompts.DIGEST_SECTION_V1, title=title, body=body))
        return "\n".join(rendered)

    def _deterministic_actions(self, evidence: DiagnosticEvidence) -> list[RecommendedAction]:
        """Actions implied directly by the evidence, independent of any model.

        These exist so that a diagnosis produced with every provider down still tells the
        technician what to do next.
        """
        actions: list[RecommendedAction] = []
        tag = evidence.tag or "the asset"
        criticality = evidence.equipment.criticality if evidence.equipment else None
        urgent = Severity.CRITICAL if criticality and criticality.value == "A" else Severity.HIGH

        for breach in evidence.breaches:
            if breach.exceeded:
                actions.append(
                    RecommendedAction(
                        action=(
                            f"Take {tag} out of service and inspect for "
                            f"{breach.reading.parameter.replace('_', ' ')} damage — the last reading "
                            f"({breach.reading.value:g}{breach.reading.unit}) is past the OEM limit "
                            f"of {breach.limit:g}."
                        ),
                        urgency=urgent,
                        owner_role="Maintenance Supervisor",
                        due_within_days=0,
                        rationale=breach.narrative,
                    )
                )
            else:
                actions.append(
                    RecommendedAction(
                        action=(
                            f"Re-measure {breach.reading.parameter.replace('_', ' ')} on {tag} and "
                            f"trend it against the OEM limit of {breach.limit:g}."
                        ),
                        urgency=Severity.WARNING,
                        owner_role="Condition Monitoring Technician",
                        due_within_days=self.settings.shift_log_lookback_days,
                        rationale=breach.narrative,
                    )
                )

        for match in evidence.precursor_matches:
            remedy = match.failure.root_cause or match.failure.failure_mode
            actions.append(
                RecommendedAction(
                    action=(
                        f"Inspect {tag} for the failure mode '{match.failure.failure_mode}' that "
                        f"followed these same symptoms on {match.failure.occurred_on.isoformat()}; "
                        f"check specifically for {remedy}."
                    ),
                    urgency=urgent,
                    owner_role="Reliability Engineer",
                    due_within_days=0,
                    rationale=match.narrative,
                )
            )

        for bypass in evidence.bypasses:
            if not bypass.within_window:
                continue
            actions.append(
                RecommendedAction(
                    action=(
                        f"Restore and function-test the bypassed alarm or interlock on {tag}, and "
                        "raise a deviation record for the period it was defeated."
                    ),
                    urgency=Severity.CRITICAL,
                    owner_role="Shift Supervisor",
                    due_within_days=0,
                    rationale=(
                        f"Protection was defeated during the anomaly window: {bypass.text}"
                    ),
                )
            )

        if not evidence.maintenance and not evidence.inspections and evidence.equipment is not None:
            actions.append(
                RecommendedAction(
                    action=(
                        f"Carry out a baseline condition survey on {tag} and file the results — "
                        f"there is no maintenance or inspection record in the last "
                        f"{self.settings.inspection_lookback_days} days to diagnose from."
                    ),
                    urgency=Severity.WARNING,
                    owner_role="Reliability Engineer",
                    due_within_days=self.settings.shift_log_lookback_days,
                    rationale="Diagnosis is currently unsupported by any measured condition data.",
                )
            )

        return actions[:MAX_RECOMMENDED_ACTIONS]

    # ==================================================================================
    # Graph access, each wrapped
    # ==================================================================================

    async def _get_equipment(self, tag: str) -> Equipment | None:
        try:
            return await self.ctx.graph.get_equipment(tag)
        except IndraError as exc:
            logger.warning(
                "equipment lookup failed", extra={"tag": tag, "error": exc.error_code, "detail": exc.message}
            )
            return None
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("equipment lookup raised an untyped error", extra={"tag": tag, "detail": str(exc)})
            return None

    async def _maintenance_history(self, tag: str, *, since: date) -> list[MaintenanceRecord]:
        if not tag:
            return []
        try:
            return await self.ctx.graph.maintenance_history(tag, since=since)
        except IndraError as exc:
            logger.warning(
                "maintenance history unavailable",
                extra={"tag": tag, "error": exc.error_code, "detail": exc.message},
            )
            return []
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(
                "maintenance history raised an untyped error", extra={"tag": tag, "detail": str(exc)}
            )
            return []

    async def _failure_history(self, tag: str) -> list[FailureEvent]:
        if not tag:
            return []
        try:
            return await self.ctx.graph.failure_history(tag, since=None)
        except IndraError as exc:
            logger.warning(
                "failure history unavailable",
                extra={"tag": tag, "error": exc.error_code, "detail": exc.message},
            )
            return []
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(
                "failure history raised an untyped error", extra={"tag": tag, "detail": str(exc)}
            )
            return []


# --------------------------------------------------------------------------------------
# Module helpers
# --------------------------------------------------------------------------------------


def _matches_any(text: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    return any(pattern.search(text) for pattern in patterns) if text else False


def _extract_matching_sentence(text: str, patterns: Sequence[re.Pattern[str]]) -> str:
    """Return the sentence that triggered the match, so evidence quotes stay tight."""
    for sentence in re.split(r"(?<=[.!?;])\s+|\n+", text):
        if _matches_any(sentence, patterns):
            return _clip(sentence.strip())
    return _clip(text)


#: Characters kept when quoting evidence into a finding. Long enough to carry the observation,
#: short enough that a reasoning chain stays readable on a phone.
_QUOTE_LIMIT: Final[int] = 220


def _clip(text: str, limit: int = _QUOTE_LIMIT) -> str:
    cleaned = " ".join(text.split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1].rstrip() + "…"


def _sources_of(records: Sequence[MaintenanceRecord]) -> list[SourceRef]:
    out: list[SourceRef] = []
    for record in records:
        out.extend(record.sources)
    return out


def _match_threshold(parameter: str, thresholds: Mapping[str, float]) -> float | None:
    """Match a reading's parameter to an OEM limit, tolerating naming drift.

    ``bearing_wear_pct`` in the manual and ``bearing wear`` on the work order are the same quantity.
    Exact match first, then a normalised match, then containment — never a fuzzy guess, because
    comparing a temperature against a vibration limit produces a confident wrong answer.
    """
    if parameter in thresholds:
        return thresholds[parameter]

    def normalise(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", name.lower())

    target = normalise(parameter)
    for name, limit in thresholds.items():
        if normalise(name) == target:
            return limit
    for name, limit in thresholds.items():
        candidate = normalise(name)
        if candidate and (candidate in target or target in candidate):
            return limit
    return None


__all__ = [
    "ANOMALY_PATTERNS",
    "BYPASS_PATTERNS",
    "BypassEvent",
    "DiagnosticEvidence",
    "DiagnosticHandler",
    "PrecursorMatch",
    "ThresholdBreach",
]
