"""Risk composition and confidence assembly for compound signals.

Two numbers leave this module and both have to survive an argument with a plant manager:

``risk_score``
    How worried to be. Composed from the rule's own weight, the strength of the constituent
    signals combined with a noisy-OR (so a *conjunction* of weak signals outranks one strong one,
    which is the entire thesis of compound-signal detection), and the asset's criticality.

``confidence``
    How much to trust the risk number. Driven by *provenance*: a signal read out of a structured
    work order is worth more than one fuzzy-matched out of OCR'd handwriting, and
    :meth:`indra.core.models.Confidence.aggregate` weights the weakest link most heavily on
    purpose.

Both are published as a decomposed breakdown, never as a bare float. A score nobody can interrogate
is a score nobody acts on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping, Sequence

from indra.core.config import Settings
from indra.core.logging import get_logger
from indra.core.models import (
    Alert,
    CompoundSignal,
    Confidence,
    Criticality,
    Equipment,
    RecommendedAction,
    Severity,
    Signal,
    SourceRef,
)
from indra.agents.proactive_intelligence_agent.rules import RuleMatch
from indra.agents.proactive_intelligence_agent.signals import (
    KIND_ALARM_BYPASS,
    KIND_DOCUMENTATION_VOID,
    KIND_EVIDENCE_VOID,
    KIND_EXPERTISE_LOSS,
    KIND_FLEET_PATTERN,
    KIND_MAINTENANCE_ANOMALY,
    KIND_MISSING_WORKORDER,
    KIND_PRECURSOR_MATCH,
    KIND_REGULATORY_EXPOSURE,
    KIND_THRESHOLD_APPROACH,
    dedupe_sources,
)

logger = get_logger(__name__)


# ======================================================================================
# Coefficients
# ======================================================================================

#: How much the asset's consequence class scales composed risk. A criticality-A asset stops
#: production or hurts somebody when it fails, so the same evidence justifies more alarm.
#: (``Settings`` has no field for this; ``indra/core`` is read-only for this agent.)
CRITICALITY_MULTIPLIER: Final[Mapping[Criticality, float]] = {
    Criticality.A: 1.00,
    Criticality.B: 0.88,
    Criticality.C: 0.76,
}

#: Floor per severity. A CRITICAL rule that fired must never render as "12% risk" on the operator's
#: screen — the rule firing at all is itself evidence, independent of the signal strengths.
SEVERITY_FLOOR: Final[Mapping[Severity, float]] = {
    Severity.CRITICAL: 0.60,
    Severity.HIGH: 0.45,
    Severity.WARNING: 0.30,
    Severity.LOW: 0.15,
    Severity.INFO: 0.05,
}

#: Trust in a signal before provenance is considered, keyed by how it was derived.
METHOD_BASE_CONFIDENCE: Final[Mapping[str, float]] = {
    "exact": 0.95,      # read directly out of a structured record
    "heuristic": 0.78,  # pattern match over prose written by a human
    "semantic": 0.72,   # fuzzy similarity — right often enough to act on, wrong often enough to say so
}

#: Applied when a signal has no citable document. Structured-record signals (a retirement date, an
#: absent work order) legitimately have none, so this is a haircut rather than a rejection.
NO_SOURCE_PENALTY: Final[float] = 0.85

#: Confidence method reported on the aggregate.
_AGGREGATE_METHOD: Final[str] = "aggregate"


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


# ======================================================================================
# Risk composition
# ======================================================================================


@dataclass(frozen=True, slots=True)
class RiskComponent:
    """One named term of the risk arithmetic, with the number it contributed."""

    key: str
    label: str
    value: float
    detail: str


@dataclass(frozen=True, slots=True)
class RiskBreakdown:
    """The composed risk score plus every term that produced it."""

    score: float
    components: tuple[RiskComponent, ...]

    @property
    def percent(self) -> float:
        return round(self.score * 100.0, 1)

    def narrative(self) -> str:
        """One line an operator or auditor can check the arithmetic against."""
        parts = " × ".join(f"{c.label} {c.value:.2f}" for c in self.components)
        return f"Risk {self.percent:.0f}% = {parts}"

    def as_dict(self) -> dict[str, float]:
        return {component.key: round(component.value, 4) for component in self.components}


def conjunction_strength(signals: Sequence[Signal]) -> float:
    """Combine signal strengths with a noisy-OR.

    ``1 - Π(1 - sᵢ)`` rather than a mean, because the whole premise is that independent weak
    observations *reinforce*. Three signals at 0.5 give 0.875, where a mean would give 0.5 and
    quietly discard the conjunction that made the finding worth raising.
    """
    if not signals:
        return 0.0
    product = 1.0
    for signal in signals:
        product *= 1.0 - _clamp(float(signal.strength))
    return _clamp(1.0 - product)


def compose_risk(match: RuleMatch, equipment: Equipment) -> RiskBreakdown:
    """Compose the 0–1 risk score for a fired rule.

    ``risk = max(severity_floor, rule_weight × noisy_or(signal strengths) × criticality)``
    """
    rule = match.rule
    conjunction = conjunction_strength(match.signals)
    criticality = CRITICALITY_MULTIPLIER.get(equipment.criticality, CRITICALITY_MULTIPLIER[Criticality.C])
    floor = SEVERITY_FLOOR.get(rule.severity, 0.0)
    raw = rule.risk_weight * conjunction * criticality
    score = _clamp(max(floor, raw))

    components = (
        RiskComponent(
            key="rule_weight",
            label="rule weight",
            value=rule.risk_weight,
            detail=f"{rule.name} is weighted {rule.risk_weight:.2f} of the maximum",
        ),
        RiskComponent(
            key="signal_conjunction",
            label="signal conjunction",
            value=conjunction,
            detail=(
                f"noisy-OR of {len(match.signals)} signal(s) at strengths "
                + ", ".join(f"{s.strength:.2f}" for s in match.signals)
            ),
        ),
        RiskComponent(
            key="criticality",
            label=f"criticality {equipment.criticality.value}",
            value=criticality,
            detail=f"criticality-{equipment.criticality.value} consequence multiplier",
        ),
        RiskComponent(
            key="severity_floor",
            label="severity floor",
            value=floor,
            detail=f"a fired {rule.severity.value} rule never scores below {floor:.2f}",
        ),
    )
    logger.debug(
        "risk composed",
        extra={
            "equipment_tag": equipment.tag,
            "rule_id": rule.rule_id,
            "risk_score": round(score, 4),
            "conjunction": round(conjunction, 4),
        },
    )
    return RiskBreakdown(score=score, components=components)


# ======================================================================================
# Confidence assembly
# ======================================================================================


def signal_confidence(signal: Signal) -> Confidence:
    """Confidence in one signal, from how it was derived and what it cites."""
    method = str(signal.data.get("method", "heuristic"))
    base = METHOD_BASE_CONFIDENCE.get(method, METHOD_BASE_CONFIDENCE["heuristic"])
    if signal.sources:
        provenance = sum(s.extraction_confidence for s in signal.sources) / len(signal.sources)
        rationale = (
            f"{signal.kind} derived by {method} matching, cited to {len(signal.sources)} "
            f"document(s) at mean extraction confidence {provenance:.2f}"
        )
    else:
        provenance = NO_SOURCE_PENALTY
        rationale = (
            f"{signal.kind} derived by {method} reasoning over structured plant records with no "
            f"citable document passage"
        )
    method_literal = method if method in {"exact", "heuristic", "semantic"} else "heuristic"
    return Confidence(
        value=_clamp(base * provenance),
        rationale=rationale,
        method=method_literal,  # type: ignore[arg-type]
    )


def assemble_confidence(match: RuleMatch) -> Confidence:
    """Aggregate per-signal confidence into the compound signal's confidence.

    Uses :meth:`Confidence.aggregate`, which is weighted 70% towards the weakest link: a compound
    signal that hangs on one fuzzy OCR read is only as good as that read, no matter how solid the
    other three signals are.
    """
    parts = [signal_confidence(signal) for signal in match.signals]
    if not parts:
        return Confidence(
            value=0.0,
            rationale=f"{match.rule.rule_id} fired with no constituent signals",
            method="aggregate",
        )
    weakest = min(parts, key=lambda c: c.value)
    aggregate = Confidence.aggregate(parts)
    return Confidence(
        value=aggregate.value,
        rationale=(
            f"{len(parts)} signal(s) combined; weakest link is {weakest.rationale} "
            f"({weakest.value:.2f})"
        ),
        method=_AGGREGATE_METHOD,  # type: ignore[arg-type]
    )


# ======================================================================================
# Compound signal assembly
# ======================================================================================


def build_compound_signal(match: RuleMatch, equipment: Equipment) -> tuple[CompoundSignal, RiskBreakdown]:
    """Turn a fired rule into the :class:`CompoundSignal` the API and the alert both carry."""
    breakdown = compose_risk(match, equipment)
    confidence = assemble_confidence(match)
    compound = CompoundSignal(
        rule_id=match.rule.rule_id,
        rule_name=match.rule.name,
        equipment_tag=equipment.tag,
        severity=match.rule.severity,
        signals=list(match.signals),
        explanation=f"{match.explanation}\n\n{breakdown.narrative()}",
        confidence=confidence,
        risk_score=breakdown.score,
    )
    return compound, breakdown


# ======================================================================================
# Recommended actions
# ======================================================================================

#: Per-rule remediation. Vague advice is not actionable on a plant floor, so every entry names an
#: owner role and a deadline; the runtime fills in the asset-specific numbers.
_ACTION_TEMPLATES: Final[Mapping[str, tuple[tuple[str, Severity, str, int], ...]]] = {
    "maintenance_precursor_match": (
        ("Take vibration and temperature readings on {tag} this shift and compare them against the "
         "readings recorded before the previous failure", Severity.CRITICAL, "Reliability Engineer", 1),
        ("Raise a priority work order for {tag} and plan a controlled shutdown rather than waiting "
         "for an uncontrolled one", Severity.CRITICAL, "Maintenance Planner", 3),
        ("Confirm the spare is on site before the intervention window", Severity.HIGH, "Stores", 3),
    ),
    "threshold_without_workorder": (
        ("Raise a work order against {tag} for the parameter approaching its OEM limit",
         Severity.WARNING, "Maintenance Planner", 7),
        ("Increase the monitoring frequency on {tag} until the reading is back inside limits",
         Severity.WARNING, "Operations Supervisor", 2),
    ),
    "bypass_with_anomaly": (
        ("Restore the {tag} alarm to service now and record who authorised the bypass",
         Severity.CRITICAL, "Shift Supervisor", 0),
        ("Raise a management-of-change record for the bypass and review it at the next safety meeting",
         Severity.HIGH, "HSE Lead", 3),
        ("Inspect {tag} against the anomalies logged while the alarm was out of service",
         Severity.CRITICAL, "Maintenance Technician", 1),
    ),
    "fleet_pattern": (
        ("Run the same inspection on every asset of this type, not just {tag}",
         Severity.HIGH, "Reliability Engineer", 14),
        ("Review the shared root cause — duty, lubricant, spares batch or installation practice — "
         "rather than repairing each asset in isolation", Severity.HIGH, "Reliability Engineer", 30),
    ),
    "expertise_loss": (
        ("Book a structured knowledge-capture interview with {expert} before their last working day",
         Severity.HIGH, "Maintenance Manager", 14),
        ("Write and file an SOP for {tag} from that interview and attach it to the asset record",
         Severity.HIGH, "Reliability Engineer", 30),
        ("Pair a junior engineer with {expert} on the next {tag} intervention",
         Severity.WARNING, "Maintenance Manager", 30),
    ),
    "regulatory_exposure": (
        ("Locate or produce the evidence record for this obligation on {tag} and file it against "
         "the asset", Severity.CRITICAL, "Compliance Officer", 3),
        ("If the work was never done, schedule it and notify the statutory authority before the "
         "deadline rather than after", Severity.CRITICAL, "Plant Manager", 7),
    ),
}


def recommended_actions(match: RuleMatch, equipment: Equipment) -> list[RecommendedAction]:
    """Concrete next steps for a fired rule, with owners and deadlines."""
    templates = _ACTION_TEMPLATES.get(match.rule.rule_id, ())
    expert = next(
        (
            str(signal.data.get("person_name"))
            for signal in match.signals
            if signal.kind == KIND_EXPERTISE_LOSS and signal.data.get("person_name")
        ),
        "the retiring expert",
    )
    actions: list[RecommendedAction] = []
    for text, urgency, owner, due_days in templates:
        actions.append(
            RecommendedAction(
                action=text.format(tag=equipment.tag, expert=expert),
                urgency=urgency,
                owner_role=owner,
                due_within_days=due_days,
                rationale=f"Triggered by {match.rule.name} on {equipment.tag}",
            )
        )
    if not actions:
        actions.append(
            RecommendedAction(
                action=f"Review {equipment.tag} against the evidence attached to this alert",
                urgency=match.rule.severity,
                owner_role="Reliability Engineer",
                due_within_days=7,
                rationale=f"No action template registered for rule {match.rule.rule_id}",
            )
        )
    return actions


# ======================================================================================
# Alert assembly
# ======================================================================================

#: Signal kinds ordered for the headline. The most consequential finding leads the alert title.
_HEADLINE_ORDER: Final[tuple[str, ...]] = (
    KIND_PRECURSOR_MATCH,
    KIND_ALARM_BYPASS,
    KIND_REGULATORY_EXPOSURE,
    KIND_EVIDENCE_VOID,
    KIND_THRESHOLD_APPROACH,
    KIND_FLEET_PATTERN,
    KIND_EXPERTISE_LOSS,
    KIND_DOCUMENTATION_VOID,
    KIND_MAINTENANCE_ANOMALY,
    KIND_MISSING_WORKORDER,
)


def build_alert(
    compound: CompoundSignal,
    match: RuleMatch,
    equipment: Equipment,
    *,
    settings: Settings,
) -> Alert:
    """Assemble the operator-facing alert for a compound signal.

    ``dedupe_key`` is ``tag:rule_id:severity`` (the model's own default), which is what
    ``MetadataStore.find_alert_by_dedupe_key`` suppresses on inside
    ``settings.alert_dedupe_window_s``. An operator who sees the same alert six times stops reading
    alerts, so the key deliberately ignores signal ids and timestamps: the same rule on the same
    asset at the same severity is the same problem.
    """
    sources = dedupe_sources(
        [*match.sources, *(s for signal in match.signals for s in signal.sources)],
        limit=8,
    )
    title = f"{equipment.tag}: {match.rule.name}"
    body = (
        f"{compound.explanation}\n\n"
        f"Confidence {compound.confidence.value:.2f} ({compound.confidence.band}) — "
        f"{compound.confidence.rationale}"
    )
    alert = Alert(
        title=title,
        equipment_tag=equipment.tag,
        severity=compound.severity,
        body=body,
        compound_signal=compound,
        recommended_actions=recommended_actions(match, equipment),
        sources=sources,
        risk_percent=round(compound.risk_score * 100.0, 1),
        dedupe_key=f"{equipment.tag}:{match.rule.rule_id}:{compound.severity.value}",
    )
    logger.debug(
        "alert assembled",
        extra={
            "equipment_tag": equipment.tag,
            "rule_id": match.rule.rule_id,
            "risk_percent": alert.risk_percent,
            "dedupe_key": alert.dedupe_key,
            "dedupe_window_s": settings.alert_dedupe_window_s,
        },
    )
    return alert


def headline_signal(signals: Sequence[Signal]) -> Signal | None:
    """The signal that should lead a summary, by consequence rather than by strength."""
    for kind in _HEADLINE_ORDER:
        for signal in signals:
            if signal.kind == kind:
                return signal
    return signals[0] if signals else None


def collect_sources(signals: Sequence[Signal], *, limit: int = 8) -> list[SourceRef]:
    """Every citation across a set of signals, deduplicated and strongest first."""
    return dedupe_sources((source for signal in signals for source in signal.sources), limit=limit)


__all__ = [
    "CRITICALITY_MULTIPLIER",
    "METHOD_BASE_CONFIDENCE",
    "NO_SOURCE_PENALTY",
    "RiskBreakdown",
    "RiskComponent",
    "SEVERITY_FLOOR",
    "assemble_confidence",
    "build_alert",
    "build_compound_signal",
    "collect_sources",
    "compose_risk",
    "conjunction_strength",
    "headline_signal",
    "recommended_actions",
    "signal_confidence",
]
