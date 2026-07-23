"""The compound-signal rule engine — declarative, not a pile of ``if`` statements (D7).

Every rule in the charter table is a :class:`SignalRule` value in the :data:`RULES` tuple: an id, a
name, the signal kinds it needs, a pure predicate, a severity, a risk weight, an explanation
template, and an evidence builder. Adding a rule is adding a list entry; nothing else in the agent
changes. That is the point — ``docs/DECISIONS.md`` D7 exists because an auditor cannot be shown a
conditional, only a rule that carries its own evidence and rationale.

Two properties this module is designed to guarantee:

* **Purity.** :func:`evaluate_rules` takes a :class:`RuleContext` and returns matches. No clock, no
  store, no network. Given fixture graph state, exactly which rules fire is deterministic — and on a
  clean asset the answer is *none*.
* **Readable explanations.** The rendered ``explanation`` names the dates, the numbers and the
  documents. A shift supervisor reading it should be able to act without opening the system, and an
  auditor reading it six months later should be able to reconstruct the reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Final, Mapping, Sequence

from indra.core.config import Settings
from indra.core.logging import get_logger
from indra.core.models import Equipment, Severity, Signal, SourceRef, utcnow
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
    format_date,
)

logger = get_logger(__name__)

#: Ceiling on citations attached to one compound signal. More than this and the alert card becomes
#: a reading exercise instead of a decision aid.
MAX_EVIDENCE_PER_RULE: Final[int] = 6

_UNKNOWN: Final[str] = "not recorded"


# ======================================================================================
# Rule context
# ======================================================================================


@dataclass(frozen=True, slots=True)
class RuleContext:
    """Everything a rule may look at. Deliberately small — signals plus the asset plus settings."""

    equipment: Equipment
    signals: tuple[Signal, ...]
    settings: Settings

    @property
    def tag(self) -> str:
        return self.equipment.tag

    def of(self, *kinds: str) -> tuple[Signal, ...]:
        """Signals of the given kinds, strongest first (ties broken by id for determinism)."""
        wanted = set(kinds)
        return tuple(
            sorted(
                (s for s in self.signals if s.kind in wanted),
                key=lambda s: (-s.strength, s.signal_id),
            )
        )

    def strongest(self, kind: str) -> Signal | None:
        """The single most convincing signal of ``kind``, or ``None``."""
        found = self.of(kind)
        return found[0] if found else None

    def has(self, *kinds: str) -> bool:
        """True when at least one signal of every listed kind is present."""
        present = {s.kind for s in self.signals}
        return all(kind in present for kind in kinds)


# ======================================================================================
# Rule definition
# ======================================================================================

Predicate = Callable[[RuleContext], bool]
EvidenceBuilder = Callable[[RuleContext], list[SourceRef]]


@dataclass(frozen=True, slots=True)
class SignalRule:
    """One compound-signal rule.

    Args:
        rule_id: Stable identifier. Part of ``Alert.dedupe_key``, so renaming one re-raises alerts.
        name: Human-readable name for the alert title.
        required_kinds: Signal kinds that must all be present before ``predicate`` is even called.
            This is the *conjunction* the charter insists on: a single signal is noise.
        predicate: Pure test over the context. Called only when ``required_kinds`` are satisfied.
        severity: Escalation level when the rule fires.
        risk_weight: Ceiling this rule contributes to the composed risk score (see :mod:`.scoring`).
        explanation_template: ``str.format_map`` template rendered with :func:`build_template_vars`.
        evidence: Builds the citation list attached to the compound signal.
    """

    rule_id: str
    name: str
    required_kinds: tuple[str, ...]
    predicate: Predicate
    severity: Severity
    risk_weight: float
    explanation_template: str
    evidence: EvidenceBuilder

    def matches(self, context: RuleContext) -> bool:
        """True when every required kind is present *and* the predicate holds."""
        if not context.has(*self.required_kinds):
            return False
        try:
            return bool(self.predicate(context))
        except Exception as exc:  # noqa: BLE001 - a broken rule must not take the scan down
            logger.error(
                "rule predicate raised; treating as not-fired",
                extra={"rule_id": self.rule_id, "equipment_tag": context.tag, "error": str(exc)},
            )
            return False

    def matched_signals(self, context: RuleContext) -> tuple[Signal, ...]:
        """The constituent signals, strongest first."""
        return context.of(*self.required_kinds)

    def explain(self, context: RuleContext) -> str:
        """Render the explanation. Never raises: a missing variable degrades to a marker string."""
        variables = build_template_vars(context)
        try:
            return self.explanation_template.format_map(_SafeVars(variables)).strip()
        except Exception as exc:  # noqa: BLE001 - formatting must never break an alert
            logger.error(
                "explanation template failed to render",
                extra={"rule_id": self.rule_id, "equipment_tag": context.tag, "error": str(exc)},
            )
            return (
                f"{self.name} fired on {context.tag} from "
                f"{len(self.matched_signals(context))} signals; explanation template failed to render."
            )


@dataclass(frozen=True, slots=True)
class RuleMatch:
    """A fired rule together with its evidence, ready for :mod:`.scoring`."""

    rule: SignalRule
    signals: tuple[Signal, ...]
    explanation: str
    sources: tuple[SourceRef, ...]


class _SafeVars(dict[str, str]):
    """``format_map`` backing store that yields a marker instead of raising ``KeyError``."""

    def __missing__(self, key: str) -> str:  # pragma: no cover - defensive
        logger.debug("explanation template referenced an unknown variable", extra={"variable": key})
        return _UNKNOWN


# ======================================================================================
# Evidence builders
# ======================================================================================


def evidence_from(*kinds: str, limit: int = MAX_EVIDENCE_PER_RULE) -> EvidenceBuilder:
    """Build an evidence builder that harvests citations from signals of ``kinds``."""

    def _build(context: RuleContext) -> list[SourceRef]:
        return dedupe_sources(
            (source for signal in context.of(*kinds) for source in signal.sources),
            limit=limit,
        )

    return _build


def evidence_from_all(limit: int = MAX_EVIDENCE_PER_RULE) -> EvidenceBuilder:
    """Every citation attached to any signal on the asset."""

    def _build(context: RuleContext) -> list[SourceRef]:
        return dedupe_sources(
            (source for signal in context.signals for source in signal.sources),
            limit=limit,
        )

    return _build


# ======================================================================================
# Template variables
# ======================================================================================


def _data(signal: Signal | None, key: str, default: object = None) -> Any:
    """Read a key out of a signal's ``data`` payload without a chain of ``if signal is not None``."""
    if signal is None:
        return default
    return signal.data.get(key, default)


def _citations(sources: Sequence[SourceRef]) -> str:
    """Render citations the way the alert card shows them: ``WO_2024_0342 p.3; Shift log 14 Jun``."""
    if not sources:
        return "no document could be cited — this finding comes from structured plant records"
    return "; ".join(dict.fromkeys(source.citation for source in sources))


def _join(items: Sequence[str], *, conjunction: str = "and") -> str:
    """Oxford-free human list: ``A``, ``A and B``, ``A, B and C``."""
    cleaned = [i for i in items if i]
    if not cleaned:
        return _UNKNOWN
    if len(cleaned) == 1:
        return cleaned[0]
    return f"{', '.join(cleaned[:-1])} {conjunction} {cleaned[-1]}"


def build_template_vars(context: RuleContext) -> dict[str, str]:
    """Assemble every variable any explanation template may reference.

    One shared builder rather than a per-rule closure keeps :class:`SignalRule` to exactly the
    fields the charter specifies, and means a new rule can reuse an existing phrase without
    re-deriving it.
    """
    equipment = context.equipment
    anomalies = context.of(KIND_MAINTENANCE_ANOMALY)
    precursor = context.strongest(KIND_PRECURSOR_MATCH)
    threshold = context.strongest(KIND_THRESHOLD_APPROACH)
    workorder = context.strongest(KIND_MISSING_WORKORDER)
    bypasses = context.of(KIND_ALARM_BYPASS)
    fleet = context.strongest(KIND_FLEET_PATTERN)
    experts = context.of(KIND_EXPERTISE_LOSS)
    void = context.strongest(KIND_DOCUMENTATION_VOID)
    exposure = context.strongest(KIND_REGULATORY_EXPOSURE)
    evidence_void = context.strongest(KIND_EVIDENCE_VOID)

    variables: dict[str, str] = {
        "tag": equipment.tag,
        "equipment_name": equipment.name or equipment.equipment_type or "unnamed asset",
        "equipment_type": equipment.equipment_type or "asset",
        "criticality": equipment.criticality.value,
        "manufacturer": equipment.manufacturer or _UNKNOWN,
        "model": equipment.model or _UNKNOWN,
        "location": equipment.location or "an unrecorded location",
        "signal_count": str(len(context.signals)),
    }

    # -- maintenance anomalies ------------------------------------------------------------
    if anomalies:
        phrases: list[str] = []
        for signal in anomalies[:3]:
            phrases.append(signal.description)
        variables["anomaly_summary"] = _join(phrases) + "."
        variables["anomaly_summary_lower"] = (_join(phrases) + ".")[:1].lower() + (_join(phrases) + ".")[1:]
        variables["anomaly_count"] = str(len(anomalies))
        variables["anomaly_terms"] = _join(
            sorted({str(_data(s, "term")) for s in anomalies if _data(s, "term")})
        )
        variables["anomaly_last_date"] = format_date(_parse_iso(_data(anomalies[0], "last_seen")))
        variables["anomaly_quote"] = str(_data(anomalies[0], "quote", "")) or anomalies[0].description
        variables["anomaly_detail"] = anomalies[0].description + "."
    else:
        variables.update(
            anomaly_summary="No maintenance anomaly was recorded.",
            anomaly_summary_lower="no maintenance anomaly was recorded.",
            anomaly_count="0",
            anomaly_terms=_UNKNOWN,
            anomaly_last_date=_UNKNOWN,
            anomaly_quote=_UNKNOWN,
            anomaly_detail="",
        )

    # -- precursor match ------------------------------------------------------------------
    if precursor is not None:
        similarity_pct = float(_data(precursor, "similarity", 0.0)) * 100.0
        cost = _data(precursor, "cost_inr")
        downtime = _data(precursor, "downtime_hours")
        consequence_parts: list[str] = []
        if downtime:
            consequence_parts.append(f"{float(downtime):.0f} hours of lost production")
        if cost:
            consequence_parts.append(f"₹{float(cost):,.0f}")
        variables.update(
            precursor_mode=str(_data(precursor, "failure_mode", "failure")),
            precursor_date=format_date(_parse_iso(_data(precursor, "failure_date"))),
            precursor_finding_date=format_date(_parse_iso(_data(precursor, "finding_date"))),
            precursor_similarity_pct=f"{similarity_pct:.0f}",
            precursor_quote=str(_data(precursor, "precursor_quote", "")) or _UNKNOWN,
            precursor_root_cause=str(_data(precursor, "root_cause") or "never established"),
            precursor_consequence=(
                f", and that failure cost {_join(consequence_parts)}" if consequence_parts else ""
            ),
        )
    else:
        variables.update(
            precursor_mode=_UNKNOWN, precursor_date=_UNKNOWN, precursor_finding_date=_UNKNOWN,
            precursor_similarity_pct="0", precursor_quote=_UNKNOWN,
            precursor_root_cause=_UNKNOWN, precursor_consequence="",
        )

    # -- threshold ------------------------------------------------------------------------
    if threshold is not None:
        unit = str(_data(threshold, "unit", "") or "")
        ratio = float(_data(threshold, "ratio", 0.0))
        days_to_limit = _data(threshold, "days_to_limit")
        slope = float(_data(threshold, "slope_per_day", 0.0) or 0.0)
        variables.update(
            threshold_parameter=str(_data(threshold, "parameter", "reading")).replace("_", " "),
            threshold_value=f"{float(_data(threshold, 'value', 0.0)):g}",
            threshold_limit=f"{float(_data(threshold, 'limit', 0.0)):g}",
            threshold_unit=unit,
            threshold_pct=f"{ratio * 100:.0f}",
            threshold_headroom=f"{float(_data(threshold, 'headroom', 0.0)):g}{unit}",
            threshold_date=format_date(_parse_iso(_data(threshold, "measured_at"))),
            threshold_samples=str(_data(threshold, "sample_count", 0)),
            threshold_trend=(
                f"rising {slope:+.3g}{unit}/day" if abs(slope) > 1e-9 else "flat"
            ),
            threshold_projection=(
                f", and at the current rate of change it reaches the limit in about "
                f"{float(days_to_limit):.0f} days"
                if days_to_limit is not None else ""
            ),
        )
    else:
        variables.update(
            threshold_parameter=_UNKNOWN, threshold_value=_UNKNOWN, threshold_limit=_UNKNOWN,
            threshold_unit="", threshold_pct="0", threshold_headroom=_UNKNOWN,
            threshold_date=_UNKNOWN, threshold_samples="0", threshold_trend=_UNKNOWN,
            threshold_projection="",
        )

    # -- missing work order ---------------------------------------------------------------
    last_maintenance = _parse_iso(_data(workorder, "last_maintenance_date"))
    days_since = _data(workorder, "days_since_maintenance")
    variables.update(
        last_maintenance_date=format_date(last_maintenance, fallback="never"),
        days_since_maintenance=(f"{float(days_since):.0f}" if days_since is not None else _UNKNOWN),
        maintenance_window_days=str(_data(workorder, "window_days", context.settings.maintenance_lookback_days)),
        days_overdue=(
            f"{float(_data(workorder, 'days_overdue')):.0f}"
            if _data(workorder, "days_overdue") is not None else "0"
        ),
    )

    # -- alarm bypass ---------------------------------------------------------------------
    if bypasses:
        occurrences = sum(int(_data(s, "occurrences", 0) or 0) for s in bypasses)
        terms = sorted({t for s in bypasses for t in (_data(s, "terms") or [])})
        titles = [str(_data(s, "document_title", "an unnamed log")) for s in bypasses]
        dates = [format_date(_parse_iso(_data(s, "document_date"))) for s in bypasses]
        variables.update(
            bypass_count=str(occurrences),
            bypass_terms=_join([f"'{t}'" for t in terms], conjunction="/"),
            bypass_verb=_join(terms, conjunction="and then"),
            bypass_documents=_join(list(dict.fromkeys(titles))),
            bypass_dates=_join(list(dict.fromkeys(dates))),
            bypass_quote=str(_data(bypasses[0], "quote", "")) or bypasses[0].description,
        )
    else:
        variables.update(
            bypass_count="0", bypass_terms=_UNKNOWN, bypass_verb=_UNKNOWN,
            bypass_documents=_UNKNOWN, bypass_dates=_UNKNOWN, bypass_quote=_UNKNOWN,
        )

    # -- fleet ----------------------------------------------------------------------------
    if fleet is not None:
        tags = [str(t) for t in (_data(fleet, "affected_tags") or [])]
        downtime = float(_data(fleet, "total_downtime_hours", 0.0) or 0.0)
        cost = float(_data(fleet, "total_cost_inr", 0.0) or 0.0)
        variables.update(
            fleet_mode=str(_data(fleet, "failure_mode", "the same failure")),
            fleet_asset_count=str(_data(fleet, "affected_count", len(tags))),
            fleet_event_count=str(_data(fleet, "event_count", len(tags))),
            fleet_tags=_join(tags),
            fleet_window_days=f"{float(_data(fleet, 'window_days', 0.0)):.0f}",
            fleet_last_date=format_date(_parse_iso(_data(fleet, "latest_event_date"))),
            fleet_tag_latest=str(_data(fleet, "latest_event_tag", _UNKNOWN)),
            fleet_downtime=f"{downtime:.0f}",
            fleet_cost_clause=(f" and ₹{cost:,.0f}" if cost else ""),
            fleet_self_clause=(
                f"{context.tag} has already been hit once."
                if _data(fleet, "this_asset_affected") else
                f"{context.tag} is the same make and duty and has not failed yet."
            ),
        )
    else:
        variables.update(
            fleet_mode=_UNKNOWN, fleet_asset_count="0", fleet_event_count="0", fleet_tags=_UNKNOWN,
            fleet_window_days="0", fleet_last_date=_UNKNOWN, fleet_tag_latest=_UNKNOWN,
            fleet_downtime="0", fleet_cost_clause="", fleet_self_clause="",
        )

    # -- expertise ------------------------------------------------------------------------
    if experts:
        names = [str(_data(s, "person_name", "an expert")) for s in experts]
        first = experts[0]
        years = _data(first, "years_experience")
        variables.update(
            expert_names=_join(names),
            expert_first_name=str(names[0]).split(" ")[0],
            expert_count=str(len(experts)),
            expert_role=str(_data(first, "role") or "plant expert"),
            expert_retirement_date=format_date(_parse_iso(_data(first, "retirement_date"))),
            days_to_retirement=f"{float(_data(first, 'days_to_retirement', 0.0)):.0f}",
            expert_years_clause=(f" after {float(years):.0f} years on this plant" if years else ""),
            expert_contributions=str(_data(first, "documented_contributions", 0)),
        )
    else:
        variables.update(
            expert_names=_UNKNOWN, expert_first_name=_UNKNOWN, expert_count="0",
            expert_role=_UNKNOWN, expert_retirement_date=_UNKNOWN, days_to_retirement=_UNKNOWN,
            expert_years_clause="", expert_contributions="0",
        )

    variables["document_count"] = str(_data(void, "document_count", 0))
    variables["knowledge_documents"] = str(_data(void, "knowledge_documents", 0))
    variables["document_types_present"] = _join([str(t) for t in (_data(void, "document_types_present") or [])])

    # -- regulatory -----------------------------------------------------------------------
    if exposure is not None:
        remaining = _data(exposure, "days_to_deadline")
        overdue = bool(_data(exposure, "overdue", False))
        deadline = format_date(_parse_iso(_data(exposure, "deadline")), fallback="an unpublished date")
        if remaining is None:
            deadline_clause = "carries no published deadline in the parsed regulation"
        elif overdue:
            deadline_clause = f"was due on {deadline} and is {abs(float(remaining)):.0f} days overdue"
        else:
            deadline_clause = f"falls due on {deadline}, {float(remaining):.0f} days from now"
        variables.update(
            regulation=str(_data(exposure, "regulation", _UNKNOWN)),
            clause=str(_data(exposure, "clause", _UNKNOWN)),
            deadline=deadline,
            days_to_deadline=(f"{float(remaining):.0f}" if remaining is not None else _UNKNOWN),
            deadline_clause=deadline_clause,
            gap_status=str(_data(exposure, "status", _UNKNOWN)),
        )
    else:
        variables.update(
            regulation=_UNKNOWN, clause=_UNKNOWN, deadline=_UNKNOWN, days_to_deadline=_UNKNOWN,
            deadline_clause=_UNKNOWN, gap_status=_UNKNOWN,
        )
    variables["evidence_void_detail"] = (
        evidence_void.description if evidence_void is not None
        else "no evidence record was located"
    )

    # -- citations ------------------------------------------------------------------------
    all_sources = dedupe_sources(
        (source for signal in context.signals for source in signal.sources),
        limit=MAX_EVIDENCE_PER_RULE,
    )
    variables["evidence_citations"] = _citations(all_sources)
    variables["evidence_count"] = str(len(all_sources))
    return variables


def _parse_iso(value: object) -> Any:
    """Best-effort ISO date/datetime parse for template rendering. Returns ``None`` on failure."""
    from datetime import date as _date, datetime as _datetime

    if isinstance(value, (_date, _datetime)):
        return value
    if isinstance(value, str) and value.strip():
        text = value.strip()
        try:
            return _datetime.fromisoformat(text)
        except ValueError:
            try:
                return _date.fromisoformat(text[:10])
            except ValueError:
                return None
    return None


# ======================================================================================
# Predicates
# ======================================================================================


def _precursor_predicate(context: RuleContext) -> bool:
    """A precursor match above the configured similarity threshold, alongside a live anomaly."""
    threshold = context.settings.precursor_similarity_threshold
    return any(
        float(signal.data.get("similarity", 0.0)) >= threshold
        for signal in context.of(KIND_PRECURSOR_MATCH)
    )


def _threshold_without_workorder_predicate(context: RuleContext) -> bool:
    """A reading at or above the OEM warning ratio, with nothing open or scheduled."""
    ratio_floor = context.settings.oem_threshold_warning_ratio
    near_limit = any(
        float(signal.data.get("ratio", 0.0)) >= ratio_floor
        for signal in context.of(KIND_THRESHOLD_APPROACH)
    )
    nothing_scheduled = any(
        int(signal.data.get("open_records", 0)) == 0
        for signal in context.of(KIND_MISSING_WORKORDER)
    )
    return near_limit and nothing_scheduled


def _bypass_with_anomaly_predicate(context: RuleContext) -> bool:
    """At least one recorded bypass, and at least one live maintenance anomaly.

    No extra numeric gate: the conjunction *is* the finding. Somebody removed a protective layer
    from an asset that was already telling us something was wrong.
    """
    return bool(context.of(KIND_ALARM_BYPASS)) and bool(context.of(KIND_MAINTENANCE_ANOMALY))


def _fleet_predicate(context: RuleContext) -> bool:
    """The mode has appeared on at least ``fleet_failure_min_count`` similar assets."""
    minimum = context.settings.fleet_failure_min_count
    return any(
        int(signal.data.get("affected_count", 0)) >= minimum
        for signal in context.of(KIND_FLEET_PATTERN)
    )


def _expertise_predicate(context: RuleContext) -> bool:
    """A retiring expert inside the horizon, and literally zero transferable documentation."""
    retiring = any(
        float(signal.data.get("days_to_retirement", 1e9)) <= context.settings.retirement_horizon_days
        for signal in context.of(KIND_EXPERTISE_LOSS)
    )
    void = any(
        int(signal.data.get("knowledge_documents", 1)) == 0
        for signal in context.of(KIND_DOCUMENTATION_VOID)
    )
    return retiring and void


def _regulatory_predicate(context: RuleContext) -> bool:
    """A near or passed deadline **on the same obligation** that has no evidence behind it.

    Matching on ``gap_id`` matters: a deadline approaching on one clause and missing evidence on an
    unrelated clause is two separate problems, not a compound signal.
    """
    exposed = {
        str(signal.data.get("gap_id"))
        for signal in context.of(KIND_REGULATORY_EXPOSURE)
        if signal.data.get("gap_id")
    }
    unevidenced = {
        str(signal.data.get("gap_id"))
        for signal in context.of(KIND_EVIDENCE_VOID)
        if signal.data.get("gap_id")
    }
    return bool(exposed & unevidenced)


# ======================================================================================
# The rules
# ======================================================================================

RULES: Final[tuple[SignalRule, ...]] = (
    SignalRule(
        rule_id="maintenance_precursor_match",
        name="Current findings match a historical failure precursor",
        required_kinds=(KIND_MAINTENANCE_ANOMALY, KIND_PRECURSOR_MATCH),
        predicate=_precursor_predicate,
        severity=Severity.CRITICAL,
        risk_weight=0.95,
        explanation_template=(
            "{tag} ({equipment_name}) is showing today the same symptoms that preceded its "
            "{precursor_mode} on {precursor_date}. The {precursor_finding_date} maintenance record "
            'reads: "{anomaly_quote}". The symptoms logged before that failure read: '
            '"{precursor_quote}" — a {precursor_similarity_pct}% match{precursor_consequence}. '
            "{anomaly_summary} The root cause established at the time was {precursor_root_cause}. "
            "This is a criticality-{criticality} {equipment_type} in {location}. "
            "Evidence: {evidence_citations}."
        ),
        evidence=evidence_from(KIND_PRECURSOR_MATCH, KIND_MAINTENANCE_ANOMALY),
    ),
    SignalRule(
        rule_id="threshold_without_workorder",
        name="Reading near the OEM limit with no work order raised",
        required_kinds=(KIND_THRESHOLD_APPROACH, KIND_MISSING_WORKORDER),
        predicate=_threshold_without_workorder_predicate,
        severity=Severity.WARNING,
        risk_weight=0.65,
        explanation_template=(
            "{tag} {threshold_parameter} measured {threshold_value}{threshold_unit} on "
            "{threshold_date} — {threshold_pct}% of the OEM limit of "
            "{threshold_limit}{threshold_unit}, leaving {threshold_headroom} of headroom "
            "(trend: {threshold_trend} over {threshold_samples} readings){threshold_projection}. "
            "There is no open or scheduled work order against this asset: the last completed job "
            "was {last_maintenance_date}, {days_since_maintenance} days ago, against a "
            "{maintenance_window_days}-day maintenance window. Nobody is currently booked to look "
            "at it. Evidence: {evidence_citations}."
        ),
        evidence=evidence_from(KIND_THRESHOLD_APPROACH, KIND_MISSING_WORKORDER),
    ),
    SignalRule(
        rule_id="bypass_with_anomaly",
        name="Alarm bypassed while maintenance anomalies were open",
        required_kinds=(KIND_ALARM_BYPASS, KIND_MAINTENANCE_ANOMALY),
        predicate=_bypass_with_anomaly_predicate,
        severity=Severity.CRITICAL,
        risk_weight=0.92,
        explanation_template=(
            "The {tag} alarm was {bypass_terms} {bypass_count} time(s), recorded in "
            '{bypass_documents} on {bypass_dates}: "{bypass_quote}". At the same time, '
            "{anomaly_summary_lower} In other words the protection was taken off an asset that was "
            "already misbehaving, on a criticality-{criticality} {equipment_type} in {location}. "
            "Evidence: {evidence_citations}."
        ),
        evidence=evidence_from(KIND_ALARM_BYPASS, KIND_MAINTENANCE_ANOMALY),
    ),
    SignalRule(
        rule_id="fleet_pattern",
        name="Same failure mode repeating across similar assets",
        required_kinds=(KIND_FLEET_PATTERN,),
        predicate=_fleet_predicate,
        severity=Severity.HIGH,
        risk_weight=0.75,
        explanation_template=(
            "'{fleet_mode}' has now occurred on {fleet_asset_count} of this plant's "
            "{equipment_type} assets ({fleet_tags}) over the last {fleet_window_days} days — "
            "{fleet_event_count} events in total, most recently {fleet_tag_latest} on "
            "{fleet_last_date}. {fleet_self_clause} Across the fleet this mode has cost "
            "{fleet_downtime} hours of downtime{fleet_cost_clause}. A single failure is bad luck; "
            "{fleet_asset_count} of the same type is a design, duty or spares problem that will "
            "reach {tag} as well. Evidence: {evidence_citations}."
        ),
        evidence=evidence_from(KIND_FLEET_PATTERN),
    ),
    SignalRule(
        rule_id="expertise_loss",
        name="Retiring expert with no documented knowledge",
        required_kinds=(KIND_EXPERTISE_LOSS, KIND_DOCUMENTATION_VOID),
        predicate=_expertise_predicate,
        severity=Severity.HIGH,
        risk_weight=0.70,
        explanation_template=(
            "{expert_names} ({expert_role}) retires on {expert_retirement_date} — "
            "{days_to_retirement} days from now{expert_years_clause} — and is the plant's recorded "
            "expert on {tag}, a criticality-{criticality} {equipment_type} in {location}. "
            "There is no SOP, no root-cause analysis, no OEM manual and no inspection report for "
            "{tag} anywhere in the system: {document_count} document(s) mention the tag and none of "
            "them explain how to run it, tune it or fix it. When {expert_first_name} walks out of "
            "the gate, that knowledge goes too. Book a capture session before the last working day."
        ),
        evidence=evidence_from_all(),
    ),
    SignalRule(
        rule_id="regulatory_exposure",
        name="Compliance deadline approaching with no evidence on file",
        required_kinds=(KIND_REGULATORY_EXPOSURE, KIND_EVIDENCE_VOID),
        predicate=_regulatory_predicate,
        severity=Severity.CRITICAL,
        risk_weight=0.88,
        explanation_template=(
            "{regulation} {clause} applies to {tag} and {deadline_clause}. The compliance scan "
            "returned status '{gap_status}': {evidence_void_detail}. This is a "
            "criticality-{criticality} {equipment_type} in {location}, so an inspector arriving "
            "tomorrow would find a statutory obligation with nothing on file to demonstrate it was "
            "met. Produce or locate the evidence record before the deadline; if the work was done "
            "and simply never filed, that is a five-minute fix today and a notice of violation "
            "later. Evidence: {evidence_citations}."
        ),
        evidence=evidence_from_all(),
    ),
)

RULES_BY_ID: Final[Mapping[str, SignalRule]] = {rule.rule_id: rule for rule in RULES}


# ======================================================================================
# Engine
# ======================================================================================


def evaluate_rules(
    context: RuleContext,
    *,
    rules: Sequence[SignalRule] = RULES,
) -> list[RuleMatch]:
    """Evaluate every rule against ``context``.

    Pure: no I/O, no clock, no randomness. The returned order is severity-descending then rule id,
    so two runs over the same fixture produce the same list.
    """
    matches: list[RuleMatch] = []
    for rule in rules:
        if not rule.matches(context):
            continue
        signals = rule.matched_signals(context)
        try:
            sources = tuple(rule.evidence(context))
        except Exception as exc:  # noqa: BLE001 - evidence builder must not break the scan
            logger.error(
                "evidence builder raised; alert will carry no citations",
                extra={"rule_id": rule.rule_id, "equipment_tag": context.tag, "error": str(exc)},
            )
            sources = ()
        matches.append(
            RuleMatch(
                rule=rule,
                signals=signals,
                explanation=rule.explain(context),
                sources=sources,
            )
        )
    matches.sort(key=lambda m: (-m.rule.severity.rank, m.rule.rule_id))
    if matches:
        logger.info(
            "compound signals matched",
            extra={
                "equipment_tag": context.tag,
                "rules_fired": [m.rule.rule_id for m in matches],
                "signal_count": len(context.signals),
            },
        )
    return matches


def make_context(
    equipment: Equipment,
    signals: Sequence[Signal],
    *,
    settings: Settings,
) -> RuleContext:
    """Build a :class:`RuleContext`. Signals are frozen into a tuple in a stable order."""
    ordered = tuple(sorted(signals, key=lambda s: (s.kind, -s.strength, s.signal_id)))
    return RuleContext(equipment=equipment, signals=ordered, settings=settings)


def rule_ids() -> tuple[str, ...]:
    """Every rule id, in declaration order. Used by the API to expose the rule catalogue."""
    return tuple(rule.rule_id for rule in RULES)


def describe_rules() -> list[dict[str, Any]]:
    """Serialisable rule catalogue for ``/system`` and for audit evidence."""
    return [
        {
            "rule_id": rule.rule_id,
            "name": rule.name,
            "required_signal_kinds": list(rule.required_kinds),
            "severity": rule.severity.value,
            "risk_weight": rule.risk_weight,
            "explanation_template": rule.explanation_template,
            "described_at": utcnow().isoformat(),
        }
        for rule in RULES
    ]


__all__ = [
    "EvidenceBuilder",
    "MAX_EVIDENCE_PER_RULE",
    "Predicate",
    "RULES",
    "RULES_BY_ID",
    "RuleContext",
    "RuleMatch",
    "SignalRule",
    "build_template_vars",
    "describe_rules",
    "evaluate_rules",
    "evidence_from",
    "evidence_from_all",
    "make_context",
    "rule_ids",
]
