"""Deterministic, rule-based compliance gap detection.

**No language model is reachable from this module.** A compliance finding has to survive an
inspector asking "how do you know?", which means the same corpus and the same date must always
produce the same finding, with the evidence attached. The LLM's only role in this agent is turning
regulation *prose* into structure (:mod:`.parser`); the decision about whether an obligation is met
is arithmetic over dated evidence.

Everything here is pure: the caller performs the I/O, builds an :class:`EvidenceSnapshot`, and hands
it in. That is what makes the whole decision layer unit-testable against fixture graph state.

The status ladder, in the order it is resolved:

``MISSING``
    Nothing of the required evidence type exists for the asset.
``OUTDATED``
    The most recent acceptable evidence is older than ``frequency_days + grace_days``, or it cites a
    revision the requirement has superseded.
``INCOMPLETE``
    Evidence exists inside the window but does not record a field the obligation requires — the
    inspection happened, the report cannot prove what it found.
``COMPLIANT``
    Right evidence type, inside the window, every required field present, current revision.

``COMPLIANT`` is never reached without at least one :class:`SourceRef`. Absence of evidence is a
gap, not a pass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Final, Iterable, Literal, Mapping, Sequence

from indra.core.logging import get_logger
from indra.core.models import (
    ComplianceGap,
    ComplianceMatrixRow,
    Confidence,
    Criticality,
    DocumentMeta,
    DocumentType,
    Equipment,
    GapStatus,
    MaintenanceRecord,
    Procedure,
    RecommendedAction,
    Severity,
    SourceRef,
)

from indra.agents.compliance_agent.requirements import (
    RequirementCatalogue,
    RequirementSpec,
    normalise_field_key,
)

logger = get_logger(__name__)

EvidenceOrigin = Literal["maintenance_record", "procedure", "document", "retrieval"]

# --------------------------------------------------------------------------------------
# Implementation constants
#
# Product tunables live in ``indra.core.config``. What follows are rule tables and structural
# limits: changing them changes the *rule*, which is a code review, not a config change.
# --------------------------------------------------------------------------------------

#: Severity ladder, indexed by rank. Mirrors ``Severity.rank``.
_SEVERITY_BY_RANK: Final[tuple[Severity, ...]] = (
    Severity.INFO, Severity.LOW, Severity.WARNING, Severity.HIGH, Severity.CRITICAL,
)

#: Severity escalation by asset criticality. Class A equipment stops production or endangers life,
#: so an identical breach on class A outranks the same breach on class C.
_CRITICALITY_ESCALATION: Final[Mapping[Criticality, int]] = {
    Criticality.A: 1,
    Criticality.B: 0,
    Criticality.C: 0,
}

#: Severity adjustment by resolved status. An incomplete record is a lesser finding than a missing
#: one: the work was done, the paperwork does not prove it.
_STATUS_ESCALATION: Final[Mapping[GapStatus, int]] = {
    GapStatus.MISSING: 0,
    GapStatus.OUTDATED: 0,
    GapStatus.INCOMPLETE: -1,
    GapStatus.COMPLIANT: 0,
}

#: How many citations to attach to one finding. Enough to be defensible, few enough to read.
_MAX_EVIDENCE_REFS: Final[int] = 5

#: How many near-miss documents (right asset, wrong evidence type) to cite on a MISSING finding.
_MAX_RELATED_REFS: Final[int] = 3

#: Maintenance record types mapped to the document type they constitute as evidence.
_RECORD_TYPE_EVIDENCE: Final[Mapping[str, DocumentType]] = {
    "inspection": DocumentType.INSPECTION_REPORT,
    "calibration": DocumentType.INSPECTION_REPORT,
    "preventive": DocumentType.WORK_ORDER,
    "work_order": DocumentType.WORK_ORDER,
}

#: Synonyms accepted for a required evidence field. Industrial records are written by people, not
#: schemas: a "competent person" and an "inspector" are the same field on a Form 13.
_FIELD_ALIASES: Final[Mapping[str, frozenset[str]]] = {
    "inspector": frozenset({
        "performed_by", "inspected_by", "competent_person", "examiner", "surveyor",
        "engineer", "inspecting_authority", "checked_by", "carried_out_by",
    }),
    "surveyor": frozenset({"performed_by", "surveyed_by", "inspector", "inspected_by", "carried_out_by"}),
    "technician": frozenset({"performed_by", "tested_by", "calibrated_by", "attended_by", "carried_out_by"}),
    "tested_by": frozenset({"performed_by", "technician", "tester", "test_engineer", "carried_out_by"}),
    "examining_authority": frozenset({"certifying_surgeon", "medical_officer", "examined_by", "performed_by"}),
    "findings": frozenset({"observation", "observations", "result", "results", "remarks", "condition", "outcome"}),
    "certificate_number": frozenset({"certificate", "certificate_no", "cert_no", "form_no", "form_number", "report_no"}),
    "thickness_mm": frozenset({"thickness", "wall_thickness", "shell_thickness", "ut_thickness", "min_thickness"}),
    "test_pressure": frozenset({"hydrotest_pressure", "hydraulic_test_pressure", "pressure_bar", "test_pressure_bar"}),
    "safety_valve_setting": frozenset({"psv_set_pressure", "set_pressure", "relief_valve_setting", "safety_valve", "psv_setting"}),
    "stroke_time": frozenset({"closure_time", "closing_time", "travel_time", "valve_stroke_time"}),
    "discharge_pressure": frozenset({"delivery_pressure", "outlet_pressure", "pump_discharge_pressure"}),
    "residual_pressure": frozenset({"remote_end_pressure", "hydrant_pressure", "terminal_pressure"}),
    "resistance_ohm": frozenset({"earth_resistance", "earthing_resistance", "megger_value", "resistance"}),
    "calibration_gas": frozenset({"span_gas", "test_gas", "cal_gas", "certified_gas"}),
    "span_reading": frozenset({"span_value", "post_calibration_reading", "span_check", "span"}),
    "safe_working_load": frozenset({"swl", "rated_capacity", "rated_load", "wll"}),
    "approved_by": frozenset({"authorised_by", "authorized_by", "signed_by", "countersigned_by", "sanctioned_by"}),
    "revision": frozenset({"rev", "rev_no", "revision_no", "edition", "issue"}),
    "validity": frozenset({"valid_upto", "valid_until", "expiry", "expiry_date", "renewal_date"}),
    "licence_number": frozenset({"license_number", "licence_no", "license_no", "peso_licence"}),
    "consent_number": frozenset({"consent_no", "cto_number", "cto_no", "consent_order"}),
    "laboratory": frozenset({"lab", "testing_laboratory", "nabl_lab", "analysed_by", "analyzed_by"}),
    "sampling_location": frozenset({"location", "station", "sample_point", "monitoring_location"}),
    "station": frozenset({"sampling_location", "monitoring_station", "location", "aaqm_station"}),
    "acknowledgement": frozenset({"ack_no", "acknowledgment", "receipt_no", "submission_id"}),
    "manifest_number": frozenset({"manifest_no", "form_10_no", "manifest"}),
    "air_quantity": frozenset({"air_flow", "quantity_of_air", "airflow_m3_s", "volume_flow"}),
    "methane_percent": frozenset({"methane", "ch4", "ch4_percent", "gas_percentage"}),
    "broken_wires": frozenset({"wire_breaks", "broken_wire_count", "wires_broken"}),
    "loss_of_metallic_area": frozenset({"lma", "metallic_area_loss", "area_loss"}),
    "stopping_distance": frozenset({"brake_distance", "braking_distance", "stop_distance"}),
    "dust_concentration": frozenset({"respirable_dust", "dust_level", "dust_mg_m3"}),
    "water_table": frozenset({"water_level", "ground_water_level", "piezometric_level"}),
    "analyser_drift": frozenset({"zero_drift", "span_drift", "drift"}),
    "reference_method": frozenset({"rata", "reference_analyser", "manual_method"}),
    "lightning_protection": frozenset({"lightning_arrestor", "lightning_conductor", "lps"}),
    "quantity": frozenset({"qty", "weight", "tonnage", "volume"}),
    "tsdf": frozenset({"disposal_facility", "treatment_storage_disposal_facility", "cthwtsdf"}),
    "period": frozenset({"reporting_period", "half_year", "period_covered"}),
    "submitted_to": frozenset({"addressed_to", "authority", "recipient"}),
    "zone": frozenset({"hazardous_area_zone", "area_classification", "zone_classification"}),
    "gradient": frozenset({"slope", "test_gradient", "ramp_gradient"}),
    "test_station": frozenset({"testing_station", "test_agency", "testing_agency"}),
}

#: Key/value scraping over free-text findings. Industrial reports are semi-structured; this pulls
#: "Shell thickness: 11.8 mm" into the field key ``shell_thickness``.
_KV_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?P<key>[A-Za-z][A-Za-z0-9 _/\.\-]{1,40}?)\s*[:=]\s*(?P<value>[^;,\n\r|]{1,120})"
)

#: Revision citation inside evidence text, e.g. "as per SOP-14 Rev. 2".
_REVISION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(?:rev|revision|edition|issue|amendment)\.?\s*[:\-]?\s*([0-9]{1,3}[A-Za-z]?|[A-Z])\b",
    re.IGNORECASE,
)


# ======================================================================================
# Evidence
# ======================================================================================


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    """One dated artefact that may discharge an obligation.

    Constructed from graph state by the pure ``evidence_from_*`` helpers below. ``fields`` is the
    set of normalised field keys the artefact demonstrably records — that set is what decides
    ``INCOMPLETE`` versus ``COMPLIANT``.
    """

    tag: str
    document_type: DocumentType
    occurred_on: date | None
    source: SourceRef
    fields: frozenset[str] = frozenset()
    revision: str | None = None
    origin: EvidenceOrigin = "document"
    label: str = ""

    @property
    def is_dated(self) -> bool:
        return self.occurred_on is not None


@dataclass(frozen=True, slots=True)
class EvidenceSnapshot:
    """Everything known about the plant's evidence at one instant.

    ``degraded_sources`` names any evidence source that failed to answer during collection. It is
    carried all the way into :class:`Confidence` so a finding produced while the graph was
    unreachable never claims the same authority as one produced with the full corpus.
    """

    as_of: date
    items: Mapping[str, tuple[EvidenceItem, ...]] = field(default_factory=dict)
    degraded_sources: tuple[str, ...] = ()

    def for_tag(self, tag: str) -> tuple[EvidenceItem, ...]:
        return self.items.get(tag.strip().upper(), ())

    @property
    def total_items(self) -> int:
        return sum(len(v) for v in self.items.values())


def _present_fields(*texts: str, extra: Iterable[str] = ()) -> frozenset[str]:
    """Normalised field keys demonstrably recorded in ``texts``."""
    keys: set[str] = {normalise_field_key(item) for item in extra if item and item.strip()}
    for text in texts:
        if not text:
            continue
        for match in _KV_PATTERN.finditer(text):
            key = normalise_field_key(match.group("key"))
            if key and match.group("value").strip():
                keys.add(key)
    return frozenset(k for k in keys if k)


def _detect_revision(*texts: str) -> str | None:
    """First revision citation found in ``texts``, normalised to its identifier."""
    for text in texts:
        if not text:
            continue
        match = _REVISION_PATTERN.search(text)
        if match:
            return match.group(1).strip().upper()
    return None


def evidence_from_maintenance(record: MaintenanceRecord) -> EvidenceItem:
    """Project a maintenance or inspection record into evidence. Pure."""
    declared_types = {src.document_type for src in record.sources if src.document_type is not DocumentType.UNKNOWN}
    document_type = (
        sorted(declared_types, key=lambda d: d.value)[0]
        if declared_types
        else _RECORD_TYPE_EVIDENCE.get(record.record_type, DocumentType.WORK_ORDER)
    )
    reading_keys = [reading.parameter for reading in record.readings]
    structural: list[str] = ["date"]
    if record.performed_by:
        structural.append("performed_by")
    if record.findings.strip():
        structural.append("findings")
    if record.recommendations.strip():
        structural.append("recommendations")
    if record.sources:
        structural.append("source_document")
    structural.append(record.record_type)

    source = record.sources[0] if record.sources else SourceRef(
        document_id=f"record:{record.record_id}",
        document_title=f"{record.record_type.replace('_', ' ').title()} {record.record_id}",
        document_type=document_type,
        snippet=(record.findings or record.recommendations)[:600],
        relevance=1.0,
        retrieved_via="direct",
        document_date=record.performed_on,
    )
    return EvidenceItem(
        tag=record.equipment_tag.strip().upper(),
        document_type=document_type,
        occurred_on=record.performed_on,
        source=source,
        fields=_present_fields(record.findings, record.recommendations, extra=structural + reading_keys),
        revision=_detect_revision(record.findings, record.recommendations),
        origin="maintenance_record",
        label=record.record_type,
    )


def evidence_from_procedure(procedure: Procedure, *, tag: str) -> EvidenceItem:
    """Project an SOP into evidence for obligations discharged by a written procedure. Pure."""
    source = procedure.sources[0] if procedure.sources else SourceRef(
        document_id=f"procedure:{procedure.procedure_id}",
        document_title=procedure.title,
        document_type=DocumentType.SOP,
        snippet=" ".join(procedure.steps)[:600],
        relevance=1.0,
        retrieved_via="direct",
    )
    structural = ["procedure", "steps" if procedure.steps else "", "safety_notes" if procedure.safety_notes else ""]
    if procedure.revision:
        structural.append("revision")
    return EvidenceItem(
        tag=tag.strip().upper(),
        document_type=DocumentType.SOP,
        occurred_on=source.document_date,
        source=source,
        fields=_present_fields(procedure.title, " ".join(procedure.safety_notes), extra=structural),
        revision=(procedure.revision or _detect_revision(procedure.title)),
        origin="procedure",
        label="procedure",
    )


def evidence_from_document(meta: DocumentMeta, *, tag: str, snippet: str = "") -> EvidenceItem:
    """Project a whole document into evidence. Pure.

    A bare document reference is weaker than a maintenance record: it proves an artefact of the
    right type exists on the right date, but not that it recorded any particular field. That is why
    such evidence typically resolves to ``INCOMPLETE`` rather than ``COMPLIANT`` — which is the
    honest answer.
    """
    source = SourceRef(
        document_id=meta.document_id,
        document_title=meta.title,
        document_type=meta.document_type,
        snippet=(snippet or meta.title)[:600],
        relevance=0.6,
        extraction_confidence=1.0,
        retrieved_via="direct",
        document_date=meta.document_date,
    )
    structural = ["document"]
    if meta.document_date:
        structural.append("date")
    return EvidenceItem(
        tag=tag.strip().upper(),
        document_type=meta.document_type,
        occurred_on=meta.document_date,
        source=source,
        fields=_present_fields(meta.title, snippet, extra=structural + list(meta.tags)),
        revision=_detect_revision(meta.title, snippet),
        origin="document",
        label=meta.document_type.value,
    )


def build_snapshot(
    as_of: date,
    items: Iterable[EvidenceItem],
    *,
    degraded_sources: Sequence[str] = (),
) -> EvidenceSnapshot:
    """Group evidence by asset tag into an immutable snapshot. Pure."""
    grouped: dict[str, list[EvidenceItem]] = {}
    for item in items:
        grouped.setdefault(item.tag, []).append(item)
    frozen = {
        tag: tuple(sorted(bucket, key=lambda i: (i.occurred_on or date.min, i.source.document_id), reverse=True))
        for tag, bucket in grouped.items()
    }
    return EvidenceSnapshot(as_of=as_of, items=frozen, degraded_sources=tuple(degraded_sources))


# ======================================================================================
# Assessment
# ======================================================================================


@dataclass(frozen=True, slots=True)
class RequirementAssessment:
    """The resolved state of one requirement against one asset.

    Every field is derived arithmetically from the snapshot; nothing here is inferred.
    """

    spec: RequirementSpec
    equipment: Equipment
    status: GapStatus
    severity: Severity
    detail: str
    evidence: tuple[SourceRef, ...] = ()
    last_evidence_date: date | None = None
    deadline: date | None = None
    days_overdue: int | None = None
    penalty_risk: str | None = None
    recommended_action: RecommendedAction | None = None
    missing_fields: tuple[str, ...] = ()
    confidence: Confidence = field(default_factory=lambda: Confidence.exact("Deterministic rule check"))

    @property
    def is_gap(self) -> bool:
        return self.status is not GapStatus.COMPLIANT

    def to_gap(self) -> ComplianceGap | None:
        """Project into the shared :class:`ComplianceGap`. ``None`` when compliant."""
        if not self.is_gap:
            return None
        return ComplianceGap(
            gap_id=gap_id_for(self.spec.requirement_id, self.equipment.tag),
            requirement=self.spec.requirement,
            equipment_tag=self.equipment.tag,
            status=self.status,
            severity=self.severity,
            detail=self.detail,
            last_evidence_date=self.last_evidence_date,
            days_overdue=self.days_overdue,
            deadline=self.deadline,
            penalty_risk=self.penalty_risk,
            recommended_action=self.recommended_action,
            evidence=list(self.evidence),
            confidence=self.confidence,
        )

    def to_row(self) -> ComplianceMatrixRow:
        """Project into one row of the audit matrix."""
        return ComplianceMatrixRow(
            requirement_id=self.spec.requirement_id,
            regulation=self.spec.regulation,
            clause=self.spec.clause,
            obligation=self.spec.obligation,
            equipment_tag=self.equipment.tag,
            status=self.status,
            evidence=list(self.evidence),
            note=self.detail,
        )


def gap_id_for(requirement_id: str, tag: str) -> str:
    """Stable gap id, so the same unmet obligation keeps one identity across scans."""
    from indra.core.ids import content_id  # local import keeps the module's import surface minimal

    return content_id(f"{requirement_id}|{tag.strip().upper()}", kind="entity")


def _escalate(base: Severity, steps: int) -> Severity:
    rank = max(0, min(len(_SEVERITY_BY_RANK) - 1, base.rank + steps))
    return _SEVERITY_BY_RANK[rank]


def _resolve_severity(
    spec: RequirementSpec,
    equipment: Equipment,
    status: GapStatus,
    *,
    days_overdue: int | None,
) -> Severity:
    """Severity from a rule table, not a judgement call.

    Base severity is declared by the requirement. It is escalated for class A assets and again once
    a whole inspection interval has been missed, and de-escalated for an incomplete record.
    """
    if status is GapStatus.COMPLIANT:
        return Severity.INFO
    steps = _STATUS_ESCALATION.get(status, 0) + _CRITICALITY_ESCALATION.get(equipment.criticality, 0)
    frequency = spec.frequency_days
    if days_overdue is not None and frequency and days_overdue >= frequency:
        steps += 1
    return _escalate(spec.breach_severity, steps)


def _missing_fields(spec: RequirementSpec, item: EvidenceItem) -> tuple[str, ...]:
    """Required evidence fields this artefact does not demonstrably record."""
    missing: list[str] = []
    for required in spec.required_evidence_fields:
        accepted = {required} | set(_FIELD_ALIASES.get(required, frozenset()))
        if item.fields & accepted:
            continue
        required_tokens = set(required.split("_"))
        if any(required_tokens <= set(present.split("_")) for present in item.fields):
            continue
        missing.append(required)
    if spec.current_revision and item.revision is None:
        missing.append("revision")
    return tuple(missing)


def _revision_superseded(spec: RequirementSpec, item: EvidenceItem) -> bool:
    if item.revision is None:
        return False
    superseded = {r.strip().upper() for r in spec.superseded_revisions}
    if item.revision in superseded:
        return True
    return bool(spec.current_revision) and item.revision != spec.current_revision.strip().upper()


def _confidence_for(snapshot: EvidenceSnapshot, *, evidence_count: int, status: GapStatus) -> Confidence:
    """Confidence in the *finding*, not in the equipment.

    A rule check over a complete corpus is exact. A rule check performed while an evidence source
    was unreachable is not, and says so.
    """
    if snapshot.degraded_sources:
        return Confidence(
            value=0.7,
            rationale=(
                "Rule check ran with degraded evidence collection: "
                f"{', '.join(snapshot.degraded_sources)} did not respond. Re-run before filing."
            ),
            method="heuristic",
        )
    if status is GapStatus.MISSING:
        return Confidence(
            value=1.0,
            rationale="Deterministic rule check: no evidence of the required type exists for this asset",
            method="exact",
        )
    return Confidence(
        value=1.0,
        rationale=f"Deterministic rule check over {evidence_count} dated evidence record(s)",
        method="exact",
    )


def _due_date(last: date, spec: RequirementSpec) -> date | None:
    from datetime import timedelta

    if spec.frequency_days is None:
        return None
    return last + timedelta(days=spec.frequency_days + spec.grace_days)


def _action(
    spec: RequirementSpec,
    equipment: Equipment,
    status: GapStatus,
    *,
    severity: Severity,
    as_of: date,
    deadline: date | None,
    missing: Sequence[str],
    last_evidence: date | None,
) -> RecommendedAction:
    """A concrete next step with an owner and a date. Vague advice is not actionable."""
    evidence_label = (
        spec.evidence_types[0].value.replace("_", " ") if spec.evidence_types else "inspection report"
    )
    if status is GapStatus.INCOMPLETE:
        readable = ", ".join(f.replace("_", " ") for f in missing) or "the required particulars"
        action = (
            f"Amend the {evidence_label} for {equipment.tag} dated "
            f"{last_evidence.isoformat() if last_evidence else 'unknown'} to record {readable}, "
            "then re-file it against the asset."
        )
    elif spec.remediation:
        action = spec.remediation
    else:
        action = (
            f"Carry out {spec.obligation} on {equipment.tag} and file the resulting "
            f"{evidence_label} against the asset."
        )
    due_within = 0 if deadline is None or deadline <= as_of else (deadline - as_of).days
    rationale = f"{spec.citation}: {spec.obligation}."
    if spec.requirement.penalty:
        rationale = f"{rationale} Exposure: {spec.requirement.penalty}"
    return RecommendedAction(
        action=action,
        urgency=severity,
        owner_role=spec.owner_role,
        due_within_days=due_within,
        rationale=rationale,
        estimated_minutes=spec.remediation_minutes,
    )


def _penalty_risk(spec: RequirementSpec, status: GapStatus, days_overdue: int | None) -> str | None:
    if not spec.requirement.penalty:
        return None
    if days_overdue is not None and days_overdue > 0:
        return f"{spec.requirement.penalty} Exposure has been open for {days_overdue} day(s)."
    if status is GapStatus.MISSING:
        return f"{spec.requirement.penalty} Exposure is open now — no evidence of compliance exists."
    return spec.requirement.penalty


def _refs(items: Sequence[EvidenceItem], limit: int) -> tuple[SourceRef, ...]:
    return tuple(item.source for item in items[:limit])


def assess(
    spec: RequirementSpec,
    equipment: Equipment,
    snapshot: EvidenceSnapshot,
) -> RequirementAssessment:
    """Resolve one requirement against one asset. Pure, total, and deterministic.

    Called once per (asset × requirement) pair. Never raises: an unresolvable input produces a
    finding, because silence in a compliance system reads as a pass.
    """
    as_of = snapshot.as_of
    all_items = snapshot.for_tag(equipment.tag)
    accepted_types = set(spec.evidence_types)
    acceptable = [i for i in all_items if not accepted_types or i.document_type in accepted_types]
    related = [i for i in all_items if i not in acceptable]

    # ---------------------------------------------------------------- MISSING
    if not acceptable:
        wanted = ", ".join(t.value.replace("_", " ") for t in spec.evidence_types) or "any documented evidence"
        if related:
            detail = (
                f"No {wanted} exists for {equipment.tag}. {len(related)} other record(s) were found for "
                f"this asset (cited below) but none is of an acceptable evidence type for "
                f"{spec.citation}, so none discharges the obligation."
            )
        else:
            detail = (
                f"No {wanted} exists for {equipment.tag}. The obligation under {spec.citation} "
                f"({spec.obligation}) is undocumented — absence of evidence is a gap, not a pass."
            )
        days_overdue: int | None = None
        deadline = as_of
        if spec.frequency_days and equipment.installed_on:
            first_due = _due_date(equipment.installed_on, spec)
            if first_due is not None:
                deadline = first_due
                if as_of > first_due:
                    days_overdue = (as_of - first_due).days
        severity = _resolve_severity(spec, equipment, GapStatus.MISSING, days_overdue=days_overdue)
        return RequirementAssessment(
            spec=spec,
            equipment=equipment,
            status=GapStatus.MISSING,
            severity=severity,
            detail=detail,
            evidence=_refs(related, _MAX_RELATED_REFS),
            last_evidence_date=None,
            deadline=deadline,
            days_overdue=days_overdue,
            penalty_risk=_penalty_risk(spec, GapStatus.MISSING, days_overdue),
            recommended_action=_action(
                spec, equipment, GapStatus.MISSING, severity=severity, as_of=as_of,
                deadline=deadline, missing=(), last_evidence=None,
            ),
            confidence=_confidence_for(snapshot, evidence_count=0, status=GapStatus.MISSING),
        )

    dated = [i for i in acceptable if i.is_dated]
    undated = [i for i in acceptable if not i.is_dated]

    # ---------------------------------------------------------------- undated evidence
    if spec.frequency_days is not None and not dated:
        detail = (
            f"{len(undated)} {spec.evidence_types[0].value if spec.evidence_types else 'evidence'} record(s) "
            f"exist for {equipment.tag} but carry no date. A {spec.frequency_days}-day obligation cannot be "
            f"demonstrated without a date, so {spec.citation} is not evidenced."
        )
        severity = _resolve_severity(spec, equipment, GapStatus.INCOMPLETE, days_overdue=None)
        return RequirementAssessment(
            spec=spec,
            equipment=equipment,
            status=GapStatus.INCOMPLETE,
            severity=severity,
            detail=detail,
            evidence=_refs(undated, _MAX_EVIDENCE_REFS),
            deadline=as_of,
            penalty_risk=_penalty_risk(spec, GapStatus.INCOMPLETE, None),
            recommended_action=_action(
                spec, equipment, GapStatus.INCOMPLETE, severity=severity, as_of=as_of,
                deadline=as_of, missing=("date",), last_evidence=None,
            ),
            missing_fields=("date",),
            confidence=_confidence_for(snapshot, evidence_count=len(undated), status=GapStatus.INCOMPLETE),
        )

    ordered = sorted(dated or undated, key=lambda i: (i.occurred_on or date.min), reverse=True)
    latest = ordered[0]
    last_evidence_date = latest.occurred_on
    deadline = _due_date(last_evidence_date, spec) if last_evidence_date else None

    # ---------------------------------------------------------------- OUTDATED (stale)
    if deadline is not None and as_of > deadline:
        days_overdue = (as_of - deadline).days
        severity = _resolve_severity(spec, equipment, GapStatus.OUTDATED, days_overdue=days_overdue)
        detail = (
            f"The most recent {latest.document_type.value.replace('_', ' ')} for {equipment.tag} is dated "
            f"{last_evidence_date.isoformat() if last_evidence_date else 'unknown'}. "
            f"{spec.citation} requires {spec.obligation} every {spec.frequency_days} day(s)"
            f"{f' (+{spec.grace_days} day grace)' if spec.grace_days else ''}; the evidence became due on "
            f"{deadline.isoformat()} and is {days_overdue} day(s) overdue."
        )
        return RequirementAssessment(
            spec=spec,
            equipment=equipment,
            status=GapStatus.OUTDATED,
            severity=severity,
            detail=detail,
            evidence=_refs(ordered, _MAX_EVIDENCE_REFS),
            last_evidence_date=last_evidence_date,
            deadline=deadline,
            days_overdue=days_overdue,
            penalty_risk=_penalty_risk(spec, GapStatus.OUTDATED, days_overdue),
            recommended_action=_action(
                spec, equipment, GapStatus.OUTDATED, severity=severity, as_of=as_of,
                deadline=deadline, missing=(), last_evidence=last_evidence_date,
            ),
            confidence=_confidence_for(snapshot, evidence_count=len(ordered), status=GapStatus.OUTDATED),
        )

    # ---------------------------------------------------------------- inside the window
    in_window = [i for i in ordered if deadline is None or i.occurred_on is None or _due_date(i.occurred_on, spec) is None
                 or _due_date(i.occurred_on, spec) >= as_of]
    candidates = in_window or ordered
    # Strongest evidence first: fewest missing required fields, then most recent.
    best = min(candidates, key=lambda i: (len(_missing_fields(spec, i)), -(i.occurred_on or date.min).toordinal()))
    missing = _missing_fields(spec, best)

    # ---------------------------------------------------------------- OUTDATED (superseded revision)
    if _revision_superseded(spec, best):
        severity = _resolve_severity(spec, equipment, GapStatus.OUTDATED, days_overdue=None)
        detail = (
            f"Evidence for {equipment.tag} dated "
            f"{best.occurred_on.isoformat() if best.occurred_on else 'unknown'} cites revision "
            f"{best.revision}, which {spec.citation} has superseded"
            f"{f' (current revision {spec.current_revision})' if spec.current_revision else ''}. "
            "Work performed against a superseded revision does not discharge the obligation."
        )
        return RequirementAssessment(
            spec=spec,
            equipment=equipment,
            status=GapStatus.OUTDATED,
            severity=severity,
            detail=detail,
            evidence=_refs(candidates, _MAX_EVIDENCE_REFS),
            last_evidence_date=last_evidence_date,
            deadline=deadline,
            days_overdue=None,
            penalty_risk=_penalty_risk(spec, GapStatus.OUTDATED, None),
            recommended_action=_action(
                spec, equipment, GapStatus.OUTDATED, severity=severity, as_of=as_of,
                deadline=deadline, missing=(), last_evidence=last_evidence_date,
            ),
            confidence=_confidence_for(snapshot, evidence_count=len(candidates), status=GapStatus.OUTDATED),
        )

    # ---------------------------------------------------------------- INCOMPLETE
    if missing:
        severity = _resolve_severity(spec, equipment, GapStatus.INCOMPLETE, days_overdue=None)
        readable = ", ".join(f.replace("_", " ") for f in missing)
        detail = (
            f"A {best.document_type.value.replace('_', ' ')} dated "
            f"{best.occurred_on.isoformat() if best.occurred_on else 'unknown'} exists for {equipment.tag} "
            f"and is inside the {spec.frequency_days or 'one-off'}-day window, but it does not record: "
            f"{readable}. {spec.citation} is therefore evidenced but not demonstrated."
        )
        return RequirementAssessment(
            spec=spec,
            equipment=equipment,
            status=GapStatus.INCOMPLETE,
            severity=severity,
            detail=detail,
            evidence=_refs(candidates, _MAX_EVIDENCE_REFS),
            last_evidence_date=last_evidence_date,
            deadline=deadline,
            days_overdue=None,
            penalty_risk=_penalty_risk(spec, GapStatus.INCOMPLETE, None),
            recommended_action=_action(
                spec, equipment, GapStatus.INCOMPLETE, severity=severity, as_of=as_of,
                deadline=deadline, missing=missing, last_evidence=best.occurred_on,
            ),
            missing_fields=missing,
            confidence=_confidence_for(snapshot, evidence_count=len(candidates), status=GapStatus.INCOMPLETE),
        )

    # ---------------------------------------------------------------- COMPLIANT
    window_note = (
        f"within the {spec.frequency_days}-day window (next due {deadline.isoformat()})"
        if deadline is not None and spec.frequency_days
        else "and the obligation is one-off rather than periodic"
    )
    detail = (
        f"{best.document_type.value.replace('_', ' ').title()} dated "
        f"{best.occurred_on.isoformat() if best.occurred_on else 'undated'} for {equipment.tag} records "
        f"{spec.obligation} {window_note}. Required particulars present: "
        f"{', '.join(f.replace('_', ' ') for f in spec.required_evidence_fields) or 'none specified'}."
    )
    return RequirementAssessment(
        spec=spec,
        equipment=equipment,
        status=GapStatus.COMPLIANT,
        severity=Severity.INFO,
        detail=detail,
        evidence=_refs(candidates, _MAX_EVIDENCE_REFS),
        last_evidence_date=last_evidence_date,
        deadline=deadline,
        days_overdue=None,
        penalty_risk=None,
        recommended_action=None,
        confidence=_confidence_for(snapshot, evidence_count=len(candidates), status=GapStatus.COMPLIANT),
    )


def assess_equipment(
    equipment: Equipment,
    specs: Sequence[RequirementSpec],
    snapshot: EvidenceSnapshot,
) -> list[RequirementAssessment]:
    """Assess every applicable requirement for one asset. Pure."""
    return [assess(spec, equipment, snapshot) for spec in specs if spec.applies_to(equipment)]


def assess_scope(
    scope: Sequence[Equipment],
    catalogue: RequirementCatalogue,
    snapshot: EvidenceSnapshot,
    *,
    regulations: Sequence[str] | None = None,
) -> list[RequirementAssessment]:
    """Assess the whole scope, ordered deterministically. Pure.

    Ordering is (severity desc, regulation, clause, tag) so two runs over the same corpus produce
    byte-identical output — a requirement of any evidence pack that goes to a regulator.
    """
    specs = catalogue.for_regulations(regulations)
    assessments: list[RequirementAssessment] = []
    for equipment in scope:
        assessments.extend(assess_equipment(equipment, specs, snapshot))
    assessments.sort(
        key=lambda a: (-a.severity.rank, a.status.value, a.spec.regulation, a.spec.clause, a.equipment.tag)
    )
    logger.info(
        "compliance assessment complete",
        extra={
            "assets": len(scope),
            "requirements": len(specs),
            "assessments": len(assessments),
            "gaps": sum(1 for a in assessments if a.is_gap),
            "as_of": snapshot.as_of.isoformat(),
        },
    )
    return assessments


def utc_today(now: datetime | None = None) -> date:
    """Today's date in UTC — the reference point every assessment is anchored to."""
    from indra.core.models import utcnow

    return (now or utcnow()).date()


__all__ = [
    "EvidenceItem",
    "EvidenceOrigin",
    "EvidenceSnapshot",
    "RequirementAssessment",
    "assess",
    "assess_equipment",
    "assess_scope",
    "build_snapshot",
    "evidence_from_document",
    "evidence_from_maintenance",
    "evidence_from_procedure",
    "gap_id_for",
    "utc_today",
]
