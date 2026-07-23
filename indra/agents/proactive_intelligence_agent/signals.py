"""Signal detection: the raw observations the compound-signal rule engine reasons over.

A :class:`~indra.core.models.Signal` on its own means almost nothing. A pump that vibrates a bit
more than last month is not news; an alarm that was silenced once is not news. The value is in the
*conjunction*, which is what :mod:`.rules` evaluates. This module's job is to produce those
observations honestly: every signal carries a real :class:`~indra.core.models.SourceRef` pointing at
the document or record it came from, a bounded ``strength``, and a ``data`` payload rich enough that
the rule explanation can name dates, numbers and documents rather than saying "rule 3 matched".

Structure
---------

* :class:`SignalCollector` does **all** the I/O. It reads the graph, the vector store and the
  in-process compliance-gap ledger and produces an immutable :class:`AssetSnapshot`.
* The ``detect_*`` functions are **pure**: snapshot in, ``list[Signal]`` out. No awaits, no clock
  reads, no store access. That is what makes "given fixture graph state, exactly these rules fire"
  a testable statement.

Signal kinds emitted here
-------------------------

============================ =========================================================
``maintenance_anomaly``      Repeat findings, deteriorating intervals, anomaly vocabulary
``precursor_match``          A current finding reads like a past failure's recorded precursor
``threshold_approach``       A reading is within ``oem_threshold_warning_ratio`` of an OEM limit
``missing_workorder``        Nothing scheduled and nothing open, past the maintenance window
``alarm_bypass``             Bypass / override / inhibit / silence language in shift logs
``fleet_pattern``            The same failure mode recurring across similar assets
``expertise_loss``           An expert who knows this asset retires inside the horizon
``documentation_void``       Zero documented knowledge (no SOP, no RCA) for this asset
``regulatory_exposure``      A compliance deadline that is near or already passed
``evidence_void``            That obligation has no evidence record behind it
============================ =========================================================

``maintenance_anomaly`` also emits ``precursor_match``, and ``expertise_loss`` also emits
``documentation_void``, because each pair is derived from a single body of evidence and splitting
the read would mean walking the same records twice. The rules consume them as separate kinds.
"""

from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Awaitable, Final, Iterable, Literal, Mapping, Sequence, TypeVar

from rapidfuzz import fuzz

from indra.core.config import Settings
from indra.core.exceptions import IndraError
from indra.core.logging import get_logger
from indra.core.models import (
    Chunk,
    ConditionReading,
    Criticality,
    DocumentMeta,
    DocumentType,
    Equipment,
    FailureEvent,
    MaintenanceRecord,
    Person,
    Procedure,
    Severity,
    Signal,
    SourceRef,
    utcnow,
)

logger = get_logger(__name__)

_T = TypeVar("_T")

# ======================================================================================
# Signal kinds
# ======================================================================================

KIND_MAINTENANCE_ANOMALY: Final[str] = "maintenance_anomaly"
KIND_PRECURSOR_MATCH: Final[str] = "precursor_match"
KIND_THRESHOLD_APPROACH: Final[str] = "threshold_approach"
KIND_MISSING_WORKORDER: Final[str] = "missing_workorder"
KIND_ALARM_BYPASS: Final[str] = "alarm_bypass"
KIND_FLEET_PATTERN: Final[str] = "fleet_pattern"
KIND_EXPERTISE_LOSS: Final[str] = "expertise_loss"
KIND_DOCUMENTATION_VOID: Final[str] = "documentation_void"
KIND_REGULATORY_EXPOSURE: Final[str] = "regulatory_exposure"
KIND_EVIDENCE_VOID: Final[str] = "evidence_void"

SignalKind = Literal[
    "maintenance_anomaly",
    "precursor_match",
    "threshold_approach",
    "missing_workorder",
    "alarm_bypass",
    "fleet_pattern",
    "expertise_loss",
    "documentation_void",
    "regulatory_exposure",
    "evidence_void",
]

ALL_SIGNAL_KINDS: Final[tuple[str, ...]] = (
    KIND_MAINTENANCE_ANOMALY,
    KIND_PRECURSOR_MATCH,
    KIND_THRESHOLD_APPROACH,
    KIND_MISSING_WORKORDER,
    KIND_ALARM_BYPASS,
    KIND_FLEET_PATTERN,
    KIND_EXPERTISE_LOSS,
    KIND_DOCUMENTATION_VOID,
    KIND_REGULATORY_EXPOSURE,
    KIND_EVIDENCE_VOID,
)

#: How a signal was derived, used by :mod:`.scoring` to pick a ``Confidence.method``.
DerivationMethod = Literal["exact", "heuristic", "semantic"]


# ======================================================================================
# Detector tuning
# ======================================================================================


@dataclass(frozen=True, slots=True)
class DetectorTuning:
    """Coefficients that shape detector output.

    These are *detector calibration*, not deployment configuration: they describe how strongly a
    given observation should be believed, which is a property of the detector's algorithm rather
    than of the environment it runs in. ``indra.core.config.Settings`` deliberately owns the
    operational knobs the detectors consult (``oem_threshold_warning_ratio``,
    ``precursor_similarity_threshold``, ``fleet_failure_min_count``, the lookback windows) and this
    agent reads every one of them from settings. Everything below has no settings field and
    ``indra/core`` is read-only for this agent, so it lives here — frozen, named, and overridable in
    tests rather than inlined at the call site.
    """

    #: Minimum number of records sharing a finding term before it counts as a repeat.
    repeat_finding_min: int = 2
    #: Strength floor and ceiling for a repeat-finding anomaly.
    repeat_finding_base: float = 0.55
    repeat_finding_per_extra: float = 0.15
    #: Strength of an anomaly raised purely from deterioration vocabulary in the latest record.
    vocabulary_base: float = 0.45
    #: Successive maintenance intervals shrinking below this ratio of the earlier mean is itself
    #: a symptom — the asset is being touched more and more often.
    interval_compression_ratio: float = 0.6
    interval_compression_strength: float = 0.6
    #: An open or deferred record whose recommendations were never actioned.
    unactioned_recommendation_strength: float = 0.65
    #: Reading-to-limit ratio at which threshold strength saturates at 1.0.
    threshold_saturation_ratio: float = 1.0
    #: Per-occurrence increment for repeated bypass language in the shift logs.
    bypass_per_occurrence: float = 0.25
    bypass_base: float = 0.5
    #: Half-life (days) used to decay text-derived signal strength by document age.
    text_recency_half_life_days: float = 45.0
    #: Fleet-pattern strength grows with the number of affected peers.
    fleet_base: float = 0.5
    fleet_per_extra_asset: float = 0.15
    #: Days-to-retirement is normalised against ``settings.retirement_horizon_days``; this is the
    #: strength floor so a retirement two years out still registers.
    expertise_floor: float = 0.35
    #: Compliance deadline strength when the deadline has already passed.
    overdue_strength: float = 1.0
    #: Maximum passages scanned per asset for text pattern detection.
    max_passages: int = 240
    #: Maximum SourceRefs attached to one signal.
    max_sources_per_signal: int = 4


DEFAULT_TUNING: Final[DetectorTuning] = DetectorTuning()


# ======================================================================================
# Text patterns
# ======================================================================================

#: Alarm-handling language. An operator writing any of this in a shift log is telling us the
#: protective layer was removed. Weight reflects how unambiguous the phrase is.
_BYPASS_PATTERNS: Final[tuple[tuple[str, re.Pattern[str], float], ...]] = (
    ("bypass", re.compile(r"\bby[\s\-]?pass(?:ed|ing|es)?\b", re.IGNORECASE), 1.0),
    ("override", re.compile(r"\boverrid(?:e|es|den|ing)\b", re.IGNORECASE), 0.92),
    ("inhibit", re.compile(r"\binhibit(?:ed|ing|s|ion)?\b", re.IGNORECASE), 0.92),
    ("silenced", re.compile(r"\bsilenc(?:e|ed|es|ing)\b", re.IGNORECASE), 0.8),
    (
        "acknowledged and cleared",
        re.compile(r"\backnowledg\w*\s+(?:and|&|,)\s*(?:then\s+)?clear\w*\b", re.IGNORECASE),
        0.95,
    ),
)

#: Degradation vocabulary in maintenance findings. The weight is how strongly the term on its own
#: suggests a developing fault rather than routine housekeeping.
_ANOMALY_TERMS: Final[tuple[tuple[str, re.Pattern[str], float], ...]] = (
    ("bearing wear", re.compile(r"\bbearing\s+(?:wear|damage|pitting|spalling)\b", re.IGNORECASE), 0.95),
    ("seizure", re.compile(r"\bseiz(?:e|ed|ure|ing)\b", re.IGNORECASE), 1.0),
    ("vibration", re.compile(r"\bvibrat(?:ion|ing|es)?\b", re.IGNORECASE), 0.8),
    ("overheating", re.compile(r"\b(?:over\s?heat(?:ing|ed)?|excess(?:ive)?\s+temperature)\b", re.IGNORECASE), 0.85),
    ("leak", re.compile(r"\bleak(?:age|ing|s|ed)?\b", re.IGNORECASE), 0.75),
    ("misalignment", re.compile(r"\bmis[\s\-]?align(?:ment|ed)?\b", re.IGNORECASE), 0.8),
    ("cavitation", re.compile(r"\bcavitat(?:ion|ing)\b", re.IGNORECASE), 0.85),
    ("abnormal noise", re.compile(r"\b(?:abnormal|unusual|excess(?:ive)?)\s+(?:noise|sound|rattle)\b", re.IGNORECASE), 0.7),
    ("crack", re.compile(r"\bcrack(?:ed|ing|s)?\b", re.IGNORECASE), 0.9),
    ("corrosion", re.compile(r"\bcorros(?:ion|ive|ed)\b", re.IGNORECASE), 0.7),
    ("erratic reading", re.compile(r"\b(?:erratic|fluctuat\w+|unstable)\s+(?:reading|value|signal|trend)\b", re.IGNORECASE), 0.65),
    ("trip", re.compile(r"\b(?:tripp?ed|trip)\s+(?:on|due to)\b", re.IGNORECASE), 0.8),
)

#: Document types that carry shift-handover and incident narrative, which is where bypass language
#: is actually written down. Work orders record the *consequence*, not the operator's decision.
SHIFT_LOG_TYPES: Final[tuple[DocumentType, ...]] = (
    DocumentType.SHIFT_LOG,
    DocumentType.INCIDENT_REPORT,
)

#: Document types that constitute transferable, institutional knowledge about an asset. A stack of
#: work orders is a history; an SOP and an RCA are knowledge somebody can be trained from.
KNOWLEDGE_DOCUMENT_TYPES: Final[tuple[DocumentType, ...]] = (
    DocumentType.SOP,
    DocumentType.ROOT_CAUSE_ANALYSIS,
    DocumentType.OEM_MANUAL,
    DocumentType.INSPECTION_REPORT,
)


# ======================================================================================
# Small helpers
# ======================================================================================


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp ``value`` into ``[low, high]``. Every ``Score`` field is validated 0–1."""
    return max(low, min(high, value))


def as_datetime(value: date | datetime | None) -> datetime | None:
    """Normalise a ``date`` or naive ``datetime`` to an aware UTC ``datetime``.

    The domain models mix ``date`` (``performed_on``, ``occurred_on``) with aware ``datetime``
    (``measured_at``). Comparing them directly raises; normalising once here avoids scattering
    ``isinstance`` checks through every detector.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)


def days_between(later: datetime | date | None, earlier: datetime | date | None) -> float | None:
    """Whole-and-fractional days from ``earlier`` to ``later``; ``None`` if either is unknown."""
    a, b = as_datetime(later), as_datetime(earlier)
    if a is None or b is None:
        return None
    return (a - b).total_seconds() / 86400.0


def format_date(value: date | datetime | None, *, fallback: str = "an unrecorded date") -> str:
    """Render a date the way a shift supervisor writes it: ``14 June 2024``."""
    moment = as_datetime(value)
    if moment is None:
        return fallback
    return f"{moment.day} {moment.strftime('%B %Y')}"


def recency_weight(moment: date | datetime | None, *, now: datetime, half_life_days: float) -> float:
    """Exponential decay in ``[0, 1]``: evidence from last week outranks evidence from last year."""
    elapsed = days_between(now, moment)
    if elapsed is None:
        # No date at all: treat as middling rather than dismissing it outright.
        return 0.5
    if elapsed <= 0:
        return 1.0
    if half_life_days <= 0:
        return 1.0
    return float(math.pow(0.5, elapsed / half_life_days))


def linear_trend(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """Ordinary least squares over ``(x, y)``; returns ``(slope, intercept)``.

    Written out rather than pulled from numpy because the sample is a handful of readings and the
    call sits on the synchronous, thread-offloaded detection path.
    """
    n = len(points)
    if n < 2:
        return 0.0, points[0][1] if points else 0.0
    mean_x = sum(p[0] for p in points) / n
    mean_y = sum(p[1] for p in points) / n
    variance = sum((p[0] - mean_x) ** 2 for p in points)
    if variance <= 1e-12:
        return 0.0, mean_y
    covariance = sum((p[0] - mean_x) * (p[1] - mean_y) for p in points)
    slope = covariance / variance
    return slope, mean_y - slope * mean_x


def similarity(left: str, right: str) -> float:
    """Order-insensitive token similarity in ``[0, 1]``.

    ``token_set_ratio`` is the right tool for maintenance prose: *"bearing temp rising, slight
    knock at DE"* and *"knocking at drive end, bearing temperature rise"* are the same observation
    written by two different fitters, and any order-sensitive measure scores them far apart.
    """
    if not left.strip() or not right.strip():
        return 0.0
    return float(fuzz.token_set_ratio(left, right)) / 100.0


def dedupe_sources(sources: Iterable[SourceRef], *, limit: int) -> list[SourceRef]:
    """Collapse duplicate citations, strongest first, capped at ``limit``."""
    seen: dict[tuple[str, str | None, int | None], SourceRef] = {}
    for ref in sources:
        key = (ref.document_id, ref.chunk_id, ref.page)
        existing = seen.get(key)
        if existing is None or ref.relevance > existing.relevance:
            seen[key] = ref
    ordered = sorted(seen.values(), key=lambda r: (-r.relevance, r.document_id, r.chunk_id or ""))
    return ordered[:limit]


def _matched_terms(
    text: str,
    patterns: Sequence[tuple[str, re.Pattern[str], float]],
) -> list[tuple[str, float, int]]:
    """Return ``(term, weight, occurrences)`` for every pattern that fires in ``text``."""
    if not text:
        return []
    hits: list[tuple[str, float, int]] = []
    for term, pattern, weight in patterns:
        count = len(pattern.findall(text))
        if count:
            hits.append((term, weight, count))
    return hits


def _snippet_around(text: str, pattern: re.Pattern[str], *, window: int = 160) -> str:
    """Extract a readable quote around the first match, for the explanation text."""
    match = pattern.search(text)
    if match is None:
        return text[:window].strip()
    start = max(0, match.start() - window // 2)
    end = min(len(text), match.end() + window // 2)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{text[start:end].strip()}{suffix}"


# ======================================================================================
# Compliance gap ledger — fed by the event bus, never by importing the compliance agent
# ======================================================================================


@dataclass(frozen=True, slots=True)
class ComplianceGapRecord:
    """A gap reported by the Compliance Agent over ``Topic.GAP_DETECTED``.

    The regulatory compound-signal rule needs compliance state, and this agent is forbidden from
    importing ``indra.agents.compliance_agent``. The documented choreography in
    :mod:`indra.core.events` is ``compliance --gap.detected--> proactive``, so the gap arrives as an
    event payload and lands here.
    """

    gap_id: str
    equipment_tag: str
    regulation: str
    clause: str
    status: str
    severity: Severity
    deadline: date | None
    received_at: datetime

    @property
    def is_evidence_void(self) -> bool:
        """True when the obligation has no acceptable evidence behind it at all."""
        return self.status.lower() in {"missing", "incomplete", "outdated"}


class GapLedger:
    """Bounded in-process store of compliance gaps observed on the bus.

    Deliberately not persisted: it is a *cache of another agent's opinion*, and a stale opinion is
    worse than no opinion. Entries older than ``max_age_days`` are dropped on every read.
    """

    __slots__ = ("_by_tag", "_max_age_days", "_max_per_tag")

    def __init__(self, *, max_age_days: int = 90, max_per_tag: int = 64) -> None:
        self._by_tag: dict[str, dict[str, ComplianceGapRecord]] = {}
        self._max_age_days = max_age_days
        self._max_per_tag = max_per_tag

    def record(self, payload: Mapping[str, Any], *, now: datetime | None = None) -> ComplianceGapRecord | None:
        """Ingest a ``GapDetectedPayload``-shaped mapping. Malformed payloads are dropped, not raised.

        A neighbouring agent publishing a bad payload must degrade observability, never break a
        proactive scan.
        """
        body = payload.get("payload") if isinstance(payload.get("payload"), Mapping) else payload
        if not isinstance(body, Mapping):
            return None
        tag = str(body.get("equipment_tag") or "").strip().upper()
        gap_id = str(body.get("gap_id") or "").strip()
        if not tag or not gap_id:
            logger.debug("dropping compliance gap event without tag or id", extra={"payload_keys": sorted(body)})
            return None
        try:
            severity = Severity(str(body.get("severity") or Severity.WARNING.value))
        except ValueError:
            severity = Severity.WARNING
        record = ComplianceGapRecord(
            gap_id=gap_id,
            equipment_tag=tag,
            regulation=str(body.get("regulation") or "an unnamed regulation"),
            clause=str(body.get("clause") or "unspecified clause"),
            status=str(body.get("status") or "missing"),
            severity=severity,
            deadline=_parse_date(body.get("deadline")),
            received_at=now or utcnow(),
        )
        bucket = self._by_tag.setdefault(tag, {})
        bucket[gap_id] = record
        if len(bucket) > self._max_per_tag:
            for stale_id in sorted(bucket, key=lambda k: bucket[k].received_at)[: len(bucket) - self._max_per_tag]:
                bucket.pop(stale_id, None)
        return record

    def for_tag(self, tag: str, *, now: datetime | None = None) -> tuple[ComplianceGapRecord, ...]:
        """Return non-expired gaps for ``tag``, newest first."""
        moment = now or utcnow()
        bucket = self._by_tag.get(tag.strip().upper())
        if not bucket:
            return ()
        cutoff = moment - timedelta(days=self._max_age_days)
        live = [g for g in bucket.values() if g.received_at >= cutoff]
        for expired in [g.gap_id for g in bucket.values() if g.received_at < cutoff]:
            bucket.pop(expired, None)
        return tuple(sorted(live, key=lambda g: g.received_at, reverse=True))

    def tags(self) -> tuple[str, ...]:
        """Every tag with at least one recorded gap."""
        return tuple(sorted(t for t, bucket in self._by_tag.items() if bucket))

    def size(self) -> int:
        return sum(len(bucket) for bucket in self._by_tag.values())

    def clear(self) -> None:
        self._by_tag.clear()


def _parse_date(value: object) -> date | None:
    """Parse an ISO date out of an event payload. Never raises."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return date.fromisoformat(value.strip()[:10])
        except ValueError:
            logger.debug("unparseable deadline in gap payload", extra={"value": value})
    return None


# ======================================================================================
# Snapshots
# ======================================================================================


@dataclass(frozen=True, slots=True)
class Passage:
    """A retrieved chunk together with the document it came from."""

    chunk: Chunk
    document: DocumentMeta
    relevance: float = 0.0

    def source(self, *, retrieved_via: str = "graph") -> SourceRef:
        """Project to a citation the operator can click through to."""
        return self.chunk.to_source_ref(self.document, relevance=self.relevance, retrieved_via=retrieved_via)


@dataclass(frozen=True, slots=True)
class AssetSnapshot:
    """Everything read from the stores for one asset, gathered once per scan.

    Frozen and self-contained so the detectors are pure functions of it. Two scans over the same
    snapshot produce byte-identical signals apart from the generated ``signal_id``.
    """

    equipment: Equipment
    now: datetime
    maintenance: tuple[MaintenanceRecord, ...] = ()
    failures: tuple[FailureEvent, ...] = ()
    procedures: tuple[Procedure, ...] = ()
    readings: tuple[ConditionReading, ...] = ()
    passages: tuple[Passage, ...] = ()
    documents: tuple[DocumentMeta, ...] = ()
    experts: tuple[Person, ...] = ()
    gaps: tuple[ComplianceGapRecord, ...] = ()
    degraded: tuple[str, ...] = ()

    @property
    def tag(self) -> str:
        return self.equipment.tag

    @property
    def document_count(self) -> int:
        """Distinct documents that mention this asset at all."""
        return len({d.document_id for d in self.documents})

    @property
    def knowledge_documents(self) -> tuple[DocumentMeta, ...]:
        """Documents that transfer knowledge rather than merely record history."""
        return tuple(d for d in self.documents if d.document_type in KNOWLEDGE_DOCUMENT_TYPES)

    @property
    def knowledge_document_count(self) -> int:
        """The denominator of institutional memory: SOPs, RCAs, manuals, inspections, procedures."""
        return len({d.document_id for d in self.knowledge_documents}) + len(self.procedures)

    def readings_for(self, parameter: str) -> tuple[ConditionReading, ...]:
        """Readings of one parameter, oldest first."""
        selected = [r for r in self.readings if r.parameter == parameter]
        selected.sort(key=lambda r: as_datetime(r.measured_at) or self.now)
        return tuple(selected)

    def latest_reading(self, parameter: str) -> ConditionReading | None:
        series = self.readings_for(parameter)
        return series[-1] if series else None

    def recent_maintenance(self, *, within_days: int) -> tuple[MaintenanceRecord, ...]:
        """Maintenance inside the lookback window, newest first."""
        cutoff = self.now - timedelta(days=within_days)
        selected = [r for r in self.maintenance if (as_datetime(r.performed_on) or self.now) >= cutoff]
        selected.sort(key=lambda r: as_datetime(r.performed_on) or self.now, reverse=True)
        return tuple(selected)

    def shift_log_passages(self, *, within_days: int, types: Sequence[DocumentType] = SHIFT_LOG_TYPES) -> tuple[Passage, ...]:
        """Shift-log and incident passages inside the lookback window, newest first."""
        cutoff = (self.now - timedelta(days=within_days)).date()
        chosen: list[Passage] = []
        for passage in self.passages:
            meta = passage.document
            if meta.document_type not in types:
                continue
            stamp = meta.document_date or meta.ingested_at.date()
            if stamp < cutoff:
                continue
            chosen.append(passage)
        chosen.sort(
            key=lambda p: (p.document.document_date or p.document.ingested_at.date()),
            reverse=True,
        )
        return tuple(chosen)


@dataclass(frozen=True, slots=True)
class FleetIndex:
    """Fleet-wide failure history, built once per scan and shared by every asset.

    Fleet pattern detection is inherently cross-asset: it is the one signal that cannot be computed
    from a single asset's records, which is exactly why it is worth having.
    """

    equipment: tuple[Equipment, ...] = ()
    failures_by_tag: Mapping[str, tuple[FailureEvent, ...]] = field(default_factory=dict)
    built_at: datetime = field(default_factory=utcnow)

    def peers(self, equipment: Equipment) -> tuple[Equipment, ...]:
        """Assets of the same type, excluding the asset itself."""
        kind = (equipment.equipment_type or "unknown").strip().lower()
        return tuple(
            e for e in self.equipment
            if e.tag != equipment.tag and (e.equipment_type or "unknown").strip().lower() == kind
        )

    def mode_incidence(
        self,
        equipment: Equipment,
        *,
        since: datetime,
        include_self: bool = True,
    ) -> dict[str, list[FailureEvent]]:
        """Map ``failure_mode`` → events on same-type assets since ``since``.

        Modes are normalised to lower case so "Bearing seizure" and "bearing seizure" are one mode.
        """
        family = {e.tag for e in self.peers(equipment)}
        if include_self:
            family.add(equipment.tag)
        incidence: dict[str, list[FailureEvent]] = {}
        for tag in sorted(family):
            for event in self.failures_by_tag.get(tag, ()):  # deterministic order
                occurred = as_datetime(event.occurred_on)
                if occurred is None or occurred < since:
                    continue
                incidence.setdefault(event.failure_mode.strip().lower(), []).append(event)
        return incidence

    def base_rate(self, equipment: Equipment, *, failure_mode: str | None, window_days: float) -> float:
        """Fleet failures per asset per year for this type, optionally restricted to one mode.

        Used as the prior in :mod:`.prediction`: if every pump of this model eats a bearing every
        second year, an individual pump's risk starts well above zero regardless of its own history.
        """
        family = [equipment.tag, *(e.tag for e in self.peers(equipment))]
        if not family or window_days <= 0:
            return 0.0
        cutoff = self.built_at - timedelta(days=window_days)
        wanted = failure_mode.strip().lower() if failure_mode else None
        count = 0
        for tag in family:
            for event in self.failures_by_tag.get(tag, ()):
                occurred = as_datetime(event.occurred_on)
                if occurred is None or occurred < cutoff:
                    continue
                if wanted and event.failure_mode.strip().lower() != wanted:
                    continue
                count += 1
        asset_years = len(family) * (window_days / 365.0)
        return count / asset_years if asset_years > 0 else 0.0


# ======================================================================================
# Guarded store reads
# ======================================================================================


async def guarded_read(
    awaitable: Awaitable[_T],
    *,
    label: str,
    default: _T,
    context: Mapping[str, object] | None = None,
) -> _T:
    """Await a store call, degrading to ``default`` instead of failing the scan.

    CLAUDE.md rule 6: a dead backend costs one capability and a loud log line, never a crash. The
    caller records the degradation on :attr:`AssetSnapshot.degraded` so ``/health`` and the alert
    body can both admit that the picture is incomplete.
    """
    extra: dict[str, Any] = {"read": label, **(dict(context) if context else {})}
    try:
        return await awaitable
    except asyncio.CancelledError:
        raise
    except IndraError as exc:
        logger.warning("proactive read degraded: %s", exc.message, extra={**extra, "error_code": exc.error_code})
        return default
    except Exception as exc:  # noqa: BLE001 - boundary: any backend fault degrades, never propagates
        logger.warning("proactive read failed: %s", exc, extra={**extra, "error_type": type(exc).__name__})
        return default


class SignalCollector:
    """Reads the stores and assembles :class:`AssetSnapshot` / :class:`FleetIndex` objects.

    All I/O for this agent funnels through here. The detectors below never touch a store, which is
    what lets a unit test hand them a hand-built snapshot and assert on exactly which rules fire.
    """

    __slots__ = ("_graph", "_vectors", "_settings", "_gaps", "_tuning", "_fleet_cache", "_fleet_cache_at", "_lock")

    def __init__(
        self,
        *,
        graph: Any,
        vectors: Any,
        settings: Settings,
        gaps: GapLedger,
        tuning: DetectorTuning = DEFAULT_TUNING,
    ) -> None:
        # ``graph`` / ``vectors`` are ``indra.core.contracts.GraphStore`` and ``VectorStore``.
        # They are typed as ``Any`` here only to avoid a runtime import cycle through
        # ``contracts`` → ``models``; the public API of this class is fully typed.
        self._graph = graph
        self._vectors = vectors
        self._settings = settings
        self._gaps = gaps
        self._tuning = tuning
        self._fleet_cache: FleetIndex | None = None
        self._fleet_cache_at: float = 0.0
        self._lock = asyncio.Lock()

    # -- fleet ------------------------------------------------------------------------

    async def fleet_index(self, *, force_refresh: bool = False) -> FleetIndex:
        """Build (or reuse) the fleet-wide failure index.

        Cached in-process for ``settings.cache_ttl_s``. Deliberately *not* pushed through
        ``deps.cache``: the index holds pydantic models, and round-tripping them through a possibly
        Redis-backed cache buys nothing at plant scale while adding a serialisation failure mode.
        """
        async with self._lock:
            age = asyncio.get_running_loop().time() - self._fleet_cache_at
            if not force_refresh and self._fleet_cache is not None and age < float(self._settings.cache_ttl_s):
                return self._fleet_cache

            equipment = await guarded_read(
                self._graph.list_equipment(), label="list_equipment", default=[]
            )
            fleet = tuple(sorted(equipment, key=lambda e: e.tag))
            since = (utcnow() - timedelta(days=self._settings.retirement_horizon_days)).date()
            failures: dict[str, tuple[FailureEvent, ...]] = {}
            semaphore = asyncio.Semaphore(self._settings.ingestion_concurrency)

            async def _load(tag: str) -> None:
                async with semaphore:
                    events = await guarded_read(
                        self._graph.failure_history(tag, since=since),
                        label="failure_history",
                        default=[],
                        context={"equipment_tag": tag},
                    )
                failures[tag] = tuple(sorted(events, key=lambda e: e.occurred_on))

            if fleet:
                await asyncio.gather(*(_load(e.tag) for e in fleet))

            index = FleetIndex(equipment=fleet, failures_by_tag=failures, built_at=utcnow())
            self._fleet_cache = index
            self._fleet_cache_at = asyncio.get_running_loop().time()
            logger.debug(
                "fleet index built",
                extra={"assets": len(fleet), "failure_events": sum(len(v) for v in failures.values())},
            )
            return index

    def invalidate_fleet(self) -> None:
        """Drop the cached fleet index — called when the graph reports an update."""
        self._fleet_cache = None
        self._fleet_cache_at = 0.0

    # -- per asset --------------------------------------------------------------------

    async def snapshot(self, equipment: Equipment, *, now: datetime | None = None) -> AssetSnapshot:
        """Gather every store-backed fact this agent needs about one asset."""
        moment = now or utcnow()
        settings = self._settings
        maintenance_since = (moment - timedelta(days=settings.maintenance_lookback_days)).date()
        failure_since = (moment - timedelta(days=settings.retirement_horizon_days)).date()
        degraded: list[str] = []

        maintenance, failures, procedures, people = await asyncio.gather(
            guarded_read(
                self._graph.maintenance_history(equipment.tag, since=maintenance_since),
                label="maintenance_history", default=[], context={"equipment_tag": equipment.tag},
            ),
            guarded_read(
                self._graph.failure_history(equipment.tag, since=failure_since),
                label="failure_history", default=[], context={"equipment_tag": equipment.tag},
            ),
            guarded_read(
                self._graph.procedures_for(equipment.tag),
                label="procedures_for", default=[], context={"equipment_tag": equipment.tag},
            ),
            guarded_read(
                self._graph.get_people(retiring_within_days=settings.retirement_horizon_days),
                label="get_people", default=[], context={"equipment_tag": equipment.tag},
            ),
        )

        passages, documents = await self._passages(equipment.tag)
        if not passages and not documents:
            degraded.append("no_documents_linked")

        readings: list[ConditionReading] = []
        for record in maintenance:
            readings.extend(r for r in record.readings if r.equipment_tag.upper() == equipment.tag.upper())
        readings.sort(key=lambda r: as_datetime(r.measured_at) or moment)

        experts = tuple(
            person for person in sorted(people, key=lambda p: (p.retirement_date or date.max, p.name))
            if any(t.strip().upper() == equipment.tag.upper() for t in person.expertise_tags)
        )

        return AssetSnapshot(
            equipment=equipment,
            now=moment,
            maintenance=tuple(sorted(maintenance, key=lambda r: r.performed_on)),
            failures=tuple(sorted(failures, key=lambda e: e.occurred_on)),
            procedures=tuple(procedures),
            readings=tuple(readings),
            passages=passages,
            documents=documents,
            experts=experts,
            gaps=self._gaps.for_tag(equipment.tag, now=moment),
            degraded=tuple(degraded),
        )

    async def _passages(self, tag: str) -> tuple[tuple[Passage, ...], tuple[DocumentMeta, ...]]:
        """Fetch the chunks that mention this asset, plus their document metadata."""
        entity_key = f"Equipment:{tag.strip().upper()}"
        scored = await guarded_read(
            self._graph.chunks_for_entities([entity_key], limit=self._tuning.max_passages),
            label="chunks_for_entities", default=[], context={"equipment_tag": tag},
        )
        if not scored:
            return (), ()
        relevance = {chunk_id: float(score) for chunk_id, score in scored}
        chunks = await guarded_read(
            self._vectors.get_chunks(list(relevance)),
            label="get_chunks", default=[], context={"equipment_tag": tag},
        )
        if not chunks:
            return (), ()
        metas = await guarded_read(
            self._graph.document_meta(sorted({c.document_id for c in chunks})),
            label="document_meta", default={}, context={"equipment_tag": tag},
        )
        passages: list[Passage] = []
        for chunk in chunks:
            meta = metas.get(chunk.document_id)
            if meta is None:
                continue
            passages.append(Passage(chunk=chunk, document=meta, relevance=_clamp(relevance.get(chunk.chunk_id, 0.0))))
        passages.sort(key=lambda p: (-p.relevance, p.chunk.chunk_id))
        documents = tuple(sorted(metas.values(), key=lambda m: m.document_id))
        return tuple(passages), documents


# ======================================================================================
# Detectors — pure functions over a snapshot
# ======================================================================================


def _signal(
    *,
    kind: str,
    snapshot: AssetSnapshot,
    description: str,
    strength: float,
    sources: Sequence[SourceRef],
    method: DerivationMethod,
    observed_at: datetime | None = None,
    data: Mapping[str, Any] | None = None,
    tuning: DetectorTuning = DEFAULT_TUNING,
) -> Signal:
    """Construct a validated :class:`Signal`, clamping the strength and capping the citations."""
    payload: dict[str, Any] = {"method": method, **(dict(data) if data else {})}
    return Signal(
        kind=kind,
        equipment_tag=snapshot.tag,
        description=description,
        observed_at=observed_at or snapshot.now,
        strength=_clamp(strength),
        sources=dedupe_sources(sources, limit=tuning.max_sources_per_signal),
        data=payload,
    )


def _record_sources(
    record: MaintenanceRecord | FailureEvent,
    snapshot: AssetSnapshot,
    *,
    document_types: Sequence[DocumentType],
) -> list[SourceRef]:
    """Citations for a plant record.

    Prefers the record's own ``sources``. When the record carries none — common when it was
    reconstructed from a structured feed rather than a parsed document — fall back to the passages
    already linked to this asset from documents of a matching type and date, so the signal still
    points at something an operator can open.
    """
    if record.sources:
        return list(record.sources)
    stamp = record.performed_on if isinstance(record, MaintenanceRecord) else record.occurred_on
    candidates = [
        passage for passage in snapshot.passages
        if passage.document.document_type in document_types
        and (passage.document.document_date is None or abs((passage.document.document_date - stamp).days) <= 7)
    ]
    candidates.sort(key=lambda p: -p.relevance)
    return [p.source(retrieved_via="graph") for p in candidates[:2]]


def detect_maintenance_anomaly(
    snapshot: AssetSnapshot,
    *,
    settings: Settings,
    tuning: DetectorTuning = DEFAULT_TUNING,
) -> list[Signal]:
    """Anomalies in the maintenance record, and matches against historical failure precursors.

    Four independent heuristics, each of which a reliability engineer would recognise:

    1. **Repeat finding** — the same degradation term recorded on ``repeat_finding_min`` or more
       separate visits inside the lookback window. One fitter noting vibration is an observation;
       three noting it is a trend.
    2. **Interval compression** — the gaps between maintenance visits are shrinking. An asset being
       touched more and more often is deteriorating even when no single visit says so.
    3. **Unactioned recommendation** — a record left ``open`` or ``deferred`` that carries a written
       recommendation. The plant already knows what to do and has not done it.
    4. **Anomaly vocabulary** — deterioration language in the most recent visit.

    It then emits ``precursor_match`` signals by comparing the findings that triggered an anomaly
    against :attr:`FailureEvent.precursor_text` from this asset's own failure history. Matching only
    anomalous findings (rather than every record) guarantees that a ``precursor_match`` is always
    accompanied by a ``maintenance_anomaly``, which is the conjunction rule 1 fires on.
    """
    records = snapshot.recent_maintenance(within_days=settings.maintenance_lookback_days)
    if not records:
        return []

    signals: list[Signal] = []
    anomalous: list[tuple[MaintenanceRecord, str, list[tuple[str, float, int]]]] = []

    # -- 1 & 4: vocabulary, per record and aggregated across records ----------------------
    term_records: dict[str, list[MaintenanceRecord]] = {}
    for record in records:
        text = f"{record.findings} {record.recommendations}".strip()
        hits = _matched_terms(text, _ANOMALY_TERMS)
        if not hits:
            continue
        anomalous.append((record, text, hits))
        for term, _weight, _count in hits:
            term_records.setdefault(term, []).append(record)

    for term, hitting in sorted(term_records.items()):
        weight = next((w for t, _p, w in _ANOMALY_TERMS if t == term), 0.7)
        dates = [r.performed_on for r in hitting]
        if len(hitting) >= tuning.repeat_finding_min:
            strength = tuning.repeat_finding_base + tuning.repeat_finding_per_extra * (len(hitting) - tuning.repeat_finding_min)
            signals.append(
                _signal(
                    kind=KIND_MAINTENANCE_ANOMALY,
                    snapshot=snapshot,
                    description=(
                        f"'{term}' recorded on {len(hitting)} separate visits to {snapshot.tag} "
                        f"between {format_date(min(dates))} and {format_date(max(dates))}"
                    ),
                    strength=strength * weight,
                    sources=[s for r in hitting for s in _record_sources(r, snapshot, document_types=(DocumentType.WORK_ORDER, DocumentType.INSPECTION_REPORT))],
                    method="heuristic",
                    observed_at=as_datetime(max(dates)) or snapshot.now,
                    data={
                        "pattern": "repeat_finding",
                        "term": term,
                        "occurrences": len(hitting),
                        "first_seen": min(dates).isoformat(),
                        "last_seen": max(dates).isoformat(),
                        "record_ids": [r.record_id for r in hitting],
                        "quote": (hitting[-1].findings or hitting[-1].recommendations)[:220],
                    },
                    tuning=tuning,
                )
            )
        elif hitting[0] is records[0]:
            # Newest record only — a single mention on the latest visit is still worth surfacing.
            signals.append(
                _signal(
                    kind=KIND_MAINTENANCE_ANOMALY,
                    snapshot=snapshot,
                    description=(
                        f"'{term}' noted on the most recent {snapshot.tag} visit "
                        f"({format_date(hitting[0].performed_on)})"
                    ),
                    strength=tuning.vocabulary_base * weight,
                    sources=_record_sources(hitting[0], snapshot, document_types=(DocumentType.WORK_ORDER, DocumentType.INSPECTION_REPORT)),
                    method="heuristic",
                    observed_at=as_datetime(hitting[0].performed_on) or snapshot.now,
                    data={
                        "pattern": "anomaly_vocabulary",
                        "term": term,
                        "occurrences": 1,
                        "last_seen": hitting[0].performed_on.isoformat(),
                        "record_ids": [hitting[0].record_id],
                        "quote": (hitting[0].findings or hitting[0].recommendations)[:220],
                    },
                    tuning=tuning,
                )
            )

    # -- 2: interval compression ----------------------------------------------------------
    ordered = sorted(records, key=lambda r: r.performed_on)
    if len(ordered) >= 4:
        intervals = [
            (ordered[i + 1].performed_on - ordered[i].performed_on).days
            for i in range(len(ordered) - 1)
        ]
        split = len(intervals) // 2
        early = intervals[:split]
        late = intervals[split:]
        if early and late:
            early_mean = sum(early) / len(early)
            late_mean = sum(late) / len(late)
            if early_mean > 0 and late_mean < tuning.interval_compression_ratio * early_mean:
                signals.append(
                    _signal(
                        kind=KIND_MAINTENANCE_ANOMALY,
                        snapshot=snapshot,
                        description=(
                            f"{snapshot.tag} is being worked on more often: average gap between visits fell "
                            f"from {early_mean:.0f} days to {late_mean:.0f} days"
                        ),
                        strength=tuning.interval_compression_strength,
                        sources=[s for r in ordered[-2:] for s in _record_sources(r, snapshot, document_types=(DocumentType.WORK_ORDER,))],
                        method="heuristic",
                        observed_at=as_datetime(ordered[-1].performed_on) or snapshot.now,
                        data={
                            "pattern": "interval_compression",
                            "early_mean_days": round(early_mean, 1),
                            "late_mean_days": round(late_mean, 1),
                            "visit_count": len(ordered),
                            "last_seen": ordered[-1].performed_on.isoformat(),
                        },
                        tuning=tuning,
                    )
                )

    # -- 3: unactioned recommendations ----------------------------------------------------
    for record in records:
        if record.status == "closed" or not record.recommendations.strip():
            continue
        signals.append(
            _signal(
                kind=KIND_MAINTENANCE_ANOMALY,
                snapshot=snapshot,
                description=(
                    f"Work order {record.record_id} on {snapshot.tag} is still {record.status} "
                    f"with an outstanding recommendation from {format_date(record.performed_on)}"
                ),
                strength=tuning.unactioned_recommendation_strength,
                sources=_record_sources(record, snapshot, document_types=(DocumentType.WORK_ORDER,)),
                method="exact",
                observed_at=as_datetime(record.performed_on) or snapshot.now,
                data={
                    "pattern": "unactioned_recommendation",
                    "record_id": record.record_id,
                    "status": record.status,
                    "last_seen": record.performed_on.isoformat(),
                    "quote": record.recommendations[:220],
                },
                tuning=tuning,
            )
        )
        if record.findings.strip() and not any(
            r.data.get("pattern") == "anomaly_vocabulary" and record.record_id in r.data.get("record_ids", [])
            for r in signals
        ):
            anomalous.append((record, f"{record.findings} {record.recommendations}", []))

    # -- precursor matching ---------------------------------------------------------------
    signals.extend(
        _detect_precursor_matches(snapshot, anomalous, settings=settings, tuning=tuning)
    )
    return signals


def _detect_precursor_matches(
    snapshot: AssetSnapshot,
    anomalous: Sequence[tuple[MaintenanceRecord, str, list[tuple[str, float, int]]]],
    *,
    settings: Settings,
    tuning: DetectorTuning,
) -> list[Signal]:
    """Compare anomalous findings against the precursor text of this asset's past failures."""
    if not anomalous or not snapshot.failures:
        return []
    signals: list[Signal] = []
    seen: set[tuple[str, str]] = set()
    for record, text, _hits in anomalous:
        for event in snapshot.failures:
            precursor = event.precursor_text.strip()
            if not precursor:
                continue
            score = similarity(text, precursor)
            if score < settings.precursor_similarity_threshold:
                continue
            key = (record.record_id, event.event_id)
            if key in seen:
                continue
            seen.add(key)
            consequence = _consequence_clause(event)
            signals.append(
                _signal(
                    kind=KIND_PRECURSOR_MATCH,
                    snapshot=snapshot,
                    description=(
                        f"Findings recorded on {format_date(record.performed_on)} read {score * 100:.0f}% the same "
                        f"as the symptoms logged before the {event.failure_mode} on {format_date(event.occurred_on)}"
                        f"{consequence}"
                    ),
                    strength=score,
                    sources=[
                        *_record_sources(record, snapshot, document_types=(DocumentType.WORK_ORDER, DocumentType.INSPECTION_REPORT)),
                        *_record_sources(event, snapshot, document_types=(DocumentType.INCIDENT_REPORT, DocumentType.ROOT_CAUSE_ANALYSIS)),
                    ],
                    method="semantic",
                    observed_at=as_datetime(record.performed_on) or snapshot.now,
                    data={
                        "pattern": "precursor_match",
                        "similarity": round(score, 4),
                        "threshold": settings.precursor_similarity_threshold,
                        "record_id": record.record_id,
                        "finding_date": record.performed_on.isoformat(),
                        "event_id": event.event_id,
                        "failure_mode": event.failure_mode,
                        "failure_date": event.occurred_on.isoformat(),
                        "downtime_hours": event.downtime_hours,
                        "cost_inr": event.cost_inr,
                        "root_cause": event.root_cause,
                        "quote": text[:220],
                        "precursor_quote": precursor[:220],
                    },
                    tuning=tuning,
                )
            )
    signals.sort(key=lambda s: -s.strength)
    return signals


def _consequence_clause(event: FailureEvent) -> str:
    """Render ``, which cost 18 hours of downtime and ₹4,20,000`` when the numbers exist."""
    parts: list[str] = []
    if event.downtime_hours:
        parts.append(f"{event.downtime_hours:.0f} hours of downtime")
    if event.cost_inr:
        parts.append(f"₹{event.cost_inr:,.0f}")
    if not parts:
        return ""
    return f", which cost {' and '.join(parts)}"


def detect_threshold_approach(
    snapshot: AssetSnapshot,
    *,
    settings: Settings,
    tuning: DetectorTuning = DEFAULT_TUNING,
) -> list[Signal]:
    """Readings closing on an OEM limit.

    Fires when ``latest / limit >= settings.oem_threshold_warning_ratio``. Strength ramps linearly
    from the warning ratio (0.45) to the limit itself (1.0), so 85% of limit and 99% of limit do
    not look alike on the operator's screen. ``oem_thresholds`` are upper limits by construction
    (they come out of the manual as "maximum permissible"), so a rising value is the risk direction.
    """
    limits = snapshot.equipment.oem_thresholds
    if not limits:
        return []
    signals: list[Signal] = []
    for parameter, limit in sorted(limits.items()):
        if limit <= 0:
            continue
        latest = snapshot.latest_reading(parameter)
        if latest is None:
            continue
        ratio = latest.value / limit
        if ratio < settings.oem_threshold_warning_ratio:
            continue
        span = max(tuning.threshold_saturation_ratio - settings.oem_threshold_warning_ratio, 1e-6)
        strength = 0.45 + 0.55 * _clamp((ratio - settings.oem_threshold_warning_ratio) / span)
        series = snapshot.readings_for(parameter)
        slope_per_day, _ = linear_trend(
            [
                ((as_datetime(r.measured_at) - snapshot.now).total_seconds() / 86400.0, r.value)  # type: ignore[operator]
                for r in series
                if as_datetime(r.measured_at) is not None
            ]
        )
        days_to_limit: float | None = None
        if slope_per_day > 1e-9 and latest.value < limit:
            days_to_limit = (limit - latest.value) / slope_per_day
        unit = latest.unit or ""
        signals.append(
            _signal(
                kind=KIND_THRESHOLD_APPROACH,
                snapshot=snapshot,
                description=(
                    f"{snapshot.tag} {parameter.replace('_', ' ')} read {latest.value:g}{unit} on "
                    f"{format_date(latest.measured_at)} — {ratio * 100:.0f}% of the OEM limit of {limit:g}{unit}"
                ),
                strength=strength,
                sources=[latest.source] if latest.source else [],
                method="exact" if latest.confidence.method == "exact" else "heuristic",
                observed_at=as_datetime(latest.measured_at) or snapshot.now,
                data={
                    "parameter": parameter,
                    "value": latest.value,
                    "limit": limit,
                    "unit": unit,
                    "ratio": round(ratio, 4),
                    "warning_ratio": settings.oem_threshold_warning_ratio,
                    "headroom": round(limit - latest.value, 4),
                    "measured_at": latest.measured_at.isoformat(),
                    "slope_per_day": round(slope_per_day, 6),
                    "days_to_limit": round(days_to_limit, 1) if days_to_limit is not None else None,
                    "sample_count": len(series),
                    "reading_confidence": latest.confidence.value,
                },
                tuning=tuning,
            )
        )
    return signals


def detect_missing_workorder(
    snapshot: AssetSnapshot,
    *,
    settings: Settings,
    tuning: DetectorTuning = DEFAULT_TUNING,
) -> list[Signal]:
    """Nothing scheduled and nothing open, with the maintenance window already elapsed.

    "Scheduled maintenance" in the model is an ``open`` or ``deferred``
    :class:`~indra.core.models.MaintenanceRecord`. If one exists, the plant has the asset in hand
    and this signal stays silent — which is exactly what makes rule ``threshold_without_workorder``
    meaningful: a reading near the limit *with* an open work order is a plant doing its job.
    """
    pending = [r for r in snapshot.maintenance if r.status in {"open", "deferred"}]
    if pending:
        return []
    completed = sorted(
        (r for r in snapshot.maintenance if r.status == "closed"),
        key=lambda r: r.performed_on,
    )
    last = completed[-1] if completed else None
    elapsed = days_between(snapshot.now, last.performed_on) if last else None
    window = float(settings.maintenance_lookback_days)

    if last is None:
        description = f"No maintenance record of any kind exists for {snapshot.tag}"
        strength = 0.9
        overdue = None
    else:
        if elapsed is None or elapsed < window:
            return []
        overdue = elapsed - window
        description = (
            f"No open or scheduled work order for {snapshot.tag}; the last completed job was "
            f"{format_date(last.performed_on)}, {elapsed:.0f} days ago"
        )
        strength = _clamp(0.5 + 0.5 * _clamp(overdue / max(window, 1.0)))

    return [
        _signal(
            kind=KIND_MISSING_WORKORDER,
            snapshot=snapshot,
            description=description,
            strength=strength,
            sources=_record_sources(last, snapshot, document_types=(DocumentType.WORK_ORDER,)) if last else [],
            method="exact",
            data={
                "open_records": 0,
                "last_maintenance_date": last.performed_on.isoformat() if last else None,
                "days_since_maintenance": round(elapsed, 1) if elapsed is not None else None,
                "window_days": settings.maintenance_lookback_days,
                "days_overdue": round(overdue, 1) if overdue is not None else None,
                "record_count": len(snapshot.maintenance),
            },
            tuning=tuning,
        )
    ]


def detect_alarm_bypass(
    snapshot: AssetSnapshot,
    *,
    settings: Settings,
    tuning: DetectorTuning = DEFAULT_TUNING,
) -> list[Signal]:
    """Alarm-handling language in the shift logs.

    An alarm that was bypassed, overridden, inhibited, silenced, or "acknowledged and cleared" is a
    protective layer that was deliberately removed by a human being. That fact is almost never in a
    work order — it is in the handover note — so this reads shift logs and incident reports inside
    ``settings.shift_log_lookback_days``.

    One occurrence is emitted as one signal per document so the explanation can name the shift and
    quote the words the operator actually wrote.
    """
    passages = snapshot.shift_log_passages(within_days=settings.shift_log_lookback_days)
    if not passages:
        return []

    by_document: dict[str, list[tuple[Passage, list[tuple[str, float, int]]]]] = {}
    for passage in passages:
        hits = _matched_terms(passage.chunk.text, _BYPASS_PATTERNS)
        if hits:
            by_document.setdefault(passage.document.document_id, []).append((passage, hits))

    signals: list[Signal] = []
    for document_id in sorted(by_document):
        entries = by_document[document_id]
        meta = entries[0][0].document
        occurrences = sum(count for _p, hits in entries for _t, _w, count in hits)
        terms = sorted({term for _p, hits in entries for term, _w, _c in hits})
        peak_weight = max(weight for _p, hits in entries for _t, weight, _c in hits)
        decay = recency_weight(
            meta.document_date or meta.ingested_at,
            now=snapshot.now,
            half_life_days=tuning.text_recency_half_life_days,
        )
        strength = (tuning.bypass_base + tuning.bypass_per_occurrence * (occurrences - 1)) * peak_weight
        strength = _clamp(strength) * (0.6 + 0.4 * decay)

        quote_source = entries[0][0]
        quote_pattern = next(
            (pattern for term, pattern, _w in _BYPASS_PATTERNS if term == entries[0][1][0][0]),
            _BYPASS_PATTERNS[0][1],
        )
        signals.append(
            _signal(
                kind=KIND_ALARM_BYPASS,
                snapshot=snapshot,
                description=(
                    f"{meta.title} ({format_date(meta.document_date or meta.ingested_at)}) records the "
                    f"{snapshot.tag} alarm being {'/'.join(terms)} {occurrences} time(s)"
                ),
                strength=strength,
                sources=[p.source(retrieved_via="graph") for p, _h in entries],
                method="heuristic",
                observed_at=as_datetime(meta.document_date) or as_datetime(meta.ingested_at) or snapshot.now,
                data={
                    "document_id": document_id,
                    "document_title": meta.title,
                    "document_type": meta.document_type.value,
                    "document_date": meta.document_date.isoformat() if meta.document_date else None,
                    "terms": terms,
                    "occurrences": occurrences,
                    "recency_weight": round(decay, 4),
                    "quote": _snippet_around(quote_source.chunk.text, quote_pattern),
                },
                tuning=tuning,
            )
        )
    signals.sort(key=lambda s: -s.strength)
    return signals


def detect_fleet_pattern(
    snapshot: AssetSnapshot,
    fleet: FleetIndex,
    *,
    settings: Settings,
    tuning: DetectorTuning = DEFAULT_TUNING,
) -> list[Signal]:
    """The same failure mode recurring across similar assets.

    Fires for every asset of the affected type, including ones that have not failed yet — that is
    the whole point. If three of the five identical pumps have eaten a bearing in eighteen months,
    the other two are not lucky, they are next.
    """
    window_days = float(settings.retirement_horizon_days)
    since = snapshot.now - timedelta(days=window_days)
    incidence = fleet.mode_incidence(snapshot.equipment, since=since)
    if not incidence:
        return []

    signals: list[Signal] = []
    for mode in sorted(incidence):
        events = incidence[mode]
        affected = sorted({e.equipment_tag for e in events})
        if len(affected) < settings.fleet_failure_min_count:
            continue
        strength = _clamp(
            tuning.fleet_base + tuning.fleet_per_extra_asset * (len(affected) - settings.fleet_failure_min_count)
        )
        latest = max(events, key=lambda e: e.occurred_on)
        already_hit = snapshot.tag in affected
        signals.append(
            _signal(
                kind=KIND_FLEET_PATTERN,
                snapshot=snapshot,
                description=(
                    f"'{mode}' has occurred on {len(affected)} {snapshot.equipment.equipment_type} assets "
                    f"({', '.join(affected)}) since {format_date(since)}"
                ),
                strength=strength,
                sources=[s for e in sorted(events, key=lambda e: e.occurred_on, reverse=True)[:3] for s in e.sources],
                method="exact",
                observed_at=as_datetime(latest.occurred_on) or snapshot.now,
                data={
                    "failure_mode": mode,
                    "affected_tags": affected,
                    "affected_count": len(affected),
                    "event_count": len(events),
                    "min_count": settings.fleet_failure_min_count,
                    "window_days": window_days,
                    "equipment_type": snapshot.equipment.equipment_type,
                    "latest_event_date": latest.occurred_on.isoformat(),
                    "latest_event_tag": latest.equipment_tag,
                    "this_asset_affected": already_hit,
                    "total_downtime_hours": round(sum(e.downtime_hours or 0.0 for e in events), 1),
                    "total_cost_inr": round(sum(e.cost_inr or 0.0 for e in events), 2),
                },
                tuning=tuning,
            )
        )
    return signals


def detect_expertise_loss(
    snapshot: AssetSnapshot,
    *,
    settings: Settings,
    tuning: DetectorTuning = DEFAULT_TUNING,
) -> list[Signal]:
    """Retiring experts, and the documentation void that makes their retirement expensive.

    Emits ``expertise_loss`` per retiring expert, and a single ``documentation_void`` when the asset
    has **zero** transferable knowledge documents — no SOP, no RCA, no manual, no inspection report
    and no procedure. Rule ``expertise_loss`` fires on the conjunction, per the charter table.
    """
    horizon = float(settings.retirement_horizon_days)
    signals: list[Signal] = []
    for person in snapshot.experts:
        remaining = days_between(person.retirement_date, snapshot.now)
        if remaining is None or remaining > horizon:
            continue
        proximity = _clamp(1.0 - (max(remaining, 0.0) / horizon)) if horizon > 0 else 1.0
        strength = _clamp(tuning.expertise_floor + (1.0 - tuning.expertise_floor) * proximity)
        signals.append(
            _signal(
                kind=KIND_EXPERTISE_LOSS,
                snapshot=snapshot,
                description=(
                    f"{person.name}"
                    f"{f' ({person.role})' if person.role else ''} retires on "
                    f"{format_date(person.retirement_date)} — {max(remaining, 0):.0f} days — and is "
                    f"recorded as an expert on {snapshot.tag}"
                ),
                strength=strength,
                sources=[],
                method="exact",
                data={
                    "person_id": person.person_id,
                    "person_name": person.name,
                    "role": person.role,
                    "years_experience": person.years_experience,
                    "retirement_date": person.retirement_date.isoformat() if person.retirement_date else None,
                    "days_to_retirement": round(max(remaining, 0.0), 1),
                    "horizon_days": settings.retirement_horizon_days,
                    "documented_contributions": person.documented_contributions,
                    "expertise_tags": list(person.expertise_tags),
                },
                tuning=tuning,
            )
        )

    if signals and snapshot.knowledge_document_count == 0:
        signals.append(
            _signal(
                kind=KIND_DOCUMENTATION_VOID,
                snapshot=snapshot,
                description=(
                    f"{snapshot.tag} has no SOP, no root-cause analysis, no OEM manual and no inspection "
                    f"report in the system — {snapshot.document_count} document(s) mention it at all"
                ),
                strength=1.0,
                sources=[],
                method="exact",
                data={
                    "knowledge_documents": 0,
                    "document_count": snapshot.document_count,
                    "procedure_count": len(snapshot.procedures),
                    "document_types_present": sorted({d.document_type.value for d in snapshot.documents}),
                },
                tuning=tuning,
            )
        )
    return signals


def detect_regulatory_exposure(
    snapshot: AssetSnapshot,
    *,
    settings: Settings,
    tuning: DetectorTuning = DEFAULT_TUNING,
) -> list[Signal]:
    """Compliance deadlines that are near or passed, and obligations with no evidence behind them.

    Sourced entirely from :class:`GapLedger` — gaps arrive on ``Topic.GAP_DETECTED`` from the
    Compliance Agent. Emits ``regulatory_exposure`` for the deadline and ``evidence_void`` when the
    gap's status says nothing acceptable is on file; rule ``regulatory_exposure`` needs both.
    """
    if not snapshot.gaps:
        return []
    warning_days = float(settings.compliance_deadline_warning_days)
    signals: list[Signal] = []
    for gap in snapshot.gaps:
        remaining = days_between(gap.deadline, snapshot.now)
        if remaining is None:
            # No deadline published: an undated obligation is still exposure, but not urgency.
            in_window = gap.severity.rank >= Severity.HIGH.rank
            strength = 0.5
        else:
            in_window = remaining <= warning_days
            strength = (
                tuning.overdue_strength if remaining <= 0
                else _clamp(0.4 + 0.6 * (1.0 - remaining / max(warning_days, 1.0)))
            )
        if not in_window:
            continue
        overdue = remaining is not None and remaining < 0
        signals.append(
            _signal(
                kind=KIND_REGULATORY_EXPOSURE,
                snapshot=snapshot,
                description=(
                    f"{gap.regulation} {gap.clause} for {snapshot.tag} "
                    + (
                        f"was due {format_date(gap.deadline)} and is {abs(remaining):.0f} days overdue"
                        if overdue and remaining is not None
                        else f"falls due {format_date(gap.deadline)}"
                        if remaining is not None
                        else "has no recorded deadline"
                    )
                ),
                strength=strength,
                sources=[],
                method="exact",
                data={
                    "gap_id": gap.gap_id,
                    "regulation": gap.regulation,
                    "clause": gap.clause,
                    "status": gap.status,
                    "gap_severity": gap.severity.value,
                    "deadline": gap.deadline.isoformat() if gap.deadline else None,
                    "days_to_deadline": round(remaining, 1) if remaining is not None else None,
                    "overdue": overdue,
                    "warning_days": settings.compliance_deadline_warning_days,
                },
                tuning=tuning,
            )
        )
        if gap.is_evidence_void:
            signals.append(
                _signal(
                    kind=KIND_EVIDENCE_VOID,
                    snapshot=snapshot,
                    description=(
                        f"No acceptable evidence is on file for {gap.regulation} {gap.clause} on "
                        f"{snapshot.tag} (status: {gap.status})"
                    ),
                    strength=0.9 if gap.status.lower() == "missing" else 0.7,
                    sources=[],
                    method="exact",
                    data={
                        "gap_id": gap.gap_id,
                        "regulation": gap.regulation,
                        "clause": gap.clause,
                        "status": gap.status,
                        "deadline": gap.deadline.isoformat() if gap.deadline else None,
                    },
                    tuning=tuning,
                )
            )
    return signals


def detect_all(
    snapshot: AssetSnapshot,
    fleet: FleetIndex,
    *,
    settings: Settings,
    tuning: DetectorTuning = DEFAULT_TUNING,
) -> list[Signal]:
    """Run every detector over one snapshot.

    Pure and synchronous. The service offloads this to a worker thread because it is CPU work
    (regex, fuzzy matching, least squares) sitting on an async request path.
    """
    signals: list[Signal] = []
    signals.extend(detect_maintenance_anomaly(snapshot, settings=settings, tuning=tuning))
    signals.extend(detect_threshold_approach(snapshot, settings=settings, tuning=tuning))
    signals.extend(detect_missing_workorder(snapshot, settings=settings, tuning=tuning))
    signals.extend(detect_alarm_bypass(snapshot, settings=settings, tuning=tuning))
    signals.extend(detect_fleet_pattern(snapshot, fleet, settings=settings, tuning=tuning))
    signals.extend(detect_expertise_loss(snapshot, settings=settings, tuning=tuning))
    signals.extend(detect_regulatory_exposure(snapshot, settings=settings, tuning=tuning))
    logger.debug(
        "signals detected",
        extra={
            "equipment_tag": snapshot.tag,
            "signal_count": len(signals),
            "kinds": sorted({s.kind for s in signals}),
        },
    )
    return signals


CRITICALITY_RANK: Final[Mapping[Criticality, int]] = {
    Criticality.A: 3,
    Criticality.B: 2,
    Criticality.C: 1,
}
"""Numeric criticality used by scoring and knowledge-cliff weighting."""


__all__ = [
    "ALL_SIGNAL_KINDS",
    "AssetSnapshot",
    "CRITICALITY_RANK",
    "ComplianceGapRecord",
    "DEFAULT_TUNING",
    "DerivationMethod",
    "DetectorTuning",
    "FleetIndex",
    "GapLedger",
    "KIND_ALARM_BYPASS",
    "KIND_DOCUMENTATION_VOID",
    "KIND_EVIDENCE_VOID",
    "KIND_EXPERTISE_LOSS",
    "KIND_FLEET_PATTERN",
    "KIND_MAINTENANCE_ANOMALY",
    "KIND_MISSING_WORKORDER",
    "KIND_PRECURSOR_MATCH",
    "KIND_REGULATORY_EXPOSURE",
    "KIND_THRESHOLD_APPROACH",
    "KNOWLEDGE_DOCUMENT_TYPES",
    "Passage",
    "SHIFT_LOG_TYPES",
    "SignalCollector",
    "SignalKind",
    "as_datetime",
    "days_between",
    "dedupe_sources",
    "detect_alarm_bypass",
    "detect_all",
    "detect_expertise_loss",
    "detect_fleet_pattern",
    "detect_maintenance_anomaly",
    "detect_missing_workorder",
    "detect_regulatory_exposure",
    "detect_threshold_approach",
    "format_date",
    "guarded_read",
    "linear_trend",
    "recency_weight",
    "similarity",
]
