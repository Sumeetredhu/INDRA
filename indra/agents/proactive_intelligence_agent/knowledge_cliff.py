"""Knowledge-cliff scoring: how much irreplaceable knowledge walks out with one retirement.

The number this module produces (0–100 per asset) is the one a plant manager is asked to spend
money on, so it is built to be argued with. :attr:`KnowledgeCliffScore.factors` publishes every
term that went into the total — the four the charter names, plus the raw observations each was
derived from — and :attr:`KnowledgeCliffScore.rationale` says the same thing in a sentence a
non-numerate reader can check.

The four factors, and why each one is weighted where it is:

``retirement_pressure`` (35)
    Retiring-expert *count* and *proximity*. Proximity is normalised against
    ``settings.retirement_horizon_days``, so "retires in three weeks" and "retires in two years"
    do not score alike. Count saturates (``1 - 0.5ⁿ``): the second expert leaving adds less than
    the first because the first already took the unique knowledge with them.

``documentation_deficit`` (30)
    The denominator of institutional memory. Counted from ``graph.documents_for_tag`` — every
    document that mentions the asset — but weighted 70/30 towards *knowledge-bearing* documents
    (SOP, RCA, OEM manual, inspection report, and ``Procedure`` nodes). Forty work orders are a
    history of what was done, not an explanation of how to do it.

``criticality`` (20)
    Consequence class. The same knowledge gap on a criticality-A asset costs production or people.

``knowledge_concentration`` (15)
    One expert holding everything is worse than three sharing it. A Herfindahl index over the
    recorded holders, blended with the share of that expertise which is actually leaving. Three
    engineers who all know P-101, one retiring, is a manageable handover; one engineer who knows it
    alone is a cliff.

**CRITICAL is asserted, not computed**, when a criticality-A asset has a retiring expert and zero
knowledge documents: that combination is the charter's named failure case, and it floors the score
at ``settings.knowledge_cliff_critical_score`` regardless of what the weighted terms happened to
produce.

Interview questions are generated from **real graph gaps** — failure modes with no root cause on
file, OEM limits with no manual, maintenance the plant repeats with no procedure written down.
A question that could have been asked about any asset in any plant is worthless in a capture
session, so every question here names a tag, a mode, a date or a number pulled from the graph.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Final, Mapping, Sequence

from indra.core.config import Settings
from indra.core.exceptions import IndraError
from indra.core.logging import get_logger
from indra.core.models import (
    ConditionReading,
    Criticality,
    DocumentMeta,
    DocumentType,
    Equipment,
    FailureEvent,
    KnowledgeCliffScore,
    MaintenanceRecord,
    Person,
    Procedure,
    Severity,
    utcnow,
)
from indra.agents.proactive_intelligence_agent.signals import (
    KNOWLEDGE_DOCUMENT_TYPES,
    days_between,
    format_date,
    guarded_read,
)

logger = get_logger(__name__)


# ======================================================================================
# Coefficients
# ======================================================================================
#
# These are model calibration, not deployment configuration: they describe how the four charter
# factors trade off against one another, which is a property of the scoring model rather than of
# the environment. ``Settings`` owns the operational knobs this module consults
# (``retirement_horizon_days``, ``knowledge_cliff_critical_score``) and every one of them is read
# from settings; ``indra/core`` is read-only for this agent, so the rest live here — named, frozen,
# and overridable in a test rather than inlined at the call site.

WEIGHT_RETIREMENT: Final[float] = 35.0
WEIGHT_DOCUMENTATION: Final[float] = 30.0
WEIGHT_CRITICALITY: Final[float] = 20.0
WEIGHT_CONCENTRATION: Final[float] = 15.0

#: Consequence multiplier per criticality class, applied to :data:`WEIGHT_CRITICALITY`.
CRITICALITY_WEIGHT: Final[Mapping[Criticality, float]] = {
    Criticality.A: 1.0,
    Criticality.B: 0.6,
    Criticality.C: 0.3,
}

#: Knowledge documents at which documentation risk is considered fully retired. Three — an SOP, a
#: manual and one root-cause analysis — is the minimum from which a competent engineer can work.
KNOWLEDGE_DOCUMENT_TARGET: Final[int] = 3
#: Total mentions at which the "somebody wrote *something* down" term saturates.
MENTION_TARGET: Final[int] = 6
#: Split of the documentation term between knowledge-bearing documents and bare mentions.
KNOWLEDGE_DOCUMENT_SHARE: Final[float] = 0.7
MENTION_SHARE: Final[float] = 0.3

#: A retirement at the far edge of the horizon still carries this fraction of the retirement term.
#: Two years' notice reduces urgency; it does not remove the exposure.
PROXIMITY_FLOOR: Final[float] = 0.35

#: Severity bands, expressed as fractions of ``settings.knowledge_cliff_critical_score`` so the one
#: configured number moves the whole ladder together.
HIGH_BAND_RATIO: Final[float] = 0.75
WARNING_BAND_RATIO: Final[float] = 0.5
LOW_BAND_RATIO: Final[float] = 0.25

#: Ceiling on documents pulled per asset for the documentation denominator.
DOCUMENT_SCAN_LIMIT: Final[int] = 200
#: Ceiling on questions returned from one capture-session plan.
MAX_INTERVIEW_QUESTIONS: Final[int] = 8
#: How many LLM-suggested questions may be appended to the deterministic set.
MAX_LLM_QUESTIONS: Final[int] = 3
#: Minimum length for an LLM-suggested question to be taken seriously.
MIN_QUESTION_CHARS: Final[int] = 40


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


# ======================================================================================
# Inputs and assessment
# ======================================================================================


@dataclass(frozen=True, slots=True)
class CliffInputs:
    """Everything the score is computed from. Frozen, so scoring is a pure function of it."""

    equipment: Equipment
    now: datetime
    settings: Settings
    #: Everyone whose ``expertise_tags`` name this asset, retiring or not.
    holders: tuple[Person, ...] = ()
    #: Holders leaving inside ``settings.retirement_horizon_days`` (or already gone).
    retiring: tuple[Person, ...] = ()
    #: Every document that mentions the asset (``graph.documents_for_tag``).
    documents: tuple[DocumentMeta, ...] = ()
    procedures: tuple[Procedure, ...] = ()

    @property
    def tag(self) -> str:
        return self.equipment.tag

    @property
    def document_count(self) -> int:
        return len({d.document_id for d in self.documents})

    @property
    def knowledge_document_count(self) -> int:
        """SOPs, RCAs, manuals, inspection reports and procedure nodes — transferable knowledge."""
        documents = {d.document_id for d in self.documents if d.document_type in KNOWLEDGE_DOCUMENT_TYPES}
        return len(documents) + len(self.procedures)

    @property
    def days_to_first_retirement(self) -> int | None:
        """Days until the first holder leaves. Negative when somebody has already gone."""
        remaining = [
            days_between(person.retirement_date, self.now)
            for person in self.retiring
            if person.retirement_date is not None
        ]
        values = [r for r in remaining if r is not None]
        return int(round(min(values))) if values else None


@dataclass(frozen=True, slots=True)
class CliffAssessment:
    """The composed score, its factor breakdown, its severity and its plain-language account."""

    score: float
    severity: Severity
    factors: dict[str, float]
    rationale: str


def severity_for(score: float, *, settings: Settings) -> Severity:
    """Map a 0–100 score onto a severity using the configured critical threshold."""
    critical = float(settings.knowledge_cliff_critical_score)
    if score >= critical:
        return Severity.CRITICAL
    if score >= critical * HIGH_BAND_RATIO:
        return Severity.HIGH
    if score >= critical * WARNING_BAND_RATIO:
        return Severity.WARNING
    if score >= critical * LOW_BAND_RATIO:
        return Severity.LOW
    return Severity.INFO


def retirement_pressure(inputs: CliffInputs) -> tuple[float, float, float]:
    """Return ``(points, count_factor, proximity)`` for the retiring-expert term.

    ``count_factor`` saturates at ``1 - 0.5ⁿ`` — the first departure carries half the term on its
    own, because unique knowledge leaves with the first person to hold it.
    """
    if not inputs.retiring:
        return 0.0, 0.0, 0.0
    horizon = float(inputs.settings.retirement_horizon_days)
    count_factor = 1.0 - 0.5 ** len(inputs.retiring)
    proximities: list[float] = []
    for person in inputs.retiring:
        remaining = days_between(person.retirement_date, inputs.now)
        if remaining is None:
            proximities.append(0.5)  # recorded as retiring, date unknown: middling urgency
        elif horizon <= 0 or remaining <= 0:
            proximities.append(1.0)
        else:
            proximities.append(_clamp(1.0 - remaining / horizon))
    proximity = max(proximities)
    urgency = PROXIMITY_FLOOR + (1.0 - PROXIMITY_FLOOR) * proximity
    return WEIGHT_RETIREMENT * count_factor * urgency, count_factor, proximity


def documentation_deficit(inputs: CliffInputs) -> tuple[float, float]:
    """Return ``(points, deficit)`` for the documentation term.

    Deficit is ``1 − coverage``; coverage blends knowledge-bearing documents (70%) with bare
    mentions (30%), each saturating at its own target.
    """
    knowledge = _clamp(inputs.knowledge_document_count / float(KNOWLEDGE_DOCUMENT_TARGET))
    mentions = _clamp(inputs.document_count / float(MENTION_TARGET))
    coverage = KNOWLEDGE_DOCUMENT_SHARE * knowledge + MENTION_SHARE * mentions
    deficit = _clamp(1.0 - coverage)
    return WEIGHT_DOCUMENTATION * deficit, deficit


def knowledge_concentration(inputs: CliffInputs) -> tuple[float, float, float]:
    """Return ``(points, hhi, departing_share)`` for the concentration term.

    Shares are proportional to recorded experience, because a thirty-year fitter and a graduate
    who both carry the tag are not equal holders of the same knowledge. The Herfindahl index is
    1.0 for a sole holder and ``1/n`` for ``n`` equal ones; blending it with the share that is
    actually leaving is what separates "three know it, one retires" from "one knows it, and he is
    the one retiring".
    """
    if not inputs.holders:
        return 0.0, 0.0, 0.0
    weights = {
        person.person_id: max(float(person.years_experience or 1.0), 0.1)
        for person in inputs.holders
    }
    total = sum(weights.values())
    if total <= 0:
        return 0.0, 0.0, 0.0
    shares = {pid: weight / total for pid, weight in weights.items()}
    hhi = sum(share * share for share in shares.values())
    departing = {p.person_id for p in inputs.retiring}
    departing_share = sum(share for pid, share in shares.items() if pid in departing)
    concentration = _clamp(0.5 * hhi + 0.5 * departing_share)
    return WEIGHT_CONCENTRATION * concentration, hhi, departing_share


def compute_cliff(inputs: CliffInputs) -> CliffAssessment:
    """Compose the 0–100 knowledge-cliff score. Pure: no clock, no store, no network."""
    equipment = inputs.equipment
    retirement_points, count_factor, proximity = retirement_pressure(inputs)
    documentation_points, deficit = documentation_deficit(inputs)
    criticality_multiplier = CRITICALITY_WEIGHT.get(equipment.criticality, CRITICALITY_WEIGHT[Criticality.C])
    criticality_points = WEIGHT_CRITICALITY * criticality_multiplier
    concentration_points, hhi, departing_share = knowledge_concentration(inputs)

    subtotal = retirement_points + documentation_points + criticality_points + concentration_points

    critical_case = (
        equipment.criticality is Criticality.A
        and bool(inputs.retiring)
        and inputs.knowledge_document_count == 0
    )
    floor = float(inputs.settings.knowledge_cliff_critical_score) if critical_case else 0.0
    score = round(_clamp(max(subtotal, floor), 0.0, 100.0), 1)

    factors: dict[str, float] = {
        "retirement_pressure": round(retirement_points, 2),
        "documentation_deficit": round(documentation_points, 2),
        "criticality": round(criticality_points, 2),
        "knowledge_concentration": round(concentration_points, 2),
        "critical_floor_applied": round(floor, 2),
        "weighted_subtotal": round(subtotal, 2),
        "total": score,
        "observed_retiring_experts": float(len(inputs.retiring)),
        "observed_knowledge_holders": float(len(inputs.holders)),
        "observed_documents": float(inputs.document_count),
        "observed_knowledge_documents": float(inputs.knowledge_document_count),
        "observed_retirement_proximity": round(proximity, 4),
        "observed_expert_count_factor": round(count_factor, 4),
        "observed_documentation_coverage": round(1.0 - deficit, 4),
        "observed_concentration_hhi": round(hhi, 4),
        "observed_departing_share": round(departing_share, 4),
    }
    days = inputs.days_to_first_retirement
    if days is not None:
        factors["observed_days_to_first_retirement"] = float(days)

    assessment = CliffAssessment(
        score=score,
        severity=severity_for(score, settings=inputs.settings),
        factors=factors,
        rationale=_rationale(inputs, factors, critical_case=critical_case),
    )
    logger.debug(
        "knowledge cliff scored",
        extra={
            "equipment_tag": inputs.tag,
            "score": score,
            "severity": assessment.severity.value,
            "retiring_experts": len(inputs.retiring),
            "knowledge_documents": inputs.knowledge_document_count,
        },
    )
    return assessment


def _rationale(inputs: CliffInputs, factors: Mapping[str, float], *, critical_case: bool) -> str:
    """Render the arithmetic as prose. Every number below appears in ``factors`` as well."""
    equipment = inputs.equipment
    parts: list[str] = []

    if inputs.retiring:
        names = ", ".join(p.name for p in inputs.retiring[:3])
        first = inputs.retiring[0]
        days = inputs.days_to_first_retirement
        when = format_date(first.retirement_date, fallback="an unrecorded date")
        if days is not None and days < 0:
            timing = f"{first.name} already left on {when}, {abs(days)} days ago"
        elif days is not None:
            timing = f"{first.name} retires on {when}, {days} days from now"
        else:
            timing = f"{first.name} is flagged as retiring but carries no recorded date"
        parts.append(
            f"{len(inputs.retiring)} recorded expert(s) on {equipment.tag} are leaving ({names}); "
            f"{timing}. Retirement pressure contributes "
            f"{factors['retirement_pressure']:.1f} of {WEIGHT_RETIREMENT:.0f} points."
        )
    else:
        parts.append(
            f"No individual on file is recorded as retiring with {equipment.tag} expertise, so no "
            f"departure is currently scheduled to remove knowledge from this asset "
            f"(0 of {WEIGHT_RETIREMENT:.0f} points)."
        )

    if inputs.knowledge_document_count == 0:
        parts.append(
            f"There is no SOP, no root-cause analysis, no OEM manual and no inspection report for "
            f"{equipment.tag}: {inputs.document_count} document(s) mention it at all and none of "
            f"them explain how to run, tune or repair it "
            f"({factors['documentation_deficit']:.1f} of {WEIGHT_DOCUMENTATION:.0f} points)."
        )
    else:
        parts.append(
            f"{inputs.knowledge_document_count} knowledge document(s) and "
            f"{inputs.document_count} total mention(s) exist for {equipment.tag}, covering "
            f"{factors['observed_documentation_coverage'] * 100:.0f}% of what a handover needs "
            f"({factors['documentation_deficit']:.1f} of {WEIGHT_DOCUMENTATION:.0f} points)."
        )

    parts.append(
        f"It is a criticality-{equipment.criticality.value} "
        f"{equipment.equipment_type or 'asset'}"
        f"{f' in {equipment.location}' if equipment.location else ''} "
        f"({factors['criticality']:.1f} of {WEIGHT_CRITICALITY:.0f} points)."
    )

    if len(inputs.holders) == 1:
        parts.append(
            f"Exactly one person is recorded as holding this knowledge, so there is nobody to fall "
            f"back on ({factors['knowledge_concentration']:.1f} of {WEIGHT_CONCENTRATION:.0f} points)."
        )
    elif inputs.holders:
        parts.append(
            f"{len(inputs.holders)} people are recorded as holding this knowledge and "
            f"{factors['observed_departing_share'] * 100:.0f}% of that experience is leaving "
            f"({factors['knowledge_concentration']:.1f} of {WEIGHT_CONCENTRATION:.0f} points)."
        )
    else:
        parts.append(
            "Nobody is recorded as holding knowledge of this asset, which means no retirement will "
            "remove it — but it also means the plant cannot name who to ask (0 points)."
        )

    if critical_case:
        parts.append(
            f"Criticality-A equipment with a retiring expert and zero documented knowledge is the "
            f"named critical case, so the score is floored at "
            f"{inputs.settings.knowledge_cliff_critical_score:.0f}."
        )
    return " ".join(parts)


# ======================================================================================
# Graph gaps → interview questions
# ======================================================================================


@dataclass(frozen=True, slots=True)
class KnowledgeGaps:
    """Concrete holes in the graph, each of which becomes one targeted interview question."""

    equipment: Equipment
    #: ``(failure_mode, occurred_on)`` for modes with no root cause and no RCA document on file.
    failure_modes_without_rca: tuple[tuple[str, date], ...] = ()
    #: Repeated maintenance work with no procedure written for it: ``(work description, count)``.
    repeated_work_without_procedure: tuple[tuple[str, int], ...] = ()
    #: Procedures that exist as a title with no steps behind them.
    empty_procedures: tuple[str, ...] = ()
    #: OEM limits with no manual in the system: ``(parameter, limit)``.
    limits_without_manual: tuple[tuple[str, float], ...] = ()
    #: Monitored parameters with readings but no documented interpretation: ``(parameter, value)``.
    parameters_without_guidance: tuple[tuple[str, float], ...] = ()
    #: Unresolved recommendations still open on the asset.
    open_recommendations: tuple[str, ...] = ()
    has_any_procedure: bool = False
    has_oem_manual: bool = False

    @property
    def is_empty(self) -> bool:
        return not any(
            (
                self.failure_modes_without_rca,
                self.repeated_work_without_procedure,
                self.empty_procedures,
                self.limits_without_manual,
                self.parameters_without_guidance,
                self.open_recommendations,
            )
        )


def build_interview_questions(
    gaps: KnowledgeGaps,
    *,
    person: Person | None = None,
    limit: int = MAX_INTERVIEW_QUESTIONS,
) -> list[str]:
    """Turn graph gaps into capture-session questions.

    Every question names something real — a mode, a date, a limit, a parameter, an open
    recommendation. Nothing here is answerable with "yes"; a capture session that produces yes/no
    answers has captured nothing.
    """
    tag = gaps.equipment.tag
    who = person.name.split(" ")[0] if person and person.name else None
    lead = f"{who}, " if who else ""
    questions: list[str] = []

    for mode, occurred in gaps.failure_modes_without_rca:
        questions.append(
            f"{lead}walk me through the {mode} on {tag} in {format_date(occurred)}. "
            f"No root cause was ever written down — what did you find when you opened it up, "
            f"and what do you believe actually caused it?"
        )

    for description, count in gaps.repeated_work_without_procedure:
        questions.append(
            f"{tag} has had '{description}' done {count} times and there is no procedure on file "
            f"for it. Talk me through how you do that job step by step — including the parts you "
            f"would never find in a manual."
        )

    for title in gaps.empty_procedures:
        questions.append(
            f"The procedure '{title}' exists for {tag} as a title with no steps behind it. "
            f"What are the actual steps, and where in that sequence do people usually go wrong?"
        )

    for parameter, value in gaps.limits_without_manual:
        readable = parameter.replace("_", " ")
        questions.append(
            f"The OEM limit for {tag} {readable} is {value:g} and there is no manual in the system "
            f"to explain it. At what value do you personally start worrying, and what do you check "
            f"first when it gets there?"
        )

    for parameter, value in gaps.parameters_without_guidance:
        readable = parameter.replace("_", " ")
        questions.append(
            f"{tag} {readable} is currently reading {value:g}. What does that number mean to you "
            f"in the context of this machine, and what would you look at before deciding whether "
            f"it matters?"
        )

    for recommendation in gaps.open_recommendations:
        questions.append(
            f"There is an open recommendation on {tag}: \"{recommendation}\". What has to happen "
            f"for that to be closed out properly, and what goes wrong if the next person does it "
            f"the obvious way instead?"
        )

    if not gaps.has_any_procedure:
        questions.append(
            f"If you had to hand {tag} over tomorrow with nothing written down, what are the three "
            f"things about this specific machine a junior engineer would get wrong in the first month?"
        )

    questions.append(
        f"What early signs on {tag} tell you something is developing before any instrument shows "
        f"it — the noise, the smell, the feel — and what do you do the moment you notice one?"
    )
    if person is not None and person.expertise_tags:
        others = [t for t in person.expertise_tags if t.strip().upper() != tag.upper()]
        if others:
            questions.append(
                f"{lead}you are also the recorded expert on {', '.join(others[:3])}. "
                f"What do you know about how {tag} interacts with those that is not obvious from "
                f"the drawings?"
            )

    deduped = list(dict.fromkeys(q.strip() for q in questions if q.strip()))
    return deduped[:limit]


# ======================================================================================
# Analyzer — the only part of this module that touches a store
# ======================================================================================


class KnowledgeCliffAnalyzer:
    """Gathers the graph facts behind the score, composes it, and plans the capture session.

    Store access is funnelled through :func:`~indra.agents.proactive_intelligence_agent.signals.guarded_read`,
    so a dead graph backend costs a degraded score and a warning line rather than a failed scan.
    """

    __slots__ = ("_graph", "_llm", "_settings")

    def __init__(self, *, graph: Any, settings: Settings, llm: Any | None = None) -> None:
        # ``graph`` is ``indra.core.contracts.GraphStore`` and ``llm`` an ``LLMRouter``; both are
        # typed ``Any`` here only to avoid a runtime import cycle through ``contracts`` → ``models``.
        self._graph = graph
        self._settings = settings
        self._llm = llm

    # -- gathering --------------------------------------------------------------------

    async def gather(self, equipment: Equipment, *, now: datetime | None = None) -> CliffInputs:
        """Read every fact the score depends on for one asset."""
        moment = now or utcnow()
        documents, procedures, people = await asyncio.gather(
            guarded_read(
                self._graph.documents_for_tag(equipment.tag, limit=DOCUMENT_SCAN_LIMIT),
                label="documents_for_tag", default=[], context={"equipment_tag": equipment.tag},
            ),
            guarded_read(
                self._graph.procedures_for(equipment.tag),
                label="procedures_for", default=[], context={"equipment_tag": equipment.tag},
            ),
            guarded_read(
                self._graph.get_people(),
                label="get_people", default=[], context={"equipment_tag": equipment.tag},
            ),
        )
        holders = self._holders(people, equipment.tag)
        retiring = self._retiring(holders, now=moment)
        return CliffInputs(
            equipment=equipment,
            now=moment,
            settings=self._settings,
            holders=holders,
            retiring=retiring,
            documents=tuple(documents),
            procedures=tuple(procedures),
        )

    def _holders(self, people: Sequence[Person], tag: str) -> tuple[Person, ...]:
        wanted = tag.strip().upper()
        return tuple(
            sorted(
                (p for p in people if any(t.strip().upper() == wanted for t in p.expertise_tags)),
                key=lambda p: (p.retirement_date or date.max, p.name),
            )
        )

    def _retiring(self, holders: Sequence[Person], *, now: datetime) -> tuple[Person, ...]:
        """Holders inside the retirement horizon, including any whose date has already passed."""
        horizon = float(self._settings.retirement_horizon_days)
        selected: list[Person] = []
        for person in holders:
            if person.retirement_date is None:
                continue
            remaining = days_between(person.retirement_date, now)
            if remaining is None or remaining <= horizon:
                selected.append(person)
        return tuple(selected)

    # -- scoring ----------------------------------------------------------------------

    async def score(
        self,
        equipment: Equipment,
        *,
        now: datetime | None = None,
        with_questions: bool = True,
    ) -> KnowledgeCliffScore:
        """Score one asset and, when the score warrants it, plan the capture session."""
        inputs = await self.gather(equipment, now=now)
        assessment = compute_cliff(inputs)
        questions: list[str] = []
        if with_questions and assessment.severity.rank >= Severity.WARNING.rank:
            person = inputs.retiring[0] if inputs.retiring else None
            questions = await self.interview_questions(equipment, person=person)
        return KnowledgeCliffScore(
            equipment_tag=equipment.tag,
            score=assessment.score,
            severity=assessment.severity,
            retiring_experts=list(inputs.retiring),
            document_count=inputs.document_count,
            criticality=equipment.criticality,
            days_to_first_retirement=inputs.days_to_first_retirement,
            factors=assessment.factors,
            interview_questions=questions,
            rationale=assessment.rationale,
        )

    async def score_many(
        self,
        equipment: Sequence[Equipment],
        *,
        now: datetime | None = None,
        with_questions: bool = True,
        concurrency: int | None = None,
    ) -> list[KnowledgeCliffScore]:
        """Score a fleet, worst first. One slow asset never blocks the rest."""
        if not equipment:
            return []
        moment = now or utcnow()
        semaphore = asyncio.Semaphore(max(1, concurrency or self._settings.ingestion_concurrency))

        async def _one(asset: Equipment) -> KnowledgeCliffScore | None:
            async with semaphore:
                try:
                    return await self.score(asset, now=moment, with_questions=with_questions)
                except asyncio.CancelledError:
                    raise
                except IndraError as exc:
                    logger.warning(
                        "knowledge cliff scoring degraded",
                        extra={"equipment_tag": asset.tag, "error": exc.message},
                    )
                except Exception as exc:  # noqa: BLE001 - one bad asset must not kill the sweep
                    logger.error(
                        "knowledge cliff scoring failed",
                        extra={"equipment_tag": asset.tag, "error": f"{type(exc).__name__}: {exc}"},
                    )
                return None

        results = await asyncio.gather(*(_one(asset) for asset in equipment))
        scores = [score for score in results if score is not None]
        scores.sort(key=lambda s: (-s.score, s.equipment_tag))
        return scores

    # -- interview planning -----------------------------------------------------------

    async def gaps(self, equipment: Equipment, *, now: datetime | None = None) -> KnowledgeGaps:
        """Find the real holes in the graph for this asset."""
        moment = now or utcnow()
        lookback = (moment.date().replace(year=moment.year - 3)
                    if moment.year > 3 else moment.date())
        failures, maintenance, procedures, documents, readings = await asyncio.gather(
            guarded_read(
                self._graph.failure_history(equipment.tag),
                label="failure_history", default=[], context={"equipment_tag": equipment.tag},
            ),
            guarded_read(
                self._graph.maintenance_history(equipment.tag, since=lookback),
                label="maintenance_history", default=[], context={"equipment_tag": equipment.tag},
            ),
            guarded_read(
                self._graph.procedures_for(equipment.tag),
                label="procedures_for", default=[], context={"equipment_tag": equipment.tag},
            ),
            guarded_read(
                self._graph.documents_for_tag(equipment.tag, limit=DOCUMENT_SCAN_LIMIT),
                label="documents_for_tag", default=[], context={"equipment_tag": equipment.tag},
            ),
            guarded_read(
                self._graph.readings_for(equipment.tag),
                label="readings_for", default=[], context={"equipment_tag": equipment.tag},
            ),
        )
        return self._build_gaps(
            equipment,
            failures=failures,
            maintenance=maintenance,
            procedures=procedures,
            documents=documents,
            readings=readings,
        )

    @staticmethod
    def _build_gaps(
        equipment: Equipment,
        *,
        failures: Sequence[FailureEvent],
        maintenance: Sequence[MaintenanceRecord],
        procedures: Sequence[Procedure],
        documents: Sequence[DocumentMeta],
        readings: Sequence[ConditionReading],
    ) -> KnowledgeGaps:
        """Pure gap analysis over already-read records."""
        types_present = {d.document_type for d in documents}
        has_rca_document = DocumentType.ROOT_CAUSE_ANALYSIS in types_present
        has_sop_document = DocumentType.SOP in types_present
        has_manual = DocumentType.OEM_MANUAL in types_present

        # -- failure modes with nothing explaining them -----------------------------------
        explained: dict[str, bool] = {}
        latest: dict[str, date] = {}
        for event in failures:
            mode = event.failure_mode.strip()
            if not mode:
                continue
            key = mode.lower()
            explained[key] = explained.get(key, False) or bool((event.root_cause or "").strip())
            if key not in latest or event.occurred_on > latest[key]:
                latest[key] = event.occurred_on
        modes_without_rca = tuple(
            (mode, latest[mode])
            for mode in sorted(latest)
            if not explained.get(mode, False) and not has_rca_document
        )

        # -- work the plant repeats with nothing written down -----------------------------
        counts: dict[str, int] = {}
        for record in maintenance:
            label = (record.findings or record.recommendations or record.record_type).strip()
            label = " ".join(label.split())[:80]
            if not label:
                continue
            counts[label.lower()] = counts.get(label.lower(), 0) + 1
        documented = bool(procedures) or has_sop_document
        repeated = tuple(
            (label, count)
            for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            if count >= 2 and not documented
        )[:2]

        empty_procedures = tuple(
            procedure.title for procedure in procedures if not procedure.steps
        )

        limits_without_manual = (
            tuple(sorted(equipment.oem_thresholds.items()))[:2] if not has_manual else ()
        )

        latest_by_parameter: dict[str, ConditionReading] = {}
        for reading in readings:
            current = latest_by_parameter.get(reading.parameter)
            if current is None or reading.measured_at > current.measured_at:
                latest_by_parameter[reading.parameter] = reading
        parameters_without_guidance = tuple(
            (parameter, latest_by_parameter[parameter].value)
            for parameter in sorted(latest_by_parameter)
            if parameter not in equipment.oem_thresholds
        )[:2]

        open_recommendations = tuple(
            " ".join(record.recommendations.split())[:160]
            for record in sorted(maintenance, key=lambda r: r.performed_on, reverse=True)
            if record.status in {"open", "deferred"} and record.recommendations.strip()
        )[:2]

        return KnowledgeGaps(
            equipment=equipment,
            failure_modes_without_rca=modes_without_rca,
            repeated_work_without_procedure=repeated,
            empty_procedures=empty_procedures,
            limits_without_manual=limits_without_manual,
            parameters_without_guidance=parameters_without_guidance,
            open_recommendations=open_recommendations,
            has_any_procedure=documented,
            has_oem_manual=has_manual,
        )

    async def interview_questions(
        self,
        equipment: Equipment,
        *,
        person: Person | None = None,
        now: datetime | None = None,
        limit: int = MAX_INTERVIEW_QUESTIONS,
    ) -> list[str]:
        """Plan a capture session against this asset's actual gaps."""
        gaps = await self.gaps(equipment, now=now)
        questions = build_interview_questions(gaps, person=person, limit=limit)
        extra = await self._llm_questions(equipment, gaps, person=person)
        for question in extra:
            if len(questions) >= limit:
                break
            if question not in questions:
                questions.append(question)
        logger.info(
            "interview questions generated",
            extra={
                "equipment_tag": equipment.tag,
                "question_count": len(questions),
                "llm_suggested": len(extra),
                "gap_free": gaps.is_empty,
            },
        )
        return questions

    async def _llm_questions(
        self,
        equipment: Equipment,
        gaps: KnowledgeGaps,
        *,
        person: Person | None,
    ) -> list[str]:
        """Optionally deepen the question set with the model. Never required, never trusted blindly.

        Skipped entirely in deterministic mode (tests, recorded demo), and any suggestion that does
        not name the asset or one of its real gaps is discarded — a generic question is exactly what
        this method exists to avoid.
        """
        if self._llm is None or self._settings.deterministic or gaps.is_empty:
            return []
        anchors = {equipment.tag.lower()}
        anchors.update(mode.lower() for mode, _d in gaps.failure_modes_without_rca)
        anchors.update(parameter.lower() for parameter, _v in gaps.limits_without_manual)
        anchors.update(parameter.lower() for parameter, _v in gaps.parameters_without_guidance)
        prompt = (
            f"Asset {equipment.tag} ({equipment.name or equipment.equipment_type}), criticality "
            f"{equipment.criticality.value}. The expert being interviewed is "
            f"{person.name if person else 'the outgoing specialist'}"
            f"{f' ({person.role})' if person and person.role else ''}.\n"
            f"Documented gaps in the plant's records:\n"
            + "\n".join(
                [
                    *(f"- failure mode '{m}' on {d.isoformat()} has no root-cause analysis"
                      for m, d in gaps.failure_modes_without_rca),
                    *(f"- '{label}' has been done {count} times with no procedure on file"
                      for label, count in gaps.repeated_work_without_procedure),
                    *(f"- OEM limit {p}={v:g} has no manual in the system"
                      for p, v in gaps.limits_without_manual),
                    *(f"- parameter {p} reads {v:g} with no documented interpretation"
                      for p, v in gaps.parameters_without_guidance),
                ]
            )
            + "\n\nWrite open-ended knowledge-capture questions that target these specific gaps."
        )
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "questions": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["questions"],
        }
        try:
            payload, provider = await self._llm.generate_json(
                prompt,
                schema=schema,
                system=(
                    "You plan knowledge-capture interviews for industrial plants. Every question "
                    "must be open-ended, must name the specific asset, mode, parameter or number "
                    "given to you, and must be answerable only by someone who has actually worked "
                    "on that machine. Never ask a yes/no question. Never ask something that would "
                    "apply to any plant."
                ),
            )
        except IndraError as exc:
            logger.warning(
                "LLM interview-question enrichment unavailable",
                extra={"equipment_tag": equipment.tag, "error": exc.message},
            )
            return []
        except Exception as exc:  # noqa: BLE001 - enrichment is optional; never fail the plan
            logger.warning(
                "LLM interview-question enrichment raised",
                extra={"equipment_tag": equipment.tag, "error": f"{type(exc).__name__}: {exc}"},
            )
            return []
        if provider == "stub":
            # The deterministic stub cannot write a plant-specific question; its output would be
            # exactly the generic filler this method is meant to keep out.
            return []
        raw = payload.get("questions") if isinstance(payload, Mapping) else None
        if not isinstance(raw, list):
            return []
        accepted: list[str] = []
        for item in raw:
            question = " ".join(str(item).split())
            if len(question) < MIN_QUESTION_CHARS or not question.endswith("?"):
                continue
            if not any(anchor in question.lower() for anchor in anchors):
                continue
            accepted.append(question)
            if len(accepted) >= MAX_LLM_QUESTIONS:
                break
        return accepted


__all__ = [
    "CRITICALITY_WEIGHT",
    "CliffAssessment",
    "CliffInputs",
    "DOCUMENT_SCAN_LIMIT",
    "HIGH_BAND_RATIO",
    "KNOWLEDGE_DOCUMENT_TARGET",
    "KnowledgeCliffAnalyzer",
    "KnowledgeGaps",
    "LOW_BAND_RATIO",
    "MAX_INTERVIEW_QUESTIONS",
    "MENTION_TARGET",
    "PROXIMITY_FLOOR",
    "WARNING_BAND_RATIO",
    "WEIGHT_CONCENTRATION",
    "WEIGHT_CRITICALITY",
    "WEIGHT_DOCUMENTATION",
    "WEIGHT_RETIREMENT",
    "build_interview_questions",
    "compute_cliff",
    "documentation_deficit",
    "knowledge_concentration",
    "retirement_pressure",
    "severity_for",
]
