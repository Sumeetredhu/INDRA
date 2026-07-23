"""Query classification: a deterministic lexical prior, then a model as second opinion.

The order matters and is not an implementation detail. Every provider being down is a normal
operating condition for INDRA (``docs/DECISIONS.md`` D1/D2), and a copilot that cannot route a
question without a network call is a copilot that stops working on the plant floor. So:

1. A weighted keyword/pattern lexicon scores all seven :class:`QueryType` values. This alone
   produces a sane route and a calibrated confidence, with no I/O of any kind.
2. The model is then asked to confirm or override, with the prior handed to it explicitly so it is
   correcting a colleague rather than guessing cold.
3. The model wins only when it is *more* confident than the prior. A confident lexical signal —
   "why did P-101 fail" — is not overturned by a hedging model.

Every failure mode of step 2 (no provider, rate limit, malformed JSON, unknown label) degrades to
the step-1 result with a warning, never to an exception.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Sequence

from indra.core.config import Settings
from indra.core.contracts import LLMRouter
from indra.core.exceptions import IndraError
from indra.core.logging import get_logger
from indra.core.models import Confidence, IndraModel, QueryType
from indra.agents.copilot_agent import prompts

logger = get_logger(__name__)


# --------------------------------------------------------------------------------------
# Lexicon
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class KeywordRule:
    """One lexical cue and the routing evidence it contributes.

    ``weight`` is not a deployment tunable — it is part of the classification algorithm, calibrated
    so that a single decisive phrase ("root cause") outscores two weak ones ("what", "P-101").
    """

    query_type: QueryType
    weight: float
    pattern: re.Pattern[str]
    label: str


def _rule(query_type: QueryType, weight: float, pattern: str, label: str) -> KeywordRule:
    return KeywordRule(
        query_type=query_type,
        weight=weight,
        pattern=re.compile(pattern, re.IGNORECASE),
        label=label,
    )


#: Cue strength tiers. Decisive cues are phrases that essentially settle the route on their own.
_DECISIVE: Final[float] = 1.0
_STRONG: Final[float] = 0.6
_WEAK: Final[float] = 0.3

#: Score at which the lexical prior is treated as fully saturated — roughly one decisive cue, or
#: two strong ones. Above this, extra matches no longer raise confidence.
_PRIOR_SATURATION: Final[float] = 1.2

#: Confidence assigned when nothing at all matched and the route falls back to FACTUAL. Low on
#: purpose: an unrouted question is exactly the case where the model's opinion should win.
_UNMATCHED_CONFIDENCE: Final[float] = 0.2

KEYWORD_RULES: Final[tuple[KeywordRule, ...]] = (
    # -- diagnostic ------------------------------------------------------------------
    _rule(QueryType.DIAGNOSTIC, _DECISIVE, r"\broot\s+cause\b", "root cause"),
    _rule(QueryType.DIAGNOSTIC, _DECISIVE, r"\bwhy\s+(did|does|is|was|are|has|have|do)\b", "why did"),
    _rule(QueryType.DIAGNOSTIC, _DECISIVE, r"\bwhat\s+(caused|went\s+wrong)\b", "what caused"),
    _rule(QueryType.DIAGNOSTIC, _STRONG, r"\btroubleshoot|\bdiagnos", "troubleshoot"),
    _rule(QueryType.DIAGNOSTIC, _STRONG, r"\b(reason|cause)s?\s+(for|of|behind)\b", "reason for"),
    _rule(QueryType.DIAGNOSTIC, _STRONG, r"\bkeeps?\s+(failing|tripping|leaking|overheating)\b", "keeps failing"),
    _rule(QueryType.DIAGNOSTIC, _STRONG, r"\bfail(ed|ure)\b.{0,30}\b(last|previous|in)\b", "failed last"),
    _rule(QueryType.DIAGNOSTIC, _WEAK, r"\bexplain\b", "explain"),
    _rule(QueryType.DIAGNOSTIC, _WEAK, r"\b(abnormal|anomal|unexpected|sudden)\w*\b", "anomaly"),
    # -- procedural ------------------------------------------------------------------
    _rule(QueryType.PROCEDURAL, _DECISIVE, r"\bhow\s+(do|does|should)\s+(i|we|you|one)\b", "how do I"),
    _rule(QueryType.PROCEDURAL, _DECISIVE, r"\bhow\s+to\b", "how to"),
    _rule(QueryType.PROCEDURAL, _DECISIVE, r"\b(sop|standard\s+operating\s+procedure)\b", "SOP"),
    _rule(QueryType.PROCEDURAL, _STRONG, r"\bprocedure|\bstep[-\s]?by[-\s]?step\b", "procedure"),
    _rule(QueryType.PROCEDURAL, _STRONG, r"\b(what|which)\s+steps\b", "what steps"),
    _rule(QueryType.PROCEDURAL, _STRONG, r"\b(lockout|tagout|loto|permit\s+to\s+work|isolat\w+)\b", "isolation"),
    _rule(QueryType.PROCEDURAL, _STRONG, r"\b(replace|install|dismantle|overhaul|commission|align)\b", "work verb"),
    _rule(QueryType.PROCEDURAL, _WEAK, r"\b(start[-\s]?up|shut[-\s]?down|changeover)\b", "startup/shutdown"),
    # -- predictive ------------------------------------------------------------------
    _rule(QueryType.PREDICTIVE, _DECISIVE, r"\bwill\s+\S+\s+fail\b|\bwill\s+it\s+fail\b", "will it fail"),
    _rule(QueryType.PREDICTIVE, _DECISIVE, r"\b(predict|forecast|prognos)\w*\b", "predict"),
    _rule(QueryType.PREDICTIVE, _STRONG, r"\b(remaining\s+(useful\s+)?life|rul)\b", "remaining life"),
    _rule(QueryType.PREDICTIVE, _STRONG, r"\b(probability|likelihood|chance)\s+of\b", "probability of"),
    _rule(QueryType.PREDICTIVE, _STRONG, r"\brisk\s+(of|score|level)\b", "risk of"),
    _rule(QueryType.PREDICTIVE, _STRONG, r"\bwhen\s+(will|is|should)\b.{0,40}\b(fail|need|due|replace)\w*\b", "when will"),
    _rule(QueryType.PREDICTIVE, _WEAK, r"\bnext\s+\d+\s+(day|week|month)s?\b", "horizon"),
    _rule(QueryType.PREDICTIVE, _WEAK, r"\b(expected|likely)\s+to\b", "likely to"),
    # -- comparative -----------------------------------------------------------------
    _rule(QueryType.COMPARATIVE, _DECISIVE, r"\bvs\.?\b|\bversus\b", "versus"),
    _rule(QueryType.COMPARATIVE, _DECISIVE, r"\bcompare[ds]?\b|\bcomparison\b", "compare"),
    _rule(QueryType.COMPARATIVE, _DECISIVE, r"\bdifference[s]?\s+between\b", "difference between"),
    _rule(QueryType.COMPARATIVE, _STRONG, r"\b(which|who)\s+(one|of\s+(these|them|the))\b", "which of"),
    _rule(QueryType.COMPARATIVE, _STRONG, r"\b(better|worse|higher|lower|faster|more)\s+than\b", "better than"),
    _rule(QueryType.COMPARATIVE, _WEAK, r"\bboth\b|\beach\s+of\b", "both"),
    # -- compliance ------------------------------------------------------------------
    _rule(QueryType.COMPLIANCE, _DECISIVE, r"\bcomplian\w*\b|\bconform\w*\b", "compliance"),
    _rule(QueryType.COMPLIANCE, _DECISIVE, r"\b(regulation|statutory|legal\s+requirement)\w*\b", "regulation"),
    _rule(QueryType.COMPLIANCE, _STRONG, r"\b(clause|section)\s+\d+", "clause number"),
    _rule(QueryType.COMPLIANCE, _STRONG, r"\baudit\w*\b|\binspector\b", "audit"),
    _rule(QueryType.COMPLIANCE, _STRONG, r"\b(certificat|licen[cs]|permit)\w*\b", "certificate"),
    _rule(QueryType.COMPLIANCE, _WEAK, r"\b(penalt|violat|non[-\s]?complian)\w*\b", "penalty"),
    # -- knowledge gap ---------------------------------------------------------------
    _rule(QueryType.KNOWLEDGE_GAP, _DECISIVE, r"\bwhat\s+(don'?t|do\s+not)\s+we\s+know\b", "what don't we know"),
    _rule(QueryType.KNOWLEDGE_GAP, _DECISIVE, r"\b(un|not\s+)documented\b", "undocumented"),
    _rule(QueryType.KNOWLEDGE_GAP, _DECISIVE, r"\bknowledge\s+(gap|cliff)\b", "knowledge gap"),
    _rule(QueryType.KNOWLEDGE_GAP, _STRONG, r"\b(gaps?|blind\s+spots?)\s+(in|about|for)\b", "gaps in"),
    _rule(QueryType.KNOWLEDGE_GAP, _STRONG, r"\bwhat('s| is)\s+missing\b|\bmissing\s+(records|documents|data)\b", "missing"),
    _rule(QueryType.KNOWLEDGE_GAP, _STRONG, r"\bno\s+(records?|documentation|documents?)\s+(for|of|about)\b", "no records"),
    _rule(QueryType.KNOWLEDGE_GAP, _WEAK, r"\bincomplete\b|\bpoorly\s+documented\b", "incomplete"),
    # -- factual ---------------------------------------------------------------------
    _rule(QueryType.FACTUAL, _STRONG, r"\bwhat('s| is| are)\b(?!.{0,20}\bmissing\b)", "what is"),
    _rule(QueryType.FACTUAL, _STRONG, r"\b(limit|threshold|rating|setpoint|set\s+point|tolerance)\b", "limit"),
    _rule(QueryType.FACTUAL, _STRONG, r"\b(spec|specification|datasheet|nameplate|model\s+number)\w*\b", "specification"),
    _rule(QueryType.FACTUAL, _WEAK, r"\bwho\s+(is|was|are|owns|maintains|performed)\b", "who"),
    _rule(QueryType.FACTUAL, _WEAK, r"\bwhen\s+was\b", "when was"),
    _rule(QueryType.FACTUAL, _WEAK, r"\bwhere\s+(is|are)\b", "where is"),
    _rule(QueryType.FACTUAL, _WEAK, r"\bhow\s+(many|much|often)\b", "how many"),
    _rule(QueryType.FACTUAL, _WEAK, r"\blist\s+(all|the|every)\b", "list"),
)

#: Plant tag grammar: one to four letters, a separator, two to five digits, optional suffix letter.
#: Matches P-101, PSV-2201A, TI 1004. Deliberately strict — loose matching turns every "A-1" in
#: prose into an equipment reference and poisons routing.
EQUIPMENT_TAG_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b([A-Z]{1,4})[-\s]?(\d{2,5})([A-Z])?\b"
)


# --------------------------------------------------------------------------------------
# Result model
# --------------------------------------------------------------------------------------


class QueryClassification(IndraModel):
    """The routing decision, with the whole derivation attached.

    The Copilot records this as the first :class:`~indra.core.models.ReasoningStep` of every
    answer, so "why did INDRA treat this as diagnostic?" is answerable from the response itself.
    """

    query_type: QueryType
    confidence: Confidence
    method: str
    prior_type: QueryType
    prior_confidence: float
    scores: dict[str, float]
    matched_cues: list[str]
    equipment_tags: list[str]
    llm_type: QueryType | None = None
    llm_confidence: float | None = None
    provider_used: str | None = None

    @property
    def overridden(self) -> bool:
        """True when the model's label displaced the lexical prior."""
        return self.llm_type is not None and self.llm_type is not self.prior_type \
            and self.query_type is self.llm_type


# --------------------------------------------------------------------------------------
# Classifier
# --------------------------------------------------------------------------------------


class QueryClassifier:
    """Routes a question to exactly one :class:`QueryType`.

    Args:
        settings: Supplies the regulation vocabulary that reinforces the compliance route, so the
            lexicon tracks configured regulations instead of hardcoding Indian statute names.
        llm: Router used for the confirmation pass. May be unavailable at any time.
    """

    def __init__(self, settings: Settings, llm: LLMRouter) -> None:
        self._settings = settings
        self._llm = llm
        self._regulation_rules: tuple[KeywordRule, ...] = tuple(
            _rule(QueryType.COMPLIANCE, _DECISIVE, re.escape(name), f"regulation:{name}")
            for name in settings.regulations
            if name.strip()
        )

    # -- public ---------------------------------------------------------------------
    def prior(self, query: str) -> QueryClassification:
        """Score the query lexically. Pure, synchronous, no I/O — this is the safety net."""
        scores: dict[QueryType, float] = {qt: 0.0 for qt in QueryType}
        cues: list[str] = []

        for rule in (*KEYWORD_RULES, *self._regulation_rules):
            if rule.pattern.search(query):
                scores[rule.query_type] += rule.weight
                cues.append(f"{rule.query_type.value}:{rule.label}")

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_type, top_score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0

        if top_score <= 0.0:
            # Nothing matched. A bare "P-101?" is a lookup until something says otherwise.
            return QueryClassification(
                query_type=QueryType.FACTUAL,
                confidence=Confidence(
                    value=_UNMATCHED_CONFIDENCE,
                    rationale="No routing cue matched; defaulting to a direct lookup.",
                    method="heuristic",
                ),
                method="keyword_prior",
                prior_type=QueryType.FACTUAL,
                prior_confidence=_UNMATCHED_CONFIDENCE,
                scores={qt.value: 0.0 for qt in QueryType},
                matched_cues=[],
                equipment_tags=extract_equipment_tags(query),
            )

        # Confidence combines two independent things: how strong the winning evidence is
        # (saturation) and how cleanly it beat the alternative (separation).
        saturation = min(1.0, top_score / _PRIOR_SATURATION)
        separation = top_score / (top_score + runner_up) if (top_score + runner_up) > 0 else 1.0
        value = round(min(1.0, saturation * separation), 4)
        top_cues = [cue.split(":", 1)[1] for cue in cues if cue.startswith(f"{top_type.value}:")]

        return QueryClassification(
            query_type=top_type,
            confidence=Confidence(
                value=value,
                rationale=f"Lexical cues {top_cues} score {top_score:.2f} against {runner_up:.2f} for the runner-up.",
                method="heuristic",
            ),
            method="keyword_prior",
            prior_type=top_type,
            prior_confidence=value,
            scores={qt.value: round(score, 4) for qt, score in scores.items()},
            matched_cues=cues,
            equipment_tags=extract_equipment_tags(query),
        )

    async def classify(self, query: str) -> QueryClassification:
        """Full classification: lexical prior, then a model confirmation pass.

        Never raises. A model that is missing, rate-limited, or returns garbage leaves the prior
        standing and logs the reason.
        """
        prior = self.prior(query)

        try:
            payload, provider = await self._llm.generate_json(
                prompts.render(
                    prompts.CLASSIFICATION_PROMPT_V1,
                    query=query,
                    prior_type=prior.prior_type.value,
                    prior_rationale=prior.confidence.rationale,
                ),
                schema=prompts.CLASSIFICATION_SCHEMA_V1,
                system=prompts.CLASSIFICATION_SYSTEM_V1,
                temperature=0.0,
            )
        except IndraError as exc:
            logger.warning(
                "classification model pass unavailable; keeping lexical prior",
                extra={"query_type": prior.query_type.value, "error": exc.error_code, "detail": exc.message},
            )
            return prior
        except Exception as exc:  # pragma: no cover - defensive; router should raise IndraError
            logger.warning(
                "classification model pass raised an untyped error; keeping lexical prior",
                extra={"query_type": prior.query_type.value, "detail": str(exc)},
            )
            return prior

        parsed = _parse_classification(payload)
        if parsed is None:
            logger.warning(
                "classification model returned an unusable label; keeping lexical prior",
                extra={"query_type": prior.query_type.value, "provider": provider, "payload": payload},
            )
            return prior

        llm_type, llm_confidence, llm_rationale, llm_tags = parsed
        tags = prior.equipment_tags or [t for t in llm_tags if EQUIPMENT_TAG_PATTERN.fullmatch(t.upper())]

        if llm_type is prior.prior_type:
            # Agreement: the two methods are independent, so agreement genuinely raises confidence
            # above either alone. Capped, because two correlated readers are not proof.
            value = round(min(1.0, max(prior.prior_confidence, llm_confidence)), 4)
            return prior.model_copy(
                update={
                    "confidence": Confidence(
                        value=value,
                        rationale=f"Lexical prior and model agree on '{llm_type.value}'. {llm_rationale}",
                        method="llm",
                    ),
                    "method": "llm_confirmed",
                    "llm_type": llm_type,
                    "llm_confidence": llm_confidence,
                    "provider_used": provider,
                    "equipment_tags": tags,
                }
            )

        if llm_confidence > prior.prior_confidence:
            logger.info(
                "classification overridden by model",
                extra={
                    "prior": prior.prior_type.value,
                    "chosen": llm_type.value,
                    "prior_confidence": prior.prior_confidence,
                    "llm_confidence": llm_confidence,
                    "provider": provider,
                },
            )
            return prior.model_copy(
                update={
                    "query_type": llm_type,
                    "confidence": Confidence(
                        value=round(llm_confidence, 4),
                        rationale=(
                            f"Model overrode the lexical prior '{prior.prior_type.value}' "
                            f"({prior.prior_confidence:.2f}) with '{llm_type.value}' "
                            f"({llm_confidence:.2f}). {llm_rationale}"
                        ),
                        method="llm",
                    ),
                    "method": "llm_override",
                    "llm_type": llm_type,
                    "llm_confidence": llm_confidence,
                    "provider_used": provider,
                    "equipment_tags": tags,
                }
            )

        # Disagreement the model is not confident enough to win. Keep the prior, but record the
        # dissent and discount confidence — a contested route is a weaker route.
        contested = round(prior.prior_confidence * (1.0 - llm_confidence / 2.0), 4)
        return prior.model_copy(
            update={
                "confidence": Confidence(
                    value=max(contested, _UNMATCHED_CONFIDENCE),
                    rationale=(
                        f"Lexical prior '{prior.prior_type.value}' kept; model preferred "
                        f"'{llm_type.value}' at lower confidence ({llm_confidence:.2f})."
                    ),
                    method="heuristic",
                ),
                "method": "prior_contested",
                "llm_type": llm_type,
                "llm_confidence": llm_confidence,
                "provider_used": provider,
                "equipment_tags": tags,
            }
        )


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _parse_classification(
    payload: dict[str, object],
) -> tuple[QueryType, float, str, list[str]] | None:
    """Validate the model's JSON. Returns ``None`` when it cannot be trusted."""
    raw_type = payload.get("query_type")
    if not isinstance(raw_type, str):
        return None
    try:
        query_type = QueryType(raw_type.strip().lower())
    except ValueError:
        return None

    raw_conf = payload.get("confidence")
    confidence = float(raw_conf) if isinstance(raw_conf, (int, float)) else 0.0
    confidence = max(0.0, min(1.0, confidence))

    raw_rationale = payload.get("rationale")
    rationale = raw_rationale.strip() if isinstance(raw_rationale, str) else ""

    raw_tags = payload.get("equipment_tags")
    tags = [t.strip().upper() for t in raw_tags if isinstance(t, str) and t.strip()] \
        if isinstance(raw_tags, list) else []

    return query_type, confidence, rationale, tags


def extract_equipment_tags(text: str, *, registry: Sequence[str] | None = None) -> list[str]:
    """Pull plant tags out of free text, normalised to ``LETTERS-DIGITS[SUFFIX]``.

    Order-preserving and de-duplicated, because "compare P-101 and P-102" must keep the subjects in
    the order the operator wrote them. When ``registry`` is supplied, only known tags survive —
    used where a false tag is more costly than a missed one.
    """
    seen: set[str] = set()
    out: list[str] = []
    known = {t.strip().upper() for t in registry} if registry is not None else None

    for match in EQUIPMENT_TAG_PATTERN.finditer(text.upper()):
        letters, digits, suffix = match.group(1), match.group(2), match.group(3) or ""
        tag = f"{letters}-{digits}{suffix}"
        if known is not None and tag not in known:
            continue
        if tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


__all__ = [
    "EQUIPMENT_TAG_PATTERN",
    "KEYWORD_RULES",
    "KeywordRule",
    "QueryClassification",
    "QueryClassifier",
    "extract_equipment_tags",
]
