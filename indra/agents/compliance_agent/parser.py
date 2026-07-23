"""Turn an uploaded regulation document into checkable obligations.

This is **the only module in the Compliance Agent that is allowed to call a language model**, and it
is allowed to call one for exactly one purpose: converting regulation *prose* into the structured
:class:`~indra.core.models.RegulatoryRequirement` shape. Whether an obligation is met is decided by
arithmetic over dated evidence in :mod:`.gap_detection`, which cannot reach a model at all.

Three guarantees make model output safe to put in front of a regulator:

1. **The quoted text is never the model's.** ``RawRequirement.text`` is always taken verbatim from
   the source document block. The model normalises the *duty* ("monthly pressure vessel
   inspection"); it never gets to write what the statute says.
2. **Every requirement must be grounded.** A returned requirement is kept only if its clause label
   actually occurs in the source block and its obligation shares real vocabulary with that block
   (:data:`OBLIGATION_GROUNDING_RATIO`). A hallucinated clause is dropped, not filed. This is also
   what keeps the deterministic stub provider — which synthesises schema-valid nonsense when no API
   key is configured — from ever polluting the catalogue.
3. **There is a model-free path.** When the model is unavailable, unusable, or ungrounded,
   :func:`heuristic_requirements` extracts what can be extracted deterministically: the clause
   label, the verbatim text, the stated periodicity, and the penalty sentence. It refuses to guess
   an obligation that the block does not state, because a fabricated obligation manufactures a
   fabricated gap, and a false gap in front of an inspector is worse than a missed one.

The parsed output is merged into the same :class:`~.requirements.RequirementCatalogue` as the
seeded YAML baseline, with ``provenance="parsed"`` so it outranks the seed for the same clause.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Final, Iterable, Literal, Mapping, Sequence

from pydantic import ValidationError

from indra.core.config import Settings, get_settings
from indra.core.contracts import LLMRouter
from indra.core.exceptions import ComplianceError, IndraError
from indra.core.logging import get_logger
from indra.core.models import (
    DocumentMeta,
    DocumentType,
    RegulatoryRequirement,
    Severity,
    SourceRef,
    utcnow,
)

from indra.agents.compliance_agent.requirements import (
    RawRequirement,
    RegulationFile,
    RequirementSpec,
    normalise_field_key,
    slugify,
    spec_from_raw,
)

logger = get_logger(__name__)

ParseMethod = Literal["llm", "heuristic", "mixed", "none"]

# --------------------------------------------------------------------------------------
# Implementation constants
#
# Product tunables live in ``indra.core.config``. What follows are structural facts about the shape
# of statutory prose and hard limits on how much of a document one parse is allowed to consume.
# Changing them changes the *parser*, which is a code review, not a config change.
# --------------------------------------------------------------------------------------

#: Longest verbatim quote retained per requirement. Long enough to be the operative sentence, short
#: enough that the audit PDF stays a document rather than a reprint of the Act.
MAX_QUOTE_CHARS: Final[int] = 1600

#: Characters of source prose handed to the model in one call. Keeps a batch inside the smallest
#: context window in the provider chain with room for the schema and the reply.
BATCH_CHAR_BUDGET: Final[int] = 6000

#: Hard ceiling on model calls for one document. A 300-page Act must not silently spend a day's
#: quota; anything past this is parsed by the model-free path and flagged in the warnings.
MAX_MODEL_CALLS: Final[int] = 8

#: Clause blocks considered per document, in document order.
MAX_CLAUSE_BLOCKS: Final[int] = 120

#: Shortest block that can carry an obligation. Below this it is a heading, not a duty.
MIN_BLOCK_CHARS: Final[int] = 60

#: Fraction of an obligation's content words that must appear in its source block for the
#: obligation to count as grounded in the document rather than invented by the model.
OBLIGATION_GROUNDING_RATIO: Final[float] = 0.34

#: Longest acceptable normalised obligation. A duty that needs a paragraph has not been atomised.
MAX_OBLIGATION_CHARS: Final[int] = 160

#: Requirements accepted from one document. A regulation with more atomic duties than this is
#: almost certainly a mis-split, and an unbounded catalogue makes every audit unreadable.
MAX_REQUIREMENTS_PER_DOCUMENT: Final[int] = 80

#: Words carrying no discriminating power when checking obligation grounding.
_STOPWORDS: Final[frozenset[str]] = frozenset({
    "the", "and", "for", "with", "shall", "any", "all", "such", "that", "this", "from", "into",
    "every", "each", "which", "been", "have", "has", "was", "were", "are", "not", "must", "may",
    "other", "than", "then", "their", "there", "where", "when", "under", "upon", "per", "its",
    "of", "in", "to", "a", "an", "or", "be", "as", "at", "by", "on", "is", "it", "no",
})

#: Adverbial periodicities, mapped to days.
_PERIOD_ADVERBS: Final[Mapping[str, int]] = {
    "daily": 1,
    "weekly": 7,
    "fortnightly": 14,
    "monthly": 30,
    "bi-monthly": 61,
    "bimonthly": 61,
    "quarterly": 91,
    "half-yearly": 182,
    "halfyearly": 182,
    "half yearly": 182,
    "six-monthly": 182,
    "six monthly": 182,
    "semi-annually": 182,
    "semi-annual": 182,
    "annually": 365,
    "annual": 365,
    "yearly": 365,
    "biennial": 730,
    "biennially": 730,
    "triennial": 1095,
    "triennially": 1095,
}

#: Period units, mapped to days. Statutory months are calendar months; 30 days is the checkable
#: reading and the one the Inspectorate's own registers use.
_PERIOD_UNITS: Final[Mapping[str, int]] = {
    "day": 1, "week": 7, "fortnight": 14, "month": 30, "quarter": 91, "year": 365,
}

#: Number words that appear in statutory intervals.
_NUMBER_WORDS: Final[Mapping[str, int]] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "twenty-four": 24, "thirty": 30, "sixty": 60, "ninety": 90,
}

#: Sanity bounds on a parsed periodicity. Outside these the value is a misreading, not an interval.
_MIN_FREQUENCY_DAYS: Final[int] = 1
_MAX_FREQUENCY_DAYS: Final[int] = 3650

#: Clause heading grammar: "Section 41(b)", "Rule 12(3)", "Clause 5.3.2", "Reg. 7", "4.2.1".
_CLAUSE_HEADING: Final[re.Pattern[str]] = re.compile(
    r"^[ \t]*(?P<label>"
    r"(?:section|sec|rule|clause|regulation|reg|article|art|para|paragraph|schedule|annexure|appendix)"
    r"\s*\.?\s*\d{1,4}[A-Za-z]?(?:\.\d{1,3}){0,3}(?:\s*\([0-9A-Za-z]{1,3}\)){0,3}"
    r"|\d{1,3}(?:\.\d{1,3}){1,3}(?:\s*\([0-9A-Za-z]{1,3}\)){0,2}"
    r")"
    r"\s*[-—:.–)]?\s",
    re.IGNORECASE | re.MULTILINE,
)

#: Interval phrasing: "at intervals not exceeding one month", "once in every twelve months".
_INTERVAL_PHRASE: Final[re.Pattern[str]] = re.compile(
    r"(?:at\s+intervals?\s+(?:of|not\s+exceeding)|at\s+least\s+once\s+in\s+every(?:\s+period\s+of)?"
    r"|once\s+in\s+every(?:\s+period\s+of)?|at\s+least\s+once\s+(?:every|in)|not\s+less\s+than\s+once"
    r"\s+in\s+every|once\s+every|every|within)\s+"
    r"(?P<count>\d{1,4}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirty|sixty|ninety)?"
    r"\s*(?P<unit>day|week|fortnight|month|quarter|year)s?\b",
    re.IGNORECASE,
)

#: Bare adverbial periodicity, e.g. "shall be inspected monthly".
_ADVERB_PHRASE: Final[re.Pattern[str]] = re.compile(
    r"\b(" + "|".join(sorted((re.escape(k) for k in _PERIOD_ADVERBS), key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

#: Sentence that states the consequence of breach.
_PENALTY_SENTENCE: Final[re.Pattern[str]] = re.compile(
    r"[^.]*\b(?:penalt|fine|imprison|punish|prosecut|liable to|offence|contravention)\w*\b[^.]*\.",
    re.IGNORECASE,
)

#: Duty verbs. A block with none of these states a definition, not an obligation.
_DUTY_PHRASE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:shall|must|is\s+required\s+to|are\s+required\s+to|shall\s+be\s+required)\b",
    re.IGNORECASE,
)

#: Evidence-type inference for the model-free path, in priority order. Deliberately small: the
#: parser only claims an evidence type the block's own vocabulary names.
_EVIDENCE_CUES: Final[tuple[tuple[re.Pattern[str], DocumentType], ...]] = (
    (re.compile(r"\b(?:examin|inspect|test(?:ed|ing)?|survey|certificat|register|thorough)\w*\b", re.I),
     DocumentType.INSPECTION_REPORT),
    (re.compile(r"\b(?:procedure|permit[- ]to[- ]work|standard operating|sop|written scheme|method statement)\b", re.I),
     DocumentType.SOP),
    (re.compile(r"\b(?:return|statement|schedule|log ?sheet|record sheet|monitoring data)\b", re.I),
     DocumentType.SPREADSHEET),
    (re.compile(r"\b(?:maintenance|repair|work order|servicing|overhaul)\b", re.I),
     DocumentType.WORK_ORDER),
)

#: Aliases accepted for ``evidence_types`` values returned by the model.
_EVIDENCE_ALIASES: Final[Mapping[str, DocumentType]] = {
    "inspection": DocumentType.INSPECTION_REPORT,
    "inspection_report": DocumentType.INSPECTION_REPORT,
    "inspection_certificate": DocumentType.INSPECTION_REPORT,
    "certificate": DocumentType.INSPECTION_REPORT,
    "test_report": DocumentType.INSPECTION_REPORT,
    "test_certificate": DocumentType.INSPECTION_REPORT,
    "examination_report": DocumentType.INSPECTION_REPORT,
    "register": DocumentType.INSPECTION_REPORT,
    "sop": DocumentType.SOP,
    "procedure": DocumentType.SOP,
    "written_scheme": DocumentType.SOP,
    "permit": DocumentType.SOP,
    "work_order": DocumentType.WORK_ORDER,
    "maintenance_record": DocumentType.WORK_ORDER,
    "job_card": DocumentType.WORK_ORDER,
    "shift_log": DocumentType.SHIFT_LOG,
    "log": DocumentType.SHIFT_LOG,
    "logbook": DocumentType.SHIFT_LOG,
    "incident_report": DocumentType.INCIDENT_REPORT,
    "accident_report": DocumentType.INCIDENT_REPORT,
    "root_cause_analysis": DocumentType.ROOT_CAUSE_ANALYSIS,
    "rca": DocumentType.ROOT_CAUSE_ANALYSIS,
    "spreadsheet": DocumentType.SPREADSHEET,
    "return": DocumentType.SPREADSHEET,
    "monitoring_data": DocumentType.SPREADSHEET,
    "oem_manual": DocumentType.OEM_MANUAL,
    "manual": DocumentType.OEM_MANUAL,
    "drawing": DocumentType.PID_DRAWING,
    "pid_drawing": DocumentType.PID_DRAWING,
    "email": DocumentType.EMAIL,
    "regulation": DocumentType.REGULATION,
}


# ======================================================================================
# Prompt and schema — the model's entire surface area in this agent
# ======================================================================================

REGULATION_SYSTEM_V1: Final[str] = """\
You convert Indian industrial safety and environmental regulation text into atomic, machine-checkable
obligations for a plant compliance system. You return JSON only, strictly matching the requested
schema, and you never write prose outside the JSON object.

Rules you must follow without exception:
1. Use ONLY the clause text supplied. You have no other knowledge of this regulation.
2. One JSON entry per atomic duty. If a clause states two independent duties on different cycles,
   emit two entries. If a clause states no duty at all, emit nothing for it.
3. Copy the clause label exactly as it appears in the supplied text (e.g. "Section 41(b)").
   Never invent, renumber, or normalise a clause label.
4. The obligation is a short noun phrase naming the duty and its cycle, e.g.
   "monthly pressure vessel inspection". Under twelve words. No sentences, no "shall".
5. frequency_days is the stated interval in days, or null when the duty is one-off or the text
   states no interval. One month = 30, six months = 182, one year = 365.
6. applies_to_types names plant equipment classes the clause binds, in snake_case
   (pressure_vessel, crane, storage_tank). Use ["*"] only when the duty binds the whole
   installation rather than identifiable assets. Never guess a class the text does not indicate.
7. required_evidence_fields names the particulars the record itself must show for the duty to be
   demonstrated (inspector, findings, test_pressure, certificate_number). Only fields the text
   actually requires.
8. If you are unsure whether the text states a duty, omit it. An invented obligation is a defect.
"""

REGULATION_PROMPT_V1: Final[str] = """\
Extract every atomic compliance obligation from the regulation clauses below.

REGULATION: {regulation}
SOURCE DOCUMENT: {document_title}

CLAUSES
{clauses}

For each obligation return an object with:
- clause: the clause label exactly as printed above
- obligation: short noun phrase naming the duty and its cycle (under twelve words)
- frequency_days: integer interval in days, or null
- applies_to_types: snake_case equipment classes bound by the clause, or ["*"] for plant-wide
- applies_to_tags: specific plant tags named in the text (usually empty)
- evidence_types: which of inspection_report, work_order, sop, shift_log, incident_report,
  root_cause_analysis, spreadsheet, oem_manual would discharge the duty
- required_evidence_fields: particulars the record must show
- penalty: the stated consequence of breach, quoted from the text, or null
- owner_role: the plant role accountable (e.g. "Safety Officer"), or null
- severity: one of INFO, LOW, WARNING, HIGH, CRITICAL — how serious a breach is
- grace_days: integer days of tolerance the text allows past the interval, 0 if none stated
- remediation: one sentence naming the concrete action that closes the gap

Emit nothing for a clause that states no duty.
"""

REGULATION_SCHEMA_V1: Final[dict[str, Any]] = {
    "type": "object",
    "required": ["requirements"],
    "properties": {
        "requirements": {
            "type": "array",
            "maxItems": MAX_REQUIREMENTS_PER_DOCUMENT,
            "items": {
                "type": "object",
                "required": ["clause", "obligation"],
                "properties": {
                    "clause": {"type": "string", "minLength": 1, "maxLength": 60},
                    "obligation": {"type": "string", "minLength": 3, "maxLength": MAX_OBLIGATION_CHARS},
                    "frequency_days": {"type": ["integer", "null"]},
                    "applies_to_types": {"type": "array", "items": {"type": "string"}},
                    "applies_to_tags": {"type": "array", "items": {"type": "string"}},
                    "evidence_types": {"type": "array", "items": {"type": "string"}},
                    "required_evidence_fields": {"type": "array", "items": {"type": "string"}},
                    "penalty": {"type": ["string", "null"]},
                    "owner_role": {"type": ["string", "null"]},
                    "severity": {
                        "type": "string",
                        "enum": ["INFO", "LOW", "WARNING", "HIGH", "CRITICAL"],
                    },
                    "grace_days": {"type": ["integer", "null"]},
                    "remediation": {"type": ["string", "null"]},
                },
            },
        }
    },
}


# ======================================================================================
# Deterministic text structure
# ======================================================================================


@dataclass(frozen=True, slots=True)
class ClauseBlock:
    """One clause of a regulation document, located in the source text.

    ``text`` is verbatim source. It is what gets quoted in the audit package, so it is never
    rewritten by anything downstream.
    """

    clause: str
    text: str
    char_start: int
    char_end: int
    ordinal: int

    @property
    def slug(self) -> str:
        return slugify(self.clause)

    @property
    def is_substantive(self) -> bool:
        """True when the block is long enough and phrased as a duty rather than a definition."""
        return len(self.text) >= MIN_BLOCK_CHARS and bool(_DUTY_PHRASE.search(self.text))


def normalise_clause_label(raw: str) -> str:
    """Tidy a clause label without changing its identity.

    ``"SECTION  41(b) "`` → ``"Section 41(b)"``. Case and spacing are cosmetic; the digits, letters
    and brackets are the clause's identity and are preserved exactly.
    """
    label = " ".join(raw.split()).strip(" .:-–—)")
    match = re.match(
        r"^(section|sec|rule|clause|regulation|reg|article|art|para|paragraph|schedule|annexure|appendix)"
        r"\s*\.?\s*(.+)$",
        label,
        re.IGNORECASE,
    )
    if match is None:
        return label
    keyword = {"sec": "Section", "reg": "Regulation", "art": "Article", "para": "Paragraph"}.get(
        match.group(1).lower(), match.group(1).capitalize()
    )
    remainder = re.sub(r"\s*\(\s*", "(", match.group(2).strip())
    remainder = re.sub(r"\s*\)\s*", ")", remainder)
    return f"{keyword} {remainder}".strip()


def split_clauses(text: str, *, limit: int = MAX_CLAUSE_BLOCKS) -> list[ClauseBlock]:
    """Split regulation prose into clause blocks. Pure, deterministic, no model involved.

    A document with no recognisable clause headings yields one block covering the whole text, so a
    plain-language circular still parses.
    """
    if not text or not text.strip():
        return []

    matches = list(_CLAUSE_HEADING.finditer(text))
    if not matches:
        body = text.strip()
        return [ClauseBlock(clause="Unnumbered clause", text=body[:MAX_QUOTE_CHARS],
                            char_start=0, char_end=len(body), ordinal=0)]

    blocks: list[ClauseBlock] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if len(body) < MIN_BLOCK_CHARS:
            continue
        clause = normalise_clause_label(match.group("label"))
        if not clause:
            continue
        blocks.append(
            ClauseBlock(
                clause=clause,
                text=body[:MAX_QUOTE_CHARS],
                char_start=start,
                char_end=end,
                ordinal=len(blocks),
            )
        )
        if len(blocks) >= limit:
            break
    if not blocks:
        body = text.strip()
        blocks = [ClauseBlock(clause="Unnumbered clause", text=body[:MAX_QUOTE_CHARS],
                              char_start=0, char_end=len(body), ordinal=0)]
    return blocks


def frequency_from_text(text: str) -> int | None:
    """Derive a stated periodicity in days from statutory phrasing. Pure.

    Returns the *first* interval stated in the block, in document order, because that is the
    operative one; a later mention is usually a cross-reference. ``None`` when the text states no
    interval, which is an honest answer — a one-off duty has no frequency.
    """
    if not text:
        return None

    candidates: list[tuple[int, int]] = []  # (position, days)

    for match in _INTERVAL_PHRASE.finditer(text):
        unit = match.group("unit")
        if unit is None:
            continue
        per_unit = _PERIOD_UNITS.get(unit.lower())
        if per_unit is None:
            continue
        raw_count = (match.group("count") or "1").strip().lower()
        count = _NUMBER_WORDS.get(raw_count)
        if count is None:
            try:
                count = int(raw_count)
            except ValueError:
                count = 1
        candidates.append((match.start(), count * per_unit))

    for match in _ADVERB_PHRASE.finditer(text):
        days = _PERIOD_ADVERBS.get(match.group(1).lower().replace("_", "-"))
        if days is None:
            days = _PERIOD_ADVERBS.get(match.group(1).lower())
        if days is not None:
            candidates.append((match.start(), days))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    days = candidates[0][1]
    if days < _MIN_FREQUENCY_DAYS or days > _MAX_FREQUENCY_DAYS:
        return None
    return int(days)


def evidence_types_from(values: Iterable[str]) -> list[DocumentType]:
    """Map free-form evidence-type names onto :class:`DocumentType`. Unknown names are dropped."""
    resolved: list[DocumentType] = []
    for raw in values:
        if not isinstance(raw, str) or not raw.strip():
            continue
        key = slugify(raw)
        document_type = _EVIDENCE_ALIASES.get(key)
        if document_type is None:
            try:
                document_type = DocumentType(key)
            except ValueError:
                continue
        if document_type is DocumentType.UNKNOWN:
            continue
        if document_type not in resolved:
            resolved.append(document_type)
    return resolved


def _content_words(text: str) -> set[str]:
    return {
        word for word in re.findall(r"[a-z]{3,}", text.lower())
        if word not in _STOPWORDS
    }


def obligation_is_grounded(obligation: str, block: ClauseBlock) -> bool:
    """True when the obligation's vocabulary genuinely comes from its source block.

    The check is deliberately lexical. A model normalising *"examined ... at intervals not exceeding
    one month"* into *"monthly pressure vessel inspection"* keeps most of the block's nouns; a model
    inventing an obligation keeps none of them. Stemming is approximated by prefix matching so
    ``inspection`` matches ``inspected``.
    """
    wanted = _content_words(obligation)
    if not wanted:
        return False
    available = _content_words(block.text)
    if not available:
        return False
    hits = 0
    for word in wanted:
        stem = word[:5]
        if any(present.startswith(stem) or word.startswith(present[:5]) for present in available):
            hits += 1
    return (hits / len(wanted)) >= OBLIGATION_GROUNDING_RATIO


def _penalty_from_text(text: str) -> str | None:
    match = _PENALTY_SENTENCE.search(text)
    if match is None:
        return None
    sentence = " ".join(match.group(0).split())
    return sentence[:600] or None


def _evidence_from_cues(text: str) -> list[DocumentType]:
    found: list[DocumentType] = []
    for pattern, document_type in _EVIDENCE_CUES:
        if pattern.search(text) and document_type not in found:
            found.append(document_type)
    return found[:2]


# ======================================================================================
# The model-free path
# ======================================================================================


def heuristic_requirements(blocks: Sequence[ClauseBlock]) -> list[RawRequirement]:
    """Extract obligations without a model. Pure and deterministic.

    Used when no model is reachable, when the model's output is ungrandable, and for any clause
    batch past :data:`MAX_MODEL_CALLS`. It claims only what the text states outright: the clause
    label, the verbatim quote, the stated periodicity, the penalty sentence, and an evidence type
    named by the block's own vocabulary. It deliberately produces **no** ``applies_to_types``, which
    means such a requirement binds nothing until a human scopes it — an unscoped requirement is
    visible in the catalogue and inert in gap detection, which is the honest failure mode.
    """
    out: list[RawRequirement] = []
    for block in blocks:
        if not block.is_substantive:
            continue
        frequency = frequency_from_text(block.text)
        evidence = _evidence_from_cues(block.text)
        if frequency is None and not evidence:
            # No cycle and no artefact named: nothing here can be mechanically checked.
            continue
        obligation = _heuristic_obligation(block, frequency)
        if not obligation:
            continue
        try:
            out.append(
                RawRequirement(
                    clause=block.clause,
                    obligation=obligation,
                    text=block.text,
                    frequency_days=frequency,
                    applies_to_types=[],
                    applies_to_tags=[],
                    evidence_types=evidence,
                    required_evidence_fields=[],
                    penalty=_penalty_from_text(block.text),
                    severity=Severity.WARNING,
                    grace_days=0,
                    remediation=(
                        f"Scope {block.clause} to the assets it binds, then carry out the duty it "
                        "states and file the resulting record against each asset."
                    ),
                )
            )
        except ValidationError as exc:  # pragma: no cover - defensive; fields are pre-validated
            logger.debug("heuristic requirement rejected", extra={"clause": block.clause, "error": str(exc)})
    return out


def _heuristic_obligation(block: ClauseBlock, frequency: int | None) -> str:
    """Name the duty using the block's own duty verb. Never invents a subject."""
    match = re.search(
        r"\b(?:shall|must)\s+(?:be\s+)?([a-z]+(?:ed|ted|ned|d)?)\b", block.text, re.IGNORECASE
    )
    verb = match.group(1).lower() if match else ""
    cadence = {
        1: "daily", 7: "weekly", 14: "fortnightly", 30: "monthly", 91: "quarterly",
        182: "half-yearly", 365: "annual", 730: "biennial", 1095: "triennial",
    }.get(frequency or 0, "")
    if not verb:
        return ""
    duty = f"{cadence} {verb} duty under {block.clause}".strip()
    return " ".join(duty.split())[:MAX_OBLIGATION_CHARS]


# ======================================================================================
# Parse result
# ======================================================================================


@dataclass(slots=True)
class ParsedRegulation:
    """Outcome of parsing one regulation document.

    ``method`` records how the requirements were produced, so the audit package can say whether an
    obligation came from a model pass, from the deterministic path, or from both.
    """

    regulation: str
    document_id: str
    specs: list[RequirementSpec] = field(default_factory=list)
    method: ParseMethod = "none"
    provider: str = ""
    model_calls: int = 0
    blocks_seen: int = 0
    rejected: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def requirements(self) -> list[RegulatoryRequirement]:
        """Project to the shared domain model — what :meth:`ComplianceService.parse_regulation` returns."""
        return [spec.requirement for spec in self.specs]

    def describe(self) -> dict[str, Any]:
        return {
            "regulation": self.regulation,
            "document_id": self.document_id,
            "requirements": len(self.specs),
            "method": self.method,
            "provider": self.provider,
            "model_calls": self.model_calls,
            "blocks": self.blocks_seen,
            "rejected": self.rejected,
            "warnings": len(self.warnings),
        }


# ======================================================================================
# Parser
# ======================================================================================


class RegulationParser:
    """Parses regulation documents into :class:`RequirementSpec` objects.

    Args:
        llm: The router. Only :meth:`~indra.core.contracts.LLMRouter.generate_json` is used, and
            only to normalise prose that the document already contains.
        settings: Process settings. Defaults to the process singleton.
    """

    __slots__ = ("_llm", "_settings")

    def __init__(self, llm: LLMRouter, *, settings: Settings | None = None) -> None:
        self._llm = llm
        self._settings = settings or get_settings()

    # -- public surface ------------------------------------------------------------

    async def parse(
        self,
        text: str,
        *,
        meta: DocumentMeta,
        regulation: str | None = None,
    ) -> ParsedRegulation:
        """Parse ``text`` into requirement specs attributed to ``meta``.

        Args:
            text: Extracted plain text of the regulation document.
            meta: The document the text came from. Every requirement's ``SourceRef`` points here.
            regulation: Regulation name to file the requirements under. Defaults to the best match
                against ``settings.regulations``, falling back to the document title.

        Returns:
            A :class:`ParsedRegulation`. Never empty-and-silent: if nothing could be parsed, the
            warnings say why.

        Raises:
            ComplianceError: the supplied text is empty or too short to contain an obligation.
        """
        body = (text or "").strip()
        if len(body) < MIN_BLOCK_CHARS:
            raise ComplianceError(
                f"Regulation document '{meta.title}' yielded {len(body)} characters of text, which "
                "cannot contain an obligation. Re-ingest the file — if it is a scanned PDF the OCR "
                "step produced nothing, and the requirement catalogue must not be built from an "
                "empty document.",
                context={"document_id": meta.document_id, "characters": len(body)},
            )

        name = regulation or self.resolve_regulation_name(meta)
        blocks = split_clauses(body)
        result = ParsedRegulation(regulation=name, document_id=meta.document_id, blocks_seen=len(blocks))

        raws, providers = await self._model_pass(blocks, regulation=name, meta=meta, result=result)

        covered = {slugify(raw.clause) for raw in raws}
        uncovered = [block for block in blocks if block.slug not in covered]
        fallback = heuristic_requirements(uncovered) if uncovered else []
        if fallback:
            result.warnings.append(
                f"{len(fallback)} clause(s) were structured without model assistance and carry no "
                "equipment scope; scope them before relying on them in an audit."
            )

        combined = self._deduplicate(raws + fallback)
        if not combined:
            result.method = "none"
            result.warnings.append(
                "No mechanically checkable obligation could be extracted from this document."
            )
            logger.warning("regulation document produced no requirements", extra=result.describe())
            return result

        result.method = (
            "mixed" if raws and fallback else ("llm" if raws else "heuristic")
        )
        result.provider = ", ".join(sorted(providers))
        result.specs = self._to_specs(combined, regulation=name, meta=meta, blocks=blocks)
        logger.info("regulation document parsed", extra=result.describe())
        return result

    def resolve_regulation_name(self, meta: DocumentMeta) -> str:
        """Match a document to a configured regulation name, else use its title.

        Filing a parsed clause under a name that is not in ``settings.regulations`` would make it
        invisible to a scoped audit, so the configured names win whenever the title mentions one.
        """
        haystack = slugify(f"{meta.title} {meta.filename}")
        best: tuple[int, str] | None = None
        for configured in self._settings.regulations:
            needle = slugify(configured)
            if needle and needle in haystack:
                score = len(needle)
                if best is None or score > best[0]:
                    best = (score, configured)
        if best is not None:
            return best[1]
        return " ".join(meta.title.split())[:120] or meta.filename

    # -- model pass ----------------------------------------------------------------

    async def _model_pass(
        self,
        blocks: Sequence[ClauseBlock],
        *,
        regulation: str,
        meta: DocumentMeta,
        result: ParsedRegulation,
    ) -> tuple[list[RawRequirement], set[str]]:
        """Run the batched model extraction. Never raises — a failure degrades to the model-free path."""
        by_slug = {block.slug: block for block in blocks}
        collected: list[RawRequirement] = []
        providers: set[str] = set()

        for batch in self._batches(blocks):
            if result.model_calls >= MAX_MODEL_CALLS:
                result.warnings.append(
                    f"Model budget of {MAX_MODEL_CALLS} call(s) exhausted; the remaining "
                    f"{len(blocks) - sum(len(b) for b in [batch])} clause(s) were parsed deterministically."
                )
                break
            prompt = self._render_prompt(batch, regulation=regulation, meta=meta)
            try:
                payload, provider = await self._llm.generate_json(
                    prompt,
                    schema=REGULATION_SCHEMA_V1,
                    system=REGULATION_SYSTEM_V1,
                    temperature=0.0,
                )
            except IndraError as exc:
                result.warnings.append(
                    f"Model extraction unavailable ({exc.error_code}); clauses "
                    f"{', '.join(b.clause for b in batch)} were parsed deterministically."
                )
                logger.warning(
                    "regulation model pass failed; falling back to deterministic extraction",
                    extra={"document_id": meta.document_id, "error": exc.error_code, "detail": exc.message},
                )
                continue
            except Exception as exc:  # pragma: no cover - router should raise IndraError
                result.warnings.append("Model extraction raised an untyped error; used the deterministic path.")
                logger.warning(
                    "regulation model pass raised an untyped error",
                    extra={"document_id": meta.document_id, "detail": f"{type(exc).__name__}: {exc}"},
                )
                continue

            result.model_calls += 1
            providers.add(provider)
            accepted, rejected = self._coerce(payload, by_slug=by_slug, batch=batch)
            result.rejected += rejected
            collected.extend(accepted)
            if rejected:
                logger.info(
                    "ungrounded requirements discarded",
                    extra={"document_id": meta.document_id, "provider": provider,
                           "accepted": len(accepted), "rejected": rejected},
                )

        if result.rejected and not collected:
            result.warnings.append(
                f"The model returned {result.rejected} requirement(s) that could not be traced back "
                "to the document text; all were discarded as ungrounded."
            )
        return collected, providers

    @staticmethod
    def _batches(blocks: Sequence[ClauseBlock]) -> list[list[ClauseBlock]]:
        """Group clause blocks into prompt-sized batches, preserving document order."""
        batches: list[list[ClauseBlock]] = []
        current: list[ClauseBlock] = []
        size = 0
        for block in blocks:
            length = len(block.text) + len(block.clause) + 16
            if current and size + length > BATCH_CHAR_BUDGET:
                batches.append(current)
                current, size = [], 0
            current.append(block)
            size += length
        if current:
            batches.append(current)
        return batches

    @staticmethod
    def _render_prompt(batch: Sequence[ClauseBlock], *, regulation: str, meta: DocumentMeta) -> str:
        clauses = "\n\n".join(f"[{block.clause}]\n{block.text}" for block in batch)
        try:
            return REGULATION_PROMPT_V1.format(
                regulation=regulation,
                document_title=meta.title,
                clauses=clauses,
            )
        except (KeyError, IndexError, ValueError) as exc:  # pragma: no cover - template is a constant
            raise ComplianceError(
                "Regulation prompt template could not be rendered; a placeholder is missing. "
                "Check REGULATION_PROMPT_V1 against its call site.",
                context={"missing": str(exc)},
                cause=exc,
            ) from exc

    def _coerce(
        self,
        payload: Mapping[str, Any],
        *,
        by_slug: Mapping[str, ClauseBlock],
        batch: Sequence[ClauseBlock],
    ) -> tuple[list[RawRequirement], int]:
        """Turn model output into validated requirements, discarding anything ungrounded.

        Returns ``(accepted, rejected_count)``.
        """
        entries = payload.get("requirements")
        if not isinstance(entries, list):
            return [], 0

        batch_slugs = {block.slug: block for block in batch}
        accepted: list[RawRequirement] = []
        rejected = 0

        for entry in entries[:MAX_REQUIREMENTS_PER_DOCUMENT]:
            if not isinstance(entry, Mapping):
                rejected += 1
                continue
            clause_raw = str(entry.get("clause") or "").strip()
            obligation = " ".join(str(entry.get("obligation") or "").split())
            if not clause_raw or not obligation:
                rejected += 1
                continue

            clause = normalise_clause_label(clause_raw)
            block = batch_slugs.get(slugify(clause)) or by_slug.get(slugify(clause))
            if block is None:
                rejected += 1
                logger.debug("clause not present in source document", extra={"clause": clause})
                continue
            if not obligation_is_grounded(obligation, block):
                rejected += 1
                logger.debug(
                    "obligation not grounded in its clause text",
                    extra={"clause": clause, "obligation": obligation[:80]},
                )
                continue

            raw = self._build_raw(entry, clause=clause, obligation=obligation, block=block)
            if raw is None:
                rejected += 1
                continue
            accepted.append(raw)
        return accepted, rejected

    def _build_raw(
        self,
        entry: Mapping[str, Any],
        *,
        clause: str,
        obligation: str,
        block: ClauseBlock,
    ) -> RawRequirement | None:
        """Assemble one validated :class:`RawRequirement`. The quoted text is always the source's."""
        frequency = _as_int(entry.get("frequency_days"))
        derived = frequency_from_text(block.text)
        if frequency is not None and not (_MIN_FREQUENCY_DAYS <= frequency <= _MAX_FREQUENCY_DAYS):
            frequency = None
        if frequency is None:
            frequency = derived
        elif derived is not None and derived != frequency:
            logger.info(
                "model periodicity differs from the interval stated in the clause; keeping the model value",
                extra={"clause": clause, "model_days": frequency, "text_days": derived},
            )

        evidence = evidence_types_from(_as_str_list(entry.get("evidence_types")))
        if not evidence:
            evidence = _evidence_from_cues(block.text)

        penalty = entry.get("penalty")
        penalty_text = " ".join(str(penalty).split()) if isinstance(penalty, str) and penalty.strip() else None
        if not penalty_text:
            penalty_text = _penalty_from_text(block.text)

        owner = entry.get("owner_role")
        owner_role = " ".join(str(owner).split())[:80] if isinstance(owner, str) and owner.strip() else None

        remediation = entry.get("remediation")
        remediation_text = (
            " ".join(str(remediation).split())[:600]
            if isinstance(remediation, str) and remediation.strip()
            else f"Carry out {obligation} and file the resulting record against the asset."
        )

        severity_raw = str(entry.get("severity") or Severity.HIGH.value).strip().upper()
        try:
            severity = Severity(severity_raw)
        except ValueError:
            severity = Severity.HIGH

        grace = _as_int(entry.get("grace_days")) or 0
        grace = max(0, min(grace, _MAX_FREQUENCY_DAYS))

        fields = [normalise_field_key(v) for v in _as_str_list(entry.get("required_evidence_fields"))]

        payload: dict[str, Any] = {
            "clause": clause,
            "obligation": obligation[:MAX_OBLIGATION_CHARS],
            "text": block.text[:MAX_QUOTE_CHARS],
            "frequency_days": frequency,
            "applies_to_types": _as_str_list(entry.get("applies_to_types")),
            "applies_to_tags": _as_str_list(entry.get("applies_to_tags")),
            "evidence_types": evidence,
            "required_evidence_fields": [f for f in fields if f],
            "penalty": penalty_text,
            "severity": severity,
            "grace_days": grace,
            "remediation": remediation_text,
        }
        if owner_role:
            payload["owner_role"] = owner_role

        try:
            return RawRequirement.model_validate(payload)
        except ValidationError as exc:
            logger.debug(
                "model requirement failed catalogue schema",
                extra={"clause": clause, "error": exc.errors()[0].get("msg") if exc.errors() else str(exc)},
            )
            return None

    # -- assembly ------------------------------------------------------------------

    @staticmethod
    def _deduplicate(raws: Sequence[RawRequirement]) -> list[RawRequirement]:
        """Collapse duplicates on (clause, obligation), keeping the first occurrence."""
        seen: dict[tuple[str, str], RawRequirement] = {}
        for raw in raws:
            key = (slugify(raw.clause), slugify(raw.obligation))
            seen.setdefault(key, raw)
        return sorted(seen.values(), key=lambda r: (r.clause, r.obligation))[:MAX_REQUIREMENTS_PER_DOCUMENT]

    def _to_specs(
        self,
        raws: Sequence[RawRequirement],
        *,
        regulation: str,
        meta: DocumentMeta,
        blocks: Sequence[ClauseBlock],
    ) -> list[RequirementSpec]:
        """Wrap validated requirements as parsed specs citing the uploaded document."""
        by_slug = {block.slug: block for block in blocks}
        file_model = RegulationFile(
            regulation=regulation,
            title=meta.title,
            authority="",
            version=meta.document_date.isoformat() if meta.document_date else "",
            disclaimer=(
                f"Machine-parsed from '{meta.title}' (document {meta.document_id}) on "
                f"{utcnow().date().isoformat()}. Clause text is quoted verbatim from that document; "
                "the obligation, periodicity and scope are a machine normalisation and must be "
                "verified against the gazetted text before they are relied on in an audit."
            ),
            requirements=list(raws),
        )
        specs: list[RequirementSpec] = []
        for raw in raws:
            block = by_slug.get(slugify(raw.clause))
            source = SourceRef(
                document_id=meta.document_id,
                document_title=meta.title,
                document_type=DocumentType.REGULATION,
                snippet=(block.text if block else raw.text)[:600],
                char_start=block.char_start if block else None,
                char_end=block.char_end if block else None,
                relevance=1.0,
                extraction_confidence=1.0,
                retrieved_via="direct",
                document_date=meta.document_date,
            )
            specs.append(
                spec_from_raw(raw, regulation=regulation, file=file_model, provenance="parsed", source=source)
            )
        return specs


# ======================================================================================
# Small coercions
# ======================================================================================


def _as_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            return None
    return None


def _as_str_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if isinstance(item, (str, int, float)) and str(item).strip()]
    return []


__all__ = [
    "BATCH_CHAR_BUDGET",
    "MAX_CLAUSE_BLOCKS",
    "MAX_MODEL_CALLS",
    "MAX_OBLIGATION_CHARS",
    "MAX_QUOTE_CHARS",
    "OBLIGATION_GROUNDING_RATIO",
    "REGULATION_PROMPT_V1",
    "REGULATION_SCHEMA_V1",
    "REGULATION_SYSTEM_V1",
    "ClauseBlock",
    "ParseMethod",
    "ParsedRegulation",
    "RegulationParser",
    "evidence_types_from",
    "frequency_from_text",
    "heuristic_requirements",
    "normalise_clause_label",
    "obligation_is_grounded",
    "split_clauses",
]
