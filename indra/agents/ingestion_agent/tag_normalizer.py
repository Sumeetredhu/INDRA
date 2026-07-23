"""OCR-error-tolerant plant tag resolution — ``docs/DECISIONS.md`` D5.

``P-l0l``, ``P—1O1``, ``PIOI`` and ``P 101`` are all the pump the technician means. Getting from
those strings to ``P-101`` is three separate problems, and this module keeps them separate:

1. **Separator normalisation** — en/em dashes, non-breaking hyphens, spaces and underscores all
   collapse to a single ASCII hyphen.
2. **Glyph confusion repair** — the confusable set (``l/1/I/|``, ``O/0/o/Q``, ``S/5``, ``B/8``,
   ``Z/2``, ``G/6``, ``A/4``) is resolved *positionally*: in the letter zone of a tag a ``0`` is an
   ``O``; in the digit zone an ``O`` is a ``0``. This is why a bare edit-distance match fails here
   and a structural one succeeds.
3. **Registry reconciliation** — ``rapidfuzz`` against the live equipment registry with
   ``settings.pid_tag_fuzzy_threshold``.

**The module never corrects silently.** When two registry candidates score within
:data:`AMBIGUITY_MARGIN` of each other, both are returned in ``alternatives`` and the confidence is
damped. A silently wrong tag on a plant floor sends a technician to the wrong pump; an admitted
ambiguity sends them to the right one with a question.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Final, Iterable, Sequence

from rapidfuzz import fuzz, process

from indra.core.config import Settings, get_settings
from indra.core.logging import get_logger
from indra.core.models import Confidence

logger = get_logger(__name__)


# --------------------------------------------------------------------------------------
# Grammar
# --------------------------------------------------------------------------------------

TAG_GRAMMAR: Final[re.Pattern[str]] = re.compile(r"^[A-Z]{1,3}[-–—]?\d{2,4}[A-Z]?$")
"""Plant tag grammar: 1–3 letter prefix, optional separator, 2–4 digits, optional suffix letter."""

TAG_CANDIDATE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z]{1,3}[-–—‑‒−_ ]?[0-9]{2,4}[A-Za-z]?)(?![A-Za-z0-9])"
)
"""Finds *clean* tag-shaped substrings in born-digital text. Strict on purpose: false positives
here become bogus graph nodes."""

TAG_OCR_CANDIDATE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z0-9|]{1,3}[-–—‑‒−_ ]?[A-Za-z0-9|]{2,4}[A-Za-z]?)(?![A-Za-z0-9])"
)
"""Permissive variant for OCR output, where the digits may have been read as letters. Only used on
text that carries an OCR confidence, and always filtered through :meth:`PlantTagNormalizer.normalize`."""

_SEPARATORS: Final[str] = "-–—‑‒−_ \t·•."
_SEPARATOR_RE: Final[re.Pattern[str]] = re.compile(f"[{re.escape(_SEPARATORS)}]+")
_JUNK_RE: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9|\-]")

#: Glyph confusion resolved towards a **letter** (used in the tag's prefix/suffix zone).
LETTER_FIX: Final[dict[str, str]] = {
    "0": "O", "1": "I", "|": "I", "l": "I", "5": "S", "8": "B", "2": "Z", "6": "G", "4": "A",
    "o": "O", "q": "Q", "Q": "O",
}

#: Glyph confusion resolved towards a **digit** (used in the tag's numeric zone).
DIGIT_FIX: Final[dict[str, str]] = {
    "O": "0", "o": "0", "Q": "0", "D": "0",
    "l": "1", "I": "1", "i": "1", "|": "1", "L": "1",
    "S": "5", "s": "5", "B": "8", "Z": "2", "z": "2", "G": "6", "A": "4", "T": "7",
}

#: Equipment-type hint by tag prefix. Shared with the P&ID vision pipeline and entity extraction.
EQUIPMENT_PREFIXES: Final[dict[str, str]] = {
    "P": "pump", "PU": "pump", "CP": "pump", "BP": "pump",
    "V": "vessel", "VS": "vessel", "D": "vessel", "KOD": "vessel",
    "T": "tank", "TK": "tank", "ST": "tank",
    "E": "heat_exchanger", "HE": "heat_exchanger", "EX": "heat_exchanger", "AC": "heat_exchanger",
    "C": "compressor", "K": "compressor", "CO": "compressor",
    "F": "filter", "FL": "filter", "STR": "filter",
    "R": "reactor", "RX": "reactor",
    "M": "motor", "MT": "motor",
    "CV": "valve", "HV": "valve", "XV": "valve", "PV": "valve", "TV": "valve", "LV": "valve",
    "FV": "valve", "PSV": "valve", "PRV": "valve", "SV": "valve", "BV": "valve", "GV": "valve",
    "PI": "instrument", "TI": "instrument", "FI": "instrument", "LI": "instrument",
    "PT": "instrument", "TT": "instrument", "FT": "instrument", "LT": "instrument",
    "PG": "instrument", "TG": "instrument", "AI": "instrument", "VI": "instrument",
    "PIC": "instrument", "TIC": "instrument", "FIC": "instrument", "LIC": "instrument",
    "PS": "instrument", "TS": "instrument", "FS": "instrument", "LS": "instrument",
    "AE": "instrument", "AT": "instrument", "ZI": "instrument",
}

#: Prefixes that are decidedly *not* equipment tags even though they fit the grammar.
_PREFIX_BLOCKLIST: Final[frozenset[str]] = frozenset(
    {"NO", "SR", "PG", "REV", "FIG", "TAB", "SEC", "PPM", "RPM", "MPA", "KPA", "PSI", "USD",
     "INR", "IST", "GMT", "UTC", "ISO", "IEC", "IEE", "API", "ANSI", "ASME", "ASTM", "BS",
     "IS", "DIN", "EN", "AM", "PM", "ID", "OD", "NB", "SCH", "QTY", "MAX", "MIN", "AVG"}
)


# --------------------------------------------------------------------------------------
# Scoring constants
#
# Local to tag resolution rather than in Settings: these describe how much a single OCR glyph
# repair should cost in trust, which is a property of the algorithm, not a deployment knob. The one
# genuine deployment knob — the rapidfuzz cutoff — comes from ``settings.pid_tag_fuzzy_threshold``.
# --------------------------------------------------------------------------------------

CORRECTION_PENALTY: Final[float] = 0.09
"""Confidence lost per glyph repaired. Three repairs on a five-character tag is barely a read."""

AMBIGUITY_MARGIN: Final[float] = 5.0
"""rapidfuzz points. Two registry candidates closer than this are reported as genuinely ambiguous."""

AMBIGUITY_DAMPING: Final[float] = 0.72
"""Multiplier applied when the top two candidates are within :data:`AMBIGUITY_MARGIN`."""

UNVERIFIED_CEILING: Final[float] = 0.74
"""A grammar-valid tag with no registry to check against can never be more than 'probably right'."""

MAX_ALTERNATIVES: Final[int] = 4
_MAX_PREFIX_LEN: Final[int] = 3
_MIN_DIGITS: Final[int] = 2
_MAX_DIGITS: Final[int] = 4
_MAX_STRUCTURAL: Final[int] = 5
"""How many structural readings are carried forward into registry reconciliation."""

# -- structural plausibility costs -------------------------------------------------------
#
# Counting glyph repairs alone picks the wrong reading: ``P-l0l`` needs two repairs to become
# ``P-101`` but only one to become ``P-10L``, and ``P-10L`` is not a tag any plant has ever issued.
# These costs encode the actual conventions of plant tag numbering so that the *plausible* reading
# wins even when it is the more heavily corrected one.

_DIGIT_COUNT_COST: Final[dict[int, float]] = {2: 0.6, 3: 0.0, 4: 0.3}
"""Tag numbers are three digits by convention; two or four are possible but less likely."""

_TRAIN_SUFFIXES: Final[frozenset[str]] = frozenset("ABCDEFGH")
"""Suffix letters denote redundant trains (P-101A / P-101B). Later letters are vanishingly rare."""

_ODD_SUFFIX_COST: Final[float] = 0.8
_AMBIGUOUS_SUFFIX_COST: Final[float] = 0.7
"""A suffix read from a confusable glyph is much more likely to have been a digit."""

_CORRECTION_COST: Final[float] = 1.0
_COST_WEIGHT: Final[float] = 0.06
"""Converts structural implausibility into rapidfuzz-comparable points during reconciliation."""

_AMBIGUOUS_GLYPHS: Final[frozenset[str]] = frozenset("lI1|Oo0QSs5Bb8Zz2Gg6Aa4Dd7Tt")


@dataclass(frozen=True, slots=True)
class TagResolution:
    """Full result of tag normalisation, including the reasoning behind the number.

    :meth:`PlantTagNormalizer.normalize` projects this down to the three-tuple the
    :class:`indra.core.contracts.TagNormalizer` protocol requires; everything inside the ingestion
    agent uses the rich form so that ``Confidence.rationale`` can say something true.
    """

    tag: str | None
    confidence: float
    alternatives: list[str] = field(default_factory=list)
    rationale: str = ""
    corrections: int = 0
    grammar_valid: bool = False
    registry_matched: bool = False
    raw: str = ""
    ambiguous: bool = False

    def as_confidence(self) -> Confidence:
        """Project into the domain :class:`Confidence` used on every extracted entity."""
        return Confidence(
            value=max(0.0, min(1.0, self.confidence)),
            rationale=self.rationale or "Plant tag normalisation",
            method="ocr" if self.corrections else "exact" if self.registry_matched else "heuristic",
        )


def _clean(raw: str) -> str:
    """Collapse separators and strip junk, preserving case (``l`` vs ``I`` still matters here)."""
    collapsed = _SEPARATOR_RE.sub("-", raw.strip())
    collapsed = _JUNK_RE.sub("", collapsed)
    return collapsed.strip("-")


def _fix_letters(text: str) -> tuple[str, int]:
    """Resolve ``text`` into A–Z, returning ``(letters, corrections)`` or ``("", -1)`` if impossible."""
    out: list[str] = []
    fixes = 0
    for ch in text:
        if "A" <= ch.upper() <= "Z":
            out.append(ch.upper())
            continue
        replacement = LETTER_FIX.get(ch)
        if replacement is None:
            return "", -1
        fixes += 1
        out.append(replacement)
    return "".join(out), fixes


def _fix_digits(text: str) -> tuple[str, int]:
    out: list[str] = []
    fixes = 0
    for ch in text:
        if ch.isdigit():
            out.append(ch)
            continue
        replacement = DIGIT_FIX.get(ch)
        if replacement is None:
            return "", -1
        fixes += 1
        out.append(replacement)
    return "".join(out), fixes


@dataclass(frozen=True, slots=True)
class StructuralCandidate:
    """One grammar-valid reading of a raw string, with its implausibility cost."""

    tag: str
    corrections: int
    cost: float


def _structural_candidates(cleaned: str) -> list[StructuralCandidate]:
    """Enumerate every grammar-valid reading of ``cleaned``, cheapest first.

    The search space is tiny — prefix length 1–3 × suffix present/absent — so brute force is both
    exhaustive and fast, and it avoids the failure mode of a single greedy split ("PI101" read as
    prefix ``P`` + number ``I101``). Ranking is by :attr:`StructuralCandidate.cost`, not by raw
    correction count; see ``_DIGIT_COUNT_COST`` for why.
    """
    if not cleaned:
        return []

    flat = cleaned.replace("-", "")
    explicit_prefix = cleaned.partition("-")[0] if "-" in cleaned else None

    found: dict[str, StructuralCandidate] = {}
    for prefix_len in range(1, _MAX_PREFIX_LEN + 1):
        if prefix_len >= len(flat):
            continue
        # Honour an explicit separator: the prefix must end exactly where the author put the dash.
        if explicit_prefix is not None and prefix_len != len(explicit_prefix):
            continue
        prefix_raw, rest = flat[:prefix_len], flat[prefix_len:]
        for suffix_len in (0, 1):
            if suffix_len and len(rest) < _MIN_DIGITS + 1:
                continue
            split = len(rest) - suffix_len
            digits_raw, suffix_raw = rest[:split], rest[split:]
            if not (_MIN_DIGITS <= len(digits_raw) <= _MAX_DIGITS):
                continue
            prefix, p_fix = _fix_letters(prefix_raw)
            if p_fix < 0:
                continue
            digits, d_fix = _fix_digits(digits_raw)
            if d_fix < 0:
                continue
            suffix, s_fix = _fix_letters(suffix_raw) if suffix_raw else ("", 0)
            if s_fix < 0:
                continue
            candidate = f"{prefix}-{digits}{suffix}"
            if not TAG_GRAMMAR.match(candidate):
                continue

            corrections = p_fix + d_fix + s_fix
            cost = _CORRECTION_COST * corrections + _DIGIT_COUNT_COST.get(len(digits), 0.5)
            if suffix:
                if suffix not in _TRAIN_SUFFIXES:
                    cost += _ODD_SUFFIX_COST
                if suffix_raw in _AMBIGUOUS_GLYPHS:
                    cost += _AMBIGUOUS_SUFFIX_COST

            existing = found.get(candidate)
            if existing is None or cost < existing.cost:
                found[candidate] = StructuralCandidate(candidate, corrections, round(cost, 4))

    return sorted(found.values(), key=lambda c: (c.cost, len(c.tag)))


class PlantTagNormalizer:
    """Implements :class:`indra.core.contracts.TagNormalizer`.

    Stateless apart from an optional default registry, so a single instance is safe to share across
    the whole ingestion pipeline and across threads.
    """

    name: str = "plant_tag_normalizer"

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        registry: Sequence[str] | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._registry: tuple[str, ...] = tuple(self._prepare(registry or ()))

    # -- registry ---------------------------------------------------------------------
    @staticmethod
    def _prepare(registry: Iterable[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in registry:
            tag = (item or "").strip().upper()
            if tag and tag not in seen:
                seen.add(tag)
                out.append(tag)
        return out

    def set_registry(self, registry: Sequence[str]) -> None:
        """Replace the default registry. Called after the equipment list is loaded from the graph."""
        self._registry = tuple(self._prepare(registry))
        logger.debug("tag registry loaded", extra={"registry_size": len(self._registry)})

    @property
    def registry(self) -> tuple[str, ...]:
        return self._registry

    # -- protocol surface -------------------------------------------------------------
    def normalize(
        self,
        raw: str,
        *,
        registry: Sequence[str] | None = None,
    ) -> tuple[str | None, float, list[str]]:
        """Return ``(tag_or_None, confidence, alternatives)`` per the protocol."""
        resolution = self.resolve(raw, registry=registry)
        return resolution.tag, resolution.confidence, list(resolution.alternatives)

    # -- rich surface -----------------------------------------------------------------
    def resolve(self, raw: str, *, registry: Sequence[str] | None = None) -> TagResolution:
        """Normalise ``raw`` into a plant tag with a full audit trail.

        Args:
            raw: The surface form, typically straight out of OCR.
            registry: Known-good tags to reconcile against. Falls back to the instance registry.

        Returns:
            A :class:`TagResolution`. ``tag`` is ``None`` when nothing defensible could be produced.
        """
        source = (raw or "").strip()
        if not source:
            return TagResolution(None, 0.0, rationale="Empty input", raw=raw or "")

        known = self._prepare(registry) if registry is not None else list(self._registry)
        cleaned = _clean(source)
        if not cleaned:
            return TagResolution(None, 0.0, rationale=f"No tag-like characters in {source!r}",
                                 raw=source)

        candidates = _structural_candidates(cleaned)
        candidates = [c for c in candidates if c.tag.split("-", 1)[0] not in _PREFIX_BLOCKLIST]
        if not candidates:
            blocked = _structural_candidates(cleaned)
            if blocked:
                prefix = blocked[0].tag.split("-", 1)[0]
                return TagResolution(
                    None, 0.0,
                    rationale=f"{blocked[0].tag!r} matches the tag grammar but {prefix!r} is a known "
                              f"non-equipment prefix (standard, unit, or document reference)",
                    raw=source, grammar_valid=True,
                )
            return self._registry_only(source, cleaned, known)

        if not known:
            return self._unverified(source, candidates)
        return self._reconcile(source, candidates, known)

    # -- internals --------------------------------------------------------------------
    @staticmethod
    def _unverified(source: str, candidates: list[StructuralCandidate]) -> TagResolution:
        """No registry to check against: report the most plausible reading, capped in confidence."""
        best = candidates[0]
        alternatives = [c.tag for c in candidates[1:MAX_ALTERNATIVES + 1]]
        confidence = max(0.15, UNVERIFIED_CEILING - CORRECTION_PENALTY * best.corrections)
        if alternatives and abs(candidates[1].cost - best.cost) < 1.0:
            confidence *= AMBIGUITY_DAMPING
        rationale = (
            f"Matches the plant tag grammar after {best.corrections} glyph correction(s) on "
            f"{source!r}; no equipment registry available to verify against"
            if best.corrections else
            f"Matches the plant tag grammar exactly; no equipment registry available to verify against"
        )
        if alternatives:
            rationale += f"; also readable as {', '.join(alternatives)}"
        return TagResolution(
            tag=best.tag,
            confidence=round(confidence, 4),
            alternatives=alternatives,
            rationale=rationale,
            corrections=best.corrections,
            grammar_valid=True,
            registry_matched=False,
            raw=source,
            ambiguous=bool(alternatives) and abs(candidates[1].cost - best.cost) < 1.0,
        )

    def _reconcile(
        self,
        source: str,
        candidates: list[StructuralCandidate],
        known: list[str],
    ) -> TagResolution:
        """Score every plausible reading against the live registry and pick the best combination.

        Reconciling *all* the structural readings rather than only the cheapest one is what makes
        ``PIOI`` resolve to ``P-101`` when the registry contains ``P-101``: the more heavily
        corrected reading wins because it is the one the plant actually has.
        """
        cutoff = float(self._settings.pid_tag_fuzzy_threshold)

        scored: list[tuple[float, str, StructuralCandidate, float]] = []
        for candidate in candidates[:_MAX_STRUCTURAL]:
            if candidate.tag in known:
                pairs: list[tuple[str, float]] = [(candidate.tag, 100.0)]
            else:
                pairs = self._fuzzy(candidate.tag, known, cutoff)
            for tag, score in pairs:
                combined = score - _COST_WEIGHT * candidate.cost * 100.0
                scored.append((combined, tag, candidate, score))

        if not scored:
            best = candidates[0]
            confidence = max(0.12, (UNVERIFIED_CEILING - CORRECTION_PENALTY * best.corrections) * 0.8)
            return TagResolution(
                tag=best.tag,
                confidence=round(confidence, 4),
                alternatives=[c.tag for c in candidates[1:MAX_ALTERNATIVES + 1]],
                rationale=(
                    f"Grammar-valid tag {best.tag!r} read from {source!r} with {best.corrections} "
                    f"glyph correction(s), but no registry entry scored above {cutoff:.0f}; "
                    f"treated as a new or not-yet-registered asset"
                ),
                corrections=best.corrections,
                grammar_valid=True,
                registry_matched=False,
                raw=source,
            )

        scored.sort(key=lambda item: (-item[0], item[2].cost))
        best_combined, best_tag, best_candidate, best_score = scored[0]

        alternatives: list[str] = []
        runner_up: tuple[float, str] | None = None
        for combined, tag, _candidate, _score in scored[1:]:
            if tag == best_tag or tag in alternatives:
                continue
            if runner_up is None:
                runner_up = (combined, tag)
            if len(alternatives) < MAX_ALTERNATIVES:
                alternatives.append(tag)

        ambiguous = runner_up is not None and (best_combined - runner_up[0]) <= AMBIGUITY_MARGIN
        confidence = (best_score / 100.0) - CORRECTION_PENALTY * best_candidate.corrections
        rationale = (
            f"rapidfuzz matched {source!r} → {best_tag} at {best_score:.0f}/100 "
            f"(cutoff {cutoff:.0f}) via the reading {best_candidate.tag!r} "
            f"with {best_candidate.corrections} glyph correction(s)"
        )
        if ambiguous and runner_up is not None:
            confidence *= AMBIGUITY_DAMPING
            rationale += (
                f"; {runner_up[1]} is within {AMBIGUITY_MARGIN:.0f} points of the same read, so the "
                f"correction is genuinely ambiguous and both are reported rather than one chosen"
            )
            logger.info(
                "ambiguous tag read reported rather than corrected",
                extra={"raw": source, "chosen": best_tag, "alternative": runner_up[1]},
            )

        return TagResolution(
            tag=best_tag,
            confidence=round(max(0.1, min(1.0, confidence)), 4),
            alternatives=alternatives,
            rationale=rationale,
            corrections=best_candidate.corrections,
            grammar_valid=True,
            registry_matched=True,
            raw=source,
            ambiguous=ambiguous,
        )

    def _registry_only(self, source: str, cleaned: str, known: list[str]) -> TagResolution:
        """Last resort: the string does not fit the grammar at all, so try the registry directly."""
        if not known:
            return TagResolution(
                None, 0.0,
                rationale=f"{source!r} does not match the plant tag grammar "
                          f"(1-3 letters, 2-4 digits, optional suffix letter)",
                raw=source,
            )
        cutoff = float(self._settings.pid_tag_fuzzy_threshold)
        matches = self._fuzzy(cleaned.upper(), known, cutoff)
        if not matches:
            return TagResolution(
                None, 0.0,
                rationale=f"{source!r} matches neither the tag grammar nor any registry entry "
                          f"above {cutoff:.0f}",
                raw=source,
            )
        top_tag, top_score = matches[0]
        ambiguous = len(matches) > 1 and (top_score - matches[1][1]) <= AMBIGUITY_MARGIN
        confidence = (top_score / 100.0) * (AMBIGUITY_DAMPING if ambiguous else 0.85)
        return TagResolution(
            tag=top_tag,
            confidence=round(max(0.1, min(1.0, confidence)), 4),
            alternatives=[tag for tag, _ in matches[1:MAX_ALTERNATIVES + 1]],
            rationale=(
                f"{source!r} is not grammar-valid; recovered by fuzzy registry match to {top_tag} "
                f"at {top_score:.0f}/100"
                + ("; a second candidate scored within the ambiguity margin" if ambiguous else "")
            ),
            corrections=0,
            grammar_valid=False,
            registry_matched=True,
            raw=source,
            ambiguous=ambiguous,
        )

    @staticmethod
    def _fuzzy(query: str, known: list[str], cutoff: float) -> list[tuple[str, float]]:
        """rapidfuzz lookup, wrapped so a scorer failure degrades to 'no match' rather than a crash."""
        if not query or not known:
            return []
        try:
            raw = process.extract(
                query, known, scorer=fuzz.WRatio, limit=MAX_ALTERNATIVES + 1, score_cutoff=cutoff
            )
        except (ValueError, TypeError) as exc:  # pragma: no cover - defensive
            logger.warning("rapidfuzz lookup failed; skipping registry reconciliation",
                           extra={"query": query, "error": str(exc)})
            return []
        return [(str(choice), float(score)) for choice, score, _ in raw]


def equipment_type_for(tag: str) -> str:
    """Map a normalised tag to a coarse equipment type using its prefix.

    Returns ``"unknown"`` when the prefix is not in :data:`EQUIPMENT_PREFIXES`. Longest prefix wins,
    so ``PIC-101`` is an instrument rather than a pump.
    """
    prefix = (tag or "").split("-", 1)[0].upper()
    for length in range(min(len(prefix), 3), 0, -1):
        hit = EQUIPMENT_PREFIXES.get(prefix[:length])
        if hit and length == len(prefix):
            return hit
    return EQUIPMENT_PREFIXES.get(prefix, "unknown")


__all__ = [
    "AMBIGUITY_DAMPING",
    "AMBIGUITY_MARGIN",
    "CORRECTION_PENALTY",
    "DIGIT_FIX",
    "EQUIPMENT_PREFIXES",
    "LETTER_FIX",
    "PlantTagNormalizer",
    "StructuralCandidate",
    "TAG_CANDIDATE_RE",
    "TAG_GRAMMAR",
    "TAG_OCR_CANDIDATE_RE",
    "TagResolution",
    "UNVERIFIED_CEILING",
    "equipment_type_for",
]
