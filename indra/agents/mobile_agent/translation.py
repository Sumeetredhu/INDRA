"""Language detection and tag-safe machine translation for the field interface.

Two jobs, both load-bearing for the multilingual demo:

**Detection** is done over Unicode script ranges rather than a statistical model, because the five
supported languages live in three mutually exclusive scripts (Devanagari, Tamil, Kannada, Latin) and
a script test is exact, instant, and needs no wheel. The one genuine ambiguity — Hindi versus
Marathi, which share Devanagari — is broken with a discriminative common-word prior. Those two
languages differ in exactly the high-frequency function words a technician's sentence is made of
(``है``/``आहे``, ``और``/``आणि``, ``नहीं``/``नाही``), so a token-level vote over ~40 markers is both
cheap and reliable on utterance-length input.

**Translation** goes through the LLM router with a deliberately tight prompt, and implements
``docs/DECISIONS.md`` **D11**: plant tags are replaced with sentinels *before* the text reaches the
model and restored *after*. This is not defensive politeness — ``P-101`` round-tripped through
machine translation comes back as ``पी-१०१``, and every graph lookup, alert join, and equipment
registry hit downstream of that point fails silently. The mask is the reason the answer stays
connected to the plant.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Mapping

from indra.core.config import Settings
from indra.core.exceptions import IndraError, TranslationError
from indra.core.ids import content_hash
from indra.core.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from indra.core.contracts import CacheStore, LLMRouter

logger = get_logger(__name__)


# ======================================================================================
# Plant-tag masking (D11)
# ======================================================================================

TAG_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b[A-Z]{1,6}-[A-Z0-9]{1,8}(?:-[A-Z0-9]{1,8}){0,2}\b"
)
"""Plant-tag grammar: ``<LETTERS>-<ALNUM>`` with up to two extra hyphenated groups.

Matches ``P-101``, ``HX-205A``, ``TIC-101``, ``PSV-401B``, ``OISD-STD-118``, ``WO-2024-0342``.
A candidate is only treated as a tag if it also contains at least one digit, which keeps ordinary
hyphenated English (``FAIL-SAFE``) out of the mask. False positives are harmless — a masked token is
restored verbatim — whereas a false negative corrupts an identifier, so the pattern errs wide.
"""

_SENTINEL_PREFIX: Final[str] = "__TAG"
_SENTINEL_SUFFIX: Final[str] = "__"

_SENTINEL_STRICT: Final[re.Pattern[str]] = re.compile(r"_{1,4}\s*TAG\s*(\d{1,3})\s*_{1,4}", re.IGNORECASE)
"""Primary recovery pattern. Tolerates the whitespace and underscore-count drift models introduce."""

_SENTINEL_LOOSE: Final[re.Pattern[str]] = re.compile(r"_*\bTAG\s*[-_]?\s*(\d{1,3})\b_*", re.IGNORECASE)
"""Last-resort recovery for a model that dropped the underscores entirely."""

_INDIC_DIGITS: Final[Mapping[int, str]] = {
    **{0x0966 + i: str(i) for i in range(10)},  # Devanagari
    **{0x09E6 + i: str(i) for i in range(10)},  # Bengali
    **{0x0AE6 + i: str(i) for i in range(10)},  # Gujarati
    **{0x0BE6 + i: str(i) for i in range(10)},  # Tamil
    **{0x0C66 + i: str(i) for i in range(10)},  # Telugu
    **{0x0CE6 + i: str(i) for i in range(10)},  # Kannada
}
"""Indic digit → ASCII. Applied only inside sentinel recovery, never to user-visible text."""


@dataclass(frozen=True, slots=True)
class MaskedText:
    """Text with every plant tag replaced by a translation-proof sentinel."""

    text: str
    mapping: dict[str, str]
    """Sentinel → original tag, e.g. ``{"__TAG0__": "P-101"}``."""

    @property
    def tag_count(self) -> int:
        return len(self.mapping)

    @property
    def tags(self) -> list[str]:
        return list(self.mapping.values())


def mask_tags(text: str) -> MaskedText:
    """Replace plant tags with sentinels so machine translation cannot corrupt them (D11).

    Identical tags share one sentinel, which keeps the sentinel count low and makes restoration
    idempotent when a model repeats a token.
    """
    if not text:
        return MaskedText(text=text, mapping={})

    assigned: dict[str, str] = {}
    mapping: dict[str, str] = {}

    def _swap(match: re.Match[str]) -> str:
        raw = match.group(0)
        if not any(ch.isdigit() for ch in raw):
            return raw
        sentinel = assigned.get(raw)
        if sentinel is None:
            sentinel = f"{_SENTINEL_PREFIX}{len(assigned)}{_SENTINEL_SUFFIX}"
            assigned[raw] = sentinel
            mapping[sentinel] = raw
        return sentinel

    return MaskedText(text=TAG_PATTERN.sub(_swap, text), mapping=mapping)


def restore_tags(text: str, mapping: Mapping[str, str]) -> str:
    """Put the original plant tags back, tolerating sentinel drift introduced by the model.

    Three passes, cheapest first: exact substring, then a whitespace/underscore-tolerant regex, then
    a loose ``TAG<n>`` match. Anything still unresolved is logged rather than guessed at — a wrong
    tag substitution is worse than a visible placeholder.
    """
    if not mapping:
        return text

    by_index: dict[int, str] = {}
    for sentinel, tag in mapping.items():
        digits = "".join(ch for ch in sentinel if ch.isdigit())
        if digits:
            by_index[int(digits)] = tag

    restored = text
    for sentinel, tag in mapping.items():
        restored = restored.replace(sentinel, tag)

    if not _needs_recovery(restored, by_index):
        return restored

    normalised = restored.translate(dict(_INDIC_DIGITS))

    def _sub(match: re.Match[str]) -> str:
        index = int(match.group(1))
        return by_index.get(index, match.group(0))

    normalised = _SENTINEL_STRICT.sub(_sub, normalised)
    if _needs_recovery(normalised, by_index):
        normalised = _SENTINEL_LOOSE.sub(_sub, normalised)

    missing = [tag for index, tag in sorted(by_index.items()) if tag not in normalised]
    if missing:
        logger.warning(
            "translation dropped plant tags; downstream graph lookups may fail",
            extra={"missing_tags": missing, "expected_tags": len(by_index)},
        )
    return normalised


def _needs_recovery(text: str, by_index: Mapping[int, str]) -> bool:
    """True when at least one expected tag is absent from ``text``."""
    return any(tag not in text for tag in by_index.values())


# ======================================================================================
# Script-based language detection
# ======================================================================================

_SCRIPT_RANGES: Final[tuple[tuple[str, int, int], ...]] = (
    ("devanagari", 0x0900, 0x097F),
    ("devanagari", 0xA8E0, 0xA8FF),
    ("bengali", 0x0980, 0x09FF),
    ("gujarati", 0x0A80, 0x0AFF),
    ("tamil", 0x0B80, 0x0BFF),
    ("telugu", 0x0C00, 0x0C7F),
    ("kannada", 0x0C80, 0x0CFF),
    ("malayalam", 0x0D00, 0x0D7F),
    ("latin", 0x0041, 0x024F),
)

_SCRIPT_TO_LANGUAGE: Final[Mapping[str, str]] = {
    "tamil": "ta",
    "kannada": "kn",
    "telugu": "te",
    "bengali": "bn",
    "gujarati": "gu",
    "malayalam": "ml",
    "latin": "en",
}

_MARATHI_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "आहे", "आहेत", "नाही", "नाहीत", "आणि", "मध्ये", "तुम्ही", "तुमच्या", "पाहिजे",
        "कसे", "कसा", "कशी", "काय", "कुठे", "केव्हा", "झाला", "झाली", "झाले", "होते",
        "करावे", "करा", "मी", "आम्ही", "मला", "त्याचा", "त्याची", "चा", "ची", "चे",
        "वर", "खूप", "दुरुस्ती", "दाब", "गळती", "तपासणी", "बिघाड",
    }
)
"""Marathi-only high-frequency tokens. Every entry has a distinct Hindi counterpart below."""

_HINDI_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "है", "हैं", "नहीं", "और", "में", "आप", "आपके", "चाहिए", "कैसे", "कैसा", "कैसी",
        "क्या", "क्यों", "कहाँ", "कहां", "कब", "हुआ", "हुई", "था", "थी", "थे", "करना",
        "करें", "मैं", "हम", "मुझे", "उसका", "उसकी", "यह", "वह", "रहा", "रही", "गया",
        "बहुत", "मरम्मत", "दबाव", "रिसाव", "जाँच", "जांच", "खराबी",
    }
)
"""Hindi-only high-frequency tokens. Shared words (``पंप``, ``कृपया``, ``का``) are deliberately absent."""

_MARATHI_SUFFIXES: Final[tuple[str, ...]] = ("च्या", "ांनी", "ामध्ये", "ावर", "ाची", "ाचा", "ाचे")
"""Weak positional prior. Only consulted when the token vote ties."""

_TOKEN_SPLIT: Final[re.Pattern[str]] = re.compile(r"[\s।॥.,;:!?()\[\]{}\"'`/\\|]+")

_LANGUAGE_NAMES: Final[Mapping[str, str]] = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "ta": "Tamil",
    "kn": "Kannada",
    "te": "Telugu",
    "bn": "Bengali",
    "gu": "Gujarati",
    "ml": "Malayalam",
}


def language_name(code: str) -> str:
    """Human-readable language name for prompts and UI labels."""
    return _LANGUAGE_NAMES.get(code.lower(), code)


def script_of(text: str) -> str:
    """Return the dominant Unicode script of ``text``, or ``"unknown"``.

    Plant tags and digits are stripped first: a Hindi sentence containing ``P-101`` is still Hindi,
    and letting three Latin characters outvote the utterance would break the whole pipeline.
    """
    stripped = TAG_PATTERN.sub(" ", text)
    counts: dict[str, int] = {}
    for char in stripped:
        if not char.isalpha():
            continue
        code = ord(char)
        for script, low, high in _SCRIPT_RANGES:
            if low <= code <= high:
                counts[script] = counts.get(script, 0) + 1
                break
    if not counts:
        return "unknown"
    return max(sorted(counts), key=lambda script: counts[script])


def _tokens(text: str) -> list[str]:
    return [token for token in _TOKEN_SPLIT.split(text) if token]


def disambiguate_devanagari(text: str) -> str:
    """Separate Hindi from Marathi with a discriminative common-word vote.

    Returns ``"hi"`` or ``"mr"``. Hindi wins ties: it is by far the more likely input at an Indian
    refinery, so an unresolvable utterance should land on the higher prior rather than a coin flip.
    """
    tokens = _tokens(text)
    marathi = sum(1 for token in tokens if token in _MARATHI_MARKERS)
    hindi = sum(1 for token in tokens if token in _HINDI_MARKERS)
    if marathi != hindi:
        return "mr" if marathi > hindi else "hi"
    suffix_votes = sum(1 for token in tokens if token.endswith(_MARATHI_SUFFIXES))
    return "mr" if suffix_votes > 0 else "hi"


def detect_language(text: str, *, supported: tuple[str, ...], default: str) -> str:
    """Detect the language of ``text``, constrained to the configured supported set."""
    if not text or not text.strip():
        return default
    script = script_of(text)
    if script == "unknown":
        return default
    detected = disambiguate_devanagari(text) if script == "devanagari" else _SCRIPT_TO_LANGUAGE.get(script, default)
    if detected not in supported:
        logger.warning(
            "detected an unsupported language; falling back to the default",
            extra={"detected_language": detected, "script": script, "default_language": default},
        )
        return default
    return detected


# ======================================================================================
# Translator
# ======================================================================================

_TRANSLATION_SYSTEM_PROMPT: Final[str] = (
    "You are a precise translation engine for industrial plant maintenance text.\n"
    "Translate the user's message from {source} into {target}.\n"
    "Rules, all mandatory:\n"
    "1. Output ONLY the translation. No preamble, no quotes, no notes, no romanisation.\n"
    "2. Copy every placeholder of the form __TAG0__, __TAG1__ ... EXACTLY as written, including the\n"
    "   underscores and the ASCII digit. They are equipment identifiers, not words. Never translate,\n"
    "   transliterate, reorder the digits of, or drop a placeholder.\n"
    "3. Keep all numbers, units, and percentages in ASCII digits.\n"
    "4. Preserve line breaks and sentence order.\n"
    "5. Translate technical maintenance vocabulary the way a plant technician would say it."
)

_CACHE_PREFIX: Final[str] = "mobile:translation:"
_CACHE_TTL_S: Final[int] = 86_400
"""Translations of the same string never change, so a long TTL is safe and saves LLM budget."""


class ScriptTranslator:
    """Implements :class:`indra.core.contracts.Translator`.

    Detection is pure Python over Unicode ranges; translation is one tight LLM call with plant tags
    masked (D11). Both are safe to call with no API key — the router's stub provider answers, and
    the tag mask still round-trips, which is the property the graph depends on.
    """

    name: Final[str] = "script_translator"

    def __init__(self, settings: Settings, llm: LLMRouter, *, cache: CacheStore | None = None) -> None:
        self._settings = settings
        self._llm = llm
        self._cache = cache
        self._supported: tuple[str, ...] = tuple(settings.supported_languages)
        self._default: str = settings.default_language

    # -- detection ---------------------------------------------------------------------
    async def detect(self, text: str) -> str:
        """Return an ISO-639-1 code from ``settings.supported_languages``.

        Async to satisfy the contract; the work is a single pass over the string and stays well under
        a millisecond for utterance-length input, so dispatching it to a thread would cost more than
        it saves.
        """
        return detect_language(text, supported=self._supported, default=self._default)

    def detect_sync(self, text: str) -> str:
        """Synchronous detection, for callers already inside a worker thread."""
        return detect_language(text, supported=self._supported, default=self._default)

    # -- translation -------------------------------------------------------------------
    async def translate(self, text: str, *, target: str, source: str | None = None) -> str:
        """Translate ``text`` into ``target``, preserving plant tags verbatim (D11).

        Raises:
            TranslationError: every provider in the router chain failed. Callers that must not fail
                (the voice pipeline) catch this and fall back to the untranslated text with an
                explicit uncertainty flag, rather than silently pretending translation happened.
        """
        if not text or not text.strip():
            return text

        target_code = self._normalise(target)
        source_code = self._normalise(source) if source else self.detect_sync(text)
        if source_code == target_code:
            return text

        masked = mask_tags(text)
        cache_key = self._cache_key(masked.text, source_code, target_code)
        cached = await self._cache_get(cache_key)
        if cached is not None:
            return restore_tags(cached, masked.mapping)

        system = _TRANSLATION_SYSTEM_PROMPT.format(
            source=language_name(source_code), target=language_name(target_code)
        )
        try:
            raw, provider = await self._llm.generate(
                masked.text,
                system=system,
                temperature=self._settings.llm_temperature,
                max_tokens=self._settings.llm_max_output_tokens,
            )
        except IndraError as exc:
            raise TranslationError(
                f"Translation {source_code}->{target_code} failed; no provider produced output. "
                "Show the untranslated text and flag it rather than blocking the technician.",
                context={"source": source_code, "target": target_code, "chars": len(text)},
                cause=exc,
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive: router contract violation
            raise TranslationError(
                f"Translation {source_code}->{target_code} raised an untyped error from the LLM router.",
                context={"source": source_code, "target": target_code},
                cause=exc,
            ) from exc

        cleaned = _strip_model_chatter(raw)
        if not cleaned:
            raise TranslationError(
                f"Translation {source_code}->{target_code} returned empty output from {provider}.",
                context={"source": source_code, "target": target_code, "provider": provider},
            )

        await self._cache_set(cache_key, cleaned)
        result = restore_tags(cleaned, masked.mapping)
        logger.info(
            "translated text",
            extra={
                "source_language": source_code,
                "target_language": target_code,
                "provider": provider,
                "masked_tags": masked.tag_count,
                "chars_in": len(text),
                "chars_out": len(result),
            },
        )
        return result

    # -- helpers -----------------------------------------------------------------------
    def _normalise(self, code: str | None) -> str:
        if not code:
            return self._default
        lowered = code.strip().lower().split("-")[0]
        return lowered if lowered in self._supported else self._default

    def _cache_key(self, masked_text: str, source: str, target: str) -> str:
        return f"{_CACHE_PREFIX}{source}:{target}:{content_hash(masked_text)[:32]}"

    async def _cache_get(self, key: str) -> str | None:
        if self._cache is None:
            return None
        try:
            value = await self._cache.get(key)
        except Exception as exc:  # pragma: no cover - cache is best-effort by contract
            logger.debug("translation cache read failed", extra={"error": str(exc)})
            return None
        return value if isinstance(value, str) else None

    async def _cache_set(self, key: str, value: str) -> None:
        if self._cache is None:
            return
        try:
            await self._cache.set(key, value, ttl_s=_CACHE_TTL_S)
        except Exception as exc:  # pragma: no cover - cache is best-effort by contract
            logger.debug("translation cache write failed", extra={"error": str(exc)})


_CHATTER_PREFIXES: Final[tuple[str, ...]] = (
    "translation:", "translated text:", "here is the translation:", "sure,", "output:",
)

_QUOTE_PAIRS: Final[tuple[tuple[str, str], ...]] = (('"', '"'), ("'", "'"), ("“", "”"), ("«", "»"))


def _strip_model_chatter(raw: str) -> str:
    """Remove the preamble a chatty model adds despite instructions.

    Only leading boilerplate and a matched pair of wrapping quotes are removed; the body is never
    edited, because an over-eager cleaner that trims real content is worse than a noisy translation.
    """
    text = raw.strip()
    lowered = text.lower()
    for prefix in _CHATTER_PREFIXES:
        if lowered.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    for opening, closing in _QUOTE_PAIRS:
        if len(text) >= 2 and text.startswith(opening) and text.endswith(closing):
            return text[1:-1].strip()
    return text


__all__ = [
    "MaskedText",
    "ScriptTranslator",
    "TAG_PATTERN",
    "detect_language",
    "disambiguate_devanagari",
    "language_name",
    "mask_tags",
    "restore_tags",
    "script_of",
]
