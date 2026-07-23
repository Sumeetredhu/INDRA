"""Deterministic offline chat provider — the last link in the chain, and the one that never breaks.

This is not a mock. It is a real :class:`~indra.core.contracts.ChatProvider` that runs through the
same prompt construction, the same JSON extraction, and the same schema validation as Gemini does
(see :class:`indra.llm.base.BaseChatProvider`). Tests and demo-safe mode therefore exercise the
production code path; only the token source differs.

Two output modes, chosen by inspecting the prompt exactly as a real model would:

* **Structured.** When the prompt carries :data:`~indra.llm.base.JSON_SCHEMA_MARKER`, the schema is
  recovered and an instance is synthesised that satisfies it — types, enums, bounds, required
  fields. Field *names* drive the values, so ``confidence`` gets 0–1, ``equipment_tag`` gets a tag
  that actually appears in the prompt, and ``severity`` gets a real severity.
* **Prose.** Otherwise it writes an industrial answer that cites the passage indices present in the
  prompt. Citation shape matters: the Copilot's grounding checks, the citation renderer, and the
  "Explain How I Know This" panel all parse ``[n]`` markers, and a stub that omitted them would let
  a whole class of bug through the test suite.

Determinism comes from ``settings.llm_seed`` mixed with a digest of the prompt: the same question
always produces the same answer, across processes and machines, which is what makes a recorded demo
reproducible and snapshot tests possible.
"""

from __future__ import annotations

import hashlib
import random
import re
from typing import Any, AsyncIterator, Final, Mapping, Sequence

from indra.core.config import Settings
from indra.core.logging import get_logger
from indra.llm.base import (
    BaseChatProvider,
    JsonSchema,
    estimate_tokens,
    extract_schema_from_prompt,
)

logger = get_logger(__name__)

#: Recognises plant tags (``P-101``, ``HX-204B``, ``PSV-12``) so synthesised output talks about the
#: equipment actually under discussion instead of inventing an asset that is not in the graph.
TAG_RE: Final[re.Pattern[str]] = re.compile(r"\b([A-Z]{1,4}-\d{2,4}[A-Z]?)\b")

#: Every citation shape the retrieval prompts emit: ``[3]``, ``Passage 3``, ``Source 3``, ``[S3]``.
_CITATION_RES: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\[\s*S?(\d{1,2})\s*\]"),
    re.compile(r"\bpassage\s+(\d{1,2})\b", re.IGNORECASE),
    re.compile(r"\bsource\s+(\d{1,2})\b", re.IGNORECASE),
    re.compile(r"\bdocument\s+(\d{1,2})\b", re.IGNORECASE),
)

_FINDINGS: Final[tuple[str, ...]] = (
    "a rising trend in the recorded condition parameters",
    "repeated corrective work against the same failure mode",
    "a reading approaching the OEM-stated limit",
    "an inspection finding that was never closed out",
    "a maintenance interval that has drifted past its nominal period",
    "corroborating evidence across two independent documents",
)

_MECHANISMS: Final[tuple[str, ...]] = (
    "bearing wear progressing under sustained vibration",
    "seal degradation consistent with the recorded leak reports",
    "cavitation at the suction side during low-flow operation",
    "misalignment introduced at the last coupling replacement",
    "fouling reducing heat-transfer effectiveness",
    "lubricant contamination shortening the bearing service life",
)

_ACTIONS: Final[tuple[str, ...]] = (
    "Schedule a vibration survey within the next 7 days",
    "Raise a work order for bearing inspection at the next shutdown window",
    "Verify the last recorded reading against the field instrument",
    "Confirm the OEM threshold in the manual before acting on the trend",
    "Escalate to the reliability engineer for a root-cause review",
)

_CAVEATS: Final[tuple[str, ...]] = (
    "Part of this evidence originates from a handwritten log, so treat the exact figures as indicative.",
    "The most recent supporting document predates the last overhaul; confirm it is still current.",
    "Only two independent sources support this conclusion; a third would materially raise confidence.",
)

#: Generic industrial noun phrases used when a schema wants a string with no name-based hint.
_PHRASES: Final[tuple[str, ...]] = (
    "elevated bearing vibration",
    "scheduled preventive maintenance",
    "pressure vessel inspection record",
    "shift log observation",
    "root cause analysis finding",
    "OEM recommended operating limit",
    "condition monitoring reading",
)

_SEVERITIES: Final[tuple[str, ...]] = ("INFO", "LOW", "WARNING", "HIGH", "CRITICAL")

#: Field-name fragments that mean "this is a 0–1 score".
_SCORE_HINTS: Final[tuple[str, ...]] = (
    "confidence", "score", "probability", "relevance", "similarity", "strength", "ratio",
)

#: Field-name fragments that mean "this is a whole number".
_COUNT_HINTS: Final[tuple[str, ...]] = (
    "count", "_n", "days", "hours", "minutes", "index", "order", "hops", "total", "horizon",
)

#: Field-name fragments that mean "this is prose".
_PROSE_HINTS: Final[tuple[str, ...]] = (
    "explanation", "rationale", "reason", "summary", "answer", "detail", "narrative",
    "description", "finding", "message", "body", "text", "note",
)


class StubChatProvider(BaseChatProvider):
    """Seeded, deterministic, always-available chat provider.

    Args:
        settings: Process settings; ``llm_seed`` fixes the output stream.
    """

    name = "stub"
    supports_json = True

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)

    async def is_available(self) -> bool:
        """Always ``True``. This provider exists so the chain can never be empty."""
        return True

    # -- generation ---------------------------------------------------------------

    async def _complete(
        self,
        prompt: str,
        *,
        system: str | None,
        temperature: float,
        max_tokens: int,
        stop: Sequence[str] | None,
    ) -> str:
        rng = self._rng(prompt, system)
        schema = extract_schema_from_prompt(prompt)
        if schema is not None:
            text = self._synthesise_json(prompt, schema, rng)
        else:
            text = self._synthesise_prose(prompt, rng, max_tokens=max_tokens)
        if stop:
            for marker in stop:
                if marker and marker in text:
                    text = text.split(marker, 1)[0]
        logger.debug(
            "stub completion produced",
            extra={"structured": schema is not None, "tokens": estimate_tokens(text)},
        )
        return text

    async def _stream_tokens(
        self,
        prompt: str,
        *,
        system: str | None,
        temperature: float,
    ) -> AsyncIterator[str]:
        """Stream in sentence-sized pieces so SSE consumers see realistic chunking."""
        text = await self._complete(
            prompt,
            system=system,
            temperature=temperature,
            max_tokens=self.settings.llm_max_output_tokens,
            stop=None,
        )
        for piece in re.split(r"(?<=[.!?])\s+", text):
            if piece:
                yield piece + " "

    # -- determinism --------------------------------------------------------------

    def _rng(self, prompt: str, system: str | None) -> random.Random:
        """Seed a generator from the settings seed and a digest of the request.

        Mixing the prompt in means two different questions get two different answers, while the
        same question is byte-identical on every run — the property a recorded demo depends on.
        """
        digest = hashlib.sha256(f"{system or ''}\x00{prompt}".encode("utf-8")).digest()
        seed = self.settings.llm_seed ^ int.from_bytes(digest[:8], "big")
        return random.Random(seed)

    # -- prose --------------------------------------------------------------------

    def _synthesise_prose(self, prompt: str, rng: random.Random, *, max_tokens: int) -> str:
        """Write a citation-shaped industrial answer grounded in the prompt's own passages."""
        citations = self._citation_indices(prompt)
        tags = self._tags(prompt)
        subject = tags[0] if tags else "the asset under review"
        question = self._question(prompt)

        primary = self._cite(citations, rng, count=2)
        secondary = self._cite(citations, rng, count=1, offset=2)

        sentences: list[str] = []
        sentences.append(
            f"Based on the retrieved plant records for {subject}, the evidence points to "
            f"{rng.choice(_FINDINGS)}{primary}."
        )
        sentences.append(
            f"The most consistent explanation across those sources is "
            f"{rng.choice(_MECHANISMS)}{secondary}."
        )
        if len(tags) > 1:
            sentences.append(
                f"The same pattern appears on {tags[1]}, which shares the service and duty cycle, "
                f"so treat this as a fleet-level observation rather than a single-asset anomaly."
            )
        if question:
            sentences.append(
                f"Directly addressing the question — {question} — the records support the "
                f"conclusion above but do not settle it beyond doubt."
            )
        sentences.append(f"{rng.choice(_CAVEATS)}")
        sentences.append(f"Recommended action: {rng.choice(_ACTIONS)}.")

        if not citations:
            sentences.append(
                "No indexed passages were supplied with this request, so this response is "
                "reasoning-only and must be verified against plant records before acting."
            )

        text = " ".join(sentences)
        budget = max(64, max_tokens)
        if estimate_tokens(text) > budget:
            from indra.llm.base import truncate_to_tokens  # noqa: PLC0415 - avoids a cycle at import

            text = truncate_to_tokens(text, budget)
        return text

    @staticmethod
    def _citation_indices(prompt: str) -> list[int]:
        """Collect every passage index the prompt exposes, in ascending order."""
        found: set[int] = set()
        for pattern in _CITATION_RES:
            for match in pattern.finditer(prompt):
                try:
                    value = int(match.group(1))
                except ValueError:  # pragma: no cover - regex guarantees digits
                    continue
                if 1 <= value <= 50:
                    found.add(value)
        return sorted(found)

    @staticmethod
    def _tags(prompt: str) -> list[str]:
        """Plant tags mentioned in the prompt, de-duplicated, in first-appearance order."""
        seen: list[str] = []
        for match in TAG_RE.finditer(prompt):
            tag = match.group(1)
            if tag not in seen:
                seen.append(tag)
        return seen

    @staticmethod
    def _question(prompt: str) -> str:
        """Recover the operator's question from the prompt, if one is identifiable."""
        for line in prompt.splitlines():
            stripped = line.strip()
            if stripped.endswith("?") and 8 <= len(stripped) <= 200:
                return stripped.rstrip("?")
        return ""

    @staticmethod
    def _cite(citations: Sequence[int], rng: random.Random, *, count: int, offset: int = 0) -> str:
        """Render ``[1][2]``-style markers drawn from the passages actually present."""
        if not citations:
            return ""
        pool = list(citations[offset:]) or list(citations)
        picks = pool[: max(1, min(count, len(pool)))]
        if len(pool) > len(picks) and rng.random() < 0.5:
            extra = rng.choice(pool[len(picks) :])
            if extra not in picks:
                picks.append(extra)
        return " " + "".join(f"[{index}]" for index in sorted(picks))

    # -- structured ---------------------------------------------------------------

    def _synthesise_json(self, prompt: str, schema: JsonSchema, rng: random.Random) -> str:
        """Build a schema-valid instance and render it the way a model would — fenced JSON."""
        import json  # noqa: PLC0415 - local to keep the module's import surface minimal

        context = _PromptContext(
            tags=self._tags(prompt),
            citations=self._citation_indices(prompt),
        )
        instance = _synthesise(schema, rng, context, name="root")
        return "```json\n" + json.dumps(instance, indent=2, default=str) + "\n```"


class _PromptContext:
    """Facts pulled from the prompt that make synthesised values plausible rather than random."""

    __slots__ = ("tags", "citations")

    def __init__(self, *, tags: Sequence[str], citations: Sequence[int]) -> None:
        self.tags = list(tags)
        self.citations = list(citations)


def _synthesise(schema: JsonSchema, rng: random.Random, context: _PromptContext, *, name: str) -> Any:
    """Produce a value satisfying ``schema``, biased by the field ``name``."""
    if "const" in schema:
        return schema["const"]

    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return enum[rng.randrange(len(enum))]

    branches = schema.get("anyOf") or schema.get("oneOf")
    if isinstance(branches, list) and branches:
        for branch in branches:
            if isinstance(branch, Mapping) and branch.get("type") != "null":
                return _synthesise(branch, rng, context, name=name)
        first = branches[0]
        return _synthesise(first, rng, context, name=name) if isinstance(first, Mapping) else None

    declared = schema.get("type")
    if isinstance(declared, list):
        declared = next((option for option in declared if option != "null"), "string")
    if not isinstance(declared, str):
        if isinstance(schema.get("properties"), Mapping):
            declared = "object"
        elif isinstance(schema.get("items"), Mapping):
            declared = "array"
        else:
            declared = "string"

    if declared == "object":
        return _synthesise_object(schema, rng, context)
    if declared == "array":
        return _synthesise_array(schema, rng, context, name=name)
    if declared == "boolean":
        return rng.random() < 0.5
    if declared == "integer":
        return _synthesise_integer(schema, rng, name)
    if declared == "number":
        return _synthesise_number(schema, rng, name)
    if declared == "null":
        return None
    return _synthesise_string(schema, rng, context, name)


def _synthesise_object(schema: JsonSchema, rng: random.Random, context: _PromptContext) -> dict[str, Any]:
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return {}
    required = schema.get("required")
    required_keys = set(required) if isinstance(required, list) else set(properties)
    result: dict[str, Any] = {}
    for key, sub_schema in properties.items():
        if not isinstance(sub_schema, Mapping):
            continue
        # Optional fields are populated too: a half-empty object is a worse test fixture than a
        # full one, and the schema permits every key it declares.
        if key not in required_keys and rng.random() < 0.15:
            continue
        result[key] = _synthesise(sub_schema, rng, context, name=key)
    return result


def _synthesise_array(schema: JsonSchema, rng: random.Random, context: _PromptContext, *, name: str) -> list[Any]:
    items = schema.get("items")
    minimum = schema.get("minItems")
    maximum = schema.get("maxItems")
    low = minimum if isinstance(minimum, int) else 1
    high = maximum if isinstance(maximum, int) else max(low, 3)
    length = rng.randint(low, max(low, high))
    if not isinstance(items, Mapping):
        return [rng.choice(_PHRASES) for _ in range(length)]
    return [_synthesise(items, rng, context, name=name) for _ in range(length)]


def _synthesise_integer(schema: JsonSchema, rng: random.Random, name: str) -> int:
    minimum = schema.get("minimum")
    maximum = schema.get("maximum")
    low = int(minimum) if isinstance(minimum, (int, float)) else 1
    lowered = name.lower()
    if any(hint in lowered for hint in _COUNT_HINTS):
        high = int(maximum) if isinstance(maximum, (int, float)) else max(low + 1, 30)
    else:
        high = int(maximum) if isinstance(maximum, (int, float)) else max(low + 1, 10)
    return rng.randint(low, max(low, high))


def _synthesise_number(schema: JsonSchema, rng: random.Random, name: str) -> float:
    minimum = schema.get("minimum")
    maximum = schema.get("maximum")
    lowered = name.lower()
    if any(hint in lowered for hint in _SCORE_HINTS):
        low = float(minimum) if isinstance(minimum, (int, float)) else 0.0
        high = float(maximum) if isinstance(maximum, (int, float)) else 1.0
        # Bias towards the defensible middle: a stub that always claims 0.99 confidence would let
        # confidence-banding bugs through untested.
        return round(min(high, max(low, rng.uniform(0.55, 0.92))), 3)
    low = float(minimum) if isinstance(minimum, (int, float)) else 0.0
    high = float(maximum) if isinstance(maximum, (int, float)) else max(low + 1.0, 100.0)
    return round(rng.uniform(low, high), 2)


def _synthesise_string(schema: JsonSchema, rng: random.Random, context: _PromptContext, name: str) -> str:
    lowered = name.lower()
    fmt = schema.get("format")

    if fmt == "date" or lowered.endswith("_date") or lowered.endswith("_on"):
        return f"2026-{rng.randint(1, 7):02d}-{rng.randint(1, 28):02d}"
    if fmt == "date-time" or lowered.endswith("_at"):
        return f"2026-{rng.randint(1, 7):02d}-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:00:00+00:00"
    if "severity" in lowered or "urgency" in lowered:
        return _SEVERITIES[rng.randrange(len(_SEVERITIES))]
    if "tag" in lowered or "equipment" in lowered:
        return context.tags[0] if context.tags else "P-101"
    if lowered.endswith("_id") or lowered == "id":
        return f"stub_{rng.getrandbits(32):08x}"
    if any(hint in lowered for hint in _PROSE_HINTS):
        clause = rng.choice(_FINDINGS)
        mechanism = rng.choice(_MECHANISMS)
        citation = f" [{context.citations[0]}]" if context.citations else ""
        return f"Plant records show {clause}, consistent with {mechanism}{citation}."
    if "action" in lowered or "recommend" in lowered:
        return rng.choice(_ACTIONS)
    if "method" in lowered or "type" in lowered or "kind" in lowered:
        return "heuristic"
    if "name" in lowered or "title" in lowered:
        subject = context.tags[0] if context.tags else "Unit 3"
        return f"{subject} condition summary"

    result = rng.choice(_PHRASES)
    min_length = schema.get("minLength")
    if isinstance(min_length, int) and len(result) < min_length:
        result = (result + " ") * (min_length // max(1, len(result)) + 1)
        result = result[:min_length].strip() or result[:min_length]
    max_length = schema.get("maxLength")
    if isinstance(max_length, int) and len(result) > max_length:
        result = result[:max_length]
    return result


__all__ = ["StubChatProvider", "TAG_RE"]
