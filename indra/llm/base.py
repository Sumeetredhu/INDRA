"""Shared provider machinery: HTTP plumbing, retry policy, token estimation, JSON handling.

Everything in :mod:`indra.llm` that is not provider-specific lives here so that adding a provider is
writing one ``_complete`` method, not re-implementing error mapping and schema validation.

Three things this module owns:

* **Error mapping.** Every provider funnels HTTP status codes through :func:`raise_for_status` so the
  router sees exactly three shapes: :class:`RateLimitError` (fail over now — a 429 against a *daily*
  quota is not transient), a retryable :class:`LLMError` (5xx / timeout — back off and retry), and
  :class:`ProviderUnavailableError` (misconfigured or dead — skip permanently for this call).
* **JSON contract.** :func:`build_json_prompt` embeds the requested schema in the prompt behind a
  stable marker, and :func:`parse_json_response` extracts and validates the reply. The stub provider
  reads the same marker, which is what makes tests exercise the *real* code path rather than a
  bypass.
* **Token estimation.** Deliberately heuristic by default — see :func:`estimate_tokens`.
"""

from __future__ import annotations

import json
import math
import re
import time
from typing import Any, AsyncIterator, Final, Iterable, Literal, Mapping, Sequence

import httpx
from tenacity import AsyncRetrying, RetryError, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from indra.core.config import Settings
from indra.core.exceptions import (
    EmbeddingError,
    LLMError,
    ProviderUnavailableError,
    RateLimitError,
    ResponseParsingError,
)
from indra.core.logging import get_logger

logger = get_logger(__name__)

# --------------------------------------------------------------------------------------
# Implementation constants
#
# These are *not* product tunables (those all live in ``indra.core.config``). They are internal
# derivations of settings that exist so a dead dependency costs a second rather than a timeout.
# --------------------------------------------------------------------------------------

#: Upper bound on a liveness probe. ``settings.llm_timeout_s`` governs real calls, but a probe that
#: blocks for 45s would stall startup, so probes are capped here.
PROBE_TIMEOUT_CAP_S: Final[float] = 1.5

#: How long a liveness verdict is trusted before the provider is probed again. Long enough that a
#: burst of queries costs one probe, short enough that a recovered provider rejoins the chain fast.
AVAILABILITY_TTL_S: Final[float] = 60.0

#: First backoff interval for retryable failures; grows exponentially with full jitter.
RETRY_BASE_DELAY_S: Final[float] = 0.4

#: Ceiling on a single backoff sleep, so ``llm_max_retries`` cannot turn into a minute of waiting.
RETRY_MAX_DELAY_S: Final[float] = 6.0

#: Marker that carries the JSON schema through the prompt. Providers that cannot accept a structured
#: schema parameter still receive it, and the stub provider recovers it verbatim.
JSON_SCHEMA_MARKER: Final[str] = "<<INDRA_JSON_SCHEMA>>"

#: System prompt used for every structured-output call.
JSON_SYSTEM_PROMPT: Final[str] = (
    "You are INDRA, an industrial plant intelligence engine. "
    "Respond with a single JSON object and nothing else. No prose, no markdown fences, no commentary."
)

#: Average characters per token across English industrial prose. Used by :func:`estimate_tokens`.
CHARS_PER_TOKEN: Final[float] = 4.0

#: Average tokens per whitespace-delimited word, same corpus.
TOKENS_PER_WORD: Final[float] = 1.33

JsonSchema = Mapping[str, Any]
EmbedTask = Literal["document", "query"]


# ======================================================================================
# Token estimation
# ======================================================================================


def estimate_tokens(text: str) -> int:
    """Estimate the token count of ``text``.

    Heuristic on purpose. ``tiktoken`` is installed but :func:`tiktoken.get_encoding` downloads its
    BPE table from the network on first use, and a module that reaches the internet to count tokens
    violates the "works offline with no keys" guarantee. The blend of a character-rate and a
    word-rate estimate lands within ~10% of ``cl100k_base`` on industrial prose, which is all the
    context-window budgeting in this repo needs. Use :func:`count_tokens_exact` when precision
    genuinely matters and the encoding is already cached.
    """
    if not text:
        return 0
    chars = len(text)
    words = len(text.split())
    blended = 0.5 * (chars / CHARS_PER_TOKEN) + 0.5 * (words * TOKENS_PER_WORD)
    return max(1, int(math.ceil(blended)))


def count_tokens_exact(text: str, *, encoding: str = "cl100k_base") -> int:
    """Exact token count via ``tiktoken``, falling back to :func:`estimate_tokens`.

    Never raises: a missing package, a cold cache, or an unreachable network all degrade to the
    heuristic and log once at debug level.
    """
    try:
        import tiktoken  # noqa: PLC0415 - deliberately lazy; see docstring

        encoder = tiktoken.get_encoding(encoding)
        return len(encoder.encode(text))
    except Exception as exc:  # noqa: BLE001 - any failure degrades to the heuristic
        logger.debug(
            "exact token count unavailable, using heuristic",
            extra={"encoding": encoding, "reason": type(exc).__name__},
        )
        return estimate_tokens(text)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Trim ``text`` so that :func:`estimate_tokens` reports at most ``max_tokens``.

    Cuts on a whitespace boundary so a prompt never ends mid-tag (``P-1`` instead of ``P-101``
    would silently change meaning in this domain).
    """
    if max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    budget_chars = int(max_tokens * CHARS_PER_TOKEN)
    clipped = text[:budget_chars]
    boundary = clipped.rfind(" ")
    if boundary > budget_chars // 2:
        clipped = clipped[:boundary]
    return clipped.rstrip() + " …"


# ======================================================================================
# Vector helpers
# ======================================================================================


def l2_normalise(vector: Sequence[float]) -> list[float]:
    """Return ``vector`` scaled to unit length; an all-zero vector is returned unchanged."""
    norm = math.sqrt(sum(component * component for component in vector))
    if norm <= 0.0:
        return [float(component) for component in vector]
    return [float(component) / norm for component in vector]


def conform_dimensions(vector: Sequence[float], dimensions: int, *, provider: str) -> list[float]:
    """Force ``vector`` to ``dimensions`` components, then re-normalise.

    Providers disagree on embedding width (``nomic-embed-text`` is 768, MiniLM is 384). The vector
    store holds one matrix, so a mismatched vector must be conformed rather than rejected —
    truncation keeps the highest-energy components of a transformer embedding, zero-padding is
    inert for cosine similarity. Logged at warning level because mixing widths within one corpus
    degrades recall and the operator should know.
    """
    current = len(vector)
    if current == dimensions:
        return l2_normalise(vector)
    logger.warning(
        "embedding dimension mismatch; conforming vector",
        extra={"provider": provider, "returned": current, "expected": dimensions},
    )
    if current > dimensions:
        return l2_normalise(list(vector[:dimensions]))
    padded = list(vector) + [0.0] * (dimensions - current)
    return l2_normalise(padded)


# ======================================================================================
# HTTP plumbing
# ======================================================================================


def build_http_client(settings: Settings, *, base_url: str = "", headers: Mapping[str, str] | None = None) -> httpx.AsyncClient:
    """Create the shared ``httpx`` client for a provider.

    One client per provider, reused for the process lifetime, so connection pooling actually
    happens and TLS handshakes are not repeated per token.
    """
    return httpx.AsyncClient(
        base_url=base_url,
        headers=dict(headers or {}),
        timeout=httpx.Timeout(settings.llm_timeout_s, connect=min(settings.llm_timeout_s, 10.0)),
        follow_redirects=True,
    )


def raise_for_status(response: httpx.Response, *, provider: str) -> None:
    """Translate an HTTP status into the router's three-way error vocabulary.

    Raises:
        RateLimitError: 429, or a 4xx whose body names a quota. The router fails over immediately.
        ProviderUnavailableError: 401/403/404 — configuration is wrong; retrying cannot help.
        LLMError: 5xx and anything else non-2xx; retryable.
    """
    if response.is_success:
        return

    status = response.status_code
    body = _safe_body(response)
    context = {"provider": provider, "status": status, "body": body[:400]}

    if status == 429 or "quota" in body.lower() or "rate limit" in body.lower():
        raise RateLimitError(
            f"{provider} rejected the request for rate limit or quota. "
            f"Failing over to the next provider in INDRA_LLM_PROVIDER_CHAIN.",
            context=context,
        )
    if status in (401, 403):
        raise ProviderUnavailableError(
            f"{provider} rejected the credentials (HTTP {status}). "
            f"Check the corresponding API key in your .env, or drop {provider} from "
            f"INDRA_LLM_PROVIDER_CHAIN.",
            context=context,
        )
    if status == 404:
        raise ProviderUnavailableError(
            f"{provider} returned 404 for the configured model or endpoint. "
            f"Verify the model name in settings is one your account can reach.",
            context=context,
        )
    if status >= 500:
        raise LLMError(
            f"{provider} returned HTTP {status}. Transient server-side failure; retrying with backoff.",
            context=context,
        )
    raise LLMError(
        f"{provider} returned HTTP {status}: {body[:200]}. Check the request payload for this provider.",
        context=context,
    )


def wrap_transport_error(exc: Exception, *, provider: str) -> LLMError:
    """Convert an ``httpx`` transport failure into a typed INDRA error."""
    if isinstance(exc, httpx.TimeoutException):
        return LLMError(
            f"{provider} timed out after the configured INDRA_LLM_TIMEOUT_S. "
            f"Retrying with backoff, then failing over.",
            context={"provider": provider},
            cause=exc,
        )
    return ProviderUnavailableError(
        f"{provider} is unreachable ({type(exc).__name__}). "
        f"Check network access, or run with INDRA_LLM_PROVIDER_CHAIN=stub for an offline demo.",
        context={"provider": provider},
        cause=exc,
    )


def _safe_body(response: httpx.Response) -> str:
    """Read a response body without ever raising (bodies can be streamed, binary, or absent)."""
    try:
        return response.text
    except Exception:  # noqa: BLE001 - diagnostics must not mask the real error
        return "<unreadable body>"


# ======================================================================================
# Retry policy
# ======================================================================================


def is_retryable(exc: BaseException) -> bool:
    """Retry policy in one predicate.

    A rate limit is **never** retried in place: the free tiers this project targets meter by day,
    so sleeping and asking again burns the user's latency for a guaranteed second refusal. The
    router advances to the next provider instead. Everything else that is marked ``retryable`` on
    the exception class gets backed off.
    """
    if isinstance(exc, RateLimitError):
        return False
    if isinstance(exc, ProviderUnavailableError):
        return False
    return isinstance(exc, LLMError) and exc.retryable


def retrying(settings: Settings, *, provider: str) -> AsyncRetrying:
    """Build the tenacity controller for one provider call.

    ``llm_max_retries`` counts *retries*, so the attempt budget is one more than that.
    """
    attempts = max(1, settings.llm_max_retries + 1)
    return AsyncRetrying(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential_jitter(initial=RETRY_BASE_DELAY_S, max=RETRY_MAX_DELAY_S),
        retry=retry_if_exception(is_retryable),
        reraise=True,
    )


__all_retry_error__ = RetryError  # re-exported for callers that catch tenacity directly


# ======================================================================================
# JSON prompt / response contract
# ======================================================================================


def build_json_prompt(prompt: str, schema: JsonSchema) -> str:
    """Wrap ``prompt`` with the schema contract every provider receives.

    The schema is emitted behind :data:`JSON_SCHEMA_MARKER` so that providers with no structured
    output mode still get it, and so :class:`~indra.llm.stub.StubChatProvider` can honour the very
    same contract instead of being special-cased by the router.
    """
    rendered = json.dumps(schema, indent=2, sort_keys=True, default=str)
    return (
        f"{prompt.strip()}\n\n"
        f"Return a single JSON object that validates against this schema. "
        f"Use only fields defined by the schema.\n"
        f"{JSON_SCHEMA_MARKER}\n"
        f"```json\n{rendered}\n```\n"
        f"{JSON_SCHEMA_MARKER}\n"
        f"JSON object:"
    )


def extract_schema_from_prompt(prompt: str) -> dict[str, Any] | None:
    """Recover the schema a prompt was built with, or ``None`` if it carries no schema."""
    parts = prompt.split(JSON_SCHEMA_MARKER)
    if len(parts) < 3:
        return None
    block = parts[1].strip()
    block = re.sub(r"^```(?:json)?", "", block).strip()
    block = re.sub(r"```$", "", block).strip()
    try:
        parsed = json.loads(block)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


_FENCE_RE: Final[re.Pattern[str]] = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json_object(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model reply.

    Models wrap JSON in fences, prefix it with "Here is the JSON:", and append trailing notes. This
    tries, in order: the raw string, every fenced block, then the widest balanced ``{...}`` span.

    Raises:
        ResponseParsingError: when no balanced JSON object can be recovered.
    """
    candidates: list[str] = []
    stripped = text.strip()
    if stripped:
        candidates.append(stripped)
    candidates.extend(match.group(1).strip() for match in _FENCE_RE.finditer(text))
    span = _balanced_span(text)
    if span is not None:
        candidates.append(span)

    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            return {"items": parsed}

    raise ResponseParsingError(
        "Model reply contained no parseable JSON object. Retry the call, or lower the temperature "
        "for this prompt.",
        context={"reply_head": text[:300]},
    )


def _balanced_span(text: str) -> str | None:
    """Return the outermost balanced ``{...}`` substring, respecting string literals."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def parse_json_response(text: str, schema: JsonSchema) -> dict[str, Any]:
    """Extract, coerce, and validate a model reply against ``schema``.

    Raises:
        ResponseParsingError: on unparseable output or a schema violation.
    """
    payload = extract_json_object(text)
    coerced = coerce_to_schema(payload, schema)
    validate_json_schema(coerced, schema)
    return coerced


def build_repair_prompt(original: str, schema: JsonSchema, bad_output: str, error: str) -> str:
    """Prompt for the single repair round-trip the router allows itself."""
    rendered = json.dumps(schema, indent=2, sort_keys=True, default=str)
    return (
        "Your previous reply did not satisfy the required JSON schema.\n\n"
        f"Validation error: {error}\n\n"
        f"Your previous reply:\n{bad_output[:1500]}\n\n"
        f"Original request:\n{original[:2000]}\n\n"
        f"Return ONLY a corrected JSON object matching this schema exactly.\n"
        f"{JSON_SCHEMA_MARKER}\n```json\n{rendered}\n```\n{JSON_SCHEMA_MARKER}\n"
        "Corrected JSON object:"
    )


# ======================================================================================
# Minimal JSON Schema validator
#
# A dependency-free subset: type, properties, required, additionalProperties, items, enum, const,
# anyOf/oneOf, and the numeric/length/count bounds. That covers every schema INDRA asks a model for
# and keeps the repair message specific enough for a model to act on.
# ======================================================================================

_TYPE_MAP: Final[dict[str, tuple[type, ...]]] = {
    "object": (dict,),
    "array": (list, tuple),
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "null": (type(None),),
}


def _matches_type(value: object, expected: str) -> bool:
    types = _TYPE_MAP.get(expected)
    if types is None:
        return True
    if expected in ("number", "integer") and isinstance(value, bool):
        return False  # bool is an int subclass; a flag is not a measurement
    return isinstance(value, types)


def validate_json_schema(data: object, schema: JsonSchema, *, path: str = "$") -> None:
    """Validate ``data`` against a JSON Schema subset.

    Raises:
        ResponseParsingError: with a path-qualified, model-actionable message.
    """
    for branch_key in ("anyOf", "oneOf"):
        branches = schema.get(branch_key)
        if isinstance(branches, list) and branches:
            errors: list[str] = []
            for branch in branches:
                if not isinstance(branch, Mapping):
                    continue
                try:
                    validate_json_schema(data, branch, path=path)
                    return
                except ResponseParsingError as exc:
                    errors.append(exc.message)
            raise ResponseParsingError(
                f"{path} matched none of the {branch_key} branches: {'; '.join(errors[:3])}",
                context={"path": path},
            )

    if "const" in schema and data != schema["const"]:
        raise ResponseParsingError(
            f"{path} must equal {schema['const']!r}, got {data!r}", context={"path": path}
        )

    enum = schema.get("enum")
    if isinstance(enum, list) and enum and data not in enum:
        raise ResponseParsingError(
            f"{path} must be one of {enum!r}, got {data!r}", context={"path": path}
        )

    expected_type = schema.get("type")
    if isinstance(expected_type, str):
        if not _matches_type(data, expected_type):
            raise ResponseParsingError(
                f"{path} must be of type {expected_type!r}, got {type(data).__name__}",
                context={"path": path},
            )
    elif isinstance(expected_type, list) and expected_type:
        if not any(_matches_type(data, option) for option in expected_type if isinstance(option, str)):
            raise ResponseParsingError(
                f"{path} must be one of types {expected_type!r}, got {type(data).__name__}",
                context={"path": path},
            )

    if isinstance(data, dict):
        _validate_object(data, schema, path)
    elif isinstance(data, (list, tuple)):
        _validate_array(list(data), schema, path)
    elif isinstance(data, str):
        _validate_string(data, schema, path)
    elif isinstance(data, (int, float)) and not isinstance(data, bool):
        _validate_number(float(data), schema, path)


def _validate_object(data: dict[str, Any], schema: JsonSchema, path: str) -> None:
    required = schema.get("required")
    if isinstance(required, list):
        missing = [key for key in required if key not in data]
        if missing:
            raise ResponseParsingError(
                f"{path} is missing required field(s): {missing}", context={"path": path}
            )

    properties = schema.get("properties")
    if isinstance(properties, Mapping):
        for key, sub_schema in properties.items():
            if key in data and isinstance(sub_schema, Mapping):
                validate_json_schema(data[key], sub_schema, path=f"{path}.{key}")
        if schema.get("additionalProperties") is False:
            extra = [key for key in data if key not in properties]
            if extra:
                raise ResponseParsingError(
                    f"{path} contains field(s) not defined by the schema: {extra}",
                    context={"path": path},
                )


def _validate_array(data: list[Any], schema: JsonSchema, path: str) -> None:
    min_items = schema.get("minItems")
    if isinstance(min_items, int) and len(data) < min_items:
        raise ResponseParsingError(
            f"{path} must contain at least {min_items} item(s), got {len(data)}", context={"path": path}
        )
    max_items = schema.get("maxItems")
    if isinstance(max_items, int) and len(data) > max_items:
        raise ResponseParsingError(
            f"{path} must contain at most {max_items} item(s), got {len(data)}", context={"path": path}
        )
    items = schema.get("items")
    if isinstance(items, Mapping):
        for index, element in enumerate(data):
            validate_json_schema(element, items, path=f"{path}[{index}]")


def _validate_string(data: str, schema: JsonSchema, path: str) -> None:
    min_length = schema.get("minLength")
    if isinstance(min_length, int) and len(data) < min_length:
        raise ResponseParsingError(
            f"{path} must be at least {min_length} characters", context={"path": path}
        )
    max_length = schema.get("maxLength")
    if isinstance(max_length, int) and len(data) > max_length:
        raise ResponseParsingError(
            f"{path} must be at most {max_length} characters", context={"path": path}
        )
    pattern = schema.get("pattern")
    if isinstance(pattern, str):
        try:
            compiled = re.compile(pattern)
        except re.error:
            return
        if not compiled.search(data):
            raise ResponseParsingError(
                f"{path} must match pattern {pattern!r}", context={"path": path}
            )


def _validate_number(data: float, schema: JsonSchema, path: str) -> None:
    minimum = schema.get("minimum")
    if isinstance(minimum, (int, float)) and data < float(minimum):
        raise ResponseParsingError(f"{path} must be >= {minimum}", context={"path": path})
    maximum = schema.get("maximum")
    if isinstance(maximum, (int, float)) and data > float(maximum):
        raise ResponseParsingError(f"{path} must be <= {maximum}", context={"path": path})


def coerce_to_schema(data: object, schema: JsonSchema) -> Any:
    """Best-effort scalar coercion before validation.

    Models routinely return ``"0.82"`` where a number is wanted or ``"true"`` where a boolean is.
    Coercing these is not laxity — the semantic content is correct and rejecting it would spend a
    whole repair round-trip on a quoting mistake. Anything ambiguous is left alone for the
    validator to reject with a specific message.
    """
    expected = schema.get("type")
    expected_name = expected if isinstance(expected, str) else None

    if isinstance(data, dict):
        properties = schema.get("properties")
        if isinstance(properties, Mapping):
            return {
                key: (coerce_to_schema(value, properties[key])
                      if key in properties and isinstance(properties[key], Mapping) else value)
                for key, value in data.items()
            }
        return dict(data)

    if isinstance(data, list):
        items = schema.get("items")
        if isinstance(items, Mapping):
            return [coerce_to_schema(element, items) for element in data]
        return list(data)

    if isinstance(data, str):
        text = data.strip()
        if expected_name in ("number", "integer"):
            try:
                number = float(text)
            except ValueError:
                return data
            return int(number) if expected_name == "integer" else number
        if expected_name == "boolean" and text.lower() in ("true", "false"):
            return text.lower() == "true"
        if expected_name == "array":
            return [part.strip() for part in text.split(",") if part.strip()]
        return data

    if expected_name == "integer" and isinstance(data, float) and float(data).is_integer():
        return int(data)
    if expected_name == "string" and isinstance(data, (int, float, bool)):
        return str(data)
    return data


# ======================================================================================
# Provider base classes
# ======================================================================================


class AvailabilityCache:
    """Memoises a liveness verdict for :data:`AVAILABILITY_TTL_S`.

    Without this, a chain of four providers probes the network on every single query and the
    latency of a *successful* call is dominated by checking the providers that were never used.
    """

    __slots__ = ("_verdict", "_checked_at", "_ttl")

    def __init__(self, ttl_s: float = AVAILABILITY_TTL_S) -> None:
        self._verdict: bool | None = None
        self._checked_at: float = 0.0
        self._ttl = ttl_s

    def get(self) -> bool | None:
        """Return the cached verdict, or ``None`` when a fresh probe is due."""
        if self._verdict is None or (time.monotonic() - self._checked_at) > self._ttl:
            return None
        return self._verdict

    def set(self, verdict: bool) -> bool:
        """Record a verdict and return it, so callers can ``return cache.set(...)``."""
        self._verdict = verdict
        self._checked_at = time.monotonic()
        return verdict

    def invalidate(self) -> None:
        """Force the next :meth:`get` to miss — used after a provider errors mid-call."""
        self._verdict = None


class BaseChatProvider:
    """Common behaviour for every :class:`~indra.core.contracts.ChatProvider`.

    Subclasses implement :meth:`_complete` (and optionally :meth:`_stream_tokens`); this class
    supplies the JSON contract, the streaming fallback, and availability caching.
    """

    name: str = "base"
    supports_json: bool = True

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: httpx.AsyncClient | None = None

    # -- subclass hooks -----------------------------------------------------------

    async def _complete(
        self,
        prompt: str,
        *,
        system: str | None,
        temperature: float,
        max_tokens: int,
        stop: Sequence[str] | None,
    ) -> str:
        """Produce a completion. Implemented by each provider."""
        raise NotImplementedError(f"{type(self).__name__} must implement _complete")

    async def _stream_tokens(
        self,
        prompt: str,
        *,
        system: str | None,
        temperature: float,
    ) -> AsyncIterator[str]:
        """Yield incremental text. Default implementation emits the completion in one chunk."""
        text = await self._complete(
            prompt,
            system=system,
            temperature=temperature,
            max_tokens=self.settings.llm_max_output_tokens,
            stop=None,
        )
        yield text

    async def is_available(self) -> bool:
        """Cheap liveness probe. Must never raise."""
        return True

    # -- public surface (matches ChatProvider) ------------------------------------

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
    ) -> str:
        return await self._complete(
            prompt,
            system=system,
            temperature=self.settings.llm_temperature if temperature is None else temperature,
            max_tokens=self.settings.llm_max_output_tokens if max_tokens is None else max_tokens,
            stop=stop,
        )

    async def generate_json(
        self,
        prompt: str,
        *,
        schema: dict[str, Any],
        system: str | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Generate and validate structured output.

        Every provider — including the stub — goes through this exact path, which is why a test run
        exercises the same prompt construction, extraction, and validation as production.
        """
        instructed = build_json_prompt(prompt, schema)
        raw = await self.generate(
            instructed,
            system=system or JSON_SYSTEM_PROMPT,
            temperature=0.0 if temperature is None else temperature,
        )
        return parse_json_response(raw, schema)

    async def stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        async for piece in self._stream_tokens(
            prompt,
            system=system,
            temperature=self.settings.llm_temperature if temperature is None else temperature,
        ):
            if piece:
                yield piece

    # -- lifecycle ----------------------------------------------------------------

    async def aclose(self) -> None:
        """Release the HTTP connection pool. Safe to call more than once."""
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception as exc:  # noqa: BLE001 - shutdown must not raise
                logger.debug("provider client close failed", extra={"provider": self.name, "error": str(exc)})
            self._client = None


class BaseEmbeddingProvider:
    """Common behaviour for every :class:`~indra.core.contracts.EmbeddingProvider`."""

    name: str = "base"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.dimensions: int = settings.embedding_dimensions
        self._client: httpx.AsyncClient | None = None

    async def embed(self, texts: Sequence[str], *, task: EmbedTask = "document") -> list[list[float]]:
        raise NotImplementedError(f"{type(self).__name__} must implement embed")

    async def is_available(self) -> bool:
        return True

    async def aclose(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception as exc:  # noqa: BLE001 - shutdown must not raise
                logger.debug("embedder client close failed", extra={"provider": self.name, "error": str(exc)})
            self._client = None

    def _batches(self, texts: Sequence[str]) -> Iterable[Sequence[str]]:
        """Split ``texts`` into ``settings.embedding_batch_size`` slices."""
        size = max(1, self.settings.embedding_batch_size)
        for start in range(0, len(texts), size):
            yield texts[start : start + size]

    def _check_batch(self, texts: Sequence[str], vectors: list[list[float]]) -> list[list[float]]:
        """Assert order preservation and conform widths before handing vectors to the store."""
        if len(vectors) != len(texts):
            raise EmbeddingError(
                f"{self.name} returned {len(vectors)} vectors for {len(texts)} inputs; the batch "
                f"ordering contract is broken. Reduce INDRA_EMBEDDING_BATCH_SIZE or switch provider.",
                context={"provider": self.name},
            )
        return [conform_dimensions(vector, self.dimensions, provider=self.name) for vector in vectors]


__all__ = [
    "AVAILABILITY_TTL_S",
    "AvailabilityCache",
    "BaseChatProvider",
    "BaseEmbeddingProvider",
    "CHARS_PER_TOKEN",
    "EmbedTask",
    "JSON_SCHEMA_MARKER",
    "JSON_SYSTEM_PROMPT",
    "JsonSchema",
    "PROBE_TIMEOUT_CAP_S",
    "RETRY_BASE_DELAY_S",
    "RETRY_MAX_DELAY_S",
    "RetryError",
    "build_http_client",
    "build_json_prompt",
    "build_repair_prompt",
    "coerce_to_schema",
    "conform_dimensions",
    "count_tokens_exact",
    "estimate_tokens",
    "extract_json_object",
    "extract_schema_from_prompt",
    "is_retryable",
    "l2_normalise",
    "parse_json_response",
    "raise_for_status",
    "retrying",
    "truncate_to_tokens",
    "validate_json_schema",
    "wrap_transport_error",
]
