"""The LLM router: ordered fail-over, budget accounting, and truthful provider attribution.

This is the component that lets ``Answer.provider_used`` be a fact rather than a hope. Every call
walks ``settings.llm_provider_chain`` in order and returns the name of the provider that actually
produced the text.

Routing policy, and the reasoning behind it
-------------------------------------------
**A rate limit is not a transient failure.** The free tiers this project targets meter by the day.
Sleeping 800ms and asking again buys a second refusal and costs a technician a second of their life.
So :class:`~indra.core.exceptions.RateLimitError` advances to the next provider *immediately* and
parks the offending one for a cooldown.

**A 5xx or a timeout is transient.** Those get jittered exponential backoff up to
``settings.llm_max_retries`` before the router gives up on that provider and advances.

**Refuse before the API does.** Gemini's free tier is roughly 500 generations a day; a single
rehearsal loop can eat it. The router keeps a per-provider daily counter against
``settings.gemini_daily_budget`` and stops using the provider before Google does, so the quota is
spent on the demo rather than on the run-up to it.

**The chain always ends in the stub.** :class:`~indra.llm.stub.StubChatProvider` is appended if the
configured chain omits it. That is CLAUDE.md rule 6 in one line: an exhausted quota degrades the
answer, it never fails the request.

Deterministic and offline modes
-------------------------------
``settings.deterministic`` (demo recording, or any pytest run) pins the chain to the stub and the
embedder to the hash projection — reproducible output, no network, no keys. ``settings.offline_mode``
drops the three hosted providers and leaves Ollama and the stub, which is the plant-floor
configuration.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, AsyncIterator, Callable, Final, Literal, Sequence

from indra.core.config import Settings
from indra.core.exceptions import (
    AllProvidersFailedError,
    EmbeddingError,
    LLMError,
    ProviderUnavailableError,
    RateLimitError,
    ResponseParsingError,
)
from indra.core.logging import get_logger
from indra.llm.anthropic_client import AnthropicChatProvider
from indra.llm.base import (
    JSON_SYSTEM_PROMPT,
    BaseChatProvider,
    BaseEmbeddingProvider,
    EmbedTask,
    JsonSchema,
    build_json_prompt,
    build_repair_prompt,
    parse_json_response,
    retrying,
)
from indra.llm.gemini import GeminiChatProvider, GeminiEmbeddingProvider
from indra.llm.groq_client import GroqChatProvider
from indra.llm.local_embed import HashEmbeddingProvider, LocalEmbeddingProvider
from indra.llm.ollama import OllamaChatProvider, OllamaEmbeddingProvider
from indra.llm.stub import StubChatProvider

logger = get_logger(__name__)

#: How long a rate-limited provider is parked. Long enough that a per-minute limiter recovers on
#: its own; the daily budget counter is what protects a per-day quota.
RATE_LIMIT_COOLDOWN_S: Final[float] = 300.0

#: Providers that require the public internet. Dropped when ``settings.offline_mode`` is set.
HOSTED_PROVIDERS: Final[frozenset[str]] = frozenset({"gemini", "groq", "anthropic"})

#: The provider that must always terminate the chain.
TERMINAL_CHAT_PROVIDER: Final[str] = "stub"

#: The embedder that must always terminate the embedding chain.
TERMINAL_EMBED_PROVIDER: Final[str] = "hash"

ChatProviderFactory = Callable[[Settings], BaseChatProvider]
EmbeddingProviderFactory = Callable[[Settings], BaseEmbeddingProvider]

#: Name → constructor. Adding a provider is adding a row here and a module next to it.
CHAT_PROVIDERS: Final[dict[str, ChatProviderFactory]] = {
    "gemini": GeminiChatProvider,
    "groq": GroqChatProvider,
    "ollama": OllamaChatProvider,
    "anthropic": AnthropicChatProvider,
    "stub": StubChatProvider,
}

EMBEDDING_PROVIDERS: Final[dict[str, EmbeddingProviderFactory]] = {
    "gemini": GeminiEmbeddingProvider,
    "ollama": OllamaEmbeddingProvider,
    "local": LocalEmbeddingProvider,
    "hash": HashEmbeddingProvider,
}


# ======================================================================================
# Budget accounting
# ======================================================================================


@dataclass(slots=True)
class ProviderLedger:
    """Per-provider call accounting for one UTC day.

    ``limit == 0`` means unmetered (local models, the stub). Counters roll over automatically when
    the UTC date changes, so a long-running server does not need a scheduled reset.
    """

    name: str
    limit: int = 0
    used: int = 0
    succeeded: int = 0
    failed: int = 0
    day: date = field(default_factory=lambda: datetime.now(timezone.utc).date())
    cooldown_until: float = 0.0

    def _roll(self) -> None:
        today = datetime.now(timezone.utc).date()
        if today != self.day:
            logger.info(
                "provider daily budget reset",
                extra={"provider": self.name, "previous_day_calls": self.used},
            )
            self.day = today
            self.used = 0
            self.cooldown_until = 0.0

    @property
    def remaining(self) -> int:
        """Calls left today; ``-1`` when unmetered."""
        self._roll()
        return -1 if self.limit <= 0 else max(0, self.limit - self.used)

    def is_parked(self) -> bool:
        """True while the provider is cooling down after a rate limit."""
        self._roll()
        return time.monotonic() < self.cooldown_until

    def reserve(self) -> bool:
        """Claim one call against the budget up front.

        Reserving before the call rather than counting after it means concurrent coroutines cannot
        collectively overshoot the quota.
        """
        self._roll()
        if self.limit > 0 and self.used >= self.limit:
            return False
        self.used += 1
        return True

    def release(self) -> None:
        """Return an unused reservation (the provider refused before doing any work)."""
        self.used = max(0, self.used - 1)

    def park(self, seconds: float = RATE_LIMIT_COOLDOWN_S) -> None:
        """Take the provider out of rotation for ``seconds``."""
        self.cooldown_until = time.monotonic() + seconds


# ======================================================================================
# Router
# ======================================================================================


class Router:
    """Ordered fail-over across chat and embedding providers.

    Implements :class:`indra.core.contracts.LLMRouter`. Construct it with
    :func:`build_router`, not directly, so the chain honours settings.
    """

    def __init__(
        self,
        settings: Settings,
        chat_providers: Sequence[BaseChatProvider],
        embedding_providers: Sequence[BaseEmbeddingProvider],
    ) -> None:
        if not chat_providers:
            raise ProviderUnavailableError(
                "The chat provider chain is empty. INDRA_LLM_PROVIDER_CHAIN must name at least one "
                "provider; 'stub' always works.",
            )
        if not embedding_providers:
            raise EmbeddingError(
                "The embedding provider chain is empty. INDRA_EMBEDDING_PROVIDER_CHAIN must name at "
                "least one provider; 'hash' always works.",
            )
        self.settings = settings
        self.chat_providers = list(chat_providers)
        self.embedding_providers = list(embedding_providers)
        self._ledgers: dict[str, ProviderLedger] = {
            provider.name: ProviderLedger(name=provider.name, limit=self._limit_for(provider.name, settings))
            for provider in self.chat_providers
        }
        self._embed_calls: dict[str, int] = {provider.name: 0 for provider in self.embedding_providers}
        self._lock = asyncio.Lock()

    @staticmethod
    def _limit_for(name: str, settings: Settings) -> int:
        """Daily call ceiling per provider; 0 for unmetered ones."""
        if name == "gemini":
            return max(0, settings.gemini_daily_budget)
        return 0

    # -- public surface (matches LLMRouter) ---------------------------------------

    async def generate(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> tuple[str, str]:
        """Generate text, returning ``(text, provider_name)``.

        Raises:
            AllProvidersFailedError: only if every provider in the chain failed, which cannot happen
                while the stub terminates the chain.
        """
        failures: dict[str, str] = {}
        for provider in self.chat_providers:
            ledger = self._ledgers[provider.name]
            skip = await self._skip_reason(provider, ledger)
            if skip is not None:
                failures[provider.name] = skip
                continue
            try:
                text = await self._attempt(
                    provider, ledger,
                    lambda: provider.generate(prompt, system=system, **kwargs),
                )
            except LLMError as exc:
                failures[provider.name] = exc.message
                continue
            return text, provider.name

        raise AllProvidersFailedError(
            "Every LLM provider in the chain failed. Add 'stub' to INDRA_LLM_PROVIDER_CHAIN for a "
            "guaranteed offline fallback.",
            context={"failures": failures},
        )

    async def generate_json(
        self,
        prompt: str,
        *,
        schema: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[dict[str, Any], str]:
        """Generate schema-valid structured output, returning ``(payload, provider_name)``.

        Each provider gets exactly one repair round-trip: the failed output and the specific
        validation error are handed back with a corrected-JSON instruction. A second failure means
        the provider does not understand the schema, so the router advances rather than looping.

        Raises:
            ResponseParsingError: when no provider produced schema-valid output.
            AllProvidersFailedError: when no provider was reachable at all.
        """
        instructed = build_json_prompt(prompt, schema)
        system = kwargs.pop("system", None) or JSON_SYSTEM_PROMPT
        kwargs.setdefault("temperature", 0.0)

        failures: dict[str, str] = {}
        parse_failures: dict[str, str] = {}

        for provider in self.chat_providers:
            ledger = self._ledgers[provider.name]
            skip = await self._skip_reason(provider, ledger)
            if skip is not None:
                failures[provider.name] = skip
                continue

            try:
                raw = await self._attempt(
                    provider, ledger,
                    lambda: provider.generate(instructed, system=system, **kwargs),
                )
            except LLMError as exc:
                failures[provider.name] = exc.message
                continue

            try:
                return parse_json_response(raw, schema), provider.name
            except ResponseParsingError as first_error:
                logger.warning(
                    "structured output failed validation; attempting one repair round-trip",
                    extra={"provider": provider.name, "error": first_error.message[:200]},
                )

            repair = build_repair_prompt(prompt, schema, raw, first_error.message)
            try:
                repaired = await self._attempt(
                    provider, ledger,
                    lambda: provider.generate(repair, system=system, **kwargs),
                )
            except LLMError as exc:
                failures[provider.name] = exc.message
                continue

            try:
                return parse_json_response(repaired, schema), provider.name
            except ResponseParsingError as second_error:
                parse_failures[provider.name] = second_error.message
                logger.warning(
                    "repair round-trip also failed validation; advancing to the next provider",
                    extra={"provider": provider.name, "error": second_error.message[:200]},
                )

        if parse_failures:
            raise ResponseParsingError(
                "No provider produced output matching the requested JSON schema, even after a "
                "repair round-trip. Simplify the schema, or lower the temperature for this call.",
                context={"parse_failures": parse_failures, "unavailable": failures},
            )
        raise AllProvidersFailedError(
            "Every LLM provider in the chain failed before producing structured output.",
            context={"failures": failures},
        )

    async def stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        on_provider: Callable[[str], None] | None = None,
    ) -> AsyncIterator[str]:
        """Stream incremental text from the first provider that yields anything.

        Args:
            on_provider: Called once with the provider name as soon as one is selected, so the
                caller can populate ``Answer.provider_used`` before the answer finishes.

        A provider that fails *after* emitting its first chunk is not retried on another provider —
        the client has already rendered those tokens and re-streaming would corrupt the transcript.
        """
        failures: dict[str, str] = {}
        for provider in self.chat_providers:
            ledger = self._ledgers[provider.name]
            skip = await self._skip_reason(provider, ledger)
            if skip is not None:
                failures[provider.name] = skip
                continue
            if not ledger.reserve():
                failures[provider.name] = "daily budget exhausted"
                continue

            emitted = False
            try:
                async for piece in provider.stream(prompt, system=system, temperature=temperature):
                    if not emitted:
                        emitted = True
                        ledger.succeeded += 1
                        if on_provider is not None:
                            on_provider(provider.name)
                    yield piece
            except LLMError as exc:
                ledger.failed += 1
                if isinstance(exc, RateLimitError):
                    ledger.park()
                if emitted:
                    logger.error(
                        "stream aborted mid-flight; not failing over to avoid duplicate tokens",
                        extra={"provider": provider.name, "error": exc.message[:200]},
                    )
                    raise
                ledger.release()
                failures[provider.name] = exc.message
                continue
            if emitted:
                return
            ledger.release()
            failures[provider.name] = "provider produced no tokens"

        raise AllProvidersFailedError(
            "Every LLM provider in the chain failed while streaming.",
            context={"failures": failures},
        )

    async def embed(self, texts: Sequence[str], *, task: EmbedTask = "document") -> list[list[float]]:
        """Embed a batch through the first available embedding provider.

        Raises:
            EmbeddingError: only if every embedder failed — impossible while ``hash`` terminates the
                chain, since it is pure computation.
        """
        if not texts:
            return []
        failures: dict[str, str] = {}
        for provider in self.embedding_providers:
            try:
                if not await provider.is_available():
                    failures[provider.name] = "not available"
                    continue
            except Exception as exc:  # noqa: BLE001 - a probe must never break routing
                failures[provider.name] = f"probe raised {type(exc).__name__}"
                continue
            try:
                vectors = await provider.embed(texts, task=task)
            except (EmbeddingError, LLMError) as exc:
                failures[provider.name] = exc.message
                logger.warning(
                    "embedding provider failed; falling through",
                    extra={"provider": provider.name, "error": exc.message[:200]},
                )
                continue
            self._embed_calls[provider.name] = self._embed_calls.get(provider.name, 0) + 1
            return vectors

        raise EmbeddingError(
            "Every embedding provider failed. Add 'hash' to INDRA_EMBEDDING_PROVIDER_CHAIN — it is "
            "pure computation and cannot fail.",
            context={"failures": failures},
        )

    def usage(self) -> dict[str, int]:
        """Successful calls per provider this process, for ``/metrics`` and budget guards."""
        counts = {name: ledger.succeeded for name, ledger in self._ledgers.items()}
        for name, calls in self._embed_calls.items():
            counts[f"embed:{name}"] = calls
        return counts

    # -- operations ---------------------------------------------------------------

    def budgets(self) -> dict[str, dict[str, int]]:
        """Per-provider budget state, surfaced on the ops panel."""
        return {
            name: {
                "limit": ledger.limit,
                "used": ledger.used,
                "remaining": ledger.remaining,
                "succeeded": ledger.succeeded,
                "failed": ledger.failed,
            }
            for name, ledger in self._ledgers.items()
        }

    async def health(self) -> dict[str, Any]:
        """Per-provider readiness. Never raises; a probe failure is reported, not thrown."""
        chat: dict[str, bool] = {}
        for provider in self.chat_providers:
            try:
                chat[provider.name] = await provider.is_available()
            except Exception:  # noqa: BLE001 - health must not fail
                chat[provider.name] = False
        embeddings: dict[str, bool] = {}
        for embedder in self.embedding_providers:
            try:
                embeddings[embedder.name] = await embedder.is_available()
            except Exception:  # noqa: BLE001 - health must not fail
                embeddings[embedder.name] = False
        return {
            "ok": any(chat.values()),
            "backend": ",".join(provider.name for provider in self.chat_providers),
            "detail": "chat chain and embedding chain readiness",
            "chat": chat,
            "embeddings": embeddings,
            "usage": self.usage(),
            "budgets": self.budgets(),
        }

    async def aclose(self) -> None:
        """Close every provider's connection pool. Safe to call more than once."""
        for provider in [*self.chat_providers, *self.embedding_providers]:
            await provider.aclose()

    # -- internals ----------------------------------------------------------------

    async def _skip_reason(self, provider: BaseChatProvider, ledger: ProviderLedger) -> str | None:
        """Return why this provider should be skipped, or ``None`` to use it."""
        if ledger.is_parked():
            return "parked after a rate limit"
        if ledger.remaining == 0:
            logger.warning(
                "provider daily budget exhausted; failing over",
                extra={"provider": provider.name, "limit": ledger.limit},
            )
            return "daily budget exhausted"
        try:
            if not await provider.is_available():
                return "not available"
        except Exception as exc:  # noqa: BLE001 - a probe must never break routing
            logger.info(
                "provider availability probe raised; treating as unavailable",
                extra={"provider": provider.name, "error": type(exc).__name__},
            )
            return f"probe raised {type(exc).__name__}"
        return None

    async def _attempt(
        self,
        provider: BaseChatProvider,
        ledger: ProviderLedger,
        call: Callable[[], Any],
    ) -> str:
        """Run one provider call under the retry policy, maintaining the ledger.

        Raises:
            LLMError: after the retry budget is spent, or immediately for a rate limit or a
                configuration failure. The caller advances to the next provider.
        """
        if not ledger.reserve():
            raise RateLimitError(
                f"{provider.name} daily budget of {ledger.limit} calls is exhausted. "
                f"Raise INDRA_GEMINI_DAILY_BUDGET or wait for the UTC day to roll over.",
                context={"provider": provider.name, "limit": ledger.limit},
            )
        started = time.monotonic()
        try:
            async for attempt in retrying(self.settings, provider=provider.name):
                with attempt:
                    text = await call()
        except RateLimitError as exc:
            ledger.failed += 1
            ledger.release()
            ledger.park()
            logger.warning(
                "provider rate limited; failing over immediately without retrying",
                extra={"provider": provider.name, "cooldown_s": RATE_LIMIT_COOLDOWN_S},
            )
            raise exc
        except LLMError as exc:
            ledger.failed += 1
            ledger.release()
            logger.warning(
                "provider failed after exhausting retries; advancing",
                extra={"provider": provider.name, "error": exc.message[:200]},
            )
            raise exc
        except Exception as exc:  # noqa: BLE001 - nothing untyped escapes the router
            ledger.failed += 1
            ledger.release()
            raise LLMError(
                f"{provider.name} raised an unexpected {type(exc).__name__}. This is a provider "
                f"adapter bug; the router is failing over so the request still completes.",
                context={"provider": provider.name},
                cause=exc,
            ) from exc

        ledger.succeeded += 1
        logger.info(
            "llm call served",
            extra={
                "provider": provider.name,
                "latency_ms": round((time.monotonic() - started) * 1000.0, 1),
                "budget_remaining": ledger.remaining,
            },
        )
        return text


# ======================================================================================
# Construction
# ======================================================================================


def resolve_chat_chain(settings: Settings) -> list[str]:
    """Return the effective chat provider chain for these settings.

    Applies deterministic mode, offline mode, unknown-name filtering, and the guarantee that the
    chain terminates in the stub.
    """
    if settings.deterministic:
        logger.info("deterministic mode: pinning the chat chain to the stub provider")
        return [TERMINAL_CHAT_PROVIDER]

    chain: list[str] = []
    for raw in settings.llm_provider_chain:
        name = raw.strip().lower()
        if not name:
            continue
        if name not in CHAT_PROVIDERS:
            logger.warning(
                "unknown chat provider in INDRA_LLM_PROVIDER_CHAIN; ignoring",
                extra={"provider": name, "known": sorted(CHAT_PROVIDERS)},
            )
            continue
        if settings.offline_mode and name in HOSTED_PROVIDERS:
            logger.info("offline mode: dropping hosted provider", extra={"provider": name})
            continue
        if name not in chain:
            chain.append(name)

    if TERMINAL_CHAT_PROVIDER not in chain:
        chain.append(TERMINAL_CHAT_PROVIDER)
    return chain


def resolve_embedding_chain(settings: Settings) -> list[str]:
    """Return the effective embedding chain, always terminated by the hash embedder."""
    if settings.deterministic:
        logger.info("deterministic mode: pinning the embedding chain to the hash embedder")
        return [TERMINAL_EMBED_PROVIDER]

    chain: list[str] = []
    for raw in settings.embedding_provider_chain:
        name = raw.strip().lower()
        if not name:
            continue
        if name not in EMBEDDING_PROVIDERS:
            logger.warning(
                "unknown embedding provider in INDRA_EMBEDDING_PROVIDER_CHAIN; ignoring",
                extra={"provider": name, "known": sorted(EMBEDDING_PROVIDERS)},
            )
            continue
        if settings.offline_mode and name in HOSTED_PROVIDERS:
            logger.info("offline mode: dropping hosted embedder", extra={"provider": name})
            continue
        if name not in chain:
            chain.append(name)

    if TERMINAL_EMBED_PROVIDER not in chain:
        chain.append(TERMINAL_EMBED_PROVIDER)
    return chain


def build_router(settings: Settings) -> Router:
    """Build the process router from settings.

    The only supported way to construct a :class:`Router`. Never raises for a misconfigured chain:
    unknown names are logged and dropped, and the stub plus hash embedder are appended so the
    result is always usable.
    """
    chat_names = resolve_chat_chain(settings)
    embed_names = resolve_embedding_chain(settings)

    chat: list[BaseChatProvider] = []
    for name in chat_names:
        try:
            chat.append(CHAT_PROVIDERS[name](settings))
        except Exception as exc:  # noqa: BLE001 - one bad adapter must not break the chain
            logger.error(
                "chat provider failed to construct; dropping it from the chain",
                extra={"provider": name, "error": str(exc)[:200]},
            )

    embeddings: list[BaseEmbeddingProvider] = []
    for name in embed_names:
        try:
            embeddings.append(EMBEDDING_PROVIDERS[name](settings))
        except Exception as exc:  # noqa: BLE001 - one bad adapter must not break the chain
            logger.error(
                "embedding provider failed to construct; dropping it from the chain",
                extra={"provider": name, "error": str(exc)[:200]},
            )

    if not any(provider.name == TERMINAL_CHAT_PROVIDER for provider in chat):
        chat.append(StubChatProvider(settings))
    if not any(provider.name == TERMINAL_EMBED_PROVIDER for provider in embeddings):
        embeddings.append(HashEmbeddingProvider(settings))

    logger.info(
        "llm router built",
        extra={
            "chat_chain": [provider.name for provider in chat],
            "embedding_chain": [provider.name for provider in embeddings],
            "deterministic": settings.deterministic,
            "offline": settings.offline_mode,
        },
    )
    return Router(settings, chat, embeddings)


__all__ = [
    "CHAT_PROVIDERS",
    "EMBEDDING_PROVIDERS",
    "HOSTED_PROVIDERS",
    "RATE_LIMIT_COOLDOWN_S",
    "ProviderLedger",
    "Router",
    "build_router",
    "resolve_chat_chain",
    "resolve_embedding_chain",
]
