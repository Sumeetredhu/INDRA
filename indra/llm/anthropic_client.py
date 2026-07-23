"""Optional Anthropic chat provider (``docs/DECISIONS.md`` D2).

Not in the default chain — add ``anthropic`` to ``INDRA_LLM_PROVIDER_CHAIN`` to use it. It exists
because the :class:`~indra.core.contracts.ChatProvider` protocol made the adapter nearly free, and
because having a second *high-quality* provider behind Gemini turns a quota exhaustion from a
demo-quality cliff into a shrug.

``httpx`` against the documented Messages API; the ``anthropic`` SDK is neither imported nor
required.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Final, Sequence

import httpx

from indra.core.config import Settings
from indra.core.exceptions import LLMError, ProviderUnavailableError
from indra.core.logging import get_logger
from indra.llm.base import (
    PROBE_TIMEOUT_CAP_S,
    AvailabilityCache,
    BaseChatProvider,
    build_http_client,
    raise_for_status,
    wrap_transport_error,
)

logger = get_logger(__name__)

#: Vendor endpoint and wire version — pinned, not tunable.
ANTHROPIC_API_BASE: Final[str] = "https://api.anthropic.com/v1"
ANTHROPIC_API_VERSION: Final[str] = "2023-06-01"


class AnthropicChatProvider(BaseChatProvider):
    """Chat completions against the Anthropic Messages API (``settings.anthropic_model``)."""

    name = "anthropic"
    supports_json = True

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._availability = AvailabilityCache()

    # -- lifecycle ----------------------------------------------------------------

    def _http(self, key: str) -> httpx.AsyncClient:
        if self._client is None:
            self._client = build_http_client(
                self.settings,
                base_url=ANTHROPIC_API_BASE,
                headers={
                    "x-api-key": key,
                    "anthropic-version": ANTHROPIC_API_VERSION,
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def is_available(self) -> bool:
        """True when a key is configured and ``/models`` answers. Never raises."""
        cached = self._availability.get()
        if cached is not None:
            return cached
        key = self.settings.secret("anthropic_api_key")
        if not key:
            logger.debug("anthropic unavailable: no API key configured", extra={"provider": self.name})
            return self._availability.set(False)
        try:
            response = await asyncio.wait_for(
                self._http(key).get("/models", params={"limit": 1}),
                timeout=min(self.settings.llm_timeout_s, PROBE_TIMEOUT_CAP_S),
            )
            return self._availability.set(response.is_success)
        except (asyncio.TimeoutError, httpx.HTTPError) as exc:
            logger.info(
                "anthropic liveness probe failed; skipping provider",
                extra={"provider": self.name, "reason": type(exc).__name__},
            )
            return self._availability.set(False)

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
        key = self._require_key()
        payload = self._build_payload(
            prompt, system=system, temperature=temperature, max_tokens=max_tokens,
            stop=stop, stream=False,
        )
        try:
            response = await self._http(key).post("/messages", json=payload)
        except httpx.HTTPError as exc:
            self._availability.invalidate()
            raise wrap_transport_error(exc, provider=self.name) from exc
        raise_for_status(response, provider=self.name)
        return self._extract_text(response.json())

    async def _stream_tokens(
        self,
        prompt: str,
        *,
        system: str | None,
        temperature: float,
    ) -> AsyncIterator[str]:
        key = self._require_key()
        payload = self._build_payload(
            prompt, system=system, temperature=temperature,
            max_tokens=self.settings.llm_max_output_tokens, stop=None, stream=True,
        )
        try:
            async with self._http(key).stream("POST", "/messages", json=payload) as response:
                if not response.is_success:
                    await response.aread()
                    raise_for_status(response, provider=self.name)
                async for line in response.aiter_lines():
                    piece = self._parse_sse_line(line)
                    if piece:
                        yield piece
        except httpx.HTTPError as exc:
            self._availability.invalidate()
            raise wrap_transport_error(exc, provider=self.name) from exc

    # -- helpers ------------------------------------------------------------------

    def _require_key(self) -> str:
        key = self.settings.secret("anthropic_api_key")
        if not key:
            raise ProviderUnavailableError(
                "Anthropic is in the provider chain but ANTHROPIC_API_KEY is not set. Set it in "
                ".env, or remove 'anthropic' from INDRA_LLM_PROVIDER_CHAIN.",
                context={"provider": self.name},
            )
        return key

    def _build_payload(
        self,
        prompt: str,
        *,
        system: str | None,
        temperature: float,
        max_tokens: int,
        stop: Sequence[str] | None,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.settings.anthropic_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
        }
        if system:
            payload["system"] = system
        if stop:
            payload["stop_sequences"] = list(stop)
        return payload

    def _extract_text(self, body: dict[str, Any]) -> str:
        blocks = body.get("content")
        if not isinstance(blocks, list) or not blocks:
            raise LLMError(
                "Anthropic returned no content blocks. Retry, or fail over to the next provider.",
                context={"provider": self.name},
            )
        text = "".join(
            str(block.get("text", "")) for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        )
        if not text.strip():
            reason = body.get("stop_reason", "unknown")
            raise LLMError(
                f"Anthropic returned an empty completion (stop_reason={reason}).",
                context={"provider": self.name, "stop_reason": str(reason)},
            )
        return text

    def _parse_sse_line(self, line: str) -> str:
        """Decode a ``content_block_delta`` frame; ignore every other event type."""
        if not line or not line.startswith("data:"):
            return ""
        raw = line[len("data:") :].strip()
        if not raw or raw == "[DONE]":
            return ""
        try:
            frame = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("discarding malformed anthropic SSE frame", extra={"provider": self.name})
            return ""
        if frame.get("type") != "content_block_delta":
            return ""
        return str((frame.get("delta") or {}).get("text") or "")


__all__ = ["ANTHROPIC_API_BASE", "ANTHROPIC_API_VERSION", "AnthropicChatProvider"]
