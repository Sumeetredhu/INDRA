"""Groq chat provider over the OpenAI-compatible REST API.

Groq is INDRA's speed tier: the same ``llama-3.3-70b`` weights everyone else serves, but at an order
of magnitude lower latency, which is why it sits second in the default chain. It speaks the
OpenAI ``/openai/v1/chat/completions`` schema, so this module is deliberately thin — no ``groq``
SDK, just ``httpx`` against a documented wire format.

Two provider-specific behaviours are handled here:

* **Native JSON mode.** ``response_format={"type": "json_object"}`` is set whenever the prompt
  carries the INDRA schema marker, which removes most of the fenced-markdown noise that
  :func:`indra.llm.base.extract_json_object` would otherwise have to clean up.
* **Per-minute rate limits.** Unlike Gemini's daily quota, Groq meters per minute — but the router's
  fail-over-don't-retry policy is still correct: with a stub at the end of the chain, serving a
  slightly weaker answer now beats sleeping 60 seconds in front of a technician.
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
    JSON_SCHEMA_MARKER,
    PROBE_TIMEOUT_CAP_S,
    AvailabilityCache,
    BaseChatProvider,
    build_http_client,
    raise_for_status,
    wrap_transport_error,
)

logger = get_logger(__name__)

#: Vendor endpoint, not a deployment tunable.
GROQ_API_BASE: Final[str] = "https://api.groq.com/openai/v1"


class GroqChatProvider(BaseChatProvider):
    """Chat completions against Groq (``settings.groq_model``)."""

    name = "groq"
    supports_json = True

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._availability = AvailabilityCache()

    # -- lifecycle ----------------------------------------------------------------

    def _http(self, key: str) -> httpx.AsyncClient:
        if self._client is None:
            self._client = build_http_client(
                self.settings,
                base_url=GROQ_API_BASE,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            )
        return self._client

    async def is_available(self) -> bool:
        """True when a key is configured and ``/models`` answers. Never raises."""
        cached = self._availability.get()
        if cached is not None:
            return cached
        key = self.settings.secret("groq_api_key")
        if not key:
            logger.debug("groq unavailable: no API key configured", extra={"provider": self.name})
            return self._availability.set(False)
        try:
            response = await asyncio.wait_for(
                self._http(key).get("/models"),
                timeout=min(self.settings.llm_timeout_s, PROBE_TIMEOUT_CAP_S),
            )
            return self._availability.set(response.is_success)
        except (asyncio.TimeoutError, httpx.HTTPError) as exc:
            logger.info(
                "groq liveness probe failed; skipping provider",
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
            response = await self._http(key).post("/chat/completions", json=payload)
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
            async with self._http(key).stream("POST", "/chat/completions", json=payload) as response:
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
        key = self.settings.secret("groq_api_key")
        if not key:
            raise ProviderUnavailableError(
                "Groq is in the provider chain but GROQ_API_KEY is not set. Set it in .env, or "
                "remove 'groq' from INDRA_LLM_PROVIDER_CHAIN.",
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
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.settings.groq_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            "seed": self.settings.llm_seed,
        }
        if stop:
            payload["stop"] = list(stop)
        if JSON_SCHEMA_MARKER in prompt:
            # Native JSON mode: the model is constrained at decode time rather than asked politely.
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _extract_text(self, body: dict[str, Any]) -> str:
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMError(
                "Groq returned no choices. Retry, or fail over to the next provider.",
                context={"provider": self.name},
            )
        message = choices[0].get("message") or {}
        text = str(message.get("content") or "")
        if not text.strip():
            finish = choices[0].get("finish_reason", "unknown")
            raise LLMError(
                f"Groq returned an empty completion (finish_reason={finish}). "
                f"Reduce the prompt size or lower max_tokens.",
                context={"provider": self.name, "finish_reason": str(finish)},
            )
        return text

    def _parse_sse_line(self, line: str) -> str:
        if not line or not line.startswith("data:"):
            return ""
        raw = line[len("data:") :].strip()
        if not raw or raw == "[DONE]":
            return ""
        try:
            chunk = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("discarding malformed groq SSE frame", extra={"provider": self.name})
            return ""
        choices = chunk.get("choices") or []
        if not choices:
            return ""
        delta = choices[0].get("delta") or {}
        return str(delta.get("content") or "")


__all__ = ["GROQ_API_BASE", "GroqChatProvider"]
