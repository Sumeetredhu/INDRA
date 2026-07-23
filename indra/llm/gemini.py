"""Google Gemini chat and embedding providers.

**REST is the primary transport.** Every call in this module is a plain ``httpx`` request against
``generativelanguage.googleapis.com``, so ``google-generativeai`` is genuinely optional rather than
optional-in-the-docstring: INDRA talks to Gemini identically whether or not the SDK is installed.
The SDK is still supported as an explicit transport (``transport="sdk"``) for deployments that
standardise on it, and the guarded import below reports honestly when it is absent.

Provider notes worth knowing:

* Gemini's free tier meters by **day**, not by minute. A 429 here is almost never transient, which
  is why :func:`indra.llm.base.is_retryable` refuses to retry it and the router fails over instead.
* ``text-embedding-004`` accepts ``outputDimensionality``, so the vector width stays whatever
  ``settings.embedding_dimensions`` says rather than forcing the store to the model's native size.
* Embedding task type matters for retrieval quality: documents are embedded as
  ``RETRIEVAL_DOCUMENT`` and queries as ``RETRIEVAL_QUERY``, which is what the asymmetric model was
  trained for.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Final, Literal, Sequence

import httpx

from indra.core.config import Settings
from indra.core.exceptions import EmbeddingError, LLMError, ProviderUnavailableError
from indra.core.logging import get_logger
from indra.llm.base import (
    PROBE_TIMEOUT_CAP_S,
    AvailabilityCache,
    BaseChatProvider,
    BaseEmbeddingProvider,
    EmbedTask,
    build_http_client,
    raise_for_status,
    wrap_transport_error,
)

logger = get_logger(__name__)

try:
    import google.generativeai as genai

    _HAS_GEMINI_SDK = True
except ImportError:  # pragma: no cover - optional dependency
    genai = None  # type: ignore[assignment]
    _HAS_GEMINI_SDK = False

#: Public REST endpoint. Not a tunable — it is the vendor's API, not a deployment choice.
GEMINI_API_BASE: Final[str] = "https://generativelanguage.googleapis.com/v1beta"

#: Maps INDRA's task vocabulary onto Gemini's ``taskType`` values.
_TASK_TYPES: Final[dict[str, str]] = {
    "document": "RETRIEVAL_DOCUMENT",
    "query": "RETRIEVAL_QUERY",
}

GeminiTransport = Literal["rest", "sdk"]


class _GeminiMixin:
    """Shared credential handling and endpoint construction for both Gemini providers."""

    settings: Settings

    def _api_key(self) -> str | None:
        return self.settings.secret("gemini_api_key")

    def _headers(self, key: str) -> dict[str, str]:
        return {"x-goog-api-key": key, "Content-Type": "application/json"}

    @staticmethod
    def _model_path(model: str) -> str:
        """Gemini wants ``models/<name>``; settings hold the bare name."""
        return model if model.startswith("models/") else f"models/{model}"


class GeminiChatProvider(_GeminiMixin, BaseChatProvider):
    """Gemini chat completions (``settings.gemini_chat_model``).

    Args:
        settings: Process settings.
        transport: ``"rest"`` (default, no SDK required) or ``"sdk"``, which reports unavailable
            when ``google-generativeai`` is not installed.
    """

    name = "gemini"
    supports_json = True

    def __init__(self, settings: Settings, *, transport: GeminiTransport = "rest") -> None:
        super().__init__(settings)
        self.transport: GeminiTransport = transport
        self._availability = AvailabilityCache()
        if transport == "sdk" and not _HAS_GEMINI_SDK:
            logger.warning(
                "gemini SDK transport requested but google-generativeai is not installed; "
                "this provider will report unavailable and the router will fail over",
                extra={"provider": self.name},
            )

    # -- lifecycle ----------------------------------------------------------------

    def _http(self, key: str) -> httpx.AsyncClient:
        if self._client is None:
            self._client = build_http_client(self.settings, base_url=GEMINI_API_BASE, headers=self._headers(key))
        return self._client

    async def is_available(self) -> bool:
        """True when a key is configured and the models endpoint answers.

        Never raises: an unreachable Gemini is a routing decision, not an error.
        """
        cached = self._availability.get()
        if cached is not None:
            return cached
        if self.transport == "sdk" and not _HAS_GEMINI_SDK:
            return self._availability.set(False)
        key = self._api_key()
        if not key:
            logger.debug("gemini unavailable: no API key configured", extra={"provider": self.name})
            return self._availability.set(False)
        try:
            client = self._http(key)
            response = await asyncio.wait_for(
                client.get("/models", params={"pageSize": 1}),
                timeout=min(self.settings.llm_timeout_s, PROBE_TIMEOUT_CAP_S),
            )
            return self._availability.set(response.is_success)
        except (asyncio.TimeoutError, httpx.HTTPError) as exc:
            logger.info(
                "gemini liveness probe failed; skipping provider",
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
        payload = self._build_payload(prompt, system=system, temperature=temperature,
                                      max_tokens=max_tokens, stop=stop)
        if self.transport == "sdk":
            return await self._complete_via_sdk(payload, key)

        path = f"/{self._model_path(self.settings.gemini_chat_model)}:generateContent"
        try:
            response = await self._http(key).post(path, json=payload)
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
        """Server-sent-event streaming via ``:streamGenerateContent?alt=sse``."""
        key = self._require_key()
        if self.transport == "sdk":
            text = await self._complete(prompt, system=system, temperature=temperature,
                                        max_tokens=self.settings.llm_max_output_tokens, stop=None)
            yield text
            return

        payload = self._build_payload(
            prompt, system=system, temperature=temperature,
            max_tokens=self.settings.llm_max_output_tokens, stop=None,
        )
        path = f"/{self._model_path(self.settings.gemini_chat_model)}:streamGenerateContent"
        try:
            async with self._http(key).stream("POST", path, params={"alt": "sse"}, json=payload) as response:
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
        key = self._api_key()
        if not key:
            raise ProviderUnavailableError(
                "Gemini is in the provider chain but GEMINI_API_KEY is not set. Set it in .env, "
                "or remove 'gemini' from INDRA_LLM_PROVIDER_CHAIN.",
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
    ) -> dict[str, Any]:
        generation: dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "candidateCount": 1,
        }
        if stop:
            generation["stopSequences"] = list(stop)
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": generation,
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        return payload

    def _extract_text(self, body: dict[str, Any]) -> str:
        """Pull the completion out of a Gemini response, explaining any refusal."""
        candidates = body.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            feedback = body.get("promptFeedback", {})
            reason = feedback.get("blockReason", "no candidates returned")
            raise LLMError(
                f"Gemini returned no completion ({reason}). Rephrase the prompt or fail over to "
                f"the next provider.",
                context={"provider": self.name, "block_reason": str(reason)},
            )
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
        if not text.strip():
            finish = candidates[0].get("finishReason", "unknown")
            raise LLMError(
                f"Gemini returned an empty completion (finishReason={finish}). "
                f"Lower max output tokens or retry.",
                context={"provider": self.name, "finish_reason": str(finish)},
            )
        return text

    def _parse_sse_line(self, line: str) -> str:
        """Decode one ``data:`` frame into text, ignoring keep-alives and malformed frames."""
        if not line or not line.startswith("data:"):
            return ""
        raw = line[len("data:") :].strip()
        if not raw or raw == "[DONE]":
            return ""
        try:
            chunk = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("discarding malformed gemini SSE frame", extra={"provider": self.name})
            return ""
        candidates = chunk.get("candidates") or []
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(part.get("text", "") for part in parts if isinstance(part, dict))

    async def _complete_via_sdk(self, payload: dict[str, Any], key: str) -> str:
        """Optional SDK transport. Blocking client, so it runs in a worker thread."""
        if not _HAS_GEMINI_SDK or genai is None:
            raise ProviderUnavailableError(
                "Gemini SDK transport was requested but google-generativeai is not installed. "
                "Install it, or use the default REST transport.",
                context={"provider": self.name},
            )

        def _call() -> str:
            genai.configure(api_key=key)
            model = genai.GenerativeModel(
                self.settings.gemini_chat_model,
                system_instruction=(payload.get("systemInstruction") or {}).get("parts", [{}])[0].get("text"),
            )
            result = model.generate_content(
                payload["contents"][0]["parts"][0]["text"],
                generation_config=payload["generationConfig"],
            )
            return str(getattr(result, "text", "") or "")

        try:
            text = await asyncio.to_thread(_call)
        except Exception as exc:  # noqa: BLE001 - third-party surface, typed on the way out
            raise LLMError(
                "Gemini SDK call failed. Switch to the REST transport with "
                "GeminiChatProvider(settings, transport='rest') if this persists.",
                context={"provider": self.name},
                cause=exc,
            ) from exc
        if not text.strip():
            raise LLMError(
                "Gemini SDK returned an empty completion.", context={"provider": self.name}
            )
        return text


class GeminiEmbeddingProvider(_GeminiMixin, BaseEmbeddingProvider):
    """Gemini embeddings (``settings.gemini_embedding_model``), batched.

    Uses ``:batchEmbedContents`` so a 32-chunk batch is one round trip rather than 32.
    """

    name = "gemini"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._availability = AvailabilityCache()

    def _http(self, key: str) -> httpx.AsyncClient:
        if self._client is None:
            self._client = build_http_client(self.settings, base_url=GEMINI_API_BASE, headers=self._headers(key))
        return self._client

    async def is_available(self) -> bool:
        cached = self._availability.get()
        if cached is not None:
            return cached
        key = self._api_key()
        if not key:
            return self._availability.set(False)
        try:
            response = await asyncio.wait_for(
                self._http(key).get("/models", params={"pageSize": 1}),
                timeout=min(self.settings.llm_timeout_s, PROBE_TIMEOUT_CAP_S),
            )
            return self._availability.set(response.is_success)
        except (asyncio.TimeoutError, httpx.HTTPError) as exc:
            logger.info(
                "gemini embedding probe failed; falling back",
                extra={"provider": self.name, "reason": type(exc).__name__},
            )
            return self._availability.set(False)

    async def embed(self, texts: Sequence[str], *, task: EmbedTask = "document") -> list[list[float]]:
        if not texts:
            return []
        key = self._api_key()
        if not key:
            raise EmbeddingError(
                "Gemini embeddings requested but GEMINI_API_KEY is not set. Remove 'gemini' from "
                "INDRA_EMBEDDING_PROVIDER_CHAIN or provide a key.",
                context={"provider": self.name},
            )

        model_path = self._model_path(self.settings.gemini_embedding_model)
        task_type = _TASK_TYPES.get(task, _TASK_TYPES["document"])
        vectors: list[list[float]] = []

        for batch in self._batches(list(texts)):
            body = {
                "requests": [
                    {
                        "model": model_path,
                        "content": {"parts": [{"text": text}]},
                        "taskType": task_type,
                        "outputDimensionality": self.dimensions,
                    }
                    for text in batch
                ]
            }
            try:
                response = await self._http(key).post(f"/{model_path}:batchEmbedContents", json=body)
            except httpx.HTTPError as exc:
                self._availability.invalidate()
                raise EmbeddingError(
                    "Gemini embedding request failed at the transport layer. Check connectivity, "
                    "or let the router fall back to the local/hash embedder.",
                    context={"provider": self.name, "batch": len(batch)},
                    cause=exc,
                ) from exc
            raise_for_status(response, provider=self.name)
            vectors.extend(self._extract_vectors(response.json(), expected=len(batch)))

        return self._check_batch(texts, vectors)

    def _extract_vectors(self, body: dict[str, Any], *, expected: int) -> list[list[float]]:
        embeddings = body.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != expected:
            raise EmbeddingError(
                f"Gemini returned {len(embeddings) if isinstance(embeddings, list) else 0} "
                f"embeddings for {expected} inputs. Reduce INDRA_EMBEDDING_BATCH_SIZE.",
                context={"provider": self.name},
            )
        result: list[list[float]] = []
        for item in embeddings:
            values = item.get("values") if isinstance(item, dict) else None
            if not isinstance(values, list) or not values:
                raise EmbeddingError(
                    "Gemini returned an embedding with no values. Retry, or fall back to the "
                    "hash embedder.",
                    context={"provider": self.name},
                )
            result.append([float(value) for value in values])
        return result


__all__ = ["GEMINI_API_BASE", "GeminiChatProvider", "GeminiEmbeddingProvider", "GeminiTransport"]
