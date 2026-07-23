"""Ollama chat and embedding providers — the offline tier.

Ollama is what makes INDRA's "works in a plant with no internet" claim real: a laptop on the shop
floor running ``ollama serve`` gives genuine language-model reasoning with zero connectivity and
zero quota. It sits third in the default chain, ahead of the stub, so an offline site degrades to a
*smaller model* rather than to canned text.

Everything here speaks HTTP against ``settings.ollama_base_url``; the ``ollama`` Python package is
not required and is not imported.

Embedding widths differ between Ollama models (``nomic-embed-text`` is 768, ``mxbai-embed-large`` is
1024). :meth:`OllamaEmbeddingProvider.embed` conforms whatever comes back to
``settings.embedding_dimensions`` via :func:`indra.llm.base.conform_dimensions`, because the vector
store holds a single matrix and a ragged one is not a store.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Sequence

import httpx

from indra.core.config import Settings
from indra.core.exceptions import EmbeddingError, LLMError
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


class _OllamaMixin:
    """Shared client construction and liveness probing."""

    settings: Settings
    name: str

    def _base_url(self) -> str:
        return self.settings.ollama_base_url.rstrip("/")

    async def _probe(self, client: httpx.AsyncClient, cache: AvailabilityCache) -> bool:
        """``GET /api/tags`` is Ollama's cheapest liveness signal. Never raises."""
        cached = cache.get()
        if cached is not None:
            return cached
        try:
            response = await asyncio.wait_for(
                client.get("/api/tags"),
                timeout=min(self.settings.llm_timeout_s, PROBE_TIMEOUT_CAP_S),
            )
            available = response.is_success
            if not available:
                logger.info(
                    "ollama responded but is not ready",
                    extra={"provider": self.name, "status": response.status_code},
                )
            return cache.set(available)
        except (asyncio.TimeoutError, httpx.HTTPError) as exc:
            logger.info(
                "ollama not reachable; skipping provider",
                extra={"provider": self.name, "url": self._base_url(), "reason": type(exc).__name__},
            )
            return cache.set(False)


class OllamaChatProvider(_OllamaMixin, BaseChatProvider):
    """Chat completions against a local Ollama daemon (``settings.ollama_model``)."""

    name = "ollama"
    supports_json = True

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._availability = AvailabilityCache()

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = build_http_client(self.settings, base_url=self._base_url())
        return self._client

    async def is_available(self) -> bool:
        return await self._probe(self._http(), self._availability)

    async def _complete(
        self,
        prompt: str,
        *,
        system: str | None,
        temperature: float,
        max_tokens: int,
        stop: Sequence[str] | None,
    ) -> str:
        payload = self._build_payload(
            prompt, system=system, temperature=temperature, max_tokens=max_tokens,
            stop=stop, stream=False,
        )
        try:
            response = await self._http().post("/api/chat", json=payload)
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
        """Ollama streams newline-delimited JSON objects rather than SSE frames."""
        payload = self._build_payload(
            prompt, system=system, temperature=temperature,
            max_tokens=self.settings.llm_max_output_tokens, stop=None, stream=True,
        )
        try:
            async with self._http().stream("POST", "/api/chat", json=payload) as response:
                if not response.is_success:
                    await response.aread()
                    raise_for_status(response, provider=self.name)
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        frame = json.loads(line)
                    except json.JSONDecodeError:
                        logger.debug("discarding malformed ollama frame", extra={"provider": self.name})
                        continue
                    piece = str((frame.get("message") or {}).get("content") or "")
                    if piece:
                        yield piece
                    if frame.get("done"):
                        break
        except httpx.HTTPError as exc:
            self._availability.invalidate()
            raise wrap_transport_error(exc, provider=self.name) from exc

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

        options: dict[str, Any] = {
            "temperature": temperature,
            "num_predict": max_tokens,
            "seed": self.settings.llm_seed,
        }
        if stop:
            options["stop"] = list(stop)
        return {
            "model": self.settings.ollama_model,
            "messages": messages,
            "stream": stream,
            "options": options,
        }

    def _extract_text(self, body: dict[str, Any]) -> str:
        text = str((body.get("message") or {}).get("content") or "")
        if not text.strip():
            raise LLMError(
                f"Ollama returned an empty completion for model '{self.settings.ollama_model}'. "
                f"Confirm the model is pulled: `ollama pull {self.settings.ollama_model}`.",
                context={"provider": self.name, "model": self.settings.ollama_model},
            )
        return text


class OllamaEmbeddingProvider(_OllamaMixin, BaseEmbeddingProvider):
    """Embeddings from a local Ollama daemon (``settings.ollama_embedding_model``).

    Prefers the batched ``/api/embed`` endpoint and falls back to the older per-text
    ``/api/embeddings`` so the provider works against both current and legacy daemons.
    """

    name = "ollama"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._availability = AvailabilityCache()
        self._use_legacy_endpoint = False

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = build_http_client(self.settings, base_url=self._base_url())
        return self._client

    async def is_available(self) -> bool:
        return await self._probe(self._http(), self._availability)

    async def embed(self, texts: Sequence[str], *, task: EmbedTask = "document") -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for batch in self._batches(list(texts)):
            vectors.extend(await self._embed_batch(list(batch)))
        return self._check_batch(texts, vectors)

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        if not self._use_legacy_endpoint:
            try:
                return await self._embed_modern(batch)
            except LLMError as exc:
                logger.info(
                    "ollama /api/embed unavailable; switching to legacy /api/embeddings",
                    extra={"provider": self.name, "reason": exc.message[:120]},
                )
                self._use_legacy_endpoint = True
        return await self._embed_legacy(batch)

    async def _embed_modern(self, batch: list[str]) -> list[list[float]]:
        body = {"model": self.settings.ollama_embedding_model, "input": batch}
        try:
            response = await self._http().post("/api/embed", json=body)
        except httpx.HTTPError as exc:
            self._availability.invalidate()
            raise EmbeddingError(
                "Ollama embedding request failed at the transport layer. Is `ollama serve` running "
                f"at {self._base_url()}?",
                context={"provider": self.name},
                cause=exc,
            ) from exc
        raise_for_status(response, provider=self.name)
        payload = response.json()
        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(batch):
            raise LLMError(
                "Ollama /api/embed returned an unexpected payload shape.",
                context={"provider": self.name},
            )
        return [[float(value) for value in vector] for vector in embeddings]

    async def _embed_legacy(self, batch: list[str]) -> list[list[float]]:
        """One request per text. Slower, but the only shape old daemons understand."""
        vectors: list[list[float]] = []
        for text in batch:
            body = {"model": self.settings.ollama_embedding_model, "prompt": text}
            try:
                response = await self._http().post("/api/embeddings", json=body)
            except httpx.HTTPError as exc:
                self._availability.invalidate()
                raise EmbeddingError(
                    "Ollama legacy embedding request failed. Pull the embedding model with "
                    f"`ollama pull {self.settings.ollama_embedding_model}`.",
                    context={"provider": self.name},
                    cause=exc,
                ) from exc
            raise_for_status(response, provider=self.name)
            values = response.json().get("embedding")
            if not isinstance(values, list) or not values:
                raise EmbeddingError(
                    "Ollama returned an empty embedding. Confirm the embedding model is pulled.",
                    context={"provider": self.name, "model": self.settings.ollama_embedding_model},
                )
            vectors.append([float(value) for value in values])
        return vectors


__all__ = ["OllamaChatProvider", "OllamaEmbeddingProvider"]
