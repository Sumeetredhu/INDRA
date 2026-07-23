"""Answering with no network: Ollama when it is there, extraction when it is not.

The offline promise INDRA makes is *always answer something*. Two tiers deliver it:

* :class:`OllamaLocalModel` — a quantised Llama on the technician's own laptop or on the site's
  edge box, spoken to over plain HTTP. No ``ollama`` Python package, no cloud, no quota.
* :class:`ExtractivePlanner` — when Ollama is unreachable, the answer is *extracted* rather than
  generated: cosine similarity over the bundle's packed index (falling back to rare-term lexical
  overlap when no query embedding can be produced), then the highest-scoring sentences from the
  best passages, quoted verbatim.

Extraction is a deliberate choice, not a consolation prize. On a plant floor a verbatim sentence
from an OEM manual with a citation attached is worth more than a fluent paraphrase from a 4-bit
model, and it cannot hallucinate a torque value. Both tiers return a fully-formed
:class:`~indra.core.models.Answer` with real :class:`~indra.core.models.SourceRef` citations, so the
offline card renders identically to the online one and the technician can see exactly where the
words came from.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Awaitable, Callable, Final, Sequence

import httpx

from indra.core.config import Settings
from indra.core.exceptions import IndraError, LLMError, ProviderUnavailableError
from indra.core.logging import get_logger
from indra.core.models import (
    Answer,
    Confidence,
    QueryType,
    ReasoningStep,
    Severity,
    SourceRef,
    UncertaintyFlag,
    UncertaintySource,
)

from indra.agents.mobile_agent.offline_bundle import LocalIndex, SearchHit

logger = get_logger(__name__)

EmbedFn = Callable[[str], Awaitable[Sequence[float]]]
"""Produces a query embedding. Injected so this module never reaches for a provider itself."""

#: Liveness probe budget. On a dead link ``connect`` fails fast; this caps the pathological case
#: where a captive portal accepts the TCP connection and then never answers.
_PROBE_TIMEOUT_S: Final[float] = 1.5

#: How long a probe result is trusted. Long enough that a burst of offline queries costs one probe,
#: short enough that a technician walking back into coverage gets the real model within a minute.
_PROBE_TTL_S: Final[float] = 45.0

#: Passages handed to the local model, and considered by the extractor. A 4-bit 8B model degrades
#: badly past a few thousand tokens of context, and the field cannot afford the latency either.
_CONTEXT_PASSAGES: Final[int] = 5

#: Sentences an extractive answer is allowed to quote. Three is a readable answer on a phone.
_EXTRACT_SENTENCES: Final[int] = 3

#: Characters of each passage placed in the local model's prompt.
_CONTEXT_CHARS: Final[int] = 700

#: Below this retrieval score the bundle simply does not contain the answer, and saying so is the
#: correct output. Matches the online path's ``settings.min_relevance_score`` in spirit; kept local
#: because it is scored against a quantised index, not the full-precision one.
_MIN_USEFUL_SCORE: Final[float] = 0.12

_SENTENCE_SPLIT: Final[re.Pattern[str]] = re.compile(r"(?<=[.!?।])\s+|\n+")
_WORD_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-_/.]*")

_LOCAL_SYSTEM_PROMPT: Final[str] = (
    "You are INDRA running offline on a plant technician's device.\n"
    "Answer ONLY from the numbered context passages below. Rules:\n"
    "1. If the passages do not contain the answer, say so plainly. Never invent a number, a torque "
    "value, a part number, or a procedure step.\n"
    "2. Cite the passage numbers you used, like [1] or [2].\n"
    "3. Be short. The technician is standing next to running machinery.\n"
    "4. Copy equipment tags (P-101, HX-205A) exactly as written."
)

_NO_BUNDLE_MESSAGE: Final[str] = (
    "There is no offline bundle on this device yet, so I have nothing to answer from. Build a "
    "bundle before the shift while connectivity is available."
)

_NO_EVIDENCE_MESSAGE: Final[str] = (
    "The offline bundle on this device contains nothing about that. Queue the question — it will be "
    "answered against the full plant brain as soon as connectivity returns."
)


class OllamaLocalModel:
    """Chat completions against a local Ollama daemon, over ``httpx``.

    Deliberately independent of :mod:`indra.llm.ollama`: that provider participates in the router's
    fail-over chain and budget ledger, which is the wrong shape for the offline path. Here Ollama is
    either reachable or it is not, and "not" is an ordinary outcome rather than a provider failure.
    """

    name: Final[str] = "ollama_local"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: httpx.AsyncClient | None = None
        self._probed_at: float = 0.0
        self._probe_result: bool | None = None
        self._lock = asyncio.Lock()

    @property
    def base_url(self) -> str:
        return self._settings.ollama_base_url.rstrip("/")

    @property
    def model(self) -> str:
        return self._settings.ollama_model

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(
                    self._settings.llm_timeout_s,
                    connect=min(self._settings.llm_timeout_s, 5.0),
                ),
                follow_redirects=True,
            )
        return self._client

    async def is_available(self) -> bool:
        """Probe ``GET /api/tags``. Cached, never raises, never blocks longer than the probe budget."""
        now = time.monotonic()
        if self._probe_result is not None and (now - self._probed_at) < _PROBE_TTL_S:
            return self._probe_result
        async with self._lock:
            now = time.monotonic()
            if self._probe_result is not None and (now - self._probed_at) < _PROBE_TTL_S:
                return self._probe_result
            available = await self._probe()
            self._probe_result = available
            self._probed_at = time.monotonic()
            return available

    async def _probe(self) -> bool:
        try:
            response = await asyncio.wait_for(self._http().get("/api/tags"), timeout=_PROBE_TIMEOUT_S)
        except (asyncio.TimeoutError, httpx.HTTPError) as exc:
            logger.info(
                "local model unreachable; offline answers will be extractive",
                extra={"url": self.base_url, "reason": type(exc).__name__},
            )
            return False
        if not response.is_success:
            logger.info(
                "local model responded but is not ready",
                extra={"url": self.base_url, "status": response.status_code},
            )
            return False
        return True

    async def generate(self, prompt: str, *, system: str | None = None) -> str:
        """Run one completion against the local daemon.

        Raises:
            ProviderUnavailableError: the daemon is unreachable or refused the request.
            LLMError: the daemon answered but produced nothing usable.
        """
        payload = {
            "model": self.model,
            "messages": (
                [{"role": "system", "content": system}] if system else []
            ) + [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": self._settings.llm_temperature,
                "num_predict": self._settings.llm_max_output_tokens,
                "seed": self._settings.llm_seed,
            },
        }
        try:
            response = await self._http().post("/api/chat", json=payload)
        except httpx.HTTPError as exc:
            self._probe_result = False
            self._probed_at = time.monotonic()
            raise ProviderUnavailableError(
                f"The local model at {self.base_url} is not reachable. Start it with `ollama serve` "
                f"and pull the model with `ollama pull {self.model}`, or accept extractive answers.",
                context={"provider": self.name, "model": self.model},
                cause=exc,
            ) from exc
        if not response.is_success:
            self._probe_result = False
            self._probed_at = time.monotonic()
            raise ProviderUnavailableError(
                f"The local model returned HTTP {response.status_code}. Confirm the model is pulled: "
                f"`ollama pull {self.model}`.",
                context={"provider": self.name, "status": response.status_code, "model": self.model},
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise LLMError(
                "The local model returned a body that is not JSON. Check the Ollama version.",
                context={"provider": self.name},
                cause=exc,
            ) from exc
        text = str((body.get("message") or {}).get("content") or "").strip()
        if not text:
            raise LLMError(
                f"The local model returned an empty completion for '{self.model}'.",
                context={"provider": self.name, "model": self.model},
            )
        return text

    async def aclose(self) -> None:
        """Release the HTTP connection pool. Safe to call more than once."""
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception as exc:  # pragma: no cover - shutdown must not raise
                logger.debug("local model client close failed", extra={"error": str(exc)})
            finally:
                self._client = None


class ExtractivePlanner:
    """Builds an answer by quoting the bundle rather than generating from it.

    Pure Python and numpy: the last thing standing when there is no model, no network, and no key.
    """

    name: Final[str] = "offline_extractive"

    @staticmethod
    def compose(question: str, hits: Sequence[SearchHit]) -> str:
        """Quote the sentences that actually answer the question, in retrieval order."""
        if not hits:
            return _NO_EVIDENCE_MESSAGE
        terms = {token.lower() for token in _WORD_RE.findall(question)}
        scored: list[tuple[float, int, int, str, SearchHit]] = []
        for rank, hit in enumerate(hits):
            for position, sentence in enumerate(_sentences(hit.entry.text)):
                overlap = _overlap(sentence, terms)
                if overlap <= 0.0 and rank > 0:
                    continue
                # Retrieval rank is the prior; term overlap picks the sentence inside the passage.
                scored.append((hit.score + overlap, rank, position, sentence, hit))
        if not scored:
            best = hits[0]
            head = _sentences(best.entry.text)
            return _cite(head[0] if head else best.entry.text[:300], hits.index(best) + 1)

        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        chosen: list[str] = []
        seen: set[str] = set()
        for _, rank, _, sentence, _hit in scored:
            key = sentence.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            chosen.append(_cite(sentence.strip(), rank + 1))
            if len(chosen) >= _EXTRACT_SENTENCES:
                break
        return " ".join(chosen)


class OfflineAnswerer:
    """The offline query path: retrieve from the bundle, then generate or extract.

    Holds no bundle of its own — the caller passes the index in — so one answerer serves every
    bundle the agent has built, and tests can drive it with a hand-made index.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        model: OllamaLocalModel | None = None,
        embed: EmbedFn | None = None,
    ) -> None:
        self._settings = settings
        self._model = model if model is not None else OllamaLocalModel(settings)
        self._embed = embed
        self._planner = ExtractivePlanner()

    @property
    def model(self) -> OllamaLocalModel:
        return self._model

    async def answer(
        self,
        question: str,
        *,
        index: LocalIndex | None,
        equipment_tag: str | None = None,
        language: str | None = None,
    ) -> Answer:
        """Answer ``question`` from the packed bundle. Always returns an :class:`Answer`.

        The reasoning chain records which retrieval path fired and which answering tier produced the
        text, so "why did the offline answer look like that?" is inspectable after the shift.
        """
        started = time.perf_counter()
        query = question.strip()
        if not query:
            return _bare_answer(
                question,
                "I did not receive a question.",
                provider="none",
                confidence=0.0,
                rationale="Empty offline query.",
                language=language or self._settings.default_language,
            )
        if index is None or index.is_empty:
            logger.warning("offline query with no bundle on the device", extra={"equipment_tag": equipment_tag})
            return _bare_answer(
                query,
                _NO_BUNDLE_MESSAGE,
                provider="none",
                confidence=0.0,
                rationale="No offline bundle is present, so there is no evidence to reason over.",
                language=language or self._settings.default_language,
            )

        hits, retrieval_step = await self._retrieve(query, index, equipment_tag=equipment_tag)
        useful = [hit for hit in hits if hit.score >= _MIN_USEFUL_SCORE]
        if not useful:
            answer = _bare_answer(
                query,
                _NO_EVIDENCE_MESSAGE,
                provider=self._planner.name,
                confidence=0.0,
                rationale=f"Nothing in the offline bundle scored above {_MIN_USEFUL_SCORE:.2f}.",
                language=language or self._settings.default_language,
            )
            answer.reasoning_chain.append(retrieval_step)
            answer.latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
            return answer

        sources = [hit.entry.to_source(relevance=hit.score) for hit in useful]
        text, provider, generation_step = await self._compose(query, useful)

        confidence = _confidence(useful, provider=provider)
        answer = Answer(
            query=query,
            query_type=QueryType.FACTUAL,
            answer_text=text,
            language=language or self._settings.default_language,
            confidence=confidence,
            reasoning_chain=[retrieval_step, generation_step],
            sources=sources,
            uncertainty_flags=_flags(provider=provider, hits=useful),
            provider_used=provider,
            latency_ms=round((time.perf_counter() - started) * 1000.0, 2),
        )
        logger.info(
            "offline answer produced",
            extra={
                "provider": provider,
                "equipment_tag": equipment_tag,
                "hits": len(useful),
                "top_score": round(useful[0].score, 4),
                "retrieval": useful[0].method,
                "latency_ms": answer.latency_ms,
            },
        )
        return answer

    # -- retrieval ---------------------------------------------------------------------
    async def _retrieve(
        self, query: str, index: LocalIndex, *, equipment_tag: str | None
    ) -> tuple[list[SearchHit], ReasoningStep]:
        """Cosine over the packed index, with lexical overlap as the guaranteed floor."""
        started = time.perf_counter()
        vector = await self._query_vector(query)
        hits: list[SearchHit] = []
        method = "lexical"
        if vector is not None:
            hits = index.search(vector, top_k=_CONTEXT_PASSAGES * 2)
            method = "semantic"
        if not hits:
            hits = index.lexical_search(query, top_k=_CONTEXT_PASSAGES * 2)
            method = "lexical"

        if equipment_tag:
            wanted = equipment_tag.strip().upper()
            preferred = [hit for hit in hits if (hit.entry.equipment_tag or "").upper() == wanted]
            if preferred:
                hits = preferred + [hit for hit in hits if hit not in preferred]

        hits = hits[:_CONTEXT_PASSAGES]
        duration = (time.perf_counter() - started) * 1000.0
        step = ReasoningStep(
            order=1,
            action=f"Searched the offline bundle ({method}) across {len(index.entries)} packed passages",
            finding=(
                f"{len(hits)} passage(s) matched"
                + (f", top score {hits[0].score:.2f}" if hits else " — nothing relevant is on this device")
            ),
            confidence=Confidence(
                value=round(hits[0].score, 4) if hits else 0.0,
                rationale=(
                    "Cosine similarity over the bundle's quantised embedding index"
                    if method == "semantic"
                    else "Rare-term overlap: no query embedding could be produced offline"
                ),
                method="semantic",
            ),
            sources=[hit.entry.to_source(relevance=hit.score) for hit in hits],
            duration_ms=round(duration, 2),
        )
        return hits, step

    async def _query_vector(self, query: str) -> Sequence[float] | None:
        if self._embed is None:
            return None
        try:
            vector = await self._embed(query)
        except IndraError as exc:
            logger.info(
                "offline query embedding failed; falling back to lexical retrieval",
                extra={"error": exc.message},
            )
            return None
        except Exception as exc:  # defensive: injected callable is external to this module
            logger.info(
                "offline query embedding raised an untyped error; falling back to lexical retrieval",
                extra={"error": str(exc)},
            )
            return None
        return vector if vector else None

    # -- generation --------------------------------------------------------------------
    async def _compose(
        self, query: str, hits: Sequence[SearchHit]
    ) -> tuple[str, str, ReasoningStep]:
        """Try the local model; fall back to extraction. Returns ``(text, provider, step)``."""
        started = time.perf_counter()
        if not self._settings.offline_mode or True:  # both modes take this path; see below
            # ``offline_mode`` forces the *offline* path, it does not forbid the local model — a
            # quantised model on the technician's own laptop is exactly what offline mode is for.
            if await self._model.is_available():
                try:
                    text = await self._model.generate(
                        _build_prompt(query, hits), system=_LOCAL_SYSTEM_PROMPT
                    )
                except IndraError as exc:
                    logger.warning(
                        "local model failed mid-answer; extracting from the bundle instead",
                        extra={"error": exc.message, "provider": self._model.name},
                    )
                else:
                    return (
                        text,
                        self._model.name,
                        ReasoningStep(
                            order=2,
                            action=f"Generated an answer with the on-device model ({self._model.model})",
                            finding=text[:300],
                            confidence=Confidence(
                                value=0.55,
                                rationale=(
                                    "A quantised local model, constrained to the bundled passages. "
                                    "Lower ceiling than the online chain by construction."
                                ),
                                method="llm",
                            ),
                            duration_ms=round((time.perf_counter() - started) * 1000.0, 2),
                        ),
                    )

        text = self._planner.compose(query, hits)
        return (
            text,
            self._planner.name,
            ReasoningStep(
                order=2,
                action="Extracted the answer verbatim from the bundled passages",
                finding=text[:300],
                confidence=Confidence(
                    value=round(min(0.75, hits[0].score), 4) if hits else 0.0,
                    rationale=(
                        "No local model was reachable, so the answer is quoted from the bundle "
                        "rather than generated. Nothing here is paraphrased."
                    ),
                    method="semantic",
                ),
                duration_ms=round((time.perf_counter() - started) * 1000.0, 2),
            ),
        )


# ======================================================================================
# Helpers
# ======================================================================================


def _build_prompt(query: str, hits: Sequence[SearchHit]) -> str:
    lines = ["Context passages from the offline bundle:", ""]
    for position, hit in enumerate(hits, start=1):
        entry = hit.entry
        label = entry.equipment_tag or entry.document_title or entry.document_id
        page = f", p.{entry.page}" if entry.page else ""
        lines.append(f"[{position}] ({label} — {entry.document_title}{page})")
        lines.append(entry.text[:_CONTEXT_CHARS].strip())
        lines.append("")
    lines.append(f"Technician's question: {query}")
    return "\n".join(lines)


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_SPLIT.split(text) if part and part.strip()]


def _overlap(sentence: str, terms: set[str]) -> float:
    """Fraction of the question's terms present in ``sentence``, in ``[0, 1]``."""
    if not terms:
        return 0.0
    tokens = {token.lower() for token in _WORD_RE.findall(sentence)}
    if not tokens:
        return 0.0
    return len(tokens & terms) / len(terms)


def _cite(sentence: str, number: int) -> str:
    stripped = sentence.strip()
    if not stripped:
        return ""
    return f"{stripped} [{number}]" if not stripped.endswith(f"[{number}]") else stripped


def _confidence(hits: Sequence[SearchHit], *, provider: str) -> Confidence:
    """Offline confidence is capped below the online path on purpose.

    A bundle is a lossy snapshot searched through a quantised index; asserting the same certainty as
    a full GraphRAG answer would be a lie the technician cannot check.
    """
    top = hits[0].score if hits else 0.0
    ceiling = 0.7 if provider == "ollama_local" else 0.65
    value = round(min(ceiling, 0.25 + 0.6 * top), 4)
    return Confidence(
        value=value,
        rationale=(
            f"Offline answer from {len(hits)} bundled passage(s), best match {top:.2f}. "
            "Capped below the online path: the bundle is a snapshot, not the live graph."
        ),
        method="llm" if provider == "ollama_local" else "semantic",
    )


def _flags(*, provider: str, hits: Sequence[SearchHit]) -> list[UncertaintyFlag]:
    flags = [
        UncertaintyFlag(
            source=UncertaintySource.STALE_DOCUMENT,
            message=(
                "Answered offline from the bundle on this device. Anything recorded in the plant "
                "since the bundle was built is not reflected here."
            ),
            severity=Severity.WARNING,
            suggested_action="Re-check online before acting on anything time-sensitive.",
        )
    ]
    if provider == "ollama_local":
        flags.append(
            UncertaintyFlag(
                source=UncertaintySource.MODEL_EXTRAPOLATION,
                message=(
                    "Generated by a small on-device model constrained to the bundled passages. It is "
                    "weaker than the online reasoning chain."
                ),
                severity=Severity.WARNING,
                suggested_action="Open the cited passages and confirm any number before acting.",
            )
        )
    if hits and all(hit.method == "lexical" for hit in hits):
        flags.append(
            UncertaintyFlag(
                source=UncertaintySource.SPARSE_EVIDENCE,
                message=(
                    "Retrieved by keyword overlap because no query embedding could be produced on "
                    "this device. Semantically-phrased questions may have missed relevant passages."
                ),
                severity=Severity.LOW,
                suggested_action="Try the question again using the wording from the manual.",
            )
        )
    return flags


def _bare_answer(
    query: str,
    message: str,
    *,
    provider: str,
    confidence: float,
    rationale: str,
    language: str,
) -> Answer:
    """An honest, source-free answer. ``Answer``'s validator adds the sparse-evidence flag itself."""
    return Answer(
        query=query or "(empty question)",
        query_type=QueryType.FACTUAL,
        answer_text=message,
        language=language,
        confidence=Confidence(value=confidence, rationale=rationale, method="heuristic"),
        provider_used=provider,
    )


__all__ = [
    "EmbedFn",
    "ExtractivePlanner",
    "OfflineAnswerer",
    "OllamaLocalModel",
]
