"""Embedding providers that need no API key: sentence-transformers, and the hash embedder.

:class:`HashEmbeddingProvider` is the floor of the whole retrieval stack. It is always available,
never touches the network, and produces vectors whose cosine similarity is genuinely meaningful —
which is what lets `git clone && pytest` exercise real GraphRAG retrieval instead of a mock.

How it works, and why it is not a toy
-------------------------------------
A random projection of a bag of features is a *signature*, and the Johnson–Lindenstrauss lemma says
that inner products in the projected space approximate inner products in the original sparse feature
space. So the design is:

1. **Feature extraction.** Word unigrams carry most of the signal; adjacent bigrams add a little word
   order; character 4-grams inside each word add morphological robustness so ``bearings`` and
   ``bearing`` are not strangers, and so an OCR slip costs similarity rather than all of it.
2. **Sublinear term weighting.** ``w · (1 + ln tf)`` — repeating a word ten times should not make a
   passage ten times more about it.
3. **Signed feature hashing.** Each feature is hashed twice with independent keyed BLAKE2b digests
   into ``(index, sign)`` pairs. Signed hashing makes collisions cancel in expectation instead of
   accumulating, so the dot product stays unbiased; two hashes halve the collision variance.
4. **L2 normalisation.** Cosine similarity becomes a plain dot product, which is what both vector
   stores assume.

The keying is derived from ``settings.llm_seed``, so vectors are byte-identical across processes and
machines — a corpus embedded today still matches a query embedded next week.

What it is *not*: this captures lexical and sub-lexical overlap, not paraphrase. "Pump is broken"
and "impeller has failed" are near-orthogonal to it. That is the honest limit of a keyless embedder,
and it is why the provider chain prefers Gemini or a local transformer when either is available.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from importlib.util import find_spec
from typing import Final, Sequence

import numpy as np

from indra.core.config import Settings
from indra.core.exceptions import EmbeddingError
from indra.core.logging import get_logger
from indra.llm.base import BaseEmbeddingProvider, EmbedTask, conform_dimensions

logger = get_logger(__name__)

# ``sentence_transformers`` pulls in torch, which costs seconds to import. Probing for the module
# spec is cheap and does not execute it; the real import happens inside a worker thread on first
# use (CLAUDE.md rule 6 — an absent optional dependency degrades one capability, silently to the
# caller and loudly in the logs).
try:
    _HAS_SENTENCE_TRANSFORMERS: bool = find_spec("sentence_transformers") is not None
except (ImportError, ValueError):  # pragma: no cover - defensive: broken import machinery
    _HAS_SENTENCE_TRANSFORMERS = False


# --------------------------------------------------------------------------------------
# Feature-extraction constants. Not product tunables — changing one invalidates every vector
# already in the store, so they are pinned here with the rationale attached.
# --------------------------------------------------------------------------------------

#: Relative weight of whole-word features. Words carry the topic.
WORD_WEIGHT: Final[float] = 1.0

#: Relative weight of adjacent word pairs. Enough word order to separate "high vibration" from
#: "vibration high" a little, not enough to dominate an unordered match.
BIGRAM_WEIGHT: Final[float] = 0.45

#: Relative weight of character n-grams. Deliberately small: they add morphological and OCR
#: robustness, but at a high weight every pair of English strings starts to look similar.
CHARGRAM_WEIGHT: Final[float] = 0.30

#: Width of the character n-grams taken inside each padded word.
CHARGRAM_SIZE: Final[int] = 4

#: Independent hash draws per feature. Two is the sweet spot: it halves collision variance for a
#: 2x cost, a third buys almost nothing at 768 dimensions.
HASH_REPLICAS: Final[int] = 2

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

#: Function words that appear in every industrial document and therefore discriminate nothing.
_STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for", "from", "has",
        "have", "in", "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "was",
        "were", "will", "with", "which", "we", "you", "they", "there", "their",
    }
)


def _tokenise(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics, keep hyphenated plant tags whole.

    ``P-101`` must survive as one token: splitting it into ``p`` and ``101`` would make every pump
    tag in the plant look alike.
    """
    return [token for token in _TOKEN_RE.findall(text.lower()) if token not in _STOPWORDS]


def _features(text: str) -> dict[str, float]:
    """Extract the weighted feature bag for ``text``.

    Returns:
        Mapping of feature string to accumulated weight, before sublinear scaling.
    """
    tokens = _tokenise(text)
    bag: dict[str, float] = {}
    if not tokens:
        return bag

    for token in tokens:
        bag[f"w:{token}"] = bag.get(f"w:{token}", 0.0) + WORD_WEIGHT
        if len(token) > CHARGRAM_SIZE:
            padded = f"^{token}$"
            for start in range(len(padded) - CHARGRAM_SIZE + 1):
                gram = padded[start : start + CHARGRAM_SIZE]
                bag[f"c:{gram}"] = bag.get(f"c:{gram}", 0.0) + CHARGRAM_WEIGHT

    for left, right in zip(tokens, tokens[1:]):
        key = f"b:{left}_{right}"
        bag[key] = bag.get(key, 0.0) + BIGRAM_WEIGHT

    return bag


def _hash_indices(feature: str, dimensions: int, key: bytes) -> list[tuple[int, float]]:
    """Map one feature to ``HASH_REPLICAS`` signed slots.

    Uses keyed BLAKE2b rather than :func:`hash`, whose per-process randomisation would make
    embeddings non-reproducible across runs — fatal for a persisted vector store.
    """
    slots: list[tuple[int, float]] = []
    for replica in range(HASH_REPLICAS):
        digest = hashlib.blake2b(
            feature.encode("utf-8"),
            key=key,
            digest_size=8,
            person=replica.to_bytes(2, "big").ljust(16, b"\x00")[:16],
        ).digest()
        value = int.from_bytes(digest, "big")
        index = value % dimensions
        sign = 1.0 if (value >> 63) & 1 else -1.0
        slots.append((index, sign))
    return slots


class HashEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic, keyless, always-available embedder. See the module docstring for the design.

    Args:
        settings: Provides ``embedding_dimensions`` and the ``llm_seed`` used to key the hashes.
    """

    name = "hash"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._key = hashlib.blake2b(
            str(settings.llm_seed).encode("utf-8"), digest_size=16
        ).digest()
        self._scale = 1.0 / math.sqrt(HASH_REPLICAS)

    async def is_available(self) -> bool:
        """Always ``True``. This is the provider that makes "no API key" a supported configuration."""
        return True

    async def embed(self, texts: Sequence[str], *, task: EmbedTask = "document") -> list[list[float]]:
        """Embed a batch.

        ``task`` is accepted for protocol compatibility but ignored: the representation is
        symmetric, so a query and a document that say the same thing land in the same place. An
        asymmetric variant would need a trained model, not a hash.
        """
        if not texts:
            return []
        try:
            return await asyncio.to_thread(self._embed_sync, list(texts))
        except Exception as exc:  # noqa: BLE001 - the last-resort embedder must report typed errors
            raise EmbeddingError(
                "Hash embedding failed, which should be impossible without a corrupt input. "
                "Check that the passages are valid UTF-8 text.",
                context={"batch": len(texts)},
                cause=exc,
            ) from exc

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        """CPU-bound projection; always called through :func:`asyncio.to_thread`."""
        dimensions = self.dimensions
        matrix = np.zeros((len(texts), dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature, weight in _features(text).items():
                # Sublinear term frequency: the tenth mention adds far less than the second.
                scaled = weight * (1.0 + math.log(max(weight, 1.0))) * self._scale
                for index, sign in _hash_indices(feature, dimensions, self._key):
                    matrix[row, index] += sign * scaled
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        np.divide(matrix, np.maximum(norms, 1e-12), out=matrix)
        return [row.astype(float).tolist() for row in matrix]


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """Sentence-transformers embedder for offline installs that have the model cached.

    Optional by design: if ``sentence-transformers`` is not installed, or the weights are not on
    disk and there is no network, :meth:`is_available` reports ``False`` and the router falls
    through to :class:`HashEmbeddingProvider`.
    """

    name = "local"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._model: object | None = None
        self._load_failed = False
        self._lock = asyncio.Lock()

    async def is_available(self) -> bool:
        """True only once the model is genuinely loaded — an unloadable model is not availability."""
        if not _HAS_SENTENCE_TRANSFORMERS or self._load_failed:
            return False
        return await self._ensure_model() is not None

    async def embed(self, texts: Sequence[str], *, task: EmbedTask = "document") -> list[list[float]]:
        if not texts:
            return []
        model = await self._ensure_model()
        if model is None:
            raise EmbeddingError(
                "Local sentence-transformers embedder is unavailable. Install "
                "`sentence-transformers` and pre-download "
                f"'{self.settings.local_embedding_model}', or rely on the hash embedder.",
                context={"provider": self.name},
            )
        vectors: list[list[float]] = []
        for batch in self._batches(list(texts)):
            try:
                encoded = await asyncio.to_thread(self._encode_sync, model, list(batch))
            except Exception as exc:  # noqa: BLE001 - third-party surface, typed on the way out
                raise EmbeddingError(
                    "Local embedding model failed while encoding a batch. Reduce "
                    "INDRA_EMBEDDING_BATCH_SIZE or switch to the hash embedder.",
                    context={"provider": self.name, "batch": len(batch)},
                    cause=exc,
                ) from exc
            vectors.extend(encoded)
        return self._check_batch(texts, vectors)

    @staticmethod
    def _encode_sync(model: object, batch: list[str]) -> list[list[float]]:
        """Run the transformer. CPU-bound, so this only ever runs in a worker thread."""
        encoded = model.encode(batch, normalize_embeddings=True)  # type: ignore[attr-defined]
        return [[float(value) for value in row] for row in encoded]

    async def _ensure_model(self) -> object | None:
        """Load the model once, in a worker thread, never raising into the caller."""
        if self._model is not None:
            return self._model
        if self._load_failed or not _HAS_SENTENCE_TRANSFORMERS:
            return None
        async with self._lock:
            if self._model is not None:
                return self._model
            if self._load_failed:
                return None
            try:
                self._model = await asyncio.to_thread(self._load_sync)
            except Exception as exc:  # noqa: BLE001 - optional dependency must never crash startup
                self._load_failed = True
                logger.warning(
                    "local embedding model unavailable; falling back to the hash embedder",
                    extra={"model": self.settings.local_embedding_model, "reason": str(exc)[:200]},
                )
                return None
            self.dimensions = self.settings.embedding_dimensions
            logger.info(
                "local embedding model loaded",
                extra={"model": self.settings.local_embedding_model},
            )
            return self._model

    def _load_sync(self) -> object:
        """Import and construct the model. Runs in a worker thread; may hit disk."""
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415 - optional dependency

        return SentenceTransformer(self.settings.local_embedding_model)


# --------------------------------------------------------------------------------------
# Self-check
# --------------------------------------------------------------------------------------


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity between two vectors. Returns 0.0 if either has no magnitude."""
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 0.0:
        return 0.0
    return float(np.dot(a, b) / denominator)


#: The claim :class:`HashEmbeddingProvider` has to earn: related industrial text must score higher
#: than unrelated industrial text. Asserted by :func:`verify_semantics` and by the unit tests.
SEMANTIC_PROBE: Final[tuple[str, str, str]] = (
    "bearing wear vibration pump",
    "pump bearing vibration high",
    "regulatory audit clause",
)


async def verify_semantics(settings: Settings) -> dict[str, float]:
    """Prove the hash embedder ranks related text above unrelated text.

    Returns:
        ``{"related": float, "unrelated": float, "margin": float}``.

    Raises:
        EmbeddingError: if the ordering does not hold, which would mean retrieval is broken.
    """
    provider = HashEmbeddingProvider(settings)
    query, related, unrelated = SEMANTIC_PROBE
    vectors = await provider.embed([query, related, unrelated], task="query")
    related_score = cosine(vectors[0], vectors[1])
    unrelated_score = cosine(vectors[0], vectors[2])
    if related_score <= unrelated_score:
        raise EmbeddingError(
            "Hash embedder failed its semantic ordering check: related industrial text did not "
            "score above unrelated text. Retrieval would be meaningless — do not ship this build.",
            context={"related": related_score, "unrelated": unrelated_score},
        )
    return {
        "related": round(related_score, 4),
        "unrelated": round(unrelated_score, 4),
        "margin": round(related_score - unrelated_score, 4),
    }


__all__ = [
    "BIGRAM_WEIGHT",
    "CHARGRAM_SIZE",
    "CHARGRAM_WEIGHT",
    "HASH_REPLICAS",
    "HashEmbeddingProvider",
    "LocalEmbeddingProvider",
    "SEMANTIC_PROBE",
    "WORD_WEIGHT",
    "conform_dimensions",
    "cosine",
    "verify_semantics",
]
