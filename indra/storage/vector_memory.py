"""In-process vector store: numpy cosine similarity over a stacked matrix.

Exact top-k, not approximate. At demo scale (thousands of chunks) an exact scan is faster than
building an index, and it removes an entire class of "why did retrieval miss that" questions.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from indra.core.exceptions import VectorStoreError
from indra.core.logging import get_logger
from indra.core.models import Chunk

logger = get_logger(__name__)


class MemoryVectorStore:
    """Implements ``contracts.VectorStore`` with numpy. Deterministic and dependency-free."""

    name = "vectors:memory"
    backend = "memory"

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._order: list[str] = []
        self._matrix: np.ndarray | None = None
        self._dirty = True

    # ------------------------------------------------------------------ writes

    async def upsert(self, chunks: Sequence[Chunk], *, embeddings: Sequence[Sequence[float]]) -> int:
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                "Chunk and embedding counts differ; refusing to write a misaligned index.",
                context={"chunks": len(chunks), "embeddings": len(embeddings)},
            )
        for chunk, embedding in zip(chunks, embeddings):
            vector = np.asarray(embedding, dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            if norm > 0:
                vector = vector / norm
            stored = chunk.model_copy(update={"embedding": vector.tolist()})
            if chunk.chunk_id not in self._chunks:
                self._order.append(chunk.chunk_id)
            self._chunks[chunk.chunk_id] = stored
        self._dirty = True
        return len(chunks)

    async def delete_document(self, document_id: str) -> int:
        removed = [cid for cid, chunk in self._chunks.items() if chunk.document_id == document_id]
        for chunk_id in removed:
            self._chunks.pop(chunk_id, None)
            self._order.remove(chunk_id)
        self._dirty = True
        return len(removed)

    # ------------------------------------------------------------------ reads

    def _rebuild(self) -> None:
        if not self._dirty:
            return
        if not self._order:
            self._matrix = None
            self._dirty = False
            return
        vectors = [self._chunks[cid].embedding or [] for cid in self._order]
        width = max((len(v) for v in vectors), default=0)
        if width == 0:
            self._matrix = None
            self._dirty = False
            return
        matrix = np.zeros((len(vectors), width), dtype=np.float32)
        for row, vector in enumerate(vectors):
            if vector:
                matrix[row, : len(vector)] = np.asarray(vector[:width], dtype=np.float32)
        self._matrix = matrix
        self._dirty = False

    async def search(
        self,
        embedding: Sequence[float],
        *,
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        self._rebuild()
        if self._matrix is None or not self._order:
            return []

        query = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(query))
        if norm == 0:
            return []
        query = query / norm
        if query.shape[0] != self._matrix.shape[1]:
            # Dimension drift means the embedding provider changed mid-run. Fail loudly rather
            # than silently returning nonsense rankings.
            raise VectorStoreError(
                "Query embedding dimension does not match the index; the embedding provider changed. "
                "Clear .cache/ and re-ingest, or pin INDRA_EMBEDDING_PROVIDER_CHAIN.",
                context={"query_dim": int(query.shape[0]), "index_dim": int(self._matrix.shape[1])},
            )

        scores = self._matrix @ query  # rows are already L2-normalised
        candidates = list(zip(self._order, scores.tolist()))

        if filters:
            candidates = [
                (cid, score) for cid, score in candidates
                if self._passes(self._chunks[cid], filters)
            ]

        # Cosine over normalised vectors is in [-1, 1]; map to [0, 1] so callers can treat it
        # as a probability-like relevance without knowing the metric.
        candidates = [(cid, (score + 1.0) / 2.0) for cid, score in candidates]
        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[:top_k]

    @staticmethod
    def _passes(chunk: Chunk, filters: dict[str, Any]) -> bool:
        for field, expected in filters.items():
            actual = getattr(chunk, field, None)
            if actual is None:
                actual = chunk.metadata.get(field)
            if isinstance(expected, (list, tuple, set)):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False
        return True

    async def get_chunks(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        return [self._chunks[cid] for cid in chunk_ids if cid in self._chunks]

    async def count(self) -> int:
        return len(self._chunks)

    async def health(self) -> dict[str, Any]:
        return {"ok": True, "backend": "memory", "detail": f"{len(self._chunks)} chunks indexed"}

    async def close(self) -> None:
        return None


__all__ = ["MemoryVectorStore"]
