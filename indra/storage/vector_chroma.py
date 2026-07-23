"""ChromaDB vector store.

``chromadb`` is an optional dependency: the import is lazy and failure is not fatal, because the
factory falls back to :class:`~indra.storage.vector_memory.MemoryVectorStore` (D1).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Sequence

from indra.core.config import Settings
from indra.core.exceptions import VectorStoreError
from indra.core.logging import get_logger
from indra.core.models import Chunk

logger = get_logger(__name__)


def chroma_available() -> bool:
    """True when the ``chromadb`` package can be imported."""
    try:
        import chromadb  # noqa: F401
    except Exception:  # noqa: BLE001 - any import failure means unavailable
        return False
    return True


class ChromaVectorStore:
    """Implements ``contracts.VectorStore`` over a persistent Chroma collection."""

    name = "vectors:chroma"
    backend = "chroma"

    def __init__(self, collection: Any) -> None:
        self._collection = collection

    @classmethod
    async def create(cls, settings: Settings) -> ChromaVectorStore:
        """Open (or create) the configured collection. Raises if chromadb is unusable."""

        def _open() -> Any:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            if settings.chroma_host:
                client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
            else:
                settings.chroma_dir.mkdir(parents=True, exist_ok=True)
                client = chromadb.PersistentClient(
                    path=str(settings.chroma_dir),
                    settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
                )
            return client.get_or_create_collection(
                name=settings.chroma_collection,
                metadata={"hnsw:space": "cosine"},
            )

        try:
            collection = await asyncio.to_thread(_open)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(
                "ChromaDB is installed but could not be opened. Delete .cache/chroma to reset, "
                "or set INDRA_STORAGE_BACKEND=memory to run without it.",
                context={"dir": str(settings.chroma_dir)},
                cause=exc,
            ) from exc
        return cls(collection)

    # ------------------------------------------------------------------ writes

    async def upsert(self, chunks: Sequence[Chunk], *, embeddings: Sequence[Sequence[float]]) -> int:
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                "Chunk and embedding counts differ; refusing to write a misaligned index.",
                context={"chunks": len(chunks), "embeddings": len(embeddings)},
            )
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "document_id": chunk.document_id,
                "index": chunk.index,
                "page": chunk.page if chunk.page is not None else -1,
                "section": chunk.section or "",
                "token_count": chunk.token_count,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "ocr_confidence": chunk.ocr_confidence if chunk.ocr_confidence is not None else -1.0,
                "entity_ids": json.dumps(chunk.entity_ids),
            }
            for chunk in chunks
        ]
        try:
            await asyncio.to_thread(
                self._collection.upsert,
                ids=ids,
                embeddings=[list(map(float, e)) for e in embeddings],
                documents=documents,
                metadatas=metadatas,
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(
                "Failed writing chunks to ChromaDB.", context={"count": len(ids)}, cause=exc
            ) from exc
        return len(ids)

    async def delete_document(self, document_id: str) -> int:
        try:
            existing = await asyncio.to_thread(
                self._collection.get, where={"document_id": document_id}, include=[]
            )
            ids = list(existing.get("ids") or [])
            if ids:
                await asyncio.to_thread(self._collection.delete, ids=ids)
            return len(ids)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(
                "Failed deleting document chunks from ChromaDB.",
                context={"document_id": document_id}, cause=exc,
            ) from exc

    # ------------------------------------------------------------------ reads

    async def search(
        self,
        embedding: Sequence[float],
        *,
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        try:
            result = await asyncio.to_thread(
                self._collection.query,
                query_embeddings=[list(map(float, embedding))],
                n_results=top_k,
                where=filters or None,
                include=["distances"],
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError("ChromaDB query failed.", cause=exc) from exc

        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        # Chroma cosine distance is 1 - cosine_similarity, in [0, 2]. Map to a 0-1 relevance so
        # callers see the same scale as the memory backend.
        return [
            (chunk_id, max(0.0, min(1.0, 1.0 - (float(distance) / 2.0))))
            for chunk_id, distance in zip(ids, distances)
        ]

    async def get_chunks(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        try:
            result = await asyncio.to_thread(
                self._collection.get, ids=list(chunk_ids), include=["documents", "metadatas"]
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError("ChromaDB fetch failed.", cause=exc) from exc

        chunks: list[Chunk] = []
        for chunk_id, text, meta in zip(
            result.get("ids") or [], result.get("documents") or [], result.get("metadatas") or []
        ):
            meta = meta or {}
            page = int(meta.get("page", -1))
            ocr = float(meta.get("ocr_confidence", -1.0))
            try:
                entity_ids = json.loads(str(meta.get("entity_ids") or "[]"))
            except ValueError:
                entity_ids = []
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=str(meta.get("document_id", "")),
                    index=int(meta.get("index", 0)),
                    text=text or "",
                    token_count=int(meta.get("token_count", 0)),
                    page=page if page > 0 else None,
                    section=str(meta.get("section") or "") or None,
                    char_start=int(meta.get("char_start", 0)),
                    char_end=int(meta.get("char_end", 0)),
                    ocr_confidence=ocr if ocr >= 0 else None,
                    entity_ids=entity_ids,
                )
            )
        return chunks

    async def count(self) -> int:
        try:
            return int(await asyncio.to_thread(self._collection.count))
        except Exception:  # noqa: BLE001
            return 0

    async def health(self) -> dict[str, Any]:
        try:
            total = await self.count()
            return {"ok": True, "backend": "chroma", "detail": f"{total} chunks indexed"}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "backend": "chroma", "detail": str(exc)}

    async def close(self) -> None:
        return None


__all__ = ["ChromaVectorStore", "chroma_available"]
