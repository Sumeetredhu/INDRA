"""Storage backend selection — the mechanism behind ``docs/DECISIONS.md`` D1.

``AUTO`` (the default) probes each real backend with a short timeout and falls back to the
in-process implementation **per store**, recording what was actually bound. A missing Neo4j does
not cost you ChromaDB.

The recorded choice matters as much as the fallback itself: ``/health`` reports the bound backend,
so the ops panel shows ``graph: memory (fallback)`` rather than a green tick that lies.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Final

from indra.core.config import Settings, StorageBackend
from indra.core.exceptions import StorageError
from indra.core.logging import get_logger
from indra.storage.blobs import FileBlobStore, MemoryBlobStore
from indra.storage.cache import MemoryCache, RedisCache
from indra.storage.event_bus import MemoryEventBus, RedisStreamEventBus
from indra.storage.graph_memory import MemoryGraphStore
from indra.storage.metadata_memory import MemoryMetadataStore
from indra.storage.vector_memory import MemoryVectorStore

logger = get_logger(__name__)

#: How long a backend gets to prove it is alive before we fall back. Deliberately short — a demo
#: cannot spend thirty seconds discovering that a container is not running.
PROBE_TIMEOUT_S: Final[float] = 4.0


@dataclass(slots=True)
class StoreBundle:
    """Everything the orchestrator needs to assemble :class:`~indra.core.deps.AgentDeps`.

    Field names match ``AgentDeps`` exactly, so the orchestrator builds deps by unpacking this.
    """

    graph: Any
    vectors: Any
    metadata: Any
    blobs: Any
    events: Any
    cache: Any
    bound_backends: dict[str, str] = field(default_factory=dict)

    async def close(self) -> None:
        """Close every store that supports it, in reverse dependency order."""
        for name in ("events", "cache", "vectors", "graph", "metadata", "blobs"):
            store = getattr(self, name, None)
            closer = getattr(store, "close", None)
            if closer is None:
                continue
            try:
                await closer()
            except Exception as exc:  # noqa: BLE001 - shutdown must not raise
                logger.warning("store close failed", extra={"store": name, "error": str(exc)})

    async def health(self) -> dict[str, Any]:
        """Per-store readiness for the ``/health`` endpoint."""
        report: dict[str, Any] = {}
        for name in ("graph", "vectors", "metadata", "blobs", "events", "cache"):
            store = getattr(self, name, None)
            probe = getattr(store, "health", None)
            if probe is None:
                report[name] = {"ok": True, "backend": self.bound_backends.get(name, "unknown"),
                                "detail": "no health probe"}
                continue
            try:
                report[name] = await probe()
            except Exception as exc:  # noqa: BLE001 - health must never raise
                report[name] = {"ok": False, "backend": self.bound_backends.get(name, "unknown"),
                                "detail": str(exc)}
        return report


async def _probe(label: str, factory: Any, *, required: bool) -> tuple[Any | None, str | None]:
    """Try to build a real backend. Returns ``(store, error_message)``.

    ``required`` is True under ``EXTERNAL``, where a failure must be loud rather than silent.
    """
    try:
        store = await asyncio.wait_for(factory(), timeout=PROBE_TIMEOUT_S)
        return store, None
    except asyncio.TimeoutError:
        message = f"{label} did not respond within {PROBE_TIMEOUT_S:.0f}s"
    except Exception as exc:  # noqa: BLE001 - every backend raises its own family
        message = f"{label} unavailable: {type(exc).__name__}: {exc}"
    if required:
        raise StorageError(
            f"{message}. INDRA_STORAGE_BACKEND=external requires every backend to be reachable; "
            "use 'auto' to fall back automatically or 'memory' to run fully in-process.",
            context={"backend": label},
        )
    return None, message


async def build_stores(settings: Settings) -> StoreBundle:
    """Construct every store, honouring ``settings.storage_backend``."""
    settings.ensure_directories()

    mode = settings.storage_backend
    force_memory = mode is StorageBackend.MEMORY
    require_external = mode is StorageBackend.EXTERNAL
    bound: dict[str, str] = {}
    notes: list[str] = []

    # ---------------------------------------------------------------- graph
    graph: Any = None
    if not force_memory:
        from indra.storage.graph_neo4j import Neo4jGraphStore

        graph, error = await _probe(
            "neo4j", lambda: Neo4jGraphStore.connect(settings), required=require_external
        )
        if error:
            notes.append(error)
    if graph is None:
        graph = MemoryGraphStore()
    bound["graph"] = getattr(graph, "backend", "memory")

    # ---------------------------------------------------------------- vectors
    vectors: Any = None
    if not force_memory:
        from indra.storage.vector_chroma import ChromaVectorStore, chroma_available

        if chroma_available():
            vectors, error = await _probe(
                "chromadb", lambda: ChromaVectorStore.create(settings), required=require_external
            )
            if error:
                notes.append(error)
        elif require_external:
            raise StorageError(
                "chromadb is not installed but INDRA_STORAGE_BACKEND=external was requested. "
                "`pip install chromadb`, or use 'auto'."
            )
        else:
            notes.append("chromadb not installed; using in-process vector store")
    if vectors is None:
        vectors = MemoryVectorStore()
    bound["vectors"] = getattr(vectors, "backend", "memory")

    # ---------------------------------------------------------------- metadata
    metadata: Any = None
    if not force_memory:
        from indra.storage.metadata_sql import SqlMetadataStore

        async def _sql() -> Any:
            store = SqlMetadataStore.from_settings(settings)
            await store.init()
            return store

        metadata, error = await _probe("metadata-sql", _sql, required=require_external)
        if error:
            notes.append(error)
    if metadata is None:
        metadata = MemoryMetadataStore()
        await metadata.init()
    bound["metadata"] = getattr(metadata, "backend", "memory")

    # ---------------------------------------------------------------- blobs
    if force_memory:
        blobs: Any = MemoryBlobStore()
    else:
        blobs = FileBlobStore(settings.raw_dir)
    bound["blobs"] = getattr(blobs, "backend", "memory")

    # ---------------------------------------------------------------- redis-backed pair
    redis_client: Any = None
    if not force_memory:
        redis_client, error = await _probe("redis", lambda: _redis(settings), required=require_external)
        if error:
            notes.append(error)

    if redis_client is not None:
        events: Any = RedisStreamEventBus(redis_client, prefix=settings.redis_stream_prefix)
        # A second client: the event bus blocks on XREAD, which would stall cache reads.
        cache_client, cache_error = await _probe(
            "redis-cache", lambda: _redis(settings), required=False
        )
        cache: Any = (
            RedisCache(cache_client, default_ttl_s=settings.cache_ttl_s)
            if cache_client is not None
            else MemoryCache(default_ttl_s=settings.cache_ttl_s)
        )
        if cache_error:
            notes.append(cache_error)
    else:
        events = MemoryEventBus()
        cache = MemoryCache(default_ttl_s=settings.cache_ttl_s)
    bound["events"] = getattr(events, "backend", "memory")
    bound["cache"] = getattr(cache, "backend", "memory")

    await events.start()

    fallbacks = [name for name, backend in bound.items() if backend == "memory"]
    if fallbacks and mode is not StorageBackend.MEMORY:
        logger.warning(
            "running with in-process fallbacks",
            extra={"fallback_stores": fallbacks, "reasons": notes[:6]},
        )
    logger.info("storage bound", extra={"backends": bound, "mode": mode.value})

    return StoreBundle(
        graph=graph, vectors=vectors, metadata=metadata,
        blobs=blobs, events=events, cache=cache, bound_backends=bound,
    )


async def _redis(settings: Settings) -> Any:
    """Open and ping a Redis client. Raises if unreachable."""
    import redis.asyncio as redis_asyncio

    client = redis_asyncio.from_url(
        settings.redis_url, decode_responses=True, socket_connect_timeout=PROBE_TIMEOUT_S
    )
    await client.ping()
    return client


__all__ = ["PROBE_TIMEOUT_S", "StoreBundle", "build_stores"]
