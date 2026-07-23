"""Best-effort cache.

A cache failure is always a miss, never an error. Nothing in INDRA may fail a request because the
cache was unavailable — that would trade a fast path for a fragile one.
"""

from __future__ import annotations

import json
import time
from typing import Any, Final

from indra.core.config import Settings
from indra.core.logging import get_logger

logger = get_logger(__name__)

_MAX_ENTRIES: Final[int] = 4096


class MemoryCache:
    """TTL dictionary with lazy expiry and a bounded size."""

    name = "cache:memory"
    backend = "memory"

    def __init__(self, *, default_ttl_s: int = 900) -> None:
        self._data: dict[str, tuple[float, Any]] = {}
        self._default_ttl = default_ttl_s

    async def get(self, key: str) -> Any | None:
        entry = self._data.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if expires_at and expires_at < time.monotonic():
            self._data.pop(key, None)
            return None
        return value

    async def set(self, key: str, value: Any, *, ttl_s: int | None = None) -> None:
        if len(self._data) >= _MAX_ENTRIES:
            # Evict the soonest-to-expire entry rather than an arbitrary one.
            oldest = min(self._data, key=lambda k: self._data[k][0])
            self._data.pop(oldest, None)
        ttl = self._default_ttl if ttl_s is None else ttl_s
        self._data[key] = (time.monotonic() + ttl if ttl > 0 else 0.0, value)

    async def invalidate(self, prefix: str) -> int:
        doomed = [key for key in self._data if key.startswith(prefix)]
        for key in doomed:
            self._data.pop(key, None)
        return len(doomed)

    async def health(self) -> dict[str, Any]:
        return {"ok": True, "backend": "memory", "detail": f"{len(self._data)} entries"}

    async def close(self) -> None:
        self._data.clear()


class RedisCache:
    """Redis-backed cache. Every operation degrades to a miss on failure."""

    name = "cache:redis"
    backend = "redis"

    def __init__(self, client: Any, *, default_ttl_s: int = 900, prefix: str = "indra:cache:") -> None:
        self._client = client
        self._default_ttl = default_ttl_s
        self._prefix = prefix

    async def get(self, key: str) -> Any | None:
        try:
            raw = await self._client.get(f"{self._prefix}{key}")
        except Exception as exc:  # noqa: BLE001 - a cache miss is always safe
            logger.warning("cache read failed; treating as miss", extra={"error": str(exc)})
            return None
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return None

    async def set(self, key: str, value: Any, *, ttl_s: int | None = None) -> None:
        try:
            payload = json.dumps(value, default=str)
        except (TypeError, ValueError):
            logger.debug("value not JSON-serialisable; skipping cache", extra={"key": key})
            return
        try:
            await self._client.set(f"{self._prefix}{key}", payload, ex=ttl_s or self._default_ttl)
        except Exception as exc:  # noqa: BLE001
            logger.warning("cache write failed; continuing", extra={"error": str(exc)})

    async def invalidate(self, prefix: str) -> int:
        removed = 0
        try:
            async for key in self._client.scan_iter(match=f"{self._prefix}{prefix}*"):
                await self._client.delete(key)
                removed += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("cache invalidation failed", extra={"error": str(exc)})
        return removed

    async def health(self) -> dict[str, Any]:
        try:
            await self._client.ping()
            return {"ok": True, "backend": "redis", "detail": "connected"}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "backend": "redis", "detail": str(exc)}

    async def close(self) -> None:
        try:
            await self._client.aclose()
        except Exception:  # noqa: BLE001 - shutdown is best effort
            logger.debug("redis cache close failed")


def build_memory_cache(settings: Settings) -> MemoryCache:
    return MemoryCache(default_ttl_s=settings.cache_ttl_s)


__all__ = ["MemoryCache", "RedisCache", "build_memory_cache"]
