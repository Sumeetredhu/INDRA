"""Inter-agent event transport (``docs/DECISIONS.md`` D8).

Redis Streams when Redis is up, in-process asyncio pub/sub otherwise. Both satisfy
``contracts.EventBus``.

**Publishing never raises into the caller.** A dead bus costs observability and proactive
follow-up; it must not cost the user their answer. Handler exceptions are caught and logged for
the same reason — one badly-behaved subscriber cannot take down ingestion.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from typing import Any, Awaitable, Callable, Final

from indra.core.config import Settings
from indra.core.ids import get_correlation_id
from indra.core.logging import get_logger

logger = get_logger(__name__)

Handler = Callable[[dict[str, Any]], Awaitable[None]]

_MAX_HISTORY: Final[int] = 500


class MemoryEventBus:
    """In-process pub/sub. Handlers run as background tasks so publishers never block."""

    name = "events:memory"
    backend = "memory"

    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = defaultdict(list)
        self._tasks: set[asyncio.Task[None]] = set()
        self._history: list[dict[str, Any]] = []
        self._running = False

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def publish(self, topic: str, payload: dict[str, Any]) -> None:
        record = {"topic": topic, "correlation_id": get_correlation_id(), **payload}
        self._history.append(record)
        if len(self._history) > _MAX_HISTORY:
            del self._history[: len(self._history) - _MAX_HISTORY]

        handlers = list(self._handlers.get(topic, ())) + list(self._handlers.get("*", ()))
        if not handlers:
            return
        for handler in handlers:
            task = asyncio.create_task(self._invoke(handler, topic, record))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    @staticmethod
    async def _invoke(handler: Handler, topic: str, record: dict[str, Any]) -> None:
        try:
            await handler(record)
        except asyncio.CancelledError:  # pragma: no cover - shutdown path
            raise
        except Exception as exc:  # noqa: BLE001 - one bad subscriber must not break the bus
            logger.warning(
                "event handler failed",
                extra={"topic": topic, "handler": getattr(handler, "__qualname__", str(handler)),
                       "error": f"{type(exc).__name__}: {exc}"},
            )

    async def subscribe(self, topic: str, handler: Handler) -> None:
        self._handlers[topic].append(handler)
        logger.debug("subscribed", extra={"topic": topic})

    async def drain(self) -> None:
        """Await in-flight handlers. Used by tests so assertions do not race the bus."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    def history(self, topic: str | None = None) -> list[dict[str, Any]]:
        if topic is None:
            return list(self._history)
        return [record for record in self._history if record["topic"] == topic]

    async def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "backend": "memory",
            "detail": f"{sum(len(v) for v in self._handlers.values())} subscribers, "
                      f"{len(self._history)} events seen",
        }


class RedisStreamEventBus:
    """Redis Streams transport. One stream per topic, one consumer task per subscription."""

    name = "events:redis"
    backend = "redis"

    def __init__(self, client: Any, *, prefix: str = "indra:events") -> None:
        self._client = client
        self._prefix = prefix
        self._consumers: list[asyncio.Task[None]] = []
        self._running = False

    def _stream(self, topic: str) -> str:
        return f"{self._prefix}:{topic}"

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False
        for task in self._consumers:
            task.cancel()
        if self._consumers:
            await asyncio.gather(*self._consumers, return_exceptions=True)
        self._consumers.clear()
        try:
            await self._client.aclose()
        except Exception:  # noqa: BLE001 - shutdown is best effort
            logger.debug("redis event bus close failed")

    async def publish(self, topic: str, payload: dict[str, Any]) -> None:
        record = {"topic": topic, "correlation_id": get_correlation_id(), **payload}
        try:
            await self._client.xadd(
                self._stream(topic),
                {"data": json.dumps(record, default=str)},
                maxlen=10_000,
                approximate=True,
            )
        except Exception as exc:  # noqa: BLE001 - never fail a request over the bus
            logger.warning(
                "event publish failed; continuing without it",
                extra={"topic": topic, "error": f"{type(exc).__name__}: {exc}"},
            )

    async def subscribe(self, topic: str, handler: Handler) -> None:
        task = asyncio.create_task(self._consume(topic, handler))
        self._consumers.append(task)

    async def _consume(self, topic: str, handler: Handler) -> None:
        stream = self._stream(topic)
        last_id = "$"
        while True:
            try:
                response = await self._client.xread({stream: last_id}, count=16, block=2000)
            except asyncio.CancelledError:  # pragma: no cover - shutdown path
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning("event stream read failed; backing off",
                               extra={"topic": topic, "error": str(exc)})
                await asyncio.sleep(2.0)
                continue
            if not response:
                continue
            for _stream_name, entries in response:
                for entry_id, fields in entries:
                    last_id = entry_id
                    raw = fields.get("data") or fields.get(b"data")
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8", errors="replace")
                    try:
                        record = json.loads(raw) if raw else {}
                    except ValueError:
                        continue
                    try:
                        await handler(record)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("event handler failed",
                                       extra={"topic": topic, "error": str(exc)})

    async def health(self) -> dict[str, Any]:
        try:
            await self._client.ping()
            return {"ok": True, "backend": "redis", "detail": f"{len(self._consumers)} consumers"}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "backend": "redis", "detail": str(exc)}


def build_memory_bus(settings: Settings) -> MemoryEventBus:
    return MemoryEventBus()


__all__ = ["MemoryEventBus", "RedisStreamEventBus", "build_memory_bus"]
