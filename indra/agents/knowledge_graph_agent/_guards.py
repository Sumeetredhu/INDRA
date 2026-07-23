"""Boundary guards: turn any store/LLM failure into a typed :class:`~indra.core.exceptions.IndraError`.

CLAUDE.md rule 2 says every external call is wrapped and raises a typed error with an actionable
message. This module is that wrapper, in one place, so the agent's business logic reads as business
logic instead of as a nest of ``try``/``except``.

Two shapes:

* :func:`guarded` — re-raise as a typed error. Use when the caller genuinely cannot continue.
* :func:`degraded` — swallow, log, and return a fallback. Use when losing this call costs one
  capability rather than the request (CLAUDE.md rule 6).

``asyncio.CancelledError`` derives from ``BaseException``, so neither helper catches it: a cancelled
request stays cancelled.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator, Awaitable, Callable, TypeVar

from indra.core.exceptions import IndraError
from indra.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


@asynccontextmanager
async def guarded(
    operation: str,
    error: type[IndraError],
    *,
    remedy: str,
    **context: object,
) -> AsyncIterator[None]:
    """Re-raise anything escaping the block as ``error``.

    Args:
        operation: What was being attempted, e.g. ``"vector search"``.
        error: The typed exception class to raise.
        remedy: What an operator should actually do about it. Required — an error message that
            does not say what to do is a log line, not an error.
        **context: Structured detail attached to the raised error.

    Raises:
        error: Wrapping the original exception. Existing :class:`IndraError` instances pass through
            unchanged so a precise error is never blunted into a generic one.
    """
    try:
        yield
    except IndraError:
        raise
    except Exception as exc:  # noqa: BLE001 - this is the boundary; that is the whole point
        raise error(
            f"{operation} failed: {exc.__class__.__name__}: {exc}. {remedy}",
            context={"operation": operation, **context},
            cause=exc,
        ) from exc


async def degraded(
    operation: str,
    call: Callable[[], Awaitable[T]],
    *,
    fallback: T,
    capability: str,
    **context: object,
) -> T:
    """Run ``call``; on any failure log a warning and return ``fallback``.

    This is the D1/rule-6 path: a dead Redis or an exhausted embedding quota degrades exactly one
    capability and says so in the logs. It never propagates into the caller's control flow.

    Args:
        operation: What was attempted, for the log record.
        call: Zero-argument coroutine factory.
        fallback: Value returned when ``call`` fails.
        capability: The single capability lost, named in the warning so the degradation is visible.
        **context: Structured extras for the log record.

    Returns:
        The call's result, or ``fallback``.
    """
    try:
        return await call()
    except Exception as exc:  # noqa: BLE001 - deliberate: degrade, never crash the request
        logger.warning(
            "degraded: %s unavailable, continuing without %s",
            operation,
            capability,
            extra={"operation": operation, "capability": capability, "error": repr(exc), **context},
        )
        return fallback


__all__ = ["degraded", "guarded"]
