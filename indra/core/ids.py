"""Identifier generation and correlation-id propagation.

Two kinds of identity live here:

* **Content-addressed ids** (:func:`content_id`) — stable across runs, machines, and re-uploads.
  These give us idempotent ingestion (see ``docs/DECISIONS.md`` D6): the same bytes always produce
  the same ``document_id``, so a demo rehearsal never duplicates the knowledge graph.
* **Correlation ids** (:func:`get_correlation_id`) — one per inbound request, carried across every
  agent hop through a :class:`contextvars.ContextVar` so that a single ``grep`` reconstructs the
  full path of a query through all six agents.
"""

from __future__ import annotations

import hashlib
import uuid
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Final, Iterator

_CORRELATION_ID: ContextVar[str | None] = ContextVar("indra_correlation_id", default=None)
_AGENT_NAME: ContextVar[str | None] = ContextVar("indra_agent_name", default=None)

_ID_PREFIXES: Final[dict[str, str]] = {
    "document": "doc",
    "chunk": "chk",
    "entity": "ent",
    "relationship": "rel",
    "query": "qry",
    "answer": "ans",
    "alert": "alt",
    "signal": "sig",
    "audit": "adt",
    "job": "job",
    "session": "ses",
    "event": "evt",
}


def new_id(kind: str = "job") -> str:
    """Return a fresh prefixed identifier, e.g. ``qry_9f2c1a...``.

    Prefixes make ids self-describing in logs and Neo4j browser output.
    """
    prefix = _ID_PREFIXES.get(kind, kind[:3].lower() or "obj")
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def content_id(payload: bytes | str, *, kind: str = "document") -> str:
    """Return a deterministic id derived from content.

    The same bytes always yield the same id, which is what makes ingestion idempotent.
    """
    data = payload.encode("utf-8") if isinstance(payload, str) else payload
    digest = hashlib.sha256(data).hexdigest()
    prefix = _ID_PREFIXES.get(kind, kind[:3].lower() or "obj")
    return f"{prefix}_{digest[:24]}"


def content_hash(payload: bytes | str) -> str:
    """Return the full SHA-256 hex digest of ``payload``."""
    data = payload.encode("utf-8") if isinstance(payload, str) else payload
    return hashlib.sha256(data).hexdigest()


def chunk_id(document_id: str, index: int) -> str:
    """Deterministic chunk id, stable for a given document and position."""
    return f"chk_{document_id.split('_', 1)[-1][:16]}_{index:04d}"


# --------------------------------------------------------------------------------------
# Correlation context
# --------------------------------------------------------------------------------------


def get_correlation_id() -> str:
    """Return the current correlation id, creating one if this is an untracked context."""
    current = _CORRELATION_ID.get()
    if current is None:
        current = new_id("job")
        _CORRELATION_ID.set(current)
    return current


def set_correlation_id(correlation_id: str) -> Token[str | None]:
    """Bind ``correlation_id`` to the current context. Returns a token for restoration."""
    return _CORRELATION_ID.set(correlation_id)


def reset_correlation_id(token: Token[str | None]) -> None:
    """Restore the correlation id captured by :func:`set_correlation_id`."""
    _CORRELATION_ID.reset(token)


def get_agent_name() -> str | None:
    """Return the agent currently handling work in this context, if any."""
    return _AGENT_NAME.get()


@contextmanager
def correlation_context(
    correlation_id: str | None = None,
    *,
    agent: str | None = None,
) -> Iterator[str]:
    """Scope a correlation id (and optionally an agent name) to a block of work.

    Example:
        >>> with correlation_context(agent="copilot_agent") as cid:
        ...     logger.info("handling query", extra={"query_id": cid})
    """
    cid = correlation_id or get_correlation_id()
    cid_token = _CORRELATION_ID.set(cid)
    agent_token = _AGENT_NAME.set(agent) if agent is not None else None
    try:
        yield cid
    finally:
        if agent_token is not None:
            _AGENT_NAME.reset(agent_token)
        _CORRELATION_ID.reset(cid_token)


__all__ = [
    "chunk_id",
    "content_hash",
    "content_id",
    "correlation_context",
    "get_agent_name",
    "get_correlation_id",
    "new_id",
    "reset_correlation_id",
    "set_correlation_id",
]
