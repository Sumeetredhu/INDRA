"""Structured JSON logging with automatic correlation-id injection.

Call :func:`configure_logging` once at process start (the FastAPI lifespan does this), then use
:func:`get_logger` everywhere. Every record automatically carries ``correlation_id`` and ``agent``
pulled from the context set by :mod:`indra.core.ids`, so a single request can be traced across all
six agents without threading a logger through every call signature.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any, Final, Mapping

from indra.core.ids import get_agent_name, get_correlation_id

_CONFIGURED: bool = False

#: Attributes present on every ``LogRecord``; anything else is user-supplied ``extra``.
_RESERVED: Final[frozenset[str]] = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName",
        "levelname", "levelno", "lineno", "message", "module", "msecs", "msg", "name",
        "pathname", "process", "processName", "relativeCreated", "stack_info",
        "taskName", "thread", "threadName",
    }
)


class CorrelationFilter(logging.Filter):
    """Attach the ambient correlation id and agent name to every record."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003 - stdlib signature
        if not hasattr(record, "correlation_id"):
            record.correlation_id = get_correlation_id()
        if not hasattr(record, "agent"):
            record.agent = get_agent_name() or "-"
        return True


class JsonFormatter(logging.Formatter):
    """Render records as single-line JSON, suitable for ``docker logs | jq``."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "agent": getattr(record, "agent", "-"),
            "correlation_id": getattr(record, "correlation_id", "-"),
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and key not in payload:
                payload[key] = _coerce(value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class ConsoleFormatter(logging.Formatter):
    """Compact, colourised, human-readable format for local development."""

    _COLOURS: Final[Mapping[str, str]] = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[1;31m",
    }
    _RESET: Final[str] = "\033[0m"

    def __init__(self, *, colour: bool = True) -> None:
        super().__init__()
        self.colour = colour

    def format(self, record: logging.LogRecord) -> str:
        level = record.levelname
        tint = self._COLOURS.get(level, "") if self.colour else ""
        reset = self._RESET if self.colour and tint else ""
        stamp = time.strftime("%H:%M:%S", time.localtime(record.created))
        agent = getattr(record, "agent", "-")
        cid = str(getattr(record, "correlation_id", "-"))[-8:]
        extras = {
            k: v for k, v in record.__dict__.items()
            if k not in _RESERVED and k not in {"correlation_id", "agent"}
        }
        tail = ""
        if extras:
            tail = "  " + " ".join(f"\033[90m{k}=\033[0m{_coerce(v)}" for k, v in extras.items()) \
                if self.colour else "  " + " ".join(f"{k}={_coerce(v)}" for k, v in extras.items())
        line = f"{stamp} {tint}{level:<8}{reset} [{agent}·{cid}] {record.getMessage()}{tail}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def _utf8_stream(stream: Any) -> Any:
    """Return ``stream`` reconfigured to survive non-ASCII output.

    A Windows console defaults to cp1252, which raises ``UnicodeEncodeError`` on the very
    characters INDRA logs constantly — the ``—CONNECTED_TO→`` arrows in graph path narratives, and
    Hindi/Tamil/Kannada text from the mobile agent. A logging call must never be able to crash a
    request, so force UTF-8 and fall back to replacement characters if even that is refused.
    """
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8", errors="replace")
            return stream
        except (ValueError, OSError):  # pragma: no cover - detached or exotic stream
            pass
    buffer = getattr(stream, "buffer", None)
    if buffer is not None:
        import io

        return io.TextIOWrapper(buffer, encoding="utf-8", errors="replace", line_buffering=True)
    return stream


def _coerce(value: Any) -> Any:
    """Make ``value`` JSON-safe without exploding on exotic objects."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple, set)):
        return [_coerce(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _coerce(v) for k, v in value.items()}
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")
        except Exception:  # pragma: no cover - defensive
            return repr(value)
    return str(value)


def configure_logging(
    level: str = "INFO",
    *,
    json_output: bool = False,
    quiet_loggers: tuple[str, ...] = (
        "httpx", "httpcore", "urllib3", "neo4j", "chromadb", "PIL", "asyncio", "watchfiles",
    ),
) -> None:
    """Install INDRA's logging configuration on the root logger.

    Idempotent — safe to call from tests, workers, and the API lifespan alike.

    Args:
        level: Root log level name.
        json_output: Emit JSON lines instead of the human-readable console format.
        quiet_loggers: Third-party loggers pinned to WARNING to keep output readable.
    """
    global _CONFIGURED

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(_utf8_stream(sys.stdout))
    handler.setFormatter(
        JsonFormatter() if json_output else ConsoleFormatter(colour=sys.stdout.isatty())
    )
    handler.addFilter(CorrelationFilter())
    root.addHandler(handler)
    root.setLevel(level.upper())

    for name in quiet_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger, configuring logging with defaults on first use.

    Always pass ``__name__``::

        logger = get_logger(__name__)
        logger.info("ingested document", extra={"document_id": doc.id, "chunks": len(chunks)})
    """
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)


__all__ = [
    "ConsoleFormatter",
    "CorrelationFilter",
    "JsonFormatter",
    "configure_logging",
    "get_logger",
]
