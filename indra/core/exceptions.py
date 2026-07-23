"""Typed exception hierarchy for INDRA.

Every external boundary — network, database, LLM, filesystem, subprocess — raises one of these
with an actionable message. Nothing in this codebase raises a bare ``Exception``.

The ``status_code`` attribute lets the API layer map any error to an HTTP response without a
translation table, and ``retryable`` drives the backoff decision in :mod:`indra.llm.router`.
"""

from __future__ import annotations

from typing import Any


class IndraError(Exception):
    """Base class for every error INDRA raises deliberately.

    Args:
        message: Human-actionable description. Say what failed *and* what to do about it.
        context: Structured detail attached to log records and (in debug mode) API responses.
        cause: The originating exception, if this wraps one.
    """

    status_code: int = 500
    retryable: bool = False
    error_code: str = "indra_error"

    def __init__(
        self,
        message: str,
        *,
        context: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = context or {}
        self.cause = cause
        if cause is not None:
            self.__cause__ = cause

    def to_dict(self) -> dict[str, Any]:
        """Serialise for API responses and structured logs."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
        }

    def __str__(self) -> str:  # pragma: no cover - trivial
        if not self.context:
            return self.message
        detail = ", ".join(f"{k}={v!r}" for k, v in sorted(self.context.items()))
        return f"{self.message} ({detail})"


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


class ConfigurationError(IndraError):
    """Settings are missing, malformed, or mutually inconsistent."""

    status_code = 500
    error_code = "configuration_error"


# --------------------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------------------


class IngestionError(IndraError):
    """Base for anything that goes wrong turning a file into knowledge."""

    status_code = 422
    error_code = "ingestion_error"


class UnsupportedFileTypeError(IngestionError):
    """The uploaded file is not a format any registered parser claims."""

    status_code = 415
    error_code = "unsupported_file_type"


class FileValidationError(IngestionError):
    """The file failed magic-number, size, or integrity validation."""

    status_code = 400
    error_code = "file_validation_error"


class ParsingError(IngestionError):
    """A parser was selected but could not extract usable content."""

    error_code = "parsing_error"


class OCRError(IngestionError):
    """Optical character recognition failed or produced no legible text."""

    error_code = "ocr_error"


class VisionError(IngestionError):
    """P&ID / engineering-drawing vision pipeline failure."""

    error_code = "vision_error"


class EmbeddingError(IngestionError):
    """Embedding generation failed for every configured provider."""

    status_code = 503
    retryable = True
    error_code = "embedding_error"


# --------------------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------------------


class StorageError(IndraError):
    """Base for persistence failures."""

    status_code = 503
    retryable = True
    error_code = "storage_error"


class GraphStoreError(StorageError):
    """Neo4j / graph backend failure."""

    error_code = "graph_store_error"


class VectorStoreError(StorageError):
    """ChromaDB / vector backend failure."""

    error_code = "vector_store_error"


class BlobStoreError(StorageError):
    """Raw-file storage failure."""

    error_code = "blob_store_error"


class MetadataStoreError(StorageError):
    """Relational metadata store failure."""

    error_code = "metadata_store_error"


class EventBusError(StorageError):
    """Inter-agent event bus failure. Never fatal to a request — log and continue."""

    error_code = "event_bus_error"


# --------------------------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------------------------


class LLMError(IndraError):
    """Base for language-model failures."""

    status_code = 503
    retryable = True
    error_code = "llm_error"


class ProviderUnavailableError(LLMError):
    """A provider is unreachable, unauthenticated, or disabled by configuration."""

    error_code = "provider_unavailable"


class RateLimitError(LLMError):
    """Provider rate limit or daily quota exhausted. Router should fail over, not retry in place."""

    status_code = 429
    error_code = "rate_limit_error"


class AllProvidersFailedError(LLMError):
    """Every provider in the fallback chain failed. This is the only truly fatal LLM error."""

    error_code = "all_providers_failed"


class ResponseParsingError(LLMError):
    """The model returned output that does not satisfy the requested schema."""

    status_code = 502
    retryable = True
    error_code = "response_parsing_error"


# --------------------------------------------------------------------------------------
# Retrieval & reasoning
# --------------------------------------------------------------------------------------


class RetrievalError(IndraError):
    """GraphRAG retrieval failed to assemble a context window."""

    status_code = 503
    error_code = "retrieval_error"


class NoEvidenceError(IndraError):
    """Retrieval succeeded but found nothing relevant.

    This is a *correct* outcome, not a bug: INDRA says "I don't know" rather than hallucinating.
    """

    status_code = 200
    error_code = "no_evidence"


class EntityNotFoundError(IndraError):
    """A referenced entity (equipment tag, person, document) is not in the graph."""

    status_code = 404
    error_code = "entity_not_found"


# --------------------------------------------------------------------------------------
# Agent plumbing
# --------------------------------------------------------------------------------------


class AgentError(IndraError):
    """An agent failed to service a request."""

    error_code = "agent_error"


class AgentUnavailableError(AgentError):
    """The agent is not registered or failed its readiness check."""

    status_code = 503
    retryable = True
    error_code = "agent_unavailable"


class AgentTimeoutError(AgentError):
    """The agent exceeded its deadline."""

    status_code = 504
    retryable = True
    error_code = "agent_timeout"


# --------------------------------------------------------------------------------------
# Mobile / compliance
# --------------------------------------------------------------------------------------


class SpeechError(IndraError):
    """Speech-to-text or text-to-speech failure."""

    status_code = 503
    error_code = "speech_error"


class TranslationError(IndraError):
    """Language detection or translation failure."""

    status_code = 503
    error_code = "translation_error"


class OfflineSyncError(IndraError):
    """Offline bundle build or sync-queue replay failure."""

    error_code = "offline_sync_error"


class ComplianceError(IndraError):
    """Regulation parsing, gap detection, or audit-package generation failure."""

    error_code = "compliance_error"


# --------------------------------------------------------------------------------------
# Security
# --------------------------------------------------------------------------------------


class AuthenticationError(IndraError):
    """Missing or invalid API key."""

    status_code = 401
    error_code = "authentication_error"


class AuthorizationError(IndraError):
    """Authenticated, but not permitted."""

    status_code = 403
    error_code = "authorization_error"


class RateLimitExceededError(IndraError):
    """Client exceeded the configured request budget."""

    status_code = 429
    error_code = "rate_limit_exceeded"


__all__ = [
    "AgentError",
    "AgentTimeoutError",
    "AgentUnavailableError",
    "AllProvidersFailedError",
    "AuthenticationError",
    "AuthorizationError",
    "BlobStoreError",
    "ComplianceError",
    "ConfigurationError",
    "EmbeddingError",
    "EntityNotFoundError",
    "EventBusError",
    "FileValidationError",
    "GraphStoreError",
    "IndraError",
    "IngestionError",
    "LLMError",
    "MetadataStoreError",
    "NoEvidenceError",
    "OCRError",
    "OfflineSyncError",
    "ParsingError",
    "ProviderUnavailableError",
    "RateLimitError",
    "RateLimitExceededError",
    "ResponseParsingError",
    "RetrievalError",
    "SpeechError",
    "StorageError",
    "TranslationError",
    "UnsupportedFileTypeError",
    "VectorStoreError",
    "VisionError",
]
