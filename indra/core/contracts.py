"""Protocol interfaces every INDRA component is written against.

This module is the seam of the whole system. Agents depend on these ``Protocol`` classes, never on
concrete implementations, which is what lets each store have both a real backend and an in-process
fallback (``docs/DECISIONS.md`` D1) and lets the LLM chain fail over without any caller noticing.

**Rule: an agent may import from ``indra.core`` and from these protocols. It may never import
another agent's package.** Cross-agent work goes through the orchestrator or the event bus.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Literal, Protocol, Sequence, runtime_checkable

from indra.core.models import (
    Alert,
    Answer,
    AuditPackage,
    Chunk,
    ComplianceGap,
    CompoundSignal,
    ConditionReading,
    DocumentMeta,
    Equipment,
    ExtractedEntity,
    ExtractedRelationship,
    FailureEvent,
    FailurePrediction,
    GraphPath,
    IngestionProgress,
    IngestionResult,
    KnowledgeCliffScore,
    MaintenanceRecord,
    MimeFamily,
    ParsedDocument,
    Person,
    Procedure,
    QueryRequest,
    RetrievalResult,
)

# ======================================================================================
# Health
# ======================================================================================


class HealthStatus(Protocol):
    """Minimal readiness surface every component exposes to ``/health``."""

    name: str

    async def health(self) -> dict[str, Any]:
        """Return ``{"ok": bool, "backend": str, "detail": str}``. Must never raise."""
        ...


# ======================================================================================
# LLM layer
# ======================================================================================


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into vectors.

    Implementations: Gemini (primary), local sentence-transformers, deterministic hash (offline
    and test). All must return vectors of identical dimension within one process.
    """

    name: str
    dimensions: int

    async def embed(self, texts: Sequence[str], *, task: Literal["document", "query"] = "document") -> list[list[float]]:
        """Embed a batch. Order-preserving. Raises ``EmbeddingError`` only if unrecoverable."""
        ...

    async def is_available(self) -> bool:
        """Cheap liveness probe used by the router to skip dead providers."""
        ...


@runtime_checkable
class ChatProvider(Protocol):
    """Generates text.

    Implementations: Gemini 2.5 Flash, Groq, Ollama (offline), Anthropic (optional), Stub
    (deterministic, used in tests and demo-safe mode).
    """

    name: str
    supports_json: bool

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
    ) -> str:
        """Return completion text. Raises ``RateLimitError`` / ``ProviderUnavailableError``."""
        ...

    async def generate_json(
        self,
        prompt: str,
        *,
        schema: dict[str, Any],
        system: str | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Return a dict validated against ``schema``. Raises ``ResponseParsingError`` on mismatch."""
        ...

    async def stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Yield incremental text. Providers without streaming yield one chunk."""
        ...

    async def is_available(self) -> bool:
        ...


class LLMRouter(Protocol):
    """Ordered fallback across chat providers, with budget accounting.

    Owns the ``gemini → groq → ollama → stub`` chain, per-provider daily budgets, retry policy, and
    the decision to fail over rather than retry in place on a rate limit.
    """

    async def generate(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> tuple[str, str]:
        """Return ``(text, provider_name)`` so the answer can report which model produced it."""
        ...

    async def generate_json(self, prompt: str, *, schema: dict[str, Any], **kwargs: Any) -> tuple[dict[str, Any], str]:
        ...

    async def embed(self, texts: Sequence[str], *, task: Literal["document", "query"] = "document") -> list[list[float]]:
        ...

    def usage(self) -> dict[str, int]:
        """Calls made per provider this process, for ``/metrics`` and budget guards."""
        ...


# ======================================================================================
# Storage layer
# ======================================================================================


class BlobStore(Protocol):
    """Raw file bytes, addressed by content hash."""

    async def put(self, content: bytes, *, filename: str, content_hash: str) -> str:
        """Store bytes, return a storage URI. Idempotent on ``content_hash``."""
        ...

    async def get(self, uri: str) -> bytes:
        ...

    async def exists(self, content_hash: str) -> str | None:
        """Return the URI if these bytes are already stored, else ``None``. Powers D6 idempotency."""
        ...

    async def path_for(self, uri: str) -> Path:
        """Local filesystem path, materialising the blob if the backend is remote."""
        ...


class VectorStore(Protocol):
    """Dense retrieval over chunks."""

    async def upsert(self, chunks: Sequence[Chunk], *, embeddings: Sequence[Sequence[float]]) -> int:
        """Insert or replace. Returns count written. Re-upserting the same ``chunk_id`` overwrites."""
        ...

    async def search(
        self,
        embedding: Sequence[float],
        *,
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        """Return ``[(chunk_id, similarity)]`` sorted descending. Similarity is 0–1, not distance."""
        ...

    async def get_chunks(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        ...

    async def delete_document(self, document_id: str) -> int:
        ...

    async def count(self) -> int:
        ...


class GraphStore(Protocol):
    """The knowledge graph. Neo4j in Docker, an in-process graph otherwise.

    Every method here has a pure-Python implementation, so GraphRAG works with no database at all.
    """

    # -- schema -------------------------------------------------------------------
    async def ensure_schema(self) -> None:
        """Create constraints and indexes. Idempotent; safe on every startup."""
        ...

    # -- writes -------------------------------------------------------------------
    async def upsert_document(self, meta: DocumentMeta) -> None:
        ...

    async def upsert_entities(self, entities: Sequence[ExtractedEntity]) -> int:
        """Merge on ``ExtractedEntity.key``. Returns nodes written."""
        ...

    async def upsert_relationships(self, relationships: Sequence[ExtractedRelationship]) -> int:
        ...

    async def upsert_equipment(self, equipment: Sequence[Equipment]) -> int:
        ...

    async def upsert_people(self, people: Sequence[Person]) -> int:
        ...

    async def upsert_maintenance(self, records: Sequence[MaintenanceRecord]) -> int:
        ...

    async def upsert_failures(self, events: Sequence[FailureEvent]) -> int:
        ...

    async def upsert_procedures(self, procedures: Sequence[Procedure]) -> int:
        ...

    async def upsert_readings(self, readings: Sequence[ConditionReading]) -> int:
        ...

    async def delete_document(self, document_id: str) -> None:
        ...

    # -- reads --------------------------------------------------------------------
    async def get_equipment(self, tag: str) -> Equipment | None:
        ...

    async def list_equipment(self, *, criticality: str | None = None) -> list[Equipment]:
        ...

    async def get_people(self, *, retiring_within_days: int | None = None) -> list[Person]:
        ...

    async def neighbours(
        self,
        entity_key: str,
        *,
        hops: int = 1,
        relation_types: Sequence[str] | None = None,
        limit: int = 50,
    ) -> list[GraphPath]:
        """Traverse outward. ``hops=1`` direct, ``2`` indirect, ``3`` temporal chains."""
        ...

    async def chunks_for_entities(self, entity_keys: Sequence[str], *, limit: int = 50) -> list[tuple[str, float]]:
        """Return ``[(chunk_id, graph_relevance)]`` for chunks mentioning these entities."""
        ...

    async def maintenance_history(self, tag: str, *, since: date | None = None) -> list[MaintenanceRecord]:
        ...

    async def failure_history(self, tag: str, *, since: date | None = None) -> list[FailureEvent]:
        ...

    async def procedures_for(self, tag: str) -> list[Procedure]:
        ...

    async def readings_for(self, tag: str, *, parameter: str | None = None,
                           since: date | None = None) -> list[ConditionReading]:
        ...

    async def documents_for_tag(self, tag: str, *, limit: int = 50) -> list[DocumentMeta]:
        """Every document mentioning this asset. Powers the knowledge-cliff denominator."""
        ...

    async def centrality(self, entity_keys: Sequence[str]) -> dict[str, float]:
        """Normalised 0–1 centrality per key, used as a graph-boost factor."""
        ...

    async def document_meta(self, document_ids: Sequence[str]) -> dict[str, DocumentMeta]:
        ...

    async def query(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Escape hatch for read-only Cypher. Rejected when the backend is in-memory."""
        ...

    async def stats(self) -> dict[str, int]:
        """Node and relationship counts by label, for the graph visualisation header."""
        ...


class MetadataStore(Protocol):
    """Relational metadata: jobs, alerts, sync queue, audit records."""

    async def init(self) -> None:
        ...

    async def save_document(self, meta: DocumentMeta) -> None:
        ...

    async def get_document(self, document_id: str) -> DocumentMeta | None:
        ...

    async def find_by_hash(self, content_hash: str) -> DocumentMeta | None:
        """Idempotency lookup (D6)."""
        ...

    async def list_documents(self, *, limit: int = 100, offset: int = 0) -> list[DocumentMeta]:
        ...

    async def save_alert(self, alert: Alert) -> None:
        ...

    async def list_alerts(self, *, unresolved_only: bool = True, limit: int = 100) -> list[Alert]:
        ...

    async def find_alert_by_dedupe_key(self, dedupe_key: str, *, within_seconds: int) -> Alert | None:
        ...

    async def enqueue_sync(self, item: dict[str, Any]) -> None:
        ...

    async def drain_sync(self, *, limit: int = 100) -> list[dict[str, Any]]:
        ...


class EventBus(Protocol):
    """Typed inter-agent messaging (D8). Redis Streams in Docker, in-memory otherwise.

    Publishing must never raise into the caller — a dead bus degrades observability, not service.
    """

    async def publish(self, topic: str, payload: dict[str, Any]) -> None:
        ...

    async def subscribe(self, topic: str, handler: Callable[[dict[str, Any]], Awaitable[None]]) -> None:
        ...

    async def start(self) -> None:
        ...

    async def stop(self) -> None:
        ...


class CacheStore(Protocol):
    """Best-effort cache. A miss is always safe; never let a cache failure fail a request."""

    async def get(self, key: str) -> Any | None:
        ...

    async def set(self, key: str, value: Any, *, ttl_s: int | None = None) -> None:
        ...

    async def invalidate(self, prefix: str) -> int:
        ...


# ======================================================================================
# Ingestion layer
# ======================================================================================


class DocumentParser(Protocol):
    """Format-specific extraction. Registered by ``MimeFamily``; the first that claims a file wins."""

    name: str
    families: tuple[MimeFamily, ...]

    def claims(self, *, filename: str, mime_family: MimeFamily, head: bytes) -> bool:
        """Return True if this parser should handle the file. Sniff, do not trust the extension."""
        ...

    async def parse(self, path: Path, meta: DocumentMeta) -> ParsedDocument:
        ...


class SymbolDetector(Protocol):
    """P&ID symbol detection. Rule-based by default, YOLO when weights are configured (D4)."""

    name: Literal["rule_based", "yolo", "template"]

    async def detect(self, image_path: Path) -> list[dict[str, Any]]:
        """Return raw detections: ``{"class", "bbox", "confidence"}``."""
        ...

    async def is_available(self) -> bool:
        ...


class TagNormalizer(Protocol):
    """OCR-error-tolerant plant tag resolution (D5). Never corrects silently."""

    def normalize(self, raw: str, *, registry: Sequence[str] | None = None) -> tuple[str | None, float, list[str]]:
        """Return ``(tag_or_None, confidence, alternatives)``."""
        ...


class EntityExtractor(Protocol):
    """Pulls typed entities out of a passage."""

    name: str

    async def extract(self, chunk: Chunk, *, meta: DocumentMeta) -> list[ExtractedEntity]:
        ...


class RelationshipExtractor(Protocol):
    """Pulls candidate edges out of a passage given its entities."""

    name: str

    async def extract(
        self,
        chunk: Chunk,
        entities: Sequence[ExtractedEntity],
        *,
        meta: DocumentMeta,
    ) -> list[ExtractedRelationship]:
        ...


ProgressCallback = Callable[[IngestionProgress], Awaitable[None]]
"""Called at every :class:`IngestionStage` transition so the UI can render a live pipeline."""


class IngestionService(Protocol):
    """The Ingestion Agent's public surface."""

    async def ingest_bytes(
        self,
        content: bytes,
        *,
        filename: str,
        on_progress: ProgressCallback | None = None,
    ) -> IngestionResult:
        ...

    async def ingest_path(self, path: Path, *, on_progress: ProgressCallback | None = None) -> IngestionResult:
        ...

    async def ingest_directory(self, directory: Path, *, concurrency: int = 4) -> list[IngestionResult]:
        ...


# ======================================================================================
# Knowledge graph / retrieval layer
# ======================================================================================


class KnowledgeGraphService(Protocol):
    """The Knowledge Graph Agent's public surface: writes, linking, and GraphRAG retrieval."""

    async def index(self, parsed: ParsedDocument) -> dict[str, int]:
        """Write a parsed document into graph + vector store. Returns counts written."""
        ...

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        equipment_tag: str | None = None,
        max_hops: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """Hybrid vector + graph retrieval with score fusion (D3)."""
        ...

    async def resolve_entities(self, text: str) -> list[str]:
        """Extract and resolve entity keys mentioned in free text."""
        ...

    async def graph_preview(self, entity_keys: Sequence[str], *, hops: int = 2, limit: int = 60) -> dict[str, Any]:
        """Nodes/edges shaped for React Flow."""
        ...


class QueryHandler(Protocol):
    """One Copilot strategy. The classifier picks exactly one per query."""

    query_type: str

    async def handle(self, request: QueryRequest, *, retrieval: RetrievalResult) -> Answer:
        ...


class CopilotService(Protocol):
    """The Copilot Agent's public surface."""

    async def answer(self, request: QueryRequest) -> Answer:
        ...

    async def classify(self, query: str) -> str:
        ...

    async def stream_answer(self, request: QueryRequest) -> AsyncIterator[str]:
        ...


# ======================================================================================
# Proactive intelligence layer
# ======================================================================================


class ProactiveService(Protocol):
    """The Proactive Intelligence Agent's public surface."""

    async def scan(self, *, equipment_tags: Sequence[str] | None = None) -> list[CompoundSignal]:
        """Evaluate every rule across the fleet. Runs on a schedule and on ingestion events."""
        ...

    async def alerts(self, *, unresolved_only: bool = True) -> list[Alert]:
        ...

    async def predict(self, tag: str, *, horizon_days: int = 30) -> FailurePrediction:
        ...

    async def knowledge_cliff(self, *, tags: Sequence[str] | None = None) -> list[KnowledgeCliffScore]:
        ...

    async def interview_questions(self, tag: str, *, person: Person | None = None) -> list[str]:
        ...


# ======================================================================================
# Mobile layer
# ======================================================================================


class SpeechToText(Protocol):
    async def transcribe(self, audio: bytes, *, language_hint: str | None = None) -> tuple[str, str, float]:
        """Return ``(transcript, detected_language, confidence)``."""
        ...


class TextToSpeech(Protocol):
    async def synthesize(self, text: str, *, language: str = "en") -> tuple[bytes, str]:
        """Return ``(audio_bytes, mime_type)``."""
        ...


class Translator(Protocol):
    async def detect(self, text: str) -> str:
        ...

    async def translate(self, text: str, *, target: str, source: str | None = None) -> str:
        """Must preserve plant tags verbatim — see D11."""
        ...


class MobileService(Protocol):
    """The Mobile Agent's public surface."""

    async def voice_query(self, audio: bytes, *, language_hint: str | None = None,
                          equipment_tag: str | None = None) -> Any:
        ...

    async def photo_query(self, image: bytes) -> Any:
        ...

    async def build_offline_bundle(self, *, budget_bytes: int | None = None) -> Any:
        ...

    async def sync(self, items: Sequence[dict[str, Any]]) -> dict[str, Any]:
        ...


# ======================================================================================
# Compliance layer
# ======================================================================================


class ComplianceService(Protocol):
    """The Compliance Agent's public surface."""

    async def parse_regulation(self, document_id: str) -> list[Any]:
        ...

    async def audit(self, *, tags: Sequence[str] | None = None,
                    regulations: Sequence[str] | None = None) -> list[ComplianceGap]:
        ...

    async def build_package(self, *, tags: Sequence[str], regulations: Sequence[str] | None = None) -> AuditPackage:
        ...

    async def export_pdf(self, package: AuditPackage) -> Path:
        ...


# ======================================================================================
# Orchestrator
# ======================================================================================


class Orchestrator(Protocol):
    """Owns agent lifecycle, dependency wiring, and cross-agent choreography.

    Nothing else constructs an agent. Route through here so that swapping a backend or a provider
    is a single change.
    """

    async def startup(self) -> None:
        ...

    async def shutdown(self) -> None:
        ...

    async def health(self) -> dict[str, Any]:
        ...

    @property
    def ingestion(self) -> IngestionService:
        ...

    @property
    def knowledge_graph(self) -> KnowledgeGraphService:
        ...

    @property
    def copilot(self) -> CopilotService:
        ...

    @property
    def proactive(self) -> ProactiveService:
        ...

    @property
    def mobile(self) -> MobileService:
        ...

    @property
    def compliance(self) -> ComplianceService:
        ...


__all__ = [
    "BlobStore", "CacheStore", "ChatProvider", "ComplianceService", "CopilotService",
    "DocumentParser", "EmbeddingProvider", "EntityExtractor", "EventBus", "GraphStore",
    "HealthStatus", "IngestionService", "KnowledgeGraphService", "LLMRouter", "MetadataStore",
    "MobileService", "Orchestrator", "ProactiveService", "ProgressCallback", "QueryHandler",
    "RelationshipExtractor", "SpeechToText", "SymbolDetector", "TagNormalizer", "TextToSpeech",
    "Translator", "VectorStore",
]
