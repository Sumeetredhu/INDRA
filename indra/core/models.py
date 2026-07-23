"""Domain models shared by every INDRA agent.

This module is the vocabulary of the system. If two agents exchange it, it is defined here — not
in an agent package. Agents may subclass or wrap these, never redefine them.

Design rules encoded below:

* **Nothing asserts without evidence.** Any model carrying a claim also carries ``sources`` and a
  :class:`Confidence`. An answer with an empty source list is a bug, not a low-confidence answer.
* **Uncertainty is structured, not prose.** :class:`UncertaintyFlag` is a first-class object so the
  UI can render it, the API can filter on it, and tests can assert on it.
* **Provenance survives every hop.** ``SourceRef`` carries document id, page, snippet, character
  offsets and the extraction confidence, so "Explain How I Know This" is a projection of data we
  already have rather than a second inference pass.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from indra.core.ids import new_id

# ======================================================================================
# Primitives
# ======================================================================================

Score = Annotated[float, Field(ge=0.0, le=1.0)]
"""A normalised 0–1 score. Used for confidence, relevance, and similarity alike."""


def utcnow() -> datetime:
    """Timezone-aware UTC now. Never use ``datetime.utcnow()`` — it is naive and lies."""
    return datetime.now(timezone.utc)


class IndraModel(BaseModel):
    """Base for every INDRA model: strict, immutable-by-convention, JSON-friendly."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=False,
        ser_json_timedelta="float",
        populate_by_name=True,
    )


# ======================================================================================
# Enumerations
# ======================================================================================


class Severity(str, Enum):
    """Alert and gap severity, ordered by escalation urgency."""

    INFO = "INFO"
    LOW = "LOW"
    WARNING = "WARNING"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @property
    def rank(self) -> int:
        return {"INFO": 0, "LOW": 1, "WARNING": 2, "HIGH": 3, "CRITICAL": 4}[self.value]


class Criticality(str, Enum):
    """Plant criticality class. ``A`` equipment stops production or endangers life when it fails."""

    A = "A"
    B = "B"
    C = "C"


class DocumentType(str, Enum):
    """What a document *is*, which determines how the Copilot weighs it as evidence."""

    OEM_MANUAL = "oem_manual"
    WORK_ORDER = "work_order"
    INSPECTION_REPORT = "inspection_report"
    SHIFT_LOG = "shift_log"
    INCIDENT_REPORT = "incident_report"
    ROOT_CAUSE_ANALYSIS = "root_cause_analysis"
    SOP = "sop"
    PID_DRAWING = "pid_drawing"
    EMAIL = "email"
    REGULATION = "regulation"
    SPREADSHEET = "spreadsheet"
    UNKNOWN = "unknown"


class MimeFamily(str, Enum):
    """Coarse format family, chosen by magic-number sniffing rather than file extension."""

    PDF = "pdf"
    IMAGE = "image"
    SPREADSHEET = "spreadsheet"
    WORD = "word"
    EMAIL = "email"
    TEXT = "text"
    UNKNOWN = "unknown"


class EntityType(str, Enum):
    """Node labels in the knowledge graph."""

    EQUIPMENT = "Equipment"
    DOCUMENT = "Document"
    PERSON = "Person"
    FAILURE_MODE = "FailureMode"
    PROCEDURE = "Procedure"
    REGULATORY_CLAUSE = "RegulatoryClause"
    CONDITION_READING = "ConditionReading"
    MEASUREMENT = "Measurement"
    DATE = "Date"
    LOCATION = "Location"
    MATERIAL = "Material"
    ORGANISATION = "Organisation"


class RelationType(str, Enum):
    """Edge types in the knowledge graph."""

    CONNECTED_TO = "CONNECTED_TO"
    MAINTAINED = "MAINTAINED"
    FAILED_WITH_MODE = "FAILED_WITH_MODE"
    MENTIONS = "MENTIONS"
    HAS_EXPERTISE = "HAS_EXPERTISE"
    REQUIRES = "REQUIRES"
    APPLIES_TO = "APPLIES_TO"
    PRECEDED_BY = "PRECEDED_BY"
    DOCUMENTED_BY = "DOCUMENTED_BY"
    INSPECTED_BY = "INSPECTED_BY"
    PART_OF = "PART_OF"
    SIMILAR_TO = "SIMILAR_TO"
    CAUSED_BY = "CAUSED_BY"
    RESOLVED_BY = "RESOLVED_BY"
    SUPERSEDES = "SUPERSEDES"


class QueryType(str, Enum):
    """How the Copilot routes an incoming question."""

    FACTUAL = "factual"
    DIAGNOSTIC = "diagnostic"
    PROCEDURAL = "procedural"
    PREDICTIVE = "predictive"
    COMPARATIVE = "comparative"
    COMPLIANCE = "compliance"
    KNOWLEDGE_GAP = "knowledge_gap"


class UncertaintySource(str, Enum):
    """Why a piece of evidence is not fully trusted."""

    LOW_OCR_CONFIDENCE = "low_ocr_confidence"
    HANDWRITTEN_SOURCE = "handwritten_source"
    VISION_INFERENCE = "vision_inference"
    STALE_DOCUMENT = "stale_document"
    CONFLICTING_SOURCES = "conflicting_sources"
    SPARSE_EVIDENCE = "sparse_evidence"
    TRANSLATED_CONTENT = "translated_content"
    MODEL_EXTRAPOLATION = "model_extrapolation"
    UNVERIFIED_TAG_MATCH = "unverified_tag_match"


class IngestionStage(str, Enum):
    """Pipeline stage, reported live to the UI during the demo's ingestion moment."""

    RECEIVED = "received"
    VALIDATED = "validated"
    STORED = "stored"
    PARSED = "parsed"
    CHUNKED = "chunked"
    EMBEDDED = "embedded"
    ENTITIES_EXTRACTED = "entities_extracted"
    RELATIONS_EXTRACTED = "relations_extracted"
    GRAPH_QUEUED = "graph_queued"
    GRAPH_WRITTEN = "graph_written"
    COMPLETE = "complete"
    FAILED = "failed"


class GapStatus(str, Enum):
    """Compliance evidence state for a single requirement."""

    COMPLIANT = "compliant"
    MISSING = "missing"
    OUTDATED = "outdated"
    INCOMPLETE = "incomplete"


# ======================================================================================
# Confidence & provenance
# ======================================================================================


class Confidence(IndraModel):
    """A score with a reason attached.

    A bare float tells an operator nothing. ``Confidence(0.72, "OCR on handwritten note")`` tells
    them whether to act or to go and look at the pump.
    """

    value: Score
    rationale: str = Field(min_length=1, description="Why this number and not a different one.")
    method: Literal["exact", "ocr", "vision", "semantic", "llm", "heuristic", "aggregate"] = "heuristic"

    @classmethod
    def exact(cls, rationale: str = "Direct match in structured source") -> Confidence:
        return cls(value=1.0, rationale=rationale, method="exact")

    @classmethod
    def aggregate(cls, parts: list[Confidence], *, rationale: str | None = None) -> Confidence:
        """Combine step confidences into an overall score.

        Uses the **minimum**, not the mean: a reasoning chain is only as trustworthy as its weakest
        link. A 0.95 retrieval step cannot rescue a 0.4 OCR read of the number the answer hinges on.
        """
        if not parts:
            return cls(value=0.0, rationale="No supporting steps", method="aggregate")
        weakest = min(parts, key=lambda c: c.value)
        mean = sum(p.value for p in parts) / len(parts)
        blended = round(0.7 * weakest.value + 0.3 * mean, 4)
        return cls(
            value=blended,
            rationale=rationale or f"Weakest link: {weakest.rationale} ({weakest.value:.2f})",
            method="aggregate",
        )

    @property
    def band(self) -> Literal["high", "medium", "low"]:
        """Coarse band used by the UI to pick a colour and by alerts to decide escalation."""
        if self.value >= 0.8:
            return "high"
        return "medium" if self.value >= 0.55 else "low"


class SourceRef(IndraModel):
    """A pointer to the exact evidence behind a claim.

    Everything needed to render a citation card *and* to let a technician open the original page.
    """

    document_id: str
    document_title: str
    document_type: DocumentType = DocumentType.UNKNOWN
    chunk_id: str | None = None
    page: int | None = Field(default=None, ge=1)
    bbox: tuple[float, float, float, float] | None = Field(
        default=None, description="Normalised (x0, y0, x1, y1) for highlight overlay on the page image."
    )
    snippet: str = Field(default="", max_length=1200)
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)
    relevance: Score = 0.0
    extraction_confidence: Score = 1.0
    retrieved_via: Literal["vector", "graph", "hybrid", "direct", "vision"] = "hybrid"
    document_date: date | None = None

    @property
    def citation(self) -> str:
        """Short human-readable citation, e.g. ``WO_2024_0342 p.3``."""
        return f"{self.document_title}{f' p.{self.page}' if self.page else ''}"


class UncertaintyFlag(IndraModel):
    """A caveat the operator must see before acting.

    The spec's example — *"Bearing wear % from handwritten note, OCR confidence 0.72 — verify with
    supervisor"* — is exactly this model rendered to text.
    """

    source: UncertaintySource
    message: str
    severity: Severity = Severity.WARNING
    affected_claim: str | None = None
    affected_source: SourceRef | None = None
    suggested_action: str | None = "Verify with supervisor before acting"


# ======================================================================================
# Documents, chunks, entities
# ======================================================================================


class DocumentMeta(IndraModel):
    """Everything known about a document before its content is considered."""

    document_id: str
    title: str
    filename: str
    content_hash: str
    mime_family: MimeFamily = MimeFamily.UNKNOWN
    mime_type: str = "application/octet-stream"
    document_type: DocumentType = DocumentType.UNKNOWN
    size_bytes: int = Field(ge=0)
    page_count: int | None = Field(default=None, ge=0)
    language: str = "en"
    document_date: date | None = None
    ingested_at: datetime = Field(default_factory=utcnow)
    source_path: str | None = None
    #: Set when this document replaces an earlier revision (D6).
    supersedes: str | None = None
    tags: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class Chunk(IndraModel):
    """A semantic passage: the atom of retrieval."""

    chunk_id: str
    document_id: str
    index: int = Field(ge=0)
    text: str = Field(min_length=1)
    token_count: int = Field(ge=0)
    page: int | None = Field(default=None, ge=1)
    section: str | None = None
    char_start: int = Field(default=0, ge=0)
    char_end: int = Field(default=0, ge=0)
    embedding: list[float] | None = Field(default=None, repr=False)
    ocr_confidence: Score | None = None
    entity_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_source_ref(self, meta: DocumentMeta, *, relevance: float = 0.0,
                      retrieved_via: str = "hybrid") -> SourceRef:
        """Project this chunk into a citation. Used everywhere an answer cites evidence."""
        return SourceRef(
            document_id=self.document_id,
            document_title=meta.title,
            document_type=meta.document_type,
            chunk_id=self.chunk_id,
            page=self.page,
            snippet=self.text[:600],
            char_start=self.char_start,
            char_end=self.char_end,
            relevance=max(0.0, min(1.0, relevance)),
            extraction_confidence=self.ocr_confidence if self.ocr_confidence is not None else 1.0,
            retrieved_via=retrieved_via,  # type: ignore[arg-type]
            document_date=meta.document_date,
        )


class ExtractedEntity(IndraModel):
    """An entity found in text or in a drawing, before it is resolved against the graph."""

    entity_id: str = Field(default_factory=lambda: new_id("entity"))
    type: EntityType
    name: str = Field(min_length=1, description="Surface form as written in the source.")
    canonical_name: str | None = Field(default=None, description="Registry-resolved form, e.g. P-101.")
    normalized_value: str | float | None = None
    unit: str | None = None
    confidence: Confidence
    document_id: str | None = None
    chunk_id: str | None = None
    page: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    #: Populated by the tag normalizer when an OCR correction was applied (D5).
    alternatives: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)

    @property
    def key(self) -> str:
        """Deduplication key: canonical name if resolved, otherwise the normalised surface form."""
        return f"{self.type.value}:{(self.canonical_name or self.name).strip().upper()}"


class ExtractedRelationship(IndraModel):
    """A candidate edge between two entities."""

    relationship_id: str = Field(default_factory=lambda: new_id("relationship"))
    type: RelationType
    source_key: str = Field(description="``ExtractedEntity.key`` of the head node.")
    target_key: str = Field(description="``ExtractedEntity.key`` of the tail node.")
    confidence: Confidence
    evidence_text: str = ""
    document_id: str | None = None
    chunk_id: str | None = None
    method: Literal["co_occurrence", "syntactic", "rule", "vision", "llm", "temporal"] = "rule"
    properties: dict[str, Any] = Field(default_factory=dict)
    observed_at: datetime | None = None


class ParsedDocument(IndraModel):
    """The Ingestion Agent's output for one file, handed to the Knowledge Graph Agent."""

    meta: DocumentMeta
    text: str = ""
    chunks: list[Chunk] = Field(default_factory=list)
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)
    tables: list[dict[str, Any]] = Field(default_factory=list)
    pid_result: PIDParseResult | None = None
    warnings: list[str] = Field(default_factory=list)
    stage: IngestionStage = IngestionStage.PARSED
    parse_duration_ms: float = 0.0


# ======================================================================================
# P&ID vision
# ======================================================================================


class SymbolClass(str, Enum):
    """Equipment symbol classes the P&ID parser recognises."""

    PUMP = "pump"
    VESSEL = "vessel"
    HEAT_EXCHANGER = "heat_exchanger"
    VALVE = "valve"
    INSTRUMENT = "instrument"
    COMPRESSOR = "compressor"
    TANK = "tank"
    FILTER = "filter"
    UNKNOWN = "unknown"


class DetectedSymbol(IndraModel):
    """One equipment symbol located in a drawing."""

    symbol_id: str = Field(default_factory=lambda: new_id("entity"))
    symbol_class: SymbolClass
    bbox: tuple[int, int, int, int] = Field(description="Pixel (x0, y0, x1, y1).")
    detection_confidence: Score
    ocr_text: str = ""
    tag: str | None = Field(default=None, description="Normalised plant tag, e.g. P-101.")
    tag_confidence: Score = 0.0
    tag_alternatives: list[str] = Field(default_factory=list)
    detector: Literal["rule_based", "yolo", "template"] = "rule_based"

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


class DetectedConnection(IndraModel):
    """A traced pipe run between two symbols."""

    source_symbol_id: str
    target_symbol_id: str
    confidence: Score
    flow_direction: Literal["forward", "reverse", "bidirectional", "unknown"] = "unknown"
    line_type: Literal["process", "instrument", "utility", "unknown"] = "process"
    pipe_spec: str | None = None
    polyline: list[tuple[int, int]] = Field(default_factory=list)


class PIDParseResult(IndraModel):
    """Structured output of the P&ID vision pipeline — INDRA's headline differentiator."""

    document_id: str
    image_width: int = Field(ge=1)
    image_height: int = Field(ge=1)
    symbols: list[DetectedSymbol] = Field(default_factory=list)
    connections: list[DetectedConnection] = Field(default_factory=list)
    detector_used: Literal["rule_based", "yolo", "template"] = "rule_based"
    overall_confidence: Score = 0.0
    warnings: list[str] = Field(default_factory=list)
    processing_ms: float = 0.0

    @property
    def tagged_symbols(self) -> list[DetectedSymbol]:
        return [s for s in self.symbols if s.tag]


# ======================================================================================
# Plant domain
# ======================================================================================


class Equipment(IndraModel):
    """A physical asset. ``tag`` is the plant-wide primary key."""

    tag: str = Field(min_length=2, description="Plant tag, e.g. P-101.")
    name: str = ""
    equipment_type: str = "unknown"
    manufacturer: str | None = None
    model: str | None = None
    criticality: Criticality = Criticality.C
    location: str | None = None
    installed_on: date | None = None
    unit: str | None = None
    specifications: dict[str, Any] = Field(default_factory=dict)
    oem_thresholds: dict[str, float] = Field(
        default_factory=dict,
        description="Named limits from the OEM manual, e.g. {'bearing_wear_pct': 85.0}.",
    )

    @field_validator("tag")
    @classmethod
    def _upper(cls, value: str) -> str:
        return value.strip().upper()


class Person(IndraModel):
    """A plant person — the unit of the knowledge cliff."""

    person_id: str = Field(default_factory=lambda: new_id("entity"))
    name: str
    role: str | None = None
    years_experience: float | None = Field(default=None, ge=0)
    retirement_date: date | None = None
    expertise_tags: list[str] = Field(default_factory=list, description="Equipment tags they know.")
    documented_contributions: int = 0
    contact: str | None = None


class ConditionReading(IndraModel):
    """A measured value with a timestamp — the raw material of trend and threshold logic."""

    equipment_tag: str
    parameter: str = Field(description="e.g. bearing_wear_pct, vibration_mm_s, temperature_c.")
    value: float
    unit: str = ""
    measured_at: datetime
    source: SourceRef | None = None
    confidence: Confidence = Field(default_factory=lambda: Confidence.exact())


class FailureEvent(IndraModel):
    """A historical failure, with the cost that makes the business case concrete."""

    event_id: str = Field(default_factory=lambda: new_id("entity"))
    equipment_tag: str
    failure_mode: str
    occurred_on: date
    root_cause: str | None = None
    downtime_hours: float | None = Field(default=None, ge=0)
    cost_inr: float | None = Field(default=None, ge=0)
    precursor_text: str = Field(default="", description="Symptoms recorded before the failure.")
    sources: list[SourceRef] = Field(default_factory=list)


class MaintenanceRecord(IndraModel):
    """A work order or inspection outcome."""

    record_id: str = Field(default_factory=lambda: new_id("entity"))
    equipment_tag: str
    record_type: Literal["work_order", "inspection", "preventive", "calibration"] = "work_order"
    performed_on: date
    performed_by: str | None = None
    findings: str = ""
    recommendations: str = ""
    readings: list[ConditionReading] = Field(default_factory=list)
    status: Literal["open", "closed", "deferred"] = "closed"
    sources: list[SourceRef] = Field(default_factory=list)


class Procedure(IndraModel):
    """An SOP, decomposed into steps so the Copilot can answer PROCEDURAL queries verbatim."""

    procedure_id: str = Field(default_factory=lambda: new_id("entity"))
    title: str
    applies_to: list[str] = Field(default_factory=list, description="Equipment tags.")
    steps: list[str] = Field(default_factory=list)
    estimated_minutes: int | None = Field(default=None, ge=0)
    required_tools: list[str] = Field(default_factory=list)
    safety_notes: list[str] = Field(default_factory=list)
    revision: str | None = None
    sources: list[SourceRef] = Field(default_factory=list)


# ======================================================================================
# Retrieval & reasoning
# ======================================================================================


class RetrievedPassage(IndraModel):
    """One scored candidate, with its score decomposed so the fusion is auditable (D3)."""

    chunk: Chunk
    document: DocumentMeta
    vector_score: float = 0.0
    graph_score: float = 0.0
    fused_score: float = 0.0
    hops: int = Field(default=0, ge=0, description="Graph distance from a query entity. 0 = direct hit.")
    matched_entities: list[str] = Field(default_factory=list)
    explanation: str = ""

    def as_source(self) -> SourceRef:
        return self.chunk.to_source_ref(
            self.document,
            relevance=max(0.0, min(1.0, self.fused_score)),
            retrieved_via="hybrid" if self.graph_score > 0 and self.vector_score > 0
            else ("graph" if self.graph_score > 0 else "vector"),
        )


class GraphPath(IndraModel):
    """A traversal result: how two entities are connected, and how strongly.

    This is the model behind "finding connections no single document contains".
    """

    nodes: list[str] = Field(description="Entity keys along the path.")
    relations: list[RelationType] = Field(default_factory=list)
    hops: int = Field(ge=0)
    confidence: Score = 1.0
    narrative: str = Field(default="", description="Human-readable rendering of the path.")


class RetrievalResult(IndraModel):
    """Everything GraphRAG found for one query."""

    query: str
    query_entities: list[str] = Field(default_factory=list)
    passages: list[RetrievedPassage] = Field(default_factory=list)
    paths: list[GraphPath] = Field(default_factory=list)
    strategy: Literal["weighted", "rrf"] = "weighted"
    total_candidates: int = 0
    retrieval_ms: float = 0.0

    @property
    def is_empty(self) -> bool:
        return not self.passages and not self.paths


class ReasoningStep(IndraModel):
    """One link in the chain the "Explain How I Know This" panel renders."""

    order: int = Field(ge=1)
    action: str = Field(description="What INDRA did, e.g. 'Retrieved maintenance history'.")
    finding: str = Field(description="What it learned.")
    confidence: Confidence
    sources: list[SourceRef] = Field(default_factory=list)
    graph_paths: list[GraphPath] = Field(default_factory=list)
    cypher: str | None = Field(default=None, description="Shown to technical users on request.")
    duration_ms: float = 0.0


class RecommendedAction(IndraModel):
    """A concrete next step. Vague advice is not actionable on a plant floor."""

    action: str
    urgency: Severity = Severity.WARNING
    owner_role: str | None = None
    due_within_days: int | None = Field(default=None, ge=0)
    rationale: str = ""
    procedure_id: str | None = None
    estimated_minutes: int | None = None


class Answer(IndraModel):
    """The Copilot's response. Every field here exists to make the answer trustworthy."""

    answer_id: str = Field(default_factory=lambda: new_id("answer"))
    query: str
    query_type: QueryType
    answer_text: str
    language: str = "en"
    confidence: Confidence
    reasoning_chain: list[ReasoningStep] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)
    uncertainty_flags: list[UncertaintyFlag] = Field(default_factory=list)
    alternative_interpretations: list[str] = Field(default_factory=list)
    recommended_actions: list[RecommendedAction] = Field(default_factory=list)
    graph_preview: dict[str, Any] | None = Field(default=None, description="Nodes/edges for React Flow.")
    cypher_queries: list[str] = Field(default_factory=list)
    related_alerts: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=utcnow)
    latency_ms: float = 0.0
    provider_used: str = "stub"

    @model_validator(mode="after")
    def _grounded(self) -> Self:
        """An answer with content but no sources is a hallucination. Force the flag."""
        if self.answer_text.strip() and not self.sources:
            no_source = any(f.source is UncertaintySource.SPARSE_EVIDENCE for f in self.uncertainty_flags)
            if not no_source:
                self.uncertainty_flags.append(
                    UncertaintyFlag(
                        source=UncertaintySource.SPARSE_EVIDENCE,
                        message="No supporting documents were retrieved for this answer.",
                        severity=Severity.HIGH,
                        suggested_action="Treat as unverified; consult plant records directly.",
                    )
                )
        return self


class QueryRequest(IndraModel):
    """An inbound question, from web, mobile, or voice."""

    query: str = Field(min_length=1, max_length=2000)
    language: str = "en"
    equipment_tag: str | None = None
    query_type: QueryType | None = Field(default=None, description="Override the classifier.")
    max_sources: int = Field(default=8, ge=1, le=50)
    include_graph_preview: bool = True
    include_cypher: bool = False
    session_id: str | None = None
    channel: Literal["web", "mobile", "voice", "photo", "offline"] = "web"


# ======================================================================================
# Proactive intelligence
# ======================================================================================


class Signal(IndraModel):
    """One observation that, alone, means little."""

    signal_id: str = Field(default_factory=lambda: new_id("signal"))
    kind: str = Field(description="e.g. maintenance_anomaly, threshold_approach, alarm_bypass.")
    equipment_tag: str
    description: str
    observed_at: datetime = Field(default_factory=utcnow)
    strength: Score = 0.5
    sources: list[SourceRef] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)


class CompoundSignal(IndraModel):
    """Several signals that, together, mean a great deal.

    This is the core of "we predict the failure before anyone asks".
    """

    rule_id: str
    rule_name: str
    equipment_tag: str
    severity: Severity
    signals: list[Signal]
    explanation: str = Field(description="Plain-language account of why these combine.")
    confidence: Confidence
    risk_score: Score = 0.0
    detected_at: datetime = Field(default_factory=utcnow)

    @property
    def signal_count(self) -> int:
        return len(self.signals)


class Alert(IndraModel):
    """A surfaced compound signal, ready for the operator's screen."""

    alert_id: str = Field(default_factory=lambda: new_id("alert"))
    title: str
    equipment_tag: str
    severity: Severity
    body: str
    compound_signal: CompoundSignal | None = None
    recommended_actions: list[RecommendedAction] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)
    risk_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    raised_at: datetime = Field(default_factory=utcnow)
    acknowledged_by: str | None = None
    acknowledged_at: datetime | None = None
    resolved: bool = False
    dedupe_key: str = ""

    @model_validator(mode="after")
    def _dedupe(self) -> Self:
        if not self.dedupe_key:
            rule = self.compound_signal.rule_id if self.compound_signal else "manual"
            self.dedupe_key = f"{self.equipment_tag}:{rule}:{self.severity.value}"
        return self


class KnowledgeCliffScore(IndraModel):
    """How much irreplaceable knowledge walks out of the gate with one retirement."""

    equipment_tag: str
    score: float = Field(ge=0.0, le=100.0)
    severity: Severity
    retiring_experts: list[Person] = Field(default_factory=list)
    document_count: int = 0
    criticality: Criticality = Criticality.C
    days_to_first_retirement: int | None = None
    factors: dict[str, float] = Field(
        default_factory=dict,
        description="Per-factor contribution, so the score is defensible rather than magic.",
    )
    interview_questions: list[str] = Field(default_factory=list)
    rationale: str = ""


class FailurePrediction(IndraModel):
    """A forward-looking risk estimate for one asset."""

    equipment_tag: str
    failure_mode: str
    probability: Score
    horizon_days: int = Field(ge=1)
    drivers: list[str] = Field(default_factory=list)
    similar_historical_events: list[FailureEvent] = Field(default_factory=list)
    confidence: Confidence
    recommended_actions: list[RecommendedAction] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)


# ======================================================================================
# Compliance
# ======================================================================================


class RegulatoryRequirement(IndraModel):
    """One atomic obligation parsed out of a regulation."""

    requirement_id: str = Field(default_factory=lambda: new_id("entity"))
    regulation: str
    clause: str = Field(description="e.g. Section 41(b).")
    text: str
    obligation: str = Field(description="Normalised duty, e.g. 'monthly pressure vessel inspection'.")
    frequency_days: int | None = Field(default=None, ge=1)
    applies_to_types: list[str] = Field(default_factory=list)
    applies_to_tags: list[str] = Field(default_factory=list)
    evidence_types: list[DocumentType] = Field(default_factory=list)
    penalty: str | None = None
    source: SourceRef | None = None


class ComplianceGap(IndraModel):
    """A requirement that is not demonstrably met."""

    gap_id: str = Field(default_factory=lambda: new_id("entity"))
    requirement: RegulatoryRequirement
    equipment_tag: str
    status: GapStatus
    severity: Severity
    detail: str
    last_evidence_date: date | None = None
    days_overdue: int | None = None
    deadline: date | None = None
    penalty_risk: str | None = None
    recommended_action: RecommendedAction | None = None
    evidence: list[SourceRef] = Field(default_factory=list)
    confidence: Confidence = Field(default_factory=lambda: Confidence.exact("Deterministic rule check"))


class ComplianceMatrixRow(IndraModel):
    """One row of the audit matrix: requirement → status → evidence."""

    requirement_id: str
    regulation: str
    clause: str
    obligation: str
    equipment_tag: str
    status: GapStatus
    evidence: list[SourceRef] = Field(default_factory=list)
    note: str = ""


class AuditPackage(IndraModel):
    """The one-click deliverable an inspector can actually take away."""

    package_id: str = Field(default_factory=lambda: new_id("audit"))
    title: str
    scope_tags: list[str]
    regulations: list[str]
    generated_at: datetime = Field(default_factory=utcnow)
    matrix: list[ComplianceMatrixRow] = Field(default_factory=list)
    gaps: list[ComplianceGap] = Field(default_factory=list)
    corrective_actions: list[RecommendedAction] = Field(default_factory=list)
    evidence_documents: list[SourceRef] = Field(default_factory=list)
    pdf_path: str | None = None

    @property
    def compliance_rate(self) -> float:
        if not self.matrix:
            return 0.0
        met = sum(1 for row in self.matrix if row.status is GapStatus.COMPLIANT)
        return round(100.0 * met / len(self.matrix), 1)


# ======================================================================================
# Mobile
# ======================================================================================


class VoiceQueryRequest(IndraModel):
    """Audio in, spoken answer out."""

    audio_base64: str | None = None
    audio_path: str | None = None
    language_hint: str | None = None
    equipment_tag: str | None = None
    session_id: str | None = None
    respond_with_audio: bool = True


class VoiceQueryResponse(IndraModel):
    """The full multilingual round trip, with every intermediate exposed for the demo."""

    transcript: str
    detected_language: str
    translated_query: str | None = None
    answer: Answer
    spoken_text: str
    audio_base64: str | None = None
    audio_mime: str = "audio/mpeg"
    stt_confidence: Score = 0.0


class PhotoQueryResponse(IndraModel):
    """The AR overlay payload: what floats over the camera view."""

    detected_tag: str | None
    tag_confidence: Score
    tag_alternatives: list[str] = Field(default_factory=list)
    equipment: Equipment | None = None
    status_line: str = ""
    last_maintenance: MaintenanceRecord | None = None
    open_alerts: list[Alert] = Field(default_factory=list)
    quick_documents: list[SourceRef] = Field(default_factory=list)
    quick_actions: list[RecommendedAction] = Field(default_factory=list)
    bbox: tuple[int, int, int, int] | None = None


class OfflineBundle(IndraModel):
    """What a technician carries into a dead zone."""

    bundle_id: str = Field(default_factory=lambda: new_id("session"))
    built_at: datetime = Field(default_factory=utcnow)
    equipment_tags: list[str] = Field(default_factory=list)
    size_bytes: int = 0
    budget_bytes: int = 0
    documents: list[DocumentMeta] = Field(default_factory=list)
    alerts: list[Alert] = Field(default_factory=list)
    procedures: list[Procedure] = Field(default_factory=list)
    excluded_tags: list[str] = Field(default_factory=list, description="Dropped to fit the budget.")
    checksum: str = ""


class SyncQueueItem(IndraModel):
    """Work performed offline, replayed when the radio comes back."""

    item_id: str = Field(default_factory=lambda: new_id("job"))
    kind: Literal["query", "observation", "photo", "acknowledgement"]
    payload: dict[str, Any]
    created_at: datetime = Field(default_factory=utcnow)
    synced: bool = False
    attempts: int = 0
    last_error: str | None = None


# ======================================================================================
# Ingestion job reporting
# ======================================================================================


class IngestionProgress(IndraModel):
    """Streamed to the UI so the demo's 0:20 mark shows a real pipeline, not a spinner."""

    job_id: str
    document_id: str | None = None
    filename: str = ""
    stage: IngestionStage
    percent: float = Field(default=0.0, ge=0.0, le=100.0)
    message: str = ""
    entities_found: int = 0
    relationships_found: int = 0
    chunks_created: int = 0
    mean_confidence: float = 0.0
    at: datetime = Field(default_factory=utcnow)


class IngestionResult(IndraModel):
    """Final outcome of ingesting one file."""

    job_id: str
    document: DocumentMeta
    chunks_created: int = 0
    entities_created: int = 0
    relationships_created: int = 0
    pid_symbols: int = 0
    pid_connections: int = 0
    duplicate_of: str | None = Field(default=None, description="Set when idempotency short-circuited (D6).")
    warnings: list[str] = Field(default_factory=list)
    duration_ms: float = 0.0
    stage: IngestionStage = IngestionStage.COMPLETE


# Resolve the forward reference used by ParsedDocument.
ParsedDocument.model_rebuild()


__all__ = [
    "Alert", "Answer", "AuditPackage", "Chunk", "ComplianceGap", "ComplianceMatrixRow",
    "CompoundSignal", "Confidence", "ConditionReading", "Criticality", "DetectedConnection",
    "DetectedSymbol", "DocumentMeta", "DocumentType", "EntityType", "Equipment",
    "ExtractedEntity", "ExtractedRelationship", "FailureEvent", "FailurePrediction",
    "GapStatus", "GraphPath", "IndraModel", "IngestionProgress", "IngestionResult",
    "IngestionStage", "KnowledgeCliffScore", "MaintenanceRecord", "MimeFamily",
    "OfflineBundle", "ParsedDocument", "Person", "PhotoQueryResponse", "PIDParseResult",
    "Procedure", "QueryRequest", "QueryType", "ReasoningStep", "RecommendedAction",
    "RegulatoryRequirement", "RelationType", "RetrievalResult", "RetrievedPassage", "Score",
    "Severity", "Signal", "SourceRef", "SymbolClass", "SyncQueueItem", "UncertaintyFlag",
    "UncertaintySource", "VoiceQueryRequest", "VoiceQueryResponse", "utcnow",
]
