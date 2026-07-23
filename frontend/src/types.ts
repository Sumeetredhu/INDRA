/**
 * TypeScript mirrors of the Pydantic models in `indra/core/models.py`.
 *
 * The API returns `model_dump(mode="json")` of those models verbatim, so these are the wire
 * contract. Keep field names identical — a rename here is a silent runtime break, not a type error.
 */

export type Severity = "INFO" | "LOW" | "WARNING" | "HIGH" | "CRITICAL";
export type Criticality = "A" | "B" | "C";
export type GapStatus = "compliant" | "missing" | "outdated" | "incomplete";
export type QueryType =
  | "factual" | "diagnostic" | "procedural" | "predictive"
  | "comparative" | "compliance" | "knowledge_gap";

export interface Confidence {
  value: number;
  rationale: string;
  method: string;
}

export interface SourceRef {
  document_id: string;
  document_title: string;
  document_type: string;
  chunk_id: string | null;
  page: number | null;
  snippet: string;
  relevance: number;
  extraction_confidence: number;
  retrieved_via: string;
  document_date: string | null;
}

export interface UncertaintyFlag {
  source: string;
  message: string;
  severity: Severity;
  affected_claim: string | null;
  suggested_action: string | null;
}

export interface GraphPath {
  nodes: string[];
  relations: string[];
  hops: number;
  confidence: number;
  narrative: string;
}

export interface ReasoningStep {
  order: number;
  action: string;
  finding: string;
  confidence: Confidence;
  sources: SourceRef[];
  graph_paths: GraphPath[];
  cypher: string | null;
  duration_ms: number;
}

export interface RecommendedAction {
  action: string;
  urgency: Severity;
  owner_role: string | null;
  due_within_days: number | null;
  rationale: string;
  estimated_minutes: number | null;
}

export interface Answer {
  answer_id: string;
  query: string;
  query_type: QueryType;
  answer_text: string;
  language: string;
  confidence: Confidence;
  reasoning_chain: ReasoningStep[];
  sources: SourceRef[];
  uncertainty_flags: UncertaintyFlag[];
  alternative_interpretations: string[];
  recommended_actions: RecommendedAction[];
  graph_preview: GraphPreview | null;
  cypher_queries: string[];
  generated_at: string;
  latency_ms: number;
  provider_used: string;
}

export interface Signal {
  signal_id: string;
  kind: string;
  equipment_tag: string;
  description: string;
  observed_at: string;
  strength: number;
  sources: SourceRef[];
}

export interface CompoundSignal {
  rule_id: string;
  rule_name: string;
  equipment_tag: string;
  severity: Severity;
  signals: Signal[];
  explanation: string;
  confidence: Confidence;
  risk_score: number;
  detected_at: string;
}

export interface Alert {
  alert_id: string;
  title: string;
  equipment_tag: string;
  severity: Severity;
  body: string;
  compound_signal: CompoundSignal | null;
  recommended_actions: RecommendedAction[];
  sources: SourceRef[];
  risk_percent: number;
  raised_at: string;
  resolved: boolean;
  dedupe_key: string;
}

export interface Person {
  person_id: string;
  name: string;
  role: string | null;
  years_experience: number | null;
  retirement_date: string | null;
  expertise_tags: string[];
  documented_contributions: number;
}

export interface KnowledgeCliffScore {
  equipment_tag: string;
  score: number;
  severity: Severity;
  retiring_experts: Person[];
  document_count: number;
  criticality: Criticality;
  days_to_first_retirement: number | null;
  factors: Record<string, number>;
  interview_questions: string[];
  rationale: string;
}

export interface Equipment {
  tag: string;
  name: string;
  equipment_type: string;
  manufacturer: string | null;
  model: string | null;
  criticality: Criticality;
  location: string | null;
  specifications: Record<string, unknown>;
  oem_thresholds: Record<string, number>;
}

export interface RegulatoryRequirement {
  requirement_id: string;
  regulation: string;
  clause: string;
  text: string;
  obligation: string;
  frequency_days: number | null;
  applies_to_types: string[];
  penalty: string | null;
  source: SourceRef | null;
}

export interface ComplianceGap {
  gap_id: string;
  requirement: RegulatoryRequirement;
  equipment_tag: string;
  status: GapStatus;
  severity: Severity;
  detail: string;
  last_evidence_date: string | null;
  days_overdue: number | null;
  deadline: string | null;
  penalty_risk: string | null;
  recommended_action: RecommendedAction | null;
  evidence: SourceRef[];
}

export interface DocumentMeta {
  document_id: string;
  title: string;
  filename: string;
  content_hash: string;
  mime_family: string;
  document_type: string;
  size_bytes: number;
  page_count: number | null;
  ingested_at: string;
}

export interface IngestionResult {
  job_id: string;
  document: DocumentMeta;
  chunks_created: number;
  entities_created: number;
  relationships_created: number;
  pid_symbols: number;
  pid_connections: number;
  duplicate_of: string | null;
  warnings: string[];
  duration_ms: number;
  stage: string;
}

export interface GraphNode {
  id: string;
  label: string;
  type: string;
  degree?: number;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  label: string;
  confidence: number;
}

export interface GraphPreview {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export type GraphStats = Record<string, number>;

export interface HealthReport {
  ok: boolean;
  settings?: Record<string, unknown>;
  stores?: Record<string, { ok: boolean; backend: string; detail: string }>;
  agents?: Record<string, { ok: boolean; backend: string; detail: string }>;
}

/** The ordered pipeline stages the ingestion agent reports. Drives the live pipeline view. */
export const PIPELINE_STAGES = [
  "received", "validated", "stored", "parsed", "chunked", "embedded",
  "entities_extracted", "relations_extracted", "graph_queued", "graph_written", "complete",
] as const;

export type PipelineStage = (typeof PIPELINE_STAGES)[number];
