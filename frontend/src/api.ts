/**
 * The single place this application talks to the network.
 *
 * No component calls `fetch` directly. Everything is typed against `types.ts`, which mirrors the
 * Pydantic models the API actually returns — the previous hand-written shapes had drifted (e.g.
 * `reasoning_chain` was typed `{label, detail}` while the API returns
 * `{order, action, finding, confidence, sources}`), which renders as silently empty panels.
 */

import type {
  Alert, Answer, ComplianceGap, Equipment, GraphPreview, GraphStats,
  HealthReport, IngestionResult, KnowledgeCliffScore,
} from "./types";
import { findAnswer, findPreview, findVoice, loadSnapshot } from "./snapshot";

const ENV_BASE = (import.meta as { env?: Record<string, string | undefined> }).env?.VITE_API_BASE;

export const defaultApiBase: string = ENV_BASE ?? "http://localhost:8000";

export class ApiError extends Error {
  readonly status: number;
  readonly code?: string;

  constructor(status: number, message: string, code?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

/** True once a request has fallen back to the recorded session, so the UI can say so. */
export type Mode = "live" | "recorded" | "unknown";

export class IndraApi {
  private readonly base: string;

  /** Flipped to "recorded" the first time a live call fails and the snapshot answers instead. */
  mode: Mode = "unknown";

  /** Called whenever `mode` changes, so the shell can re-render its status pill. */
  onModeChange?: (mode: Mode) => void;

  constructor(baseUrl: string = defaultApiBase) {
    this.base = baseUrl.replace(/\/+$/, "");
  }

  private setMode(mode: Mode): void {
    if (this.mode !== mode) {
      this.mode = mode;
      this.onModeChange?.(mode);
    }
  }

  /**
   * Try the live API; on a connection failure fall back to the recorded session.
   *
   * Only connection-level failures fall back. A 4xx/5xx from a *reachable* backend is a real error
   * and must surface — masking it with a recording would hide genuine breakage behind a demo.
   */
  private async withFallback<T>(
    live: () => Promise<T>,
    recorded: (snapshot: import("./snapshot").DemoSnapshot) => T | null,
  ): Promise<T> {
    try {
      const result = await live();
      this.setMode("live");
      return result;
    } catch (error) {
      const unreachable = error instanceof ApiError && error.status === 0;
      if (!unreachable) throw error;
      const snapshot = await loadSnapshot();
      if (!snapshot) throw error;
      const value = recorded(snapshot);
      if (value === null || value === undefined) throw error;
      this.setMode("recorded");
      return value;
    }
  }

  private url(path: string): string {
    return `${this.base}/api/v1${path}`;
  }

  private async request<T>(path: string, init?: RequestInit): Promise<T> {
    let response: Response;
    try {
      response = await fetch(this.url(path), {
        ...init,
        headers: {
          ...(init?.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
          ...(init?.headers ?? {}),
        },
      });
    } catch {
      throw new ApiError(0, `Cannot reach the INDRA API at ${this.base}. Is uvicorn running?`);
    }

    const text = await response.text();
    let payload: unknown = null;
    try {
      payload = text ? (JSON.parse(text) as unknown) : null;
    } catch {
      payload = text;
    }

    if (!response.ok) {
      const detail = payload as { message?: string; error_code?: string; detail?: string } | null;
      throw new ApiError(
        response.status,
        detail?.message ?? detail?.detail ?? `Request failed (${response.status})`,
        detail?.error_code,
      );
    }
    return payload as T;
  }

  // ---------------------------------------------------------------- system
  health(): Promise<HealthReport> {
    return this.withFallback(
      () => this.request<HealthReport>("/system/health"),
      (snap) => snap.health,
    );
  }
  metrics(): Promise<Record<string, unknown>> {
    return this.request<Record<string, unknown>>("/system/metrics");
  }
  config(): Promise<Record<string, unknown>> {
    return this.withFallback(
      () => this.request<Record<string, unknown>>("/system/config"),
      (snap) => snap.config,
    );
  }

  // ---------------------------------------------------------------- ingestion
  upload(file: File): Promise<IngestionResult> {
    const form = new FormData();
    form.append("file", file);
    return this.request<IngestionResult>("/ingest/upload", { method: "POST", body: form });
  }

  // ---------------------------------------------------------------- copilot
  ask(query: string, opts: { equipmentTag?: string } = {}): Promise<Answer> {
    return this.withFallback(
      () => this.askLive(query, opts),
      (snap) => findAnswer(snap, query),
    );
  }

  private askLive(query: string, opts: { equipmentTag?: string } = {}): Promise<Answer> {
    return this.request<Answer>("/query/ask", {
      method: "POST",
      body: JSON.stringify({
        query,
        language: "en",
        equipment_tag: opts.equipmentTag ?? null,
        max_sources: 8,
        include_graph_preview: true,
        include_cypher: true,
        channel: "web",
      }),
    });
  }

  classify(query: string): Promise<{ query_type: string }> {
    return this.request<{ query_type: string }>("/query/classify", {
      method: "POST",
      body: JSON.stringify({ query, language: "en", channel: "web" }),
    });
  }

  // ---------------------------------------------------------------- graph
  graphStats(): Promise<GraphStats> {
    return this.withFallback(
      () => this.request<GraphStats>("/graph/stats"),
      (snap) => snap.graph_stats,
    );
  }

  graphPreview(keys: string[], hops = 2): Promise<GraphPreview> {
    const params = new URLSearchParams();
    keys.forEach((key) => params.append("keys", key));
    params.set("hops", String(hops));
    return this.withFallback(
      () => this.request<GraphPreview>(`/graph/preview?${params.toString()}`),
      (snap) => findPreview(snap, keys),
    );
  }

  // ---------------------------------------------------------------- equipment
  equipment(): Promise<Equipment[]> {
    return this.withFallback(
      () => this.request<Equipment[]>("/equipment"),
      (snap) => snap.equipment,
    );
  }
  equipmentDetail(tag: string): Promise<Equipment> {
    return this.request<Equipment>(`/equipment/${encodeURIComponent(tag)}`);
  }

  // ---------------------------------------------------------------- proactive
  alerts(unresolvedOnly = true): Promise<Alert[]> {
    return this.withFallback(
      () => this.request<Alert[]>(`/alerts?unresolved_only=${String(unresolvedOnly)}`),
      (snap) => snap.alerts,
    );
  }
  scan(tags?: string[]): Promise<unknown> {
    // The route declares `tags: list[str] | None`, so FastAPI expects the body to be a JSON *array*
    // or absent. Posting `{}` returns 422 — send nothing when scanning the whole fleet.
    return this.request<unknown>("/alerts/scan", {
      method: "POST",
      ...(tags && tags.length ? { body: JSON.stringify(tags) } : {}),
    });
  }
  knowledgeCliff(): Promise<KnowledgeCliffScore[]> {
    return this.withFallback(
      () => this.request<KnowledgeCliffScore[]>("/alerts/knowledge-cliff"),
      (snap) => snap.knowledge_cliff,
    );
  }

  // ---------------------------------------------------------------- compliance
  audit(tags?: string[]): Promise<ComplianceGap[]> {
    return this.withFallback(
      () => this.request<ComplianceGap[]>("/compliance/audit", {
        method: "POST",
        body: JSON.stringify(tags && tags.length ? { tags } : {}),
      }),
      (snap) => snap.compliance_gaps,
    );
  }
  auditPdf(tags: string[]): Promise<Record<string, unknown>> {
    return this.request<Record<string, unknown>>("/compliance/package/pdf", {
      method: "POST",
      body: JSON.stringify({ tags }),
    });
  }

  // ---------------------------------------------------------------- mobile
  /**
   * The route takes multipart (`audio` file + `language_hint` form field), not JSON. When Whisper
   * is absent the backend's NullSTT decodes the payload as text, so a UTF-8 blob of the transcript
   * exercises the real pipeline — detection, tag masking, translation, copilot, translation back.
   */
  voice(transcript: string, language?: string): Promise<Record<string, unknown>> {
    const form = new FormData();
    form.append("audio", new Blob([transcript], { type: "text/plain" }), "transcript.txt");
    if (language) form.append("language_hint", language);
    return this.withFallback(
      () => this.request<Record<string, unknown>>("/mobile/voice", { method: "POST", body: form }),
      (snap) => findVoice(snap, transcript),
    );
  }

  photo(file: File): Promise<Record<string, unknown>> {
    const form = new FormData();
    form.append("image", file); // field is `image`, not `file`
    return this.request<Record<string, unknown>>("/mobile/photo", { method: "POST", body: form });
  }
  offlineBundle(budgetBytes?: number): Promise<Record<string, unknown>> {
    const query = budgetBytes ? `?budget_bytes=${String(budgetBytes)}` : "";
    return this.request<Record<string, unknown>>(`/mobile/offline-bundle${query}`, { method: "POST" });
  }
}
