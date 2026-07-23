/**
 * Read-only fallback for the hosted build.
 *
 * GitHub Pages serves a static bundle — there is no Python process behind it. Rather than show an
 * "API unreachable" banner on the public link, the client falls back to `demo-snapshot.json`: a
 * recording of a real API session produced by `scripts/build_demo_snapshot.py`.
 *
 * This is a recording, not a simulation. Every answer, alert, gap and graph in it came out of the
 * actual pipeline. The UI labels itself READ-ONLY whenever it is serving from here, because a
 * recording presented as a live system would be a lie — and this product is about earned trust.
 */

import type {
  Alert, Answer, ComplianceGap, Equipment, GraphPreview, GraphStats,
  HealthReport, KnowledgeCliffScore,
} from "./types";

export interface DemoSnapshot {
  read_only: true;
  config: Record<string, unknown>;
  health: HealthReport;
  graph_stats: GraphStats;
  graph_previews: Record<string, GraphPreview>;
  equipment: Equipment[];
  alerts: Alert[];
  knowledge_cliff: KnowledgeCliffScore[];
  compliance_gaps: ComplianceGap[];
  answers: Record<string, Answer>;
  voice: Record<string, Record<string, unknown>>;
  ingestion: Record<string, unknown>[];
  equipment_tags: string[];
}

let cache: DemoSnapshot | null = null;
let inflight: Promise<DemoSnapshot | null> | null = null;

/** Load (and memoise) the recorded session. Returns null when no snapshot ships with the build. */
export async function loadSnapshot(): Promise<DemoSnapshot | null> {
  if (cache) return cache;
  if (inflight) return inflight;

  inflight = (async () => {
    try {
      const base = (import.meta as { env?: Record<string, string | undefined> }).env?.BASE_URL ?? "/";
      const response = await fetch(`${base}demo-snapshot.json`);
      if (!response.ok) return null;
      cache = (await response.json()) as DemoSnapshot;
      return cache;
    } catch {
      return null;
    } finally {
      inflight = null;
    }
  })();

  return inflight;
}

/**
 * Best-effort answer lookup.
 *
 * Exact question match first, then a token-overlap search so a typed variation still resolves to
 * the closest recorded answer instead of falling through to an error.
 */
export function findAnswer(snapshot: DemoSnapshot, query: string): Answer | null {
  const exact = snapshot.answers[query];
  if (exact) return exact;

  const wanted = new Set(
    query.toLowerCase().replace(/[^a-z0-9\s-]/g, " ").split(/\s+/).filter((w) => w.length > 2),
  );
  if (!wanted.size) return null;

  let best: Answer | null = null;
  let bestScore = 0;
  for (const [question, answer] of Object.entries(snapshot.answers)) {
    const tokens = question.toLowerCase().replace(/[^a-z0-9\s-]/g, " ").split(/\s+/);
    const overlap = tokens.filter((t) => wanted.has(t)).length;
    const score = overlap / Math.max(wanted.size, tokens.length);
    if (score > bestScore) {
      bestScore = score;
      best = answer;
    }
  }
  return bestScore >= 0.3 ? best : null;
}

/** Nearest recorded graph preview for a focus key, falling back to the first available. */
export function findPreview(snapshot: DemoSnapshot, keys: string[]): GraphPreview {
  for (const key of keys) {
    const tag = key.replace(/^Equipment:/, "");
    const hit = snapshot.graph_previews[tag];
    if (hit) return hit;
  }
  const first = Object.values(snapshot.graph_previews)[0];
  return first ?? { nodes: [], edges: [] };
}

export function findVoice(
  snapshot: DemoSnapshot,
  transcript: string,
): Record<string, unknown> | null {
  const exact = snapshot.voice[transcript];
  if (exact) return exact;
  const entries = Object.entries(snapshot.voice);
  if (!entries.length) return null;
  const lower = transcript.toLowerCase();
  const partial = entries.find(([k]) => lower.includes(k.toLowerCase().slice(0, 8)));
  return (partial ?? entries[0])[1];
}
