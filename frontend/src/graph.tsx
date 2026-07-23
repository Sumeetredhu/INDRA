/**
 * Knowledge-graph canvas.
 *
 * Hand-rolled SVG with a small force-directed layout rather than React Flow. Two reasons: it keeps
 * the dependency list at react + react-dom so the build cannot fail on a network install, and the
 * layout we need — a query entity at the centre with its evidence radiating out — is simpler to
 * express directly than to configure.
 *
 * The simulation is deterministic: node positions seed from a hash of the node id, so the same
 * graph always renders the same way and a demo looks identical on every run.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import type { GraphEdge, GraphNode, GraphPreview } from "./types";

interface Positioned extends GraphNode {
  x: number;
  y: number;
  vx: number;
  vy: number;
  degree: number;
}

const WIDTH = 900;
const HEIGHT = 520;
const ITERATIONS = 260;

/** Deterministic pseudo-random in [0,1) from a string — keeps layout stable across runs. */
function seeded(id: string): number {
  let h = 2166136261;
  for (let i = 0; i < id.length; i += 1) {
    h ^= id.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return ((h >>> 0) % 100000) / 100000;
}

function nodeColour(type: string): string {
  const t = type.toLowerCase();
  if (t.includes("equipment")) return "#4ea3ff";
  if (t.includes("document")) return "#8b93a7";
  if (t.includes("person")) return "#c47bff";
  if (t.includes("failure")) return "#ff5d5d";
  if (t.includes("procedure")) return "#39d3a0";
  if (t.includes("regulatory") || t.includes("clause")) return "#ffb020";
  return "#6d7891";
}

function layout(nodes: GraphNode[], edges: GraphEdge[], focus: Set<string>): Positioned[] {
  const degree = new Map<string, number>();
  edges.forEach((e) => {
    degree.set(e.source, (degree.get(e.source) ?? 0) + 1);
    degree.set(e.target, (degree.get(e.target) ?? 0) + 1);
  });

  const placed: Positioned[] = nodes.map((n, i) => {
    const angle = seeded(n.id) * Math.PI * 2;
    const radius = focus.has(n.id) ? 40 : 130 + seeded(`${n.id}r`) * 190;
    return {
      ...n,
      degree: degree.get(n.id) ?? 0,
      x: WIDTH / 2 + Math.cos(angle) * radius + (i % 7) * 3,
      y: HEIGHT / 2 + Math.sin(angle) * radius + (i % 5) * 3,
      vx: 0,
      vy: 0,
    };
  });

  const index = new Map(placed.map((p) => [p.id, p]));
  const links = edges
    .map((e) => ({ a: index.get(e.source), b: index.get(e.target) }))
    .filter((l): l is { a: Positioned; b: Positioned } => Boolean(l.a && l.b));

  for (let step = 0; step < ITERATIONS; step += 1) {
    const cooling = 1 - step / ITERATIONS;

    // Repulsion — every pair pushes apart, damped by distance.
    for (let i = 0; i < placed.length; i += 1) {
      for (let j = i + 1; j < placed.length; j += 1) {
        const a = placed[i];
        const b = placed[j];
        let dx = a.x - b.x;
        let dy = a.y - b.y;
        let dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 0.01) {
          dx = seeded(a.id + b.id) - 0.5;
          dy = seeded(b.id + a.id) - 0.5;
          dist = 0.01;
        }
        const force = 2600 / (dist * dist);
        a.vx += (dx / dist) * force;
        a.vy += (dy / dist) * force;
        b.vx -= (dx / dist) * force;
        b.vy -= (dy / dist) * force;
      }
    }

    // Attraction along edges.
    links.forEach(({ a, b }) => {
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const dist = Math.sqrt(dx * dx + dy * dy) || 0.01;
      const force = (dist - 110) * 0.012;
      a.vx += (dx / dist) * force;
      a.vy += (dy / dist) * force;
      b.vx -= (dx / dist) * force;
      b.vy -= (dy / dist) * force;
    });

    placed.forEach((p) => {
      // Focus nodes are pinned near the centre — the query entity should never drift to a corner.
      const pull = focus.has(p.id) ? 0.05 : 0.004;
      p.vx += (WIDTH / 2 - p.x) * pull;
      p.vy += (HEIGHT / 2 - p.y) * pull;
      p.x += Math.max(-14, Math.min(14, p.vx * cooling));
      p.y += Math.max(-14, Math.min(14, p.vy * cooling));
      p.vx *= 0.82;
      p.vy *= 0.82;
      p.x = Math.max(36, Math.min(WIDTH - 36, p.x));
      p.y = Math.max(30, Math.min(HEIGHT - 30, p.y));
    });
  }
  return placed;
}

export function GraphCanvas({
  preview,
  focusKeys = [],
  onSelect,
}: {
  preview: GraphPreview;
  focusKeys?: string[];
  onSelect?: (node: GraphNode) => void;
}): JSX.Element {
  const focus = useMemo(() => new Set(focusKeys), [focusKeys]);
  const nodes = useMemo(
    () => layout(preview.nodes, preview.edges, focus),
    [preview.nodes, preview.edges, focus],
  );
  const byId = useMemo(() => new Map(nodes.map((n) => [n.id, n])), [nodes]);
  const [hover, setHover] = useState<string | null>(null);
  const [mounted, setMounted] = useState(false);
  const ref = useRef<SVGSVGElement>(null);

  useEffect(() => {
    const timer = window.setTimeout(() => setMounted(true), 30);
    return () => window.clearTimeout(timer);
  }, [preview]);

  if (!preview.nodes.length) {
    return <div className="empty"><p className="empty-title">No graph yet</p>
      <p className="empty-hint">Ingest documents to populate the knowledge graph.</p></div>;
  }

  const types = Array.from(new Set(preview.nodes.map((n) => n.type)));

  return (
    <div className="graphwrap">
      <svg ref={ref} viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className={`graph ${mounted ? "in" : ""}`}
           role="img" aria-label={`Knowledge graph with ${preview.nodes.length} nodes`}>
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="18" refY="5" markerWidth="5" markerHeight="5"
                  orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#3a4459" />
          </marker>
        </defs>

        <g className="edges">
          {preview.edges.map((edge) => {
            const a = byId.get(edge.source);
            const b = byId.get(edge.target);
            if (!a || !b) return null;
            const active = hover === edge.source || hover === edge.target;
            return (
              <g key={edge.id} className={active ? "edge active" : "edge"}>
                <line x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                      strokeWidth={active ? 1.9 : 0.9}
                      strokeOpacity={active ? 0.95 : 0.28 + edge.confidence * 0.25}
                      markerEnd="url(#arrow)" />
                {active ? (
                  <text x={(a.x + b.x) / 2} y={(a.y + b.y) / 2 - 5} className="edge-label">
                    {edge.label}
                  </text>
                ) : null}
              </g>
            );
          })}
        </g>

        <g className="nodes">
          {nodes.map((node) => {
            const r = focus.has(node.id) ? 15 : 7 + Math.min(7, node.degree * 0.7);
            const active = hover === node.id;
            return (
              <g key={node.id} className="node"
                 onMouseEnter={() => setHover(node.id)}
                 onMouseLeave={() => setHover(null)}
                 onClick={() => onSelect?.(node)}
                 tabIndex={0}
                 role="button"
                 aria-label={`${node.label}, ${node.type}, ${node.degree} connections`}
                 onKeyDown={(e) => { if (e.key === "Enter") onSelect?.(node); }}>
                {focus.has(node.id) ? (
                  <circle cx={node.x} cy={node.y} r={r + 7} className="node-halo"
                          fill="none" stroke={nodeColour(node.type)} />
                ) : null}
                <circle cx={node.x} cy={node.y} r={r}
                        fill={nodeColour(node.type)}
                        stroke={active ? "#fff" : "#0d1017"}
                        strokeWidth={active ? 2 : 1.2} />
                {(focus.has(node.id) || node.degree > 2 || active) ? (
                  <text x={node.x} y={node.y - r - 6} className="node-label" textAnchor="middle">
                    {node.label.length > 22 ? `${node.label.slice(0, 21)}…` : node.label}
                  </text>
                ) : null}
              </g>
            );
          })}
        </g>
      </svg>

      <div className="legend">
        {types.map((t) => (
          <span key={t}>
            <i style={{ background: nodeColour(t) }} />
            {t}
          </span>
        ))}
        <span className="legend-count">
          {preview.nodes.length} nodes · {preview.edges.length} edges
        </span>
      </div>
    </div>
  );
}
