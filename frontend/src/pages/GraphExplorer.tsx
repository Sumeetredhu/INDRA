/**
 * Interactive knowledge graph — the demo's 0:40 beat.
 *
 * Click any node to re-centre the view on it. That single interaction is what convinces a judge the
 * graph is real rather than a rendered screenshot.
 */

import { useCallback, useEffect, useState } from "react";
import type { GraphNode, GraphPreview, GraphStats } from "../types";
import { ApiError, IndraApi } from "../api";
import { EmptyState, ErrorNote, Panel, Skeleton, StatTile } from "../ui";
import { GraphCanvas } from "../graph";

export function GraphExplorerPage({ api }: { api: IndraApi }): JSX.Element {
  const [stats, setStats] = useState<GraphStats | null>(null);
  const [preview, setPreview] = useState<GraphPreview | null>(null);
  const [focus, setFocus] = useState<string[]>(["Equipment:P-101"]);
  const [hops, setHops] = useState(2);
  const [selected, setSelected] = useState<GraphNode | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [seed, setSeed] = useState("P-101");

  const load = useCallback(async (keys: string[], depth: number): Promise<void> => {
    setError(null);
    try {
      const [s, p] = await Promise.all([api.graphStats(), api.graphPreview(keys, depth)]);
      setStats(s);
      setPreview(p);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
      setPreview({ nodes: [], edges: [] });
    }
  }, [api]);

  useEffect(() => { void load(focus, hops); }, [load, focus, hops]);

  const nodeCounts = Object.entries(stats ?? {})
    .filter(([k]) => k.startsWith("node:"))
    .map(([k, v]) => [k.replace("node:", ""), v] as const)
    .sort((a, b) => b[1] - a[1]);

  return (
    <div className="page">
      <Panel kicker="Cross-document intelligence" title="The living graph">
        <div className="stats">
          <StatTile label="nodes" value={stats?.nodes ?? "—"} />
          <StatTile label="relationships" value={stats?.relationships ?? "—"} />
          <StatTile label="documents" value={stats?.documents ?? "—"} />
          <StatTile label="equipment" value={stats?.equipment ?? "—"} />
        </div>
        <form
          className="askbar"
          onSubmit={(e) => { e.preventDefault(); setFocus([`Equipment:${seed.trim().toUpperCase()}`]); }}
        >
          <input value={seed} onChange={(e) => setSeed(e.target.value)}
                 aria-label="Centre the graph on an equipment tag" placeholder="Equipment tag, e.g. P-101" />
          <button type="submit">Centre</button>
        </form>
        <div className="chips">
          {[1, 2, 3].map((h) => (
            <button key={h} type="button" className={`chip ${hops === h ? "on" : ""}`} onClick={() => setHops(h)}>
              {h} hop{h > 1 ? "s" : ""}
            </button>
          ))}
        </div>
        {error ? <ErrorNote message={error} /> : null}
      </Panel>

      <Panel kicker={`centred on ${focus.join(", ")}`} title="Click a node to re-centre">
        {preview === null ? (
          <Skeleton rows={6} />
        ) : preview.nodes.length === 0 ? (
          <EmptyState title="Nothing connected to that tag yet"
                      hint="Ingest documents mentioning it, then centre again." />
        ) : (
          <GraphCanvas
            preview={preview}
            focusKeys={focus}
            onSelect={(node) => { setSelected(node); setFocus([node.id]); }}
          />
        )}
      </Panel>

      {selected ? (
        <Panel kicker="Selected node" title={selected.label}>
          <div className="meta">
            <span className="tag">{selected.type}</span>
            <code>{selected.id}</code>
          </div>
        </Panel>
      ) : null}

      {nodeCounts.length ? (
        <Panel kicker="Composition" title="Nodes by label">
          <div className="stats">
            {nodeCounts.map(([label, count]) => (
              <StatTile key={label} label={label} value={count} />
            ))}
          </div>
        </Panel>
      ) : null}
    </div>
  );
}
