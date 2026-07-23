/**
 * Equipment register and system status.
 *
 * The system panel deliberately shows the backend *actually bound* per store, so a fallback reads
 * as "memory (fallback)" rather than a green tick that lies (`docs/DECISIONS.md` D1).
 */

import { useEffect, useState } from "react";
import type { Equipment, HealthReport } from "../types";
import { ApiError, IndraApi } from "../api";
import { EmptyState, ErrorNote, Panel, Skeleton, StatTile } from "../ui";

export function FleetPage({ api }: { api: IndraApi }): JSX.Element {
  const [equipment, setEquipment] = useState<Equipment[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState("ALL");

  useEffect(() => {
    void (async () => {
      try {
        const data = await api.equipment();
        data.sort((a, b) => a.criticality.localeCompare(b.criticality) || a.tag.localeCompare(b.tag));
        setEquipment(data);
      } catch (err) {
        setError(err instanceof ApiError ? err.message : String(err));
        setEquipment([]);
      }
    })();
  }, [api]);

  const shown = (equipment ?? []).filter((e) => filter === "ALL" || e.criticality === filter);
  const counts = (equipment ?? []).reduce<Record<string, number>>((acc, e) => {
    acc[e.criticality] = (acc[e.criticality] ?? 0) + 1;
    return acc;
  }, {});

  return (
    <div className="page">
      <Panel kicker="Asset register" title="Equipment in the graph">
        <div className="stats">
          <StatTile label="total assets" value={equipment?.length ?? "—"} />
          <StatTile label="criticality A" value={counts.A ?? 0} tone="crit" />
          <StatTile label="criticality B" value={counts.B ?? 0} tone="warn" />
          <StatTile label="criticality C" value={counts.C ?? 0} />
        </div>
        <div className="chips">
          {["ALL", "A", "B", "C"].map((f) => (
            <button key={f} type="button" className={`chip ${filter === f ? "on" : ""}`}
                    onClick={() => setFilter(f)}>
              {f === "ALL" ? "All" : `Criticality ${f}`}
            </button>
          ))}
        </div>
        {error ? <ErrorNote message={error} /> : null}
      </Panel>

      {equipment === null ? (
        <Panel><Skeleton rows={5} /></Panel>
      ) : shown.length === 0 ? (
        <Panel><EmptyState title="No equipment yet" hint="Ingest documents to populate the register." /></Panel>
      ) : (
        <Panel kicker={`${shown.length} assets`} title="Register">
          <div className="tablewrap">
            <table className="matrix">
              <thead>
                <tr><th>Tag</th><th>Name</th><th>Type</th><th>Manufacturer</th>
                    <th>Criticality</th><th>OEM thresholds</th></tr>
              </thead>
              <tbody>
                {shown.map((e) => (
                  <tr key={e.tag}>
                    <td><code>{e.tag}</code></td>
                    <td>{e.name || "—"}</td>
                    <td>{e.equipment_type}</td>
                    <td>{e.manufacturer ?? "—"}{e.model ? ` ${e.model}` : ""}</td>
                    <td><span className={`crit crit-${e.criticality}`}>{e.criticality}</span></td>
                    <td>
                      {Object.entries(e.oem_thresholds).length
                        ? Object.entries(e.oem_thresholds)
                            .map(([k, v]) => `${k.replace(/_/g, " ")} ${v}`)
                            .join(" · ")
                        : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      )}
    </div>
  );
}

export function SystemPage({ api }: { api: IndraApi }): JSX.Element {
  const [health, setHealth] = useState<HealthReport | null>(null);
  const [config, setConfig] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void (async () => {
      try {
        const [h, c] = await Promise.all([api.health(), api.config()]);
        setHealth(h);
        setConfig(c);
      } catch (err) {
        setError(err instanceof ApiError ? err.message : String(err));
      }
    })();
  }, [api]);

  function rows(section: Record<string, { ok: boolean; backend: string; detail: string }> | undefined) {
    if (!section) return null;
    return Object.entries(section).map(([name, s]) => (
      <tr key={name}>
        <td><code>{name}</code></td>
        <td><span className={s.ok ? "status compliant" : "status missing"}>{s.ok ? "ok" : "down"}</span></td>
        <td>
          {s.backend}
          {s.backend === "memory" ? <span className="tag warn">fallback</span> : null}
        </td>
        <td className="obl">{s.detail}</td>
      </tr>
    ));
  }

  return (
    <div className="page">
      <Panel kicker="Operations" title="System status">
        {error ? <ErrorNote message={error} /> : null}
        {health === null && !error ? <Skeleton rows={4} /> : null}
        {health ? (
          <div className="tablewrap">
            <table className="matrix">
              <thead><tr><th>Component</th><th>State</th><th>Backend</th><th>Detail</th></tr></thead>
              <tbody>
                {rows(health.stores)}
                {rows(health.agents)}
              </tbody>
            </table>
          </div>
        ) : null}
      </Panel>

      {config ? (
        <Panel kicker="Configuration" title="Active settings">
          <div className="stats">
            {Object.entries(config).map(([k, v]) => (
              <StatTile key={k} label={k.replace(/_/g, " ")} value={String(v)} />
            ))}
          </div>
        </Panel>
      ) : null}
    </div>
  );
}
