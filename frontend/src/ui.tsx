/**
 * Shared visual primitives.
 *
 * Two rules run through all of these:
 *  - Severity colour is semantic and identical everywhere. An operator learns the palette once.
 *  - Confidence always shows the number. A bare coloured bar asks the viewer to trust a shape;
 *    industrial operators do not, and the whole product thesis is earned trust.
 */

import type { ReactNode } from "react";
import type { Confidence, Severity, SourceRef, UncertaintyFlag } from "./types";

export function severityClass(severity: string): string {
  return `sev sev-${severity.toLowerCase()}`;
}

export function SeverityBadge({ severity }: { severity: Severity | string }): JSX.Element {
  return <span className={severityClass(String(severity))}>{String(severity)}</span>;
}

/** Circular confidence gauge. Band thresholds match `Confidence.band` on the backend. */
export function ConfidenceDial({
  value,
  size = 58,
  label,
}: {
  value: number;
  size?: number;
  label?: string;
}): JSX.Element {
  const pct = Math.max(0, Math.min(1, value));
  const radius = size / 2 - 5;
  const circumference = 2 * Math.PI * radius;
  const band = pct >= 0.8 ? "high" : pct >= 0.55 ? "medium" : "low";
  return (
    <div className="dial" style={{ width: size }} title={label ?? `${Math.round(pct * 100)}% confidence`}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="img"
           aria-label={`${Math.round(pct * 100)} percent confidence`}>
        <circle cx={size / 2} cy={size / 2} r={radius} className="dial-track" />
        <circle
          cx={size / 2} cy={size / 2} r={radius}
          className={`dial-value dial-${band}`}
          strokeDasharray={`${circumference * pct} ${circumference}`}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
        />
        <text x="50%" y="50%" className="dial-text" dominantBaseline="central" textAnchor="middle">
          {Math.round(pct * 100)}
        </text>
      </svg>
      {label ? <span className="dial-label">{label}</span> : null}
    </div>
  );
}

/** Horizontal confidence bar with the number visible, for dense lists. */
export function ConfidenceBar({ value }: { value: number }): JSX.Element {
  const pct = Math.max(0, Math.min(1, value));
  const band = pct >= 0.8 ? "high" : pct >= 0.55 ? "medium" : "low";
  return (
    <span className="cbar" title={`${Math.round(pct * 100)}% confidence`}>
      <span className="cbar-track">
        <span className={`cbar-fill cbar-${band}`} style={{ width: `${pct * 100}%` }} />
      </span>
      <span className="cbar-num">{Math.round(pct * 100)}%</span>
    </span>
  );
}

export function Panel({
  kicker, title, actions, children, className = "",
}: {
  kicker?: string;
  title?: string;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}): JSX.Element {
  return (
    <section className={`panel ${className}`}>
      {(kicker || title || actions) && (
        <header className="panel-head">
          <div>
            {kicker ? <p className="kicker">{kicker}</p> : null}
            {title ? <h2>{title}</h2> : null}
          </div>
          {actions ? <div className="panel-actions">{actions}</div> : null}
        </header>
      )}
      {children}
    </section>
  );
}

export function StatTile({
  label, value, hint, tone = "",
}: {
  label: string;
  value: string | number;
  hint?: string;
  tone?: string;
}): JSX.Element {
  return (
    <div className={`stat ${tone}`}>
      <span className="stat-value">{value}</span>
      <span className="stat-label">{label}</span>
      {hint ? <span className="stat-hint">{hint}</span> : null}
    </div>
  );
}

/** A citation card. Page number and relevance are always shown — that is the point of the card. */
export function SourceCard({ source, index }: { source: SourceRef; index?: number }): JSX.Element {
  return (
    <article className="source">
      <header>
        {index !== undefined ? <span className="source-idx">[{index + 1}]</span> : null}
        <span className="source-title">{source.document_title}</span>
        {source.page ? <span className="source-page">p.{source.page}</span> : null}
        <span className="source-rel" title="fused relevance score">
          {Math.round(source.relevance * 100)}%
        </span>
      </header>
      {source.snippet ? <p className="source-snip">{source.snippet.slice(0, 260)}</p> : null}
      <footer>
        <span className="tag">{source.document_type.replace(/_/g, " ")}</span>
        <span className="tag ghost">via {source.retrieved_via}</span>
        {source.extraction_confidence < 0.85 ? (
          <span className="tag warn">OCR {Math.round(source.extraction_confidence * 100)}%</span>
        ) : null}
      </footer>
    </article>
  );
}

/** The caveat an operator must read before acting. Deliberately hard to miss. */
export function UncertaintyBanner({ flag }: { flag: UncertaintyFlag }): JSX.Element {
  return (
    <div className={`uncert ${String(flag.severity).toLowerCase()}`} role="note">
      <strong>{flag.source.replace(/_/g, " ")}</strong>
      <p>{flag.message}</p>
      {flag.suggested_action ? <em>{flag.suggested_action}</em> : null}
    </div>
  );
}

export function EmptyState({
  title, hint, action,
}: {
  title: string;
  hint?: string;
  action?: ReactNode;
}): JSX.Element {
  return (
    <div className="empty">
      <p className="empty-title">{title}</p>
      {hint ? <p className="empty-hint">{hint}</p> : null}
      {action}
    </div>
  );
}

export function Skeleton({ rows = 3 }: { rows?: number }): JSX.Element {
  return (
    <div className="skel" aria-busy="true" aria-live="polite">
      {Array.from({ length: rows }, (_, i) => (
        <span key={i} className="skel-row" style={{ animationDelay: `${i * 90}ms` }} />
      ))}
    </div>
  );
}

export function ErrorNote({ message }: { message: string }): JSX.Element {
  return (
    <div className="errnote" role="alert">
      <strong>Degraded</strong>
      <span>{message}</span>
    </div>
  );
}

export function confidenceOf(c: Confidence | undefined): number {
  return c ? c.value : 0;
}
