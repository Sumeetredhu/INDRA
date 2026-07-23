"""Compliance service for regulation parsing, deterministic audits, and PDF evidence packs."""

from __future__ import annotations

import asyncio
import io
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from indra.agents.compliance_agent.gap_detection import (
    RequirementAssessment,
    assess_scope,
    build_snapshot,
    evidence_from_document,
    evidence_from_maintenance,
    evidence_from_procedure,
)
from indra.agents.compliance_agent.parser import RegulationParser
from indra.agents.compliance_agent.requirements import RequirementCatalogue, load_seed_catalogue
from indra.core.contracts import KnowledgeGraphService
from indra.core.deps import AgentDeps
from indra.core.events import Event, Topic
from indra.core.exceptions import ComplianceError
from indra.core.logging import get_logger
from indra.core.models import AuditPackage, ComplianceGap, DocumentMeta, Equipment, SourceRef

if TYPE_CHECKING:  # pragma: no cover - protocol declaration only
    pass

logger = get_logger(__name__)


class ComplianceAgent:
    """Continuously compare plant evidence against structured regulatory requirements."""

    name = "compliance_agent"

    def __init__(self, deps: AgentDeps) -> None:
        self._deps = deps
        self._catalogue = RequirementCatalogue()
        self._parser = RegulationParser(deps.llm, settings=deps.settings)
        self._knowledge_graph: KnowledgeGraphService | None = None
        self._last_assessments: list[RequirementAssessment] = []

    def bind(self, *, knowledge_graph: KnowledgeGraphService) -> None:
        """Keep the service relationship explicit for lifecycle checks and future retrieval use."""
        self._knowledge_graph = knowledge_graph

    async def startup(self) -> None:
        """Load shipped regulation definitions without making the application depend on one file."""
        self._catalogue = await asyncio.to_thread(load_seed_catalogue, self._deps.settings)

    async def shutdown(self) -> None:
        """The catalogue is in-memory and has no resource to close."""

    async def health(self) -> dict[str, object]:
        return {
            "ok": len(self._catalogue) > 0,
            "backend": "deterministic_audit",
            "detail": f"{len(self._catalogue)} requirements across {len(self._catalogue.regulations())} regulations",
            "catalogue": self._catalogue.describe(),
        }

    async def parse_regulation(self, document_id: str) -> list[object]:
        """Parse an uploaded regulation document and promote its clauses over seeded defaults."""
        meta = await self._deps.metadata.get_document(document_id)
        if meta is None:
            raise ComplianceError(f"Cannot parse regulation: document {document_id} was not found in metadata.")
        if not meta.source_path:
            raise ComplianceError(f"Cannot parse regulation {meta.title}: the raw source location is missing.")
        content = await self._deps.blobs.get(meta.source_path)
        text = await asyncio.to_thread(self._text_from_bytes, content, meta)
        parsed = await self._parser.parse(text, meta=meta)
        self._catalogue.add(parsed.specs)
        return list(parsed.requirements())

    async def audit(
        self,
        *,
        tags: Sequence[str] | None = None,
        regulations: Sequence[str] | None = None,
    ) -> list[ComplianceGap]:
        """Run a whole-scope evidence audit and emit each current gap as an event."""
        assessments = await self._assess(tags=tags, regulations=regulations)
        gaps = [gap for assessment in assessments if (gap := assessment.to_gap()) is not None]
        for gap in gaps:
            await self._publish(
                Topic.GAP_DETECTED,
                gap_id=gap.gap_id,
                equipment_tag=gap.equipment_tag,
                regulation=gap.requirement.regulation,
                clause=gap.requirement.clause,
                status=gap.status.value,
                severity=gap.severity.value,
                deadline=gap.deadline.isoformat() if gap.deadline else None,
            )
        return gaps

    async def build_package(
        self, *, tags: Sequence[str], regulations: Sequence[str] | None = None
    ) -> AuditPackage:
        """Build the compliance matrix, evidence list, and corrective actions for a requested scope."""
        assessments = await self._assess(tags=tags, regulations=regulations)
        gaps = [gap for assessment in assessments if (gap := assessment.to_gap()) is not None]
        actions = [gap.recommended_action for gap in gaps if gap.recommended_action is not None]
        evidence: list[SourceRef] = []
        seen: set[tuple[str, str | None]] = set()
        for assessment in assessments:
            for source in assessment.evidence:
                key = (source.document_id, source.chunk_id)
                if key not in seen:
                    seen.add(key)
                    evidence.append(source)
        return AuditPackage(
            title=f"INDRA compliance audit - {', '.join(sorted(tag.upper() for tag in tags))}",
            scope_tags=sorted({tag.upper() for tag in tags}),
            regulations=list(regulations or self._catalogue.regulations()),
            matrix=[assessment.to_row() for assessment in assessments],
            gaps=gaps,
            corrective_actions=actions,
            evidence_documents=evidence,
        )

    async def export_pdf(self, package: AuditPackage) -> Path:
        """Render a portable audit package under the configured export directory."""
        target = self._deps.settings.export_dir / f"{package.package_id}_audit.pdf"
        await asyncio.to_thread(self._render_pdf, package, target)
        package.pdf_path = str(target)
        return target

    async def _assess(
        self,
        *,
        tags: Sequence[str] | None,
        regulations: Sequence[str] | None,
    ) -> list[RequirementAssessment]:
        equipment = await self._deps.graph.list_equipment()
        wanted = {tag.strip().upper() for tag in tags or ()}
        scope = [item for item in equipment if not wanted or item.tag.upper() in wanted]
        if wanted and not scope:
            raise ComplianceError(f"No requested equipment is present in the graph: {', '.join(sorted(wanted))}.")
        evidence = await self._evidence(scope)
        snapshot = build_snapshot(datetime.now(UTC).date(), evidence)
        assessments = assess_scope(scope, self._catalogue, snapshot, regulations=regulations)
        self._last_assessments = assessments
        return assessments

    async def _evidence(self, equipment: Sequence[Equipment]) -> list[object]:
        collected: list[object] = []
        for item in equipment:
            maintenance, procedures, documents = await asyncio.gather(
                self._deps.graph.maintenance_history(item.tag),
                self._deps.graph.procedures_for(item.tag),
                self._deps.graph.documents_for_tag(item.tag),
            )
            collected.extend(evidence_from_maintenance(record) for record in maintenance)
            collected.extend(evidence_from_procedure(procedure, tag=item.tag) for procedure in procedures)
            collected.extend(evidence_from_document(meta, tag=item.tag) for meta in documents)
        return collected

    @staticmethod
    def _text_from_bytes(content: bytes, meta: DocumentMeta) -> str:
        if meta.mime_family.value == "pdf":
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(content))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        return content.decode("utf-8", errors="replace")

    @staticmethod
    def _render_pdf(package: AuditPackage, target: Path) -> None:
        """Create a compact, readable evidence package without a web or model dependency."""
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.pdfbase.pdfmetrics import stringWidth
        from reportlab.pdfgen import canvas

        target.parent.mkdir(parents=True, exist_ok=True)
        pdf = canvas.Canvas(str(target), pagesize=A4)
        width, height = A4
        x = 18 * mm
        y = height - 18 * mm

        def line(value: str, *, size: int = 9, leading: float = 4.8 * mm) -> None:
            nonlocal y
            words = value.split()
            current = ""
            for word in words or [""]:
                candidate = f"{current} {word}".strip()
                if stringWidth(candidate, "Helvetica", size) > width - 2 * x and current:
                    pdf.drawString(x, y, current)
                    y -= leading
                    current = word
                else:
                    current = candidate
            pdf.drawString(x, y, current)
            y -= leading
            if y < 20 * mm:
                pdf.showPage()
                y = height - 18 * mm

        pdf.setFont("Helvetica-Bold", 16)
        line(package.title, size=16, leading=9 * mm)
        pdf.setFont("Helvetica", 9)
        line(f"Generated: {package.generated_at.isoformat()} | Scope: {', '.join(package.scope_tags)}")
        line(f"Regulations: {', '.join(package.regulations)} | Compliance rate: {package.compliance_rate:.1f}%")
        pdf.setFont("Helvetica-Bold", 12)
        line("Compliance matrix", size=12, leading=7 * mm)
        pdf.setFont("Helvetica", 8)
        for row in package.matrix:
            line(f"{row.equipment_tag} | {row.regulation} {row.clause} | {row.status.value.upper()} | {row.obligation}", size=8)
        pdf.setFont("Helvetica-Bold", 12)
        line("Corrective actions", size=12, leading=7 * mm)
        pdf.setFont("Helvetica", 8)
        for action in package.corrective_actions:
            line(f"[{action.urgency.value}] {action.action} (owner: {action.owner_role or 'Unassigned'})", size=8)
        pdf.setFont("Helvetica-Bold", 12)
        line("Evidence documents", size=12, leading=7 * mm)
        pdf.setFont("Helvetica", 8)
        for source in package.evidence_documents:
            line(f"{source.citation}: {source.snippet[:260]}", size=8)
        pdf.save()

    async def _publish(self, topic: Topic, **payload: object) -> None:
        event = Event.make(topic, source=self.name, **payload)
        await self._deps.events.publish(event.topic.value, event.model_dump(mode="json"))


__all__ = ["ComplianceAgent"]
