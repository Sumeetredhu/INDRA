"""Generate and ingest INDRA's deterministic demo corpus in one command."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence

from indra.core.config import Settings, StorageBackend, get_settings
from indra.core.exceptions import IndraError
from indra.core.logging import get_logger
from indra.orchestrator.orchestrator import IndraOrchestrator
from scripts import demo_facts as facts
from scripts._console import Console
from scripts.generate_demo_data import generate_corpus

logger = get_logger(__name__)


async def seed(settings: Settings, *, regenerate: bool = False) -> dict[str, object]:
    """Generate (if required), ingest, scan, and return a concise demo readiness report."""
    report = generate_corpus(settings=settings, skip_existing=not regenerate)
    orchestrator = IndraOrchestrator(settings)
    await orchestrator.startup()
    try:
        results = []
        for filename in facts.CORPUS_FILENAMES:
            results.append(await orchestrator.ingestion.ingest_path(report.output_dir / filename))
        drain = getattr(orchestrator.knowledge_graph._deps.events, "drain", None)
        if callable(drain):
            await drain()
        gaps = await orchestrator.compliance.audit(tags=[facts.P101.tag])
        if callable(drain):
            await drain()
        package = await orchestrator.compliance.build_package(tags=[facts.P101.tag])
        audit_pdf = await orchestrator.compliance.export_pdf(package)
        alerts = await orchestrator.proactive.alerts()
        cliff = await orchestrator.proactive.knowledge_cliff(tags=[facts.P101.tag])
        return {
            "files": len(results),
            "chunks": sum(item.chunks_created for item in results),
            "entities": sum(item.entities_created for item in results),
            "relationships": sum(item.relationships_created for item in results),
            "alerts": len(alerts),
            "gaps": len(gaps),
            "knowledge_cliffs": len(cliff),
            "audit_pdf": str(audit_pdf),
            "output_dir": str(report.output_dir),
        }
    finally:
        await orchestrator.shutdown()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the demo seed command and return a shell-friendly exit status."""
    parser = argparse.ArgumentParser(description="Generate and ingest INDRA's demo corpus.")
    parser.add_argument("--regenerate", action="store_true", help="Rebuild every demo document before ingestion.")
    parser.add_argument("--memory", action="store_true", help="Force all stores in-process for a portable demo run.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    settings = get_settings()
    if args.memory:
        settings = settings.model_copy(update={"storage_backend": StorageBackend.MEMORY, "demo_mode": True})
    console = Console()
    try:
        result = asyncio.run(seed(settings, regenerate=args.regenerate))
    except IndraError as exc:
        logger.error("demo seeding failed", extra={"error_code": exc.error_code, "detail": exc.message})
        console.status("fail", "demo seed failed", exc.message)
        return 1
    console.banner("INDRA demo corpus seeded", str(result["output_dir"]))
    for key, value in result.items():
        console.kv(key.replace("_", " ").title(), str(value))
    console.status("pass", "demo ready", "next: uvicorn indra.main:app --reload")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
