"""INDRA FastAPI application and lifespan wiring."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from indra.api.middleware import RequestContextMiddleware, SecurityMiddleware
from indra.api.routes import alerts, compliance, equipment, graph, ingestion, mobile, query, system
from indra.core.config import Settings, get_settings
from indra.core.exceptions import IndraError
from indra.core.logging import get_logger
from indra.orchestrator.orchestrator import IndraOrchestrator

logger = get_logger(__name__)


async def _bootstrap_demo_corpus(orchestrator: IndraOrchestrator, settings: Settings) -> None:
    """Ingest the bundled demo corpus at startup so a fresh deployment is not an empty shell.

    A public URL that returns ``[]`` for every endpoint looks broken even when it is perfectly
    healthy. Enabled with ``INDRA_BOOTSTRAP_DEMO=true`` (see ``render.yaml``).

    Never fatal: any failure is logged and the API serves anyway, because an empty instance beats
    one that refuses to boot.
    """
    from pathlib import Path

    ingestible = {".pdf", ".png", ".xlsx", ".docx", ".csv", ".txt"}
    demo_dir: Path = settings.demo_dir
    try:
        if not demo_dir.is_dir() or not any(demo_dir.iterdir()):
            from scripts.generate_demo_data import main as generate

            logger.info("no demo corpus present; generating it")
            generate()

        files = sorted(
            p for p in demo_dir.glob("*")
            if p.suffix.lower() in ingestible
            # Skip the degraded scanned P&ID: it is the single heaviest file and exists only to
            # exercise OCR tag-correction. A live backend gets P-101's connectivity from the clean
            # drawing, so dropping it here keeps peak memory under a 512 MB free instance's ceiling.
            and "scanned" not in p.stem.lower()
        )
        ingested = 0
        for path in files:
            try:
                await orchestrator.ingestion.ingest_path(path)
                ingested += 1
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop the rest
                logger.warning("bootstrap ingest failed",
                               extra={"file": path.name, "error": f"{type(exc).__name__}: {exc}"})
        logger.info("demo corpus bootstrapped", extra={"ingested": ingested, "found": len(files)})
    except Exception as exc:  # noqa: BLE001 - bootstrap must never block startup
        logger.warning("demo bootstrap skipped", extra={"error": f"{type(exc).__name__}: {exc}"})


def create_app(runtime_settings: Settings | None = None) -> FastAPI:
    """Create an INDRA application with explicit settings for production and test lifecycles."""
    active_settings = runtime_settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Start the orchestrator once and close it only after in-flight requests finish."""
        orchestrator = IndraOrchestrator(active_settings)
        app.state.orchestrator = orchestrator
        await orchestrator.startup()

        bootstrap_task: asyncio.Task[None] | None = None
        if active_settings.bootstrap_demo:
            # Fire-and-forget, NOT awaited: a synchronous ingest here blocks lifespan startup, so
            # uvicorn never reports "startup complete" and the platform health check times out
            # before the corpus finishes loading. Running it as a background task lets /health
            # answer immediately; the graph populates a minute or so later. It still runs in *this*
            # process, which the in-memory backend requires — a separate process would ingest into
            # its own graph and exit.
            bootstrap_task = asyncio.create_task(_bootstrap_demo_corpus(orchestrator, active_settings))

        try:
            yield
        finally:
            if bootstrap_task is not None and not bootstrap_task.done():
                bootstrap_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await bootstrap_task
            await orchestrator.shutdown()

    application = FastAPI(
        title="INDRA - Industrial Neural Data & Reasoning Assistant",
        version="1.0.0",
        description="A proactive, explainable industrial intelligence layer.",
        lifespan=lifespan,
    )
    application.add_middleware(RequestContextMiddleware)
    application.add_middleware(SecurityMiddleware, settings=active_settings)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=active_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.exception_handler(IndraError)
    async def indra_error(_: Request, exc: IndraError) -> JSONResponse:
        logger.warning("handled INDRA error", extra={"error_code": exc.error_code, "detail": exc.message})
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict())

    @application.exception_handler(Exception)
    async def unhandled_error(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled API exception", extra={"error": str(exc)})
        return JSONResponse(
            status_code=500,
            content={
                "error_code": "internal_error",
                "message": "An unexpected INDRA error occurred. Check the correlation id in the response headers.",
            },
        )

    @application.get("/health", tags=["system"])
    async def root_health() -> dict[str, str]:
        return {"status": "ok", "service": "INDRA"}

    for route in (
        system.router,
        ingestion.router,
        query.router,
        graph.router,
        equipment.router,
        alerts.router,
        mobile.router,
        compliance.router,
    ):
        application.include_router(route, prefix=active_settings.api_prefix)
    return application


app = create_app()


__all__ = ["app", "create_app"]
