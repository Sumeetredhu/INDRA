"""System health, backend status, and lightweight metrics."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from indra.orchestrator.orchestrator import IndraOrchestrator, get_orchestrator

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/health")
async def health(orchestrator: IndraOrchestrator = Depends(get_orchestrator)) -> dict[str, object]:
    return await orchestrator.health()


@router.get("/metrics")
async def metrics(orchestrator: IndraOrchestrator = Depends(get_orchestrator)) -> dict[str, object]:
    stores = await orchestrator.health()
    graph = await orchestrator.knowledge_graph._deps.graph.stats()
    return {"health": stores, "graph": graph, "llm_usage": orchestrator.knowledge_graph._deps.llm.usage()}


@router.get("/config")
async def config(orchestrator: IndraOrchestrator = Depends(get_orchestrator)) -> dict[str, object]:
    settings = orchestrator._settings
    return {"environment": settings.environment.value, "api_prefix": settings.api_prefix, "deterministic": settings.deterministic, "offline_mode": settings.offline_mode, "storage_backend": settings.storage_backend.value}
