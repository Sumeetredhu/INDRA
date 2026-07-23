"""Graph inspection routes for the explainability panel."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from indra.orchestrator.orchestrator import IndraOrchestrator, get_orchestrator

router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("/stats")
async def stats(orchestrator: IndraOrchestrator = Depends(get_orchestrator)) -> dict[str, int]:
    return await orchestrator.knowledge_graph._deps.graph.stats()


@router.get("/preview")
async def preview(keys: list[str] = Query(...), hops: int = 2, orchestrator: IndraOrchestrator = Depends(get_orchestrator)) -> dict[str, object]:
    return await orchestrator.knowledge_graph.graph_preview(keys, hops=hops)


@router.get("/neighbours/{key}")
async def neighbours(key: str, hops: int = 1, orchestrator: IndraOrchestrator = Depends(get_orchestrator)) -> list[dict[str, object]]:
    return [item.model_dump(mode="json") for item in await orchestrator.knowledge_graph._deps.graph.neighbours(key, hops=hops)]
