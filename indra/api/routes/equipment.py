"""Equipment registry and historical condition routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from indra.orchestrator.orchestrator import IndraOrchestrator, get_orchestrator

router = APIRouter(prefix="/equipment", tags=["equipment"])


@router.get("")
async def list_equipment(orchestrator: IndraOrchestrator = Depends(get_orchestrator)) -> list[dict[str, object]]:
    return [item.model_dump(mode="json") for item in await orchestrator.knowledge_graph._deps.graph.list_equipment()]


@router.get("/{tag}")
async def detail(tag: str, orchestrator: IndraOrchestrator = Depends(get_orchestrator)) -> dict[str, object]:
    item = await orchestrator.knowledge_graph._deps.graph.get_equipment(tag)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Equipment {tag.upper()} is not in the graph.")
    return item.model_dump(mode="json")


@router.get("/{tag}/history")
async def history(tag: str, orchestrator: IndraOrchestrator = Depends(get_orchestrator)) -> dict[str, object]:
    graph = orchestrator.knowledge_graph._deps.graph
    maintenance, failures = await graph.maintenance_history(tag), await graph.failure_history(tag)
    return {"maintenance": [item.model_dump(mode="json") for item in maintenance], "failures": [item.model_dump(mode="json") for item in failures]}


@router.get("/{tag}/readings")
async def readings(tag: str, orchestrator: IndraOrchestrator = Depends(get_orchestrator)) -> list[dict[str, object]]:
    values = await orchestrator.knowledge_graph._deps.graph.readings_for(tag)
    return [item.model_dump(mode="json") for item in values]
