"""Proactive alert, prediction, and knowledge-cliff routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from indra.orchestrator.orchestrator import IndraOrchestrator, get_orchestrator

router = APIRouter(prefix="/alerts", tags=["proactive intelligence"])


@router.get("")
async def list_alerts(unresolved_only: bool = True, orchestrator: IndraOrchestrator = Depends(get_orchestrator)) -> list[dict[str, object]]:
    return [item.model_dump(mode="json") for item in await orchestrator.proactive.alerts(unresolved_only=unresolved_only)]


@router.post("/scan")
async def scan(tags: list[str] | None = None, orchestrator: IndraOrchestrator = Depends(get_orchestrator)) -> list[dict[str, object]]:
    return [item.model_dump(mode="json") for item in await orchestrator.proactive.scan(equipment_tags=tags)]


@router.get("/predict/{tag}")
async def predict(tag: str, horizon_days: int = 30, orchestrator: IndraOrchestrator = Depends(get_orchestrator)) -> dict[str, object]:
    return (await orchestrator.proactive.predict(tag, horizon_days=horizon_days)).model_dump(mode="json")


@router.get("/knowledge-cliff")
async def knowledge_cliff(tags: list[str] | None = None, orchestrator: IndraOrchestrator = Depends(get_orchestrator)) -> list[dict[str, object]]:
    return [item.model_dump(mode="json") for item in await orchestrator.proactive.knowledge_cliff(tags=tags)]
