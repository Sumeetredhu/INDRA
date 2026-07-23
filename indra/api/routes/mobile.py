"""Voice, photo, offline-bundle, and sync routes for technicians."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile

from indra.orchestrator.orchestrator import IndraOrchestrator, get_orchestrator

router = APIRouter(prefix="/mobile", tags=["mobile"])


@router.post("/voice")
async def voice(
    audio: UploadFile = File(...),
    language_hint: str | None = Form(default=None),
    equipment_tag: str | None = Form(default=None),
    orchestrator: IndraOrchestrator = Depends(get_orchestrator),
) -> dict[str, object]:
    response = await orchestrator.mobile.voice_query(await audio.read(), language_hint=language_hint, equipment_tag=equipment_tag)
    return response.model_dump(mode="json")


@router.post("/photo")
async def photo(image: UploadFile = File(...), orchestrator: IndraOrchestrator = Depends(get_orchestrator)) -> dict[str, object]:
    return (await orchestrator.mobile.photo_query(await image.read())).model_dump(mode="json")


@router.post("/offline-bundle")
async def offline_bundle(budget_bytes: int | None = None, orchestrator: IndraOrchestrator = Depends(get_orchestrator)) -> dict[str, object]:
    return await orchestrator.mobile.build_offline_bundle(budget_bytes=budget_bytes)


@router.post("/sync")
async def sync(items: list[dict[str, object]], orchestrator: IndraOrchestrator = Depends(get_orchestrator)) -> dict[str, object]:
    return await orchestrator.mobile.sync(items)
