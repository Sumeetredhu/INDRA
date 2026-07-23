"""Upload and ingestion-result routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, UploadFile

from indra.orchestrator.orchestrator import IndraOrchestrator, get_orchestrator

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("/upload")
async def upload(
    file: UploadFile = File(...), orchestrator: IndraOrchestrator = Depends(get_orchestrator)
) -> dict[str, object]:
    content = await file.read()
    result = await orchestrator.ingestion.ingest_bytes(content, filename=file.filename or "upload")
    return result.model_dump(mode="json")


@router.post("/batch")
async def batch(
    files: list[UploadFile] = File(...), orchestrator: IndraOrchestrator = Depends(get_orchestrator)
) -> list[dict[str, object]]:
    results = []
    for file in files:
        result = await orchestrator.ingestion.ingest_bytes(await file.read(), filename=file.filename or "upload")
        results.append(result.model_dump(mode="json"))
    return results
