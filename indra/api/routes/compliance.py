"""Regulatory audit and one-click PDF package routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from indra.orchestrator.orchestrator import IndraOrchestrator, get_orchestrator

router = APIRouter(prefix="/compliance", tags=["compliance"])


class AuditRequest(BaseModel):
    """Explicit JSON body for a compliance scope, shared by web and mobile clients."""

    tags: list[str] | None = None
    regulations: list[str] | None = None


class PackageRequest(AuditRequest):
    """A package must name at least one equipment tag to remain audit-scoped."""

    tags: list[str] = Field(min_length=1)


@router.post("/audit")
async def audit(
    request: AuditRequest,
    orchestrator: IndraOrchestrator = Depends(get_orchestrator),
) -> list[dict[str, object]]:
    return [
        item.model_dump(mode="json")
        for item in await orchestrator.compliance.audit(tags=request.tags, regulations=request.regulations)
    ]


@router.post("/package")
async def package(
    request: PackageRequest,
    orchestrator: IndraOrchestrator = Depends(get_orchestrator),
) -> dict[str, object]:
    return (
        await orchestrator.compliance.build_package(tags=request.tags, regulations=request.regulations)
    ).model_dump(mode="json")


@router.post("/package/pdf")
async def package_pdf(
    request: PackageRequest,
    orchestrator: IndraOrchestrator = Depends(get_orchestrator),
) -> FileResponse:
    package = await orchestrator.compliance.build_package(
        tags=request.tags, regulations=request.regulations
    )
    path = await orchestrator.compliance.export_pdf(package)
    return FileResponse(path, media_type="application/pdf", filename=path.name)
