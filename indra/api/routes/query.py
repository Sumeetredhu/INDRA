"""Grounded copilot query and streamed-response routes."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from indra.core.models import QueryRequest
from indra.orchestrator.orchestrator import IndraOrchestrator, get_orchestrator

router = APIRouter(prefix="/query", tags=["copilot"])


@router.post("/ask")
async def ask(request: QueryRequest, orchestrator: IndraOrchestrator = Depends(get_orchestrator)) -> dict[str, object]:
    return (await orchestrator.copilot.answer(request)).model_dump(mode="json")


@router.post("/classify")
async def classify(request: QueryRequest, orchestrator: IndraOrchestrator = Depends(get_orchestrator)) -> dict[str, str]:
    return {"query_type": await orchestrator.copilot.classify(request.query)}


@router.post("/stream")
async def stream(request: QueryRequest, orchestrator: IndraOrchestrator = Depends(get_orchestrator)) -> StreamingResponse:
    async def events():
        async for text in orchestrator.copilot.stream_answer(request):
            yield f"data: {json.dumps({'text': text})}\n\n"
        yield "event: done\ndata: {}\n\n"
    return StreamingResponse(events(), media_type="text/event-stream")
