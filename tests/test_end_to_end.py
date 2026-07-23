"""Product-level checks: upload, graph reasoning, compliance, and API boundaries."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from indra.core.models import QueryRequest
from indra.main import create_app
from indra.orchestrator.orchestrator import IndraOrchestrator
from tests.conftest import make_test_settings

_P101_RECORD = b"""P-101 centrifugal pump inspection report dated 2024-03-15.
P-101 is a critical process pump. Bearing wear measured 88% during routine inspection.
Vibration measured 9.6 mm/s and the operator reported rising temperature.
OEM manual requires bearing replacement at 90% wear. Keep the pump under close observation.
"""


async def test_memory_workflow_is_grounded_and_exports_audit_package(
    orchestrator: IndraOrchestrator,
) -> None:
    """The supported no-key/no-database path still produces sourced operational results."""
    first = await orchestrator.ingestion.ingest_bytes(_P101_RECORD, filename="Inspection_2024_0315_P-101.txt")
    duplicate = await orchestrator.ingestion.ingest_bytes(_P101_RECORD, filename="Inspection_2024_0315_P-101-copy.txt")

    assert first.chunks_created >= 1
    assert first.entities_created >= 1
    assert duplicate.duplicate_of == first.document.document_id

    answer = await orchestrator.copilot.answer(
        QueryRequest(query="Why is P-101 at risk of bearing failure?", equipment_tag="P-101")
    )
    assert answer.sources
    assert answer.reasoning_chain
    assert "P-101" in answer.answer_text

    prediction = await orchestrator.proactive.predict("P-101", horizon_days=30)
    assert prediction.equipment_tag == "P-101"
    assert 0.0 <= prediction.probability <= 1.0

    package = await orchestrator.compliance.build_package(tags=["P-101"])
    pdf_path = await orchestrator.compliance.export_pdf(package)
    assert pdf_path.exists()
    assert pdf_path.read_bytes().startswith(b"%PDF")


def test_api_lifespan_uploads_and_answers_without_external_services(tmp_path: Path) -> None:
    """FastAPI owns the same memory-mode lifecycle used by local demo deployments."""
    app = create_app(make_test_settings(tmp_path))
    with TestClient(app) as client:
        health = client.get("/api/v1/system/health")
        assert health.status_code == 200
        assert health.json()["ok"] is True

        uploaded = client.post(
            "/api/v1/ingest/upload",
            files={"file": ("Inspection_2024_0315_P-101.txt", _P101_RECORD, "text/plain")},
        )
        assert uploaded.status_code == 200, uploaded.text
        assert uploaded.json()["chunks_created"] >= 1

        answered = client.post(
            "/api/v1/query/ask",
            json={"query": "What should we do for P-101?", "equipment_tag": "P-101"},
        )
        assert answered.status_code == 200, answered.text
        payload = answered.json()
        assert payload["sources"]
        assert payload["answer_text"]

        audited = client.post("/api/v1/compliance/audit", json={"tags": ["P-101"]})
        assert audited.status_code == 200, audited.text
        assert isinstance(audited.json(), list)
