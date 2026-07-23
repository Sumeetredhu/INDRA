"""Capture a live API session into a static fixture the deployed frontend can serve from.

Why this exists
---------------
The GitHub Pages build is a static bundle: there is no Python process behind it. Without a fixture
the public link would render an "API unreachable" banner, which is worse than no link at all.

So we drive the *real* API once — real ingestion, real GraphRAG retrieval, real rule evaluation,
real compliance assessment — and record every response. The frontend falls back to this recording
when no live backend answers, and labels itself READ-ONLY so nobody mistakes a recording for a
running system.

Nothing here is hand-authored. If the pipeline regresses, the snapshot regresses with it.

Usage:
    python -m uvicorn indra.main:app --port 8000      # in one terminal
    python -m scripts.build_demo_snapshot             # in another
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import httpx

from indra.core.logging import configure_logging, get_logger

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = REPO_ROOT / "data" / "demo"
OUTPUT = REPO_ROOT / "frontend" / "public" / "demo-snapshot.json"

#: The questions the console offers as one-click chips. Each is answered for real and recorded.
QUESTIONS: tuple[str, ...] = (
    "Why did P-101 fail last month?",
    "What is the OEM bearing wear limit for P-101?",
    "How do I replace the P-101 bearing?",
    "Will P-101 fail in the next 30 days?",
    "Are we compliant with Factory Act Section 41(b)?",
    "What don't we know about P-101?",
)

VOICE_PHRASES: tuple[tuple[str, str], ...] = (
    ("P-101 ka kya haal hai?", "hi"),
    ("What is the status of P-101?", "en"),
    ("P-101 ची स्थिती काय आहे?", "mr"),
    ("P-101 நிலை என்ன?", "ta"),
)


def _upload_corpus(client: httpx.Client) -> list[dict[str, Any]]:
    """Ingest every demo document through the running API, recording each result."""
    results: list[dict[str, Any]] = []
    files = sorted(
        [p for p in DEMO_DIR.iterdir() if p.suffix.lower() in {".pdf", ".png", ".xlsx"}],
        key=lambda p: p.name,
    )
    if not files:
        raise SystemExit(
            f"No demo documents in {DEMO_DIR}. Run `python -m scripts.generate_demo_data` first."
        )
    for path in files:
        response = client.post(
            "/api/v1/ingest/upload",
            files={"file": (path.name, path.read_bytes(), "application/octet-stream")},
            timeout=180.0,
        )
        response.raise_for_status()
        payload = response.json()
        results.append(payload)
        logger.info(
            "captured ingestion",
            extra={
                "file": path.name,
                "chunks": payload.get("chunks_created"),
                "entities": payload.get("entities_created"),
                "pid_symbols": payload.get("pid_symbols"),
            },
        )
    return results


def _get(client: httpx.Client, path: str) -> Any:
    response = client.get(path, timeout=120.0)
    response.raise_for_status()
    return response.json()


def _post(client: httpx.Client, path: str, payload: Any = None) -> Any:
    response = client.post(path, json=payload, timeout=180.0) if payload is not None \
        else client.post(path, timeout=180.0)
    response.raise_for_status()
    return response.json()


def build(base_url: str) -> dict[str, Any]:
    """Drive the live API and return the complete recording."""
    with httpx.Client(base_url=base_url.rstrip("/")) as client:
        try:
            client.get("/health", timeout=10.0).raise_for_status()
        except httpx.HTTPError as exc:
            raise SystemExit(
                f"No INDRA API at {base_url}. Start it with:\n"
                "  python -m uvicorn indra.main:app --port 8000"
            ) from exc

        logger.info("ingesting demo corpus through the live API")
        ingestion = _upload_corpus(client)

        logger.info("capturing answers")
        answers = {}
        for question in QUESTIONS:
            answers[question] = _post(
                client,
                "/api/v1/query/ask",
                {
                    "query": question, "language": "en", "equipment_tag": None,
                    "max_sources": 8, "include_graph_preview": True,
                    "include_cypher": True, "channel": "web",
                },
            )
            logger.info("captured answer", extra={"query": question[:44]})

        logger.info("capturing voice round trips")
        voice = {}
        for transcript, language in VOICE_PHRASES:
            try:
                # Multipart, matching the route: an `audio` file plus a `language_hint` form field.
                # With Whisper absent the backend decodes the bytes as text, so the rest of the
                # pipeline — detection, tag masking, translation, copilot — runs for real.
                response = client.post(
                    "/api/v1/mobile/voice",
                    files={"audio": ("transcript.txt", transcript.encode("utf-8"), "text/plain")},
                    data={"language_hint": language},
                    timeout=180.0,
                )
                response.raise_for_status()
                voice[transcript] = response.json()
                logger.info("captured voice round trip", extra={"language": language})
            except httpx.HTTPError as exc:  # a modality may be unavailable; record the absence
                logger.warning("voice capture failed", extra={"lang": language, "error": str(exc)})

        logger.info("capturing graph, alerts and compliance")
        equipment = _get(client, "/api/v1/equipment")
        tags = [e["tag"] for e in equipment][:40]

        previews = {}
        for tag in ["P-101", "V-201", "E-301"]:
            try:
                previews[tag] = _get(
                    client, f"/api/v1/graph/preview?keys=Equipment:{tag}&hops=2"
                )
            except httpx.HTTPError:
                logger.warning("preview unavailable", extra={"tag": tag})

        snapshot: dict[str, Any] = {
            "generated_by": "scripts/build_demo_snapshot.py",
            "read_only": True,
            "config": _get(client, "/api/v1/system/config"),
            "health": _get(client, "/api/v1/system/health"),
            "graph_stats": _get(client, "/api/v1/graph/stats"),
            "graph_previews": previews,
            "equipment": equipment,
            "alerts": _get(client, "/api/v1/alerts?unresolved_only=true"),
            "knowledge_cliff": _get(client, "/api/v1/alerts/knowledge-cliff"),
            "compliance_gaps": _post(client, "/api/v1/compliance/audit", {}),
            "answers": answers,
            "voice": voice,
            "ingestion": ingestion,
            "equipment_tags": tags,
        }
        return snapshot


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    snapshot = build(args.base_url)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    size_kb = args.output.stat().st_size / 1024
    print(f"\n  snapshot written : {args.output}")
    print(f"  size             : {size_kb:,.0f} KB")
    print(f"  documents        : {len(snapshot['ingestion'])}")
    print(f"  graph nodes      : {snapshot['graph_stats'].get('nodes')}")
    print(f"  alerts           : {len(snapshot['alerts'])}")
    print(f"  compliance gaps  : {len(snapshot['compliance_gaps'])}")
    print(f"  answers recorded : {len(snapshot['answers'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
