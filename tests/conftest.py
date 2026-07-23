"""Deterministic, no-network fixtures for INDRA tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from indra.core.config import Environment, Settings, StorageBackend
from indra.orchestrator.orchestrator import IndraOrchestrator


def make_test_settings(root: Path) -> Settings:
    """Return an isolated in-memory configuration rooted in pytest's temporary directory."""
    data_dir = root / "data"
    cache_dir = root / "cache"
    return Settings(
        environment=Environment.TEST,
        demo_mode=True,
        storage_backend=StorageBackend.MEMORY,
        data_dir=data_dir,
        raw_dir=data_dir / "raw",
        processed_dir=data_dir / "processed",
        demo_dir=data_dir / "demo",
        cache_dir=cache_dir,
        chroma_dir=cache_dir / "chroma",
        export_dir=data_dir / "exports",
    )


@pytest.fixture
async def orchestrator(tmp_path: Path) -> AsyncIterator[IndraOrchestrator]:
    """A fully wired INDRA system using only its supported memory fallbacks."""
    service = IndraOrchestrator(make_test_settings(tmp_path))
    await service.startup()
    try:
        yield service
    finally:
        await service.shutdown()
