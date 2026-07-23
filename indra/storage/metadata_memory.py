"""In-process metadata store: documents, alerts, and the offline sync queue."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from indra.core.logging import get_logger
from indra.core.models import Alert, DocumentMeta, utcnow

logger = get_logger(__name__)


class MemoryMetadataStore:
    """Implements ``contracts.MetadataStore`` with dictionaries."""

    name = "metadata:memory"
    backend = "memory"

    def __init__(self) -> None:
        self._documents: dict[str, DocumentMeta] = {}
        self._by_hash: dict[str, str] = {}
        self._alerts: dict[str, Alert] = {}
        self._sync: list[dict[str, Any]] = []

    async def init(self) -> None:
        return None

    # ------------------------------------------------------------------ documents

    async def save_document(self, meta: DocumentMeta) -> None:
        self._documents[meta.document_id] = meta
        self._by_hash[meta.content_hash] = meta.document_id

    async def get_document(self, document_id: str) -> DocumentMeta | None:
        return self._documents.get(document_id)

    async def find_by_hash(self, content_hash: str) -> DocumentMeta | None:
        document_id = self._by_hash.get(content_hash)
        return self._documents.get(document_id) if document_id else None

    async def list_documents(self, *, limit: int = 100, offset: int = 0) -> list[DocumentMeta]:
        items = sorted(self._documents.values(), key=lambda m: m.ingested_at, reverse=True)
        return items[offset : offset + limit]

    # ------------------------------------------------------------------ alerts

    async def save_alert(self, alert: Alert) -> None:
        self._alerts[alert.alert_id] = alert

    async def list_alerts(self, *, unresolved_only: bool = True, limit: int = 100) -> list[Alert]:
        items = list(self._alerts.values())
        if unresolved_only:
            items = [a for a in items if not a.resolved]
        items.sort(key=lambda a: (-a.severity.rank, a.raised_at), reverse=False)
        items.sort(key=lambda a: a.severity.rank, reverse=True)
        return items[:limit]

    async def find_alert_by_dedupe_key(self, dedupe_key: str, *, within_seconds: int) -> Alert | None:
        cutoff = utcnow() - timedelta(seconds=within_seconds)
        candidates = [
            a for a in self._alerts.values()
            if a.dedupe_key == dedupe_key and a.raised_at >= cutoff
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda a: a.raised_at)

    # ------------------------------------------------------------------ sync queue

    async def enqueue_sync(self, item: dict[str, Any]) -> None:
        self._sync.append(item)

    async def drain_sync(self, *, limit: int = 100) -> list[dict[str, Any]]:
        taken = self._sync[:limit]
        del self._sync[: len(taken)]
        return taken

    async def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "backend": "memory",
            "detail": f"{len(self._documents)} documents, {len(self._alerts)} alerts",
        }

    async def close(self) -> None:
        return None


__all__ = ["MemoryMetadataStore"]
