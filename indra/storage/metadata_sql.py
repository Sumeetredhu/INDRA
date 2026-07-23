"""Relational metadata store: SQLAlchemy async, SQLite by default (``docs/DECISIONS.md`` D10).

Models are persisted as JSON documents rather than shredded into columns. At this scale the query
patterns are all "fetch by id" or "list recent", and JSON storage means a Pydantic model gaining a
field does not require a migration — which matters when six agents are evolving in parallel.
The columns that *are* broken out (``content_hash``, ``dedupe_key``, ``raised_at``) are exactly the
ones with an indexed lookup behind them.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    delete,
    select,
)
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from indra.core.config import Settings
from indra.core.exceptions import MetadataStoreError
from indra.core.logging import get_logger
from indra.core.models import Alert, DocumentMeta, utcnow

logger = get_logger(__name__)

metadata_obj = MetaData()

documents_table = Table(
    "documents",
    metadata_obj,
    Column("document_id", String(64), primary_key=True),
    Column("content_hash", String(64), nullable=False),
    Column("title", String(512), nullable=False, default=""),
    Column("ingested_at", DateTime(timezone=True), nullable=False),
    Column("payload", Text, nullable=False),
    Index("ix_documents_content_hash", "content_hash"),
    Index("ix_documents_ingested_at", "ingested_at"),
)

alerts_table = Table(
    "alerts",
    metadata_obj,
    Column("alert_id", String(64), primary_key=True),
    Column("dedupe_key", String(256), nullable=False, default=""),
    Column("equipment_tag", String(64), nullable=False, default=""),
    Column("severity_rank", Integer, nullable=False, default=0),
    Column("resolved", Boolean, nullable=False, default=False),
    Column("raised_at", DateTime(timezone=True), nullable=False),
    Column("payload", Text, nullable=False),
    Index("ix_alerts_dedupe", "dedupe_key", "raised_at"),
    Index("ix_alerts_resolved", "resolved", "severity_rank"),
)

sync_queue_table = Table(
    "sync_queue",
    metadata_obj,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("payload", Text, nullable=False),
)


def _aware(value: datetime | None) -> datetime:
    """SQLite drops tzinfo on round-trip; restore it so comparisons do not explode."""
    if value is None:
        return utcnow()
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


class SqlMetadataStore:
    """Implements ``contracts.MetadataStore`` over SQLAlchemy's async engine."""

    name = "metadata:sql"

    def __init__(self, engine: AsyncEngine, *, backend: str = "sqlite") -> None:
        self._engine = engine
        self._session = async_sessionmaker(engine, expire_on_commit=False)
        self.backend = backend

    @classmethod
    def from_settings(cls, settings: Settings) -> SqlMetadataStore:
        url = settings.database_url
        backend = "postgres" if url.startswith("postgresql") else "sqlite"
        engine = create_async_engine(url, echo=False, future=True)
        return cls(engine, backend=backend)

    async def init(self) -> None:
        try:
            async with self._engine.begin() as connection:
                await connection.run_sync(metadata_obj.create_all)
        except Exception as exc:  # noqa: BLE001 - surface as a typed error
            raise MetadataStoreError(
                "Could not create metadata tables. Check DATABASE_URL and that the driver is "
                "installed (aiosqlite for SQLite, asyncpg for PostgreSQL).",
                context={"url": self._engine.url.render_as_string(hide_password=True)},
                cause=exc,
            ) from exc

    # ------------------------------------------------------------------ documents

    async def save_document(self, meta: DocumentMeta) -> None:
        async with self._session() as session, session.begin():
            await session.execute(
                delete(documents_table).where(documents_table.c.document_id == meta.document_id)
            )
            await session.execute(
                documents_table.insert().values(
                    document_id=meta.document_id,
                    content_hash=meta.content_hash,
                    title=meta.title[:512],
                    ingested_at=_aware(meta.ingested_at),
                    payload=meta.model_dump_json(),
                )
            )

    async def get_document(self, document_id: str) -> DocumentMeta | None:
        async with self._session() as session:
            row = (
                await session.execute(
                    select(documents_table.c.payload).where(
                        documents_table.c.document_id == document_id
                    )
                )
            ).scalar_one_or_none()
        return DocumentMeta.model_validate_json(row) if row else None

    async def find_by_hash(self, content_hash: str) -> DocumentMeta | None:
        async with self._session() as session:
            row = (
                await session.execute(
                    select(documents_table.c.payload)
                    .where(documents_table.c.content_hash == content_hash)
                    .order_by(documents_table.c.ingested_at.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
        return DocumentMeta.model_validate_json(row) if row else None

    async def list_documents(self, *, limit: int = 100, offset: int = 0) -> list[DocumentMeta]:
        async with self._session() as session:
            rows = (
                await session.execute(
                    select(documents_table.c.payload)
                    .order_by(documents_table.c.ingested_at.desc())
                    .limit(limit)
                    .offset(offset)
                )
            ).scalars().all()
        return [DocumentMeta.model_validate_json(row) for row in rows]

    # ------------------------------------------------------------------ alerts

    async def save_alert(self, alert: Alert) -> None:
        async with self._session() as session, session.begin():
            await session.execute(delete(alerts_table).where(alerts_table.c.alert_id == alert.alert_id))
            await session.execute(
                alerts_table.insert().values(
                    alert_id=alert.alert_id,
                    dedupe_key=alert.dedupe_key,
                    equipment_tag=alert.equipment_tag,
                    severity_rank=alert.severity.rank,
                    resolved=alert.resolved,
                    raised_at=_aware(alert.raised_at),
                    payload=alert.model_dump_json(),
                )
            )

    async def list_alerts(self, *, unresolved_only: bool = True, limit: int = 100) -> list[Alert]:
        statement = select(alerts_table.c.payload)
        if unresolved_only:
            statement = statement.where(alerts_table.c.resolved.is_(False))
        statement = statement.order_by(
            alerts_table.c.severity_rank.desc(), alerts_table.c.raised_at.desc()
        ).limit(limit)
        async with self._session() as session:
            rows = (await session.execute(statement)).scalars().all()
        return [Alert.model_validate_json(row) for row in rows]

    async def find_alert_by_dedupe_key(self, dedupe_key: str, *, within_seconds: int) -> Alert | None:
        cutoff = utcnow() - timedelta(seconds=within_seconds)
        async with self._session() as session:
            row = (
                await session.execute(
                    select(alerts_table.c.payload)
                    .where(alerts_table.c.dedupe_key == dedupe_key)
                    .where(alerts_table.c.raised_at >= cutoff)
                    .order_by(alerts_table.c.raised_at.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
        return Alert.model_validate_json(row) if row else None

    # ------------------------------------------------------------------ sync queue

    async def enqueue_sync(self, item: dict[str, Any]) -> None:
        import json

        async with self._session() as session, session.begin():
            await session.execute(
                sync_queue_table.insert().values(
                    created_at=utcnow(), payload=json.dumps(item, default=str)
                )
            )

    async def drain_sync(self, *, limit: int = 100) -> list[dict[str, Any]]:
        import json

        async with self._session() as session, session.begin():
            rows = (
                await session.execute(
                    select(sync_queue_table.c.id, sync_queue_table.c.payload)
                    .order_by(sync_queue_table.c.id)
                    .limit(limit)
                )
            ).all()
            if rows:
                await session.execute(
                    delete(sync_queue_table).where(
                        sync_queue_table.c.id.in_([row[0] for row in rows])
                    )
                )
        return [json.loads(row[1]) for row in rows]

    async def health(self) -> dict[str, Any]:
        try:
            async with self._session() as session:
                count = (
                    await session.execute(select(documents_table.c.document_id).limit(1))
                ).first()
            return {
                "ok": True,
                "backend": self.backend,
                "detail": f"{self._engine.url.render_as_string(hide_password=True)}"
                          f"{' (empty)' if count is None else ''}",
            }
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "backend": self.backend, "detail": str(exc)}

    async def close(self) -> None:
        await self._engine.dispose()


__all__ = ["SqlMetadataStore", "alerts_table", "documents_table", "metadata_obj", "sync_queue_table"]
