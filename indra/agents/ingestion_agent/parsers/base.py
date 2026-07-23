"""Shared parser plumbing.

Every format parser produces the same three things: **blocks** (text with provenance attached),
**tables** (structured rows kept structured), and **warnings** (what could not be read). This module
defines that shape and the small amount of behaviour every parser shares.

Why parsers return :class:`ParseOutcome` and not :class:`~indra.core.models.ParsedDocument`
directly: chunking is a *pipeline stage*, not a parsing concern, and the chunker needs
:class:`~indra.agents.ingestion_agent.chunking.TextBlock` objects — page numbers, section headings
and per-block OCR confidence — which a flat string has already thrown away. The protocol method
:meth:`BaseParser.parse` still exists and still returns a ``ParsedDocument``, so
:class:`indra.core.contracts.DocumentParser` is satisfied for any caller that just wants a document.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Sequence

from indra.agents.ingestion_agent.chunking import TextBlock, iter_block_text
from indra.core.config import Settings, get_settings
from indra.core.logging import get_logger
from indra.core.models import (
    DocumentMeta,
    IngestionStage,
    MimeFamily,
    ParsedDocument,
    PIDParseResult,
)

logger = get_logger(__name__)

MAX_TABLE_ROWS: Final[int] = 500
"""Rows kept per table in :attr:`ParsedDocument.tables`.

A 40 000-row export is data, not a document: past this point the rows are still chunked as text but
the structured copy is truncated so that a single spreadsheet cannot blow out the payload the
Knowledge Graph Agent has to serialise.
"""

MAX_CELL_CHARS: Final[int] = 400
"""Cells longer than this are truncated. Real table cells are short; longer means a parse artefact."""


@dataclass(slots=True)
class ParseOutcome:
    """What a parser extracted, before chunking.

    Attributes:
        blocks: Ordered text blocks carrying page, section, kind and OCR confidence.
        tables: Structured tables, shaped by :func:`make_table`.
        warnings: Human-readable notes about anything that degraded.
        pid_result: Populated only by the P&ID vision path.
        page_count: Pages/sheets/parts, when the format has a meaningful count.
        ocr_confidence: Mean OCR confidence across the document, ``None`` when no OCR ran.
        extra: Format-specific metadata folded into ``DocumentMeta.extra``.
    """

    blocks: list[TextBlock] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    pid_result: PIDParseResult | None = None
    page_count: int | None = None
    ocr_confidence: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def add(self, block: TextBlock | None) -> None:
        """Append ``block`` when it carries any text at all."""
        if block is not None and block.text and block.text.strip():
            self.blocks.append(block)

    def warn(self, message: str) -> None:
        """Record a degradation. Deduplicated, because a per-page failure repeats N times."""
        if message and message not in self.warnings:
            self.warnings.append(message)

    @property
    def is_empty(self) -> bool:
        return not any(block.text.strip() for block in self.blocks)


def make_table(
    rows: Sequence[Sequence[object]],
    *,
    name: str,
    page: int | None = None,
    index: int = 0,
    source: str = "",
    header_row: bool = True,
) -> dict[str, Any] | None:
    """Normalise raw rows into the dict shape carried on :attr:`ParsedDocument.tables`.

    Returns ``None`` when the table has no usable content, so callers can filter with a walrus.
    """
    cleaned: list[list[str]] = []
    for row in rows:
        cells = [
            "" if cell is None else str(cell).replace("\r", " ").replace("\n", " ").strip()[:MAX_CELL_CHARS]
            for cell in row
        ]
        if any(cells):
            cleaned.append(cells)
    if not cleaned:
        return None

    width = max(len(row) for row in cleaned)
    padded = [row + [""] * (width - len(row)) for row in cleaned]
    truncated = len(padded) > MAX_TABLE_ROWS
    if header_row and len(padded) > 1:
        columns = [cell or f"col_{position}" for position, cell in enumerate(padded[0])]
        body = padded[1 : MAX_TABLE_ROWS + 1]
    else:
        columns = [f"col_{position}" for position in range(width)]
        body = padded[:MAX_TABLE_ROWS]

    return {
        "name": name,
        "index": index,
        "page": page,
        "columns": columns,
        "rows": body,
        "row_count": len(padded) - (1 if header_row and len(padded) > 1 else 0),
        "truncated": truncated,
        "source": source,
    }


class BaseParser:
    """Common behaviour for every registered parser.

    Subclasses set :attr:`name`, :attr:`families` and (optionally) :attr:`extensions`, then implement
    :meth:`parse_document`. They must not raise for content-level problems — a page that will not
    render is a warning on the outcome, not a failed ingestion (CLAUDE.md rule 6). Only a genuinely
    unreadable *file* justifies raising :class:`~indra.core.exceptions.ParsingError`.
    """

    name: str = "base"
    families: tuple[MimeFamily, ...] = ()
    extensions: frozenset[str] = frozenset()

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    @property
    def settings(self) -> Settings:
        return self._settings

    # -- contracts.DocumentParser -----------------------------------------------------
    def claims(self, *, filename: str, mime_family: MimeFamily, head: bytes) -> bool:
        """Claim by sniffed family first, by extension only as a secondary signal."""
        if mime_family in self.families:
            return True
        if not self.extensions:
            return False
        return Path(filename).suffix.lower() in self.extensions

    async def parse(self, path: Path, meta: DocumentMeta) -> ParsedDocument:
        """Protocol entry point: parse ``path`` into a :class:`ParsedDocument` without chunking.

        The ingestion service uses :meth:`parse_document` instead, because it needs the blocks to
        chunk them; this method exists so any other caller can treat a parser as a black box.
        """
        started = time.perf_counter()
        outcome = await self.parse_document(path, meta)
        return self.to_document(outcome, meta, duration_ms=(time.perf_counter() - started) * 1000.0)

    async def parse_document(self, path: Path, meta: DocumentMeta) -> ParseOutcome:
        """Extract blocks, tables and warnings from ``path``. Implemented by every subclass."""
        raise NotImplementedError(f"{type(self).__name__} does not implement parse_document")

    # -- helpers ----------------------------------------------------------------------
    @staticmethod
    def to_document(
        outcome: ParseOutcome,
        meta: DocumentMeta,
        *,
        duration_ms: float = 0.0,
    ) -> ParsedDocument:
        """Fold a :class:`ParseOutcome` into a :class:`ParsedDocument`.

        ``text`` here is the block-joined rendition. The ingestion service overwrites it with the
        chunker's output so that :attr:`Chunk.char_start` offsets index into the *same* string.
        """
        updates: dict[str, Any] = {}
        if outcome.page_count is not None and outcome.page_count != meta.page_count:
            updates["page_count"] = outcome.page_count
        if outcome.extra:
            merged = dict(meta.extra)
            merged.update(outcome.extra)
            updates["extra"] = merged
        enriched = meta.model_copy(update=updates) if updates else meta

        return ParsedDocument(
            meta=enriched,
            text=iter_block_text(outcome.blocks),
            tables=outcome.tables,
            pid_result=outcome.pid_result,
            warnings=list(outcome.warnings),
            stage=IngestionStage.PARSED,
            parse_duration_ms=round(duration_ms, 2),
        )

    def describe(self) -> dict[str, Any]:
        """Registry/health summary."""
        return {
            "name": self.name,
            "families": [family.value for family in self.families],
            "extensions": sorted(self.extensions),
        }


__all__ = [
    "MAX_CELL_CHARS",
    "MAX_TABLE_ROWS",
    "BaseParser",
    "ParseOutcome",
    "make_table",
]
