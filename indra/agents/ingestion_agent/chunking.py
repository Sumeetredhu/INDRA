"""Semantic chunking: the unit of retrieval is a passage, not a page.

Three properties matter downstream and each is enforced here:

* **Never split mid-sentence.** A chunk that ends "the bearing wear reached 87% before the" is
  useless as evidence and worse as a citation. Packing happens at sentence granularity.
* **Provenance survives.** Page number, section heading, character offsets and OCR confidence ride
  on every :class:`~indra.core.models.Chunk`, which is what makes ``SourceRef`` — and therefore the
  "Explain How I Know This" panel — a projection of stored data rather than a second guess.
* **Token counts are real.** ``tiktoken`` when it is loadable, a calibrated word-and-punctuation
  heuristic when it is not. The heuristic is used only as a fallback and logs once when it engages,
  because silently miscounting tokens silently overflows the context window.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Final, Iterable, Literal, Sequence

from indra.core.config import Settings, get_settings
from indra.core.ids import chunk_id as make_chunk_id
from indra.core.logging import get_logger
from indra.core.models import Chunk

logger = get_logger(__name__)

BlockKind = Literal["paragraph", "heading", "table", "caption", "metadata", "list"]

# --------------------------------------------------------------------------------------
# Sentence segmentation
#
# A dependency-free segmenter beats a lazy `.split(".")` and avoids pulling in spaCy for what is,
# in plant documents, a well-behaved problem: the hard cases are abbreviations, decimal numbers,
# and tag/standard references like "IS 2062" or "P-101." at the end of a line.
# --------------------------------------------------------------------------------------

_SENTENCE_BOUNDARY: Final[re.Pattern[str]] = re.compile(r"(?<=[.!?;])[\s ]+(?=[^\s])")

_ABBREVIATIONS: Final[frozenset[str]] = frozenset(
    {
        "mr", "mrs", "ms", "dr", "er", "shri", "smt", "sr", "jr", "prof",
        "no", "nos", "fig", "figs", "eq", "ref", "rev", "sec", "cl", "vol", "ch", "pt",
        "approx", "dept", "dia", "assy", "incl", "excl", "min", "max", "avg", "temp", "std",
        "qty", "wt", "hrs", "hr", "sq", "ft", "in", "mm", "cm", "kg", "psi", "rpm", "vs",
        "etc", "i.e", "e.g", "viz", "cf", "al", "st", "ltd", "pvt", "co", "inc", "corp",
        "u.s", "u.k", "a.m", "p.m", "no.s",
    }
)

_LIST_MARKER: Final[re.Pattern[str]] = re.compile(r"^\s*(?:[-*•●▪]|\(?\d{1,2}[.)]|[a-z][.)])\s+")
_HEADING_NUMBER: Final[re.Pattern[str]] = re.compile(r"^\s*\d+(?:\.\d+)*[.)]?\s+\S")
_WORDS: Final[re.Pattern[str]] = re.compile(r"\w+|[^\w\s]")

_HEADING_MAX_CHARS: Final[int] = 90
"""Longer than this and it is a sentence that happens to lack a full stop, not a heading."""

_HEURISTIC_TOKENS_PER_WORD: Final[float] = 1.33
"""Calibration for the no-tiktoken fallback: BPE splits English technical prose ~1.3 tokens/word."""

_MAX_ATOM_TOKEN_RATIO: Final[float] = 1.0
"""An atom longer than ``chunk_size_tokens`` × this is hard-split on word boundaries."""


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences without splitting on abbreviations or decimals.

    Returns the pieces with their original spacing collapsed but their content intact; joining the
    result with a single space reproduces a readable paragraph.
    """
    if not text.strip():
        return []
    pieces = _SENTENCE_BOUNDARY.split(text.strip())
    merged: list[str] = []
    for piece in pieces:
        if merged and _is_false_boundary(merged[-1]):
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    return [piece.strip() for piece in merged if piece.strip()]


def _is_false_boundary(previous: str) -> bool:
    """True when ``previous`` ends in something that only looked like a sentence terminator."""
    stripped = previous.rstrip()
    if not stripped or stripped[-1] not in ".!?;":
        return False
    tail = stripped[:-1]
    last_token = re.split(r"[\s(\[]", tail)[-1].lower() if tail else ""
    if last_token in _ABBREVIATIONS:
        return True
    if len(last_token) == 1 and last_token.isalpha():
        return True  # middle initial, e.g. "R. Sharma"
    if last_token.isdigit() and len(stripped) <= 4:
        return True  # a bare list number such as "1."
    if re.search(r"\d$", tail):
        return True  # decimal or a tag number immediately before the dot
    return False


# --------------------------------------------------------------------------------------
# Input blocks
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class TextBlock:
    """A parser's unit of output: some text plus where it came from.

    Parsers emit these instead of one flat string so that page numbers, section headings and OCR
    confidence are attached at the point where they are actually known.
    """

    text: str
    page: int | None = None
    section: str | None = None
    kind: BlockKind = "paragraph"
    ocr_confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _Atom:
    """The smallest thing the packer will move: one sentence, list item, or table row."""

    text: str
    tokens: int
    page: int | None
    section: str | None
    kind: BlockKind
    ocr_confidence: float | None
    char_start: int
    char_end: int


# --------------------------------------------------------------------------------------
# Token counting
# --------------------------------------------------------------------------------------


class TokenCounter:
    """Token counting with a hard guarantee that it always returns a number.

    ``tiktoken`` downloads its BPE ranks on first use. On an offline machine that raises, which is
    exactly the situation CLAUDE.md rule 6 exists for — so the failure is caught once, logged once,
    and every subsequent call uses the heuristic.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._encoding: object | None = None
        self._resolved = False
        self._degraded = False

    @property
    def degraded(self) -> bool:
        """True when counts come from the heuristic rather than a real BPE encoder."""
        self._ensure()
        return self._degraded

    def _ensure(self) -> None:
        if self._resolved:
            return
        self._resolved = True
        try:
            import tiktoken  # noqa: PLC0415 - lazy: first call may hit the network
        except ImportError:  # pragma: no cover - optional dependency
            self._degraded = True
            logger.warning("tiktoken is not installed; falling back to heuristic token counting")
            return
        try:
            self._encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as exc:  # noqa: BLE001 - network/cache failures surface many types
            self._degraded = True
            logger.warning(
                "tiktoken encoding unavailable (offline or no cache); using heuristic token counts. "
                "Chunk sizes stay within ~10%% of true token counts.",
                extra={"error": type(exc).__name__},
            )

    def count(self, text: str) -> int:
        """Return the token count of ``text``. Never raises, never returns a negative."""
        if not text:
            return 0
        self._ensure()
        if self._encoding is not None:
            try:
                return len(self._encoding.encode(text, disallowed_special=()))  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001 - a bad byte must not break ingestion
                self._degraded = True
                logger.warning("tiktoken encode failed; using heuristic for this text",
                               extra={"error": type(exc).__name__})
        return self._heuristic(text)

    @staticmethod
    def _heuristic(text: str) -> int:
        return max(1, int(round(len(_WORDS.findall(text)) * _HEURISTIC_TOKENS_PER_WORD)))


# --------------------------------------------------------------------------------------
# Chunker
# --------------------------------------------------------------------------------------


class SemanticChunker:
    """Packs :class:`TextBlock` output into overlapping, sentence-aligned :class:`Chunk` objects."""

    name: str = "semantic_chunker"

    def __init__(self, settings: Settings | None = None, *, counter: TokenCounter | None = None) -> None:
        self._settings = settings or get_settings()
        self._counter = counter or TokenCounter(self._settings)

    @property
    def counter(self) -> TokenCounter:
        return self._counter

    def count_tokens(self, text: str) -> int:
        """Public token count, shared with the embedding batcher."""
        return self._counter.count(text)

    async def chunk(
        self,
        blocks: Sequence[TextBlock],
        *,
        document_id: str,
    ) -> tuple[list[Chunk], str]:
        """Chunk ``blocks`` for ``document_id``.

        Returns:
            ``(chunks, full_text)`` — the flat document text is returned alongside because
            ``ParsedDocument.text`` must use the *same* character offsets the chunks recorded.
        """
        return await asyncio.to_thread(self._chunk_sync, list(blocks), document_id)

    # -- implementation ---------------------------------------------------------------
    def _chunk_sync(self, blocks: list[TextBlock], document_id: str) -> tuple[list[Chunk], str]:
        atoms, full_text = self._atomise(blocks)
        if not atoms:
            return [], full_text

        size = max(1, self._settings.chunk_size_tokens)
        overlap = max(0, min(self._settings.chunk_overlap_tokens, size - 1))
        minimum = max(0, self._settings.chunk_min_tokens)

        packed: list[list[_Atom]] = []
        index = 0
        total = len(atoms)
        while index < total:
            end = index
            tokens = 0
            while end < total and (end == index or tokens + atoms[end].tokens <= size):
                tokens += atoms[end].tokens
                end += 1
            packed.append(atoms[index:end])
            if end >= total:
                break
            # Walk back from the boundary to build the overlap, always leaving forward progress.
            back = end
            carried = 0
            while back > index + 1 and carried + atoms[back - 1].tokens <= overlap:
                back -= 1
                carried += atoms[back].tokens
            index = max(back, index + 1)

        chunks = [
            self._build(group, document_id, position)
            for position, group in enumerate(packed)
        ]
        chunks = [chunk for chunk in chunks if chunk is not None]  # type: ignore[misc]

        kept = [chunk for chunk in chunks if chunk.token_count >= minimum]
        if not kept and chunks:
            # A short document must not vanish just because it is shorter than the minimum.
            kept = [max(chunks, key=lambda c: c.token_count)]
        elif len(kept) < len(chunks):
            logger.debug(
                "dropped undersized chunks",
                extra={"document_id": document_id, "dropped": len(chunks) - len(kept),
                       "min_tokens": minimum},
            )

        for position, chunk in enumerate(kept):
            chunk.index = position
            chunk.chunk_id = make_chunk_id(document_id, position)

        logger.info(
            "document chunked",
            extra={"document_id": document_id, "chunks": len(kept), "atoms": total,
                   "chunk_size_tokens": size, "overlap_tokens": overlap,
                   "token_counter": "heuristic" if self._counter.degraded else "tiktoken"},
        )
        return kept, full_text

    def _atomise(self, blocks: list[TextBlock]) -> tuple[list[_Atom], str]:
        """Flatten blocks into sentence-level atoms and build the document text in one pass."""
        atoms: list[_Atom] = []
        parts: list[str] = []
        cursor = 0
        current_section: str | None = None
        size_limit = int(self._settings.chunk_size_tokens * _MAX_ATOM_TOKEN_RATIO)

        for block in blocks:
            text = (block.text or "").strip()
            if not text:
                continue
            if block.kind == "heading":
                current_section = text[:_HEADING_MAX_CHARS]
            section = block.section or current_section

            for piece in self._pieces(block, text):
                for fragment in self._enforce_size(piece, size_limit):
                    if parts:
                        parts.append("\n")
                        cursor += 1
                    start = cursor
                    parts.append(fragment)
                    cursor += len(fragment)
                    atoms.append(
                        _Atom(
                            text=fragment,
                            tokens=self._counter.count(fragment),
                            page=block.page,
                            section=section,
                            kind=block.kind,
                            ocr_confidence=block.ocr_confidence,
                            char_start=start,
                            char_end=cursor,
                        )
                    )
        return atoms, "".join(parts)

    @staticmethod
    def _pieces(block: TextBlock, text: str) -> list[str]:
        """Split a block into atoms according to its kind."""
        if block.kind in ("table", "metadata"):
            return [line.strip() for line in text.splitlines() if line.strip()]
        if block.kind == "heading":
            return [text]
        out: list[str] = []
        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            if _LIST_MARKER.match(stripped):
                out.append(stripped)  # list items are atoms; splitting them destroys the step
            else:
                out.extend(split_sentences(stripped) or [stripped])
        return out

    def _enforce_size(self, text: str, limit: int) -> list[str]:
        """Hard-split an atom that is on its own larger than a whole chunk.

        This only fires on pathological input — a PDF that extracted an entire page as one
        unpunctuated run. Splitting on word boundaries is the least-bad option: it still never
        splits a *word*, and the alternative is a chunk that blows the embedding input limit.
        """
        if limit <= 0 or self._counter.count(text) <= limit:
            return [text]
        words = text.split(" ")
        out: list[str] = []
        buffer: list[str] = []
        for word in words:
            buffer.append(word)
            if self._counter.count(" ".join(buffer)) >= limit:
                out.append(" ".join(buffer))
                buffer = []
        if buffer:
            out.append(" ".join(buffer))
        logger.debug("hard-split an oversized text atom", extra={"pieces": len(out), "limit": limit})
        return out or [text]

    def _build(self, group: list[_Atom], document_id: str, position: int) -> Chunk | None:
        text = " ".join(atom.text for atom in group).strip()
        if not text:
            return None
        confidences = [a.ocr_confidence for a in group if a.ocr_confidence is not None]
        pages = [a.page for a in group if a.page is not None]
        sections = [a.section for a in group if a.section]
        return Chunk(
            chunk_id=make_chunk_id(document_id, position),
            document_id=document_id,
            index=position,
            text=text,
            token_count=sum(atom.tokens for atom in group),
            page=pages[0] if pages else None,
            section=sections[0] if sections else None,
            char_start=group[0].char_start,
            char_end=group[-1].char_end,
            # The weakest OCR read in the passage governs how much the passage can be trusted.
            ocr_confidence=min(confidences) if confidences else None,
            metadata={
                "atoms": len(group),
                "kinds": sorted({atom.kind for atom in group}),
                "pages": sorted({p for p in pages}) if pages else [],
            },
        )


# --------------------------------------------------------------------------------------
# Helpers shared by the parsers
# --------------------------------------------------------------------------------------


def table_to_text(rows: Sequence[Sequence[object]], *, title: str = "") -> str:
    """Render a table as pipe-delimited lines that survive chunking and read well in a citation.

    Markdown pipe format is deliberate: it keeps the header association intact inside a text chunk,
    so a retrieved passage still shows which column a number came from.
    """
    cleaned: list[list[str]] = []
    for row in rows:
        cells = ["" if cell is None else str(cell).replace("\n", " ").strip() for cell in row]
        if any(cells):
            cleaned.append(cells)
    if not cleaned:
        return ""
    width = max(len(row) for row in cleaned)
    lines: list[str] = []
    if title:
        lines.append(title)
    header = cleaned[0] + [""] * (width - len(cleaned[0]))
    lines.append(" | ".join(header))
    lines.append(" | ".join(["---"] * width))
    for row in cleaned[1:]:
        lines.append(" | ".join(row + [""] * (width - len(row))))
    return "\n".join(lines)


def blocks_from_text(
    text: str,
    *,
    page: int | None = None,
    ocr_confidence: float | None = None,
) -> list[TextBlock]:
    """Split raw text into paragraph and heading blocks.

    Used by the text, email and OCR paths, which get a flat string and still need section headings
    so that every chunk can carry one.
    """
    blocks: list[TextBlock] = []
    for raw_paragraph in re.split(r"\n\s*\n", text):
        paragraph = raw_paragraph.strip("\n ")
        if not paragraph.strip():
            continue
        lines = [line for line in paragraph.splitlines() if line.strip()]
        if len(lines) == 1 and is_heading(lines[0]):
            blocks.append(TextBlock(text=lines[0].strip(), page=page, kind="heading",
                                    ocr_confidence=ocr_confidence))
            continue
        if lines and is_heading(lines[0]) and len(lines) > 1:
            blocks.append(TextBlock(text=lines[0].strip(), page=page, kind="heading",
                                    ocr_confidence=ocr_confidence))
            paragraph = "\n".join(lines[1:])
        kind: BlockKind = "list" if all(_LIST_MARKER.match(line) for line in lines) and lines else "paragraph"
        blocks.append(TextBlock(text=paragraph.strip(), page=page, kind=kind,
                                ocr_confidence=ocr_confidence))
    return blocks


def is_heading(line: str) -> bool:
    """Heuristic heading detection for documents with no structural markup.

    A heading is short, does not end in sentence punctuation, and is either numbered
    ("4.2 Bearing replacement"), all-caps ("SAFETY PRECAUTIONS"), or markdown-hashed.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > _HEADING_MAX_CHARS:
        return False
    if stripped.startswith("#"):
        return True
    if stripped.endswith((".", "!", "?", ",", ";", ":")) and not stripped.endswith(":"):
        return False
    letters = [ch for ch in stripped if ch.isalpha()]
    if letters and all(ch.isupper() for ch in letters) and len(letters) > 2:
        return True
    if _HEADING_NUMBER.match(stripped) and not stripped.endswith("."):
        return True
    return bool(stripped.endswith(":") and len(stripped) < 60)


def iter_block_text(blocks: Iterable[TextBlock]) -> str:
    """Join blocks the same way :class:`SemanticChunker` does, for previews and logging."""
    return "\n".join(block.text.strip() for block in blocks if block.text.strip())


__all__ = [
    "BlockKind",
    "SemanticChunker",
    "TextBlock",
    "TokenCounter",
    "blocks_from_text",
    "is_heading",
    "iter_block_text",
    "split_sentences",
    "table_to_text",
]
