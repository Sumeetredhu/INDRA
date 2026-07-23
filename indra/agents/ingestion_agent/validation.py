"""Upload validation: magic-number sniffing, size limits, extension allowlist, integrity.

``python-magic`` is deliberately **not** a dependency — it needs a native ``libmagic`` that is a
pain to ship on Windows and would be exactly the kind of import that kills a demo. The signature
table below is hand-rolled and covers every format INDRA accepts.

The order of trust is: **content first, filename second.** The extension is only used to
disambiguate containers that genuinely cannot be told apart from their bytes (an OLE2 compound file
is a ``.doc``, an ``.xls`` and an Outlook ``.msg`` all at once) and to break ties inside plain text
(``.csv`` vs ``.md`` vs ``.json``). A mismatch between the declared extension and the sniffed family
is recorded as a warning and the *sniffed* family wins.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Final

import chardet

from indra.core.config import Settings, get_settings
from indra.core.exceptions import FileValidationError, UnsupportedFileTypeError
from indra.core.ids import content_hash
from indra.core.logging import get_logger
from indra.core.models import DocumentType, MimeFamily

logger = get_logger(__name__)

# --------------------------------------------------------------------------------------
# Tunables local to sniffing.
#
# These are *format facts*, not product policy, which is why they live here rather than in
# ``Settings``: changing them changes what "a PDF" means, not how INDRA behaves. Everything that is
# genuinely a policy knob (size cap, extension allowlist) comes from settings.
# --------------------------------------------------------------------------------------

HEAD_BYTES: Final[int] = 8192
"""Bytes read from the front of a file for sniffing. Large enough for OOXML/OLE2 headers."""

_PDF_SEARCH_WINDOW: Final[int] = 1024
"""Some generators emit junk before ``%PDF-``; the spec allows the header inside the first 1 KiB."""

_TEXT_PRINTABLE_RATIO: Final[float] = 0.90
"""Fraction of decoded characters that must be printable for a blob to count as text."""

_EMAIL_HEADER_SCAN_LINES: Final[int] = 40
"""RFC 5322 headers appear at the very top; scanning further just invites false positives."""

_MIN_EMAIL_HEADERS: Final[int] = 2
"""A single ``From:`` line is a sentence. Two distinct headers is a message."""


# --------------------------------------------------------------------------------------
# Signature table
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Signature:
    """One magic-number rule."""

    label: str
    magic: bytes
    family: MimeFamily
    mime_type: str
    offset: int = 0


MAGIC_SIGNATURES: Final[tuple[Signature, ...]] = (
    Signature("png", b"\x89PNG\r\n\x1a\n", MimeFamily.IMAGE, "image/png"),
    Signature("jpeg", b"\xff\xd8\xff", MimeFamily.IMAGE, "image/jpeg"),
    Signature("gif87", b"GIF87a", MimeFamily.IMAGE, "image/gif"),
    Signature("gif89", b"GIF89a", MimeFamily.IMAGE, "image/gif"),
    Signature("bmp", b"BM", MimeFamily.IMAGE, "image/bmp"),
    Signature("tiff_le", b"II\x2a\x00", MimeFamily.IMAGE, "image/tiff"),
    Signature("tiff_be", b"MM\x00\x2a", MimeFamily.IMAGE, "image/tiff"),
    Signature("jpeg2000", b"\x00\x00\x00\x0cjP  ", MimeFamily.IMAGE, "image/jp2"),
    Signature("rtf", b"{\\rtf", MimeFamily.WORD, "application/rtf"),
)
"""Fixed-offset signatures. Containers (ZIP, OLE2, RIFF) need structural inspection instead."""

_ZIP_MAGIC: Final[bytes] = b"PK\x03\x04"
_ZIP_EMPTY_MAGIC: Final[bytes] = b"PK\x05\x06"
_OLE2_MAGIC: Final[bytes] = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_RIFF_MAGIC: Final[bytes] = b"RIFF"
_PDF_MAGIC: Final[bytes] = b"%PDF-"

EXTENSION_FAMILIES: Final[dict[str, MimeFamily]] = {
    ".pdf": MimeFamily.PDF,
    ".png": MimeFamily.IMAGE,
    ".jpg": MimeFamily.IMAGE,
    ".jpeg": MimeFamily.IMAGE,
    ".tif": MimeFamily.IMAGE,
    ".tiff": MimeFamily.IMAGE,
    ".bmp": MimeFamily.IMAGE,
    ".webp": MimeFamily.IMAGE,
    ".gif": MimeFamily.IMAGE,
    ".xlsx": MimeFamily.SPREADSHEET,
    ".xlsm": MimeFamily.SPREADSHEET,
    ".xls": MimeFamily.SPREADSHEET,
    ".csv": MimeFamily.SPREADSHEET,
    ".tsv": MimeFamily.SPREADSHEET,
    ".docx": MimeFamily.WORD,
    ".doc": MimeFamily.WORD,
    ".rtf": MimeFamily.WORD,
    ".eml": MimeFamily.EMAIL,
    ".msg": MimeFamily.EMAIL,
    ".mbox": MimeFamily.EMAIL,
    ".txt": MimeFamily.TEXT,
    ".md": MimeFamily.TEXT,
    ".markdown": MimeFamily.TEXT,
    ".json": MimeFamily.TEXT,
    ".log": MimeFamily.TEXT,
    ".xml": MimeFamily.TEXT,
    ".html": MimeFamily.TEXT,
    ".htm": MimeFamily.TEXT,
}
"""Extension → expected family. Used for integrity cross-checks and text-family tie-breaks."""

_EXTENSION_MIME: Final[dict[str, str]] = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xlsm": "application/vnd.ms-excel.sheet.macroEnabled.12",
    ".xls": "application/vnd.ms-excel",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".rtf": "application/rtf",
    ".eml": "message/rfc822",
    ".msg": "application/vnd.ms-outlook",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".json": "application/json",
    ".xml": "application/xml",
    ".html": "text/html",
    ".htm": "text/html",
}

_EMAIL_HEADER_RE: Final[re.Pattern[str]] = re.compile(
    r"^(Return-Path|Received|Message-ID|Message-Id|From|To|Cc|Bcc|Subject|Date|"
    r"MIME-Version|Content-Type|Delivered-To|Reply-To|Sender|X-[A-Za-z0-9-]+)\s*:",
    re.IGNORECASE,
)

_STRONG_EMAIL_HEADERS: Final[frozenset[str]] = frozenset(
    {"from", "to", "subject", "date", "message-id", "received", "return-path"}
)


@dataclass(frozen=True, slots=True)
class SniffResult:
    """Outcome of magic-number inspection."""

    family: MimeFamily
    mime_type: str
    label: str
    confidence: float
    detail: str = ""


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Everything the pipeline learns about a file before a parser touches it."""

    filename: str
    extension: str
    size_bytes: int
    content_hash: str
    mime_family: MimeFamily
    mime_type: str
    document_type: DocumentType
    sniff_label: str
    sniff_confidence: float
    warnings: tuple[str, ...] = ()
    head: bytes = b""

    @property
    def ok(self) -> bool:
        """A report only exists when validation passed; kept for call-site readability."""
        return self.mime_family is not MimeFamily.UNKNOWN


# --------------------------------------------------------------------------------------
# Sniffing
# --------------------------------------------------------------------------------------


def _decode_head(head: bytes) -> str:
    """Best-effort decode of a byte head for text heuristics. Never raises."""
    if not head:
        return ""
    try:
        detected = chardet.detect(head)
        encoding = str(detected.get("encoding") or "") or "utf-8"
    except (ValueError, TypeError, LookupError):  # pragma: no cover - chardet is defensive already
        encoding = "utf-8"
    for candidate in (encoding, "utf-8", "latin-1"):
        try:
            return head.decode(candidate, errors="strict")
        except (UnicodeDecodeError, LookupError):
            continue
    return head.decode("latin-1", errors="replace")


def _looks_like_text(head: bytes) -> tuple[bool, str]:
    """Return ``(is_text, decoded_head)`` using a printable-character ratio."""
    if not head:
        return False, ""
    if b"\x00" in head[:512]:
        return False, ""
    text = _decode_head(head)
    if not text:
        return False, ""
    printable = sum(1 for ch in text if ch.isprintable() or ch in "\r\n\t\f\v")
    return (printable / len(text)) >= _TEXT_PRINTABLE_RATIO, text


def _looks_like_email(text: str) -> bool:
    """RFC 5322 header block detection."""
    seen: set[str] = set()
    for line in text.splitlines()[:_EMAIL_HEADER_SCAN_LINES]:
        if not line.strip():
            break  # blank line ends the header block
        match = _EMAIL_HEADER_RE.match(line)
        if match:
            seen.add(match.group(1).lower())
        elif not line.startswith((" ", "\t")):
            break  # a non-continuation, non-header line means this is not a header block
    return len(seen) >= _MIN_EMAIL_HEADERS and bool(seen & _STRONG_EMAIL_HEADERS)


def _sniff_zip_container(payload: bytes, extension: str) -> SniffResult:
    """Distinguish OOXML flavours by looking inside the ZIP central directory."""
    try:
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            names = set(archive.namelist())
    except (zipfile.BadZipFile, OSError, ValueError):
        # Truncated head is normal — we only read HEAD_BYTES. Fall back to the extension.
        family = EXTENSION_FAMILIES.get(extension, MimeFamily.UNKNOWN)
        return SniffResult(
            family=family,
            mime_type=_EXTENSION_MIME.get(extension, "application/zip"),
            label="zip_container",
            confidence=0.55 if family is not MimeFamily.UNKNOWN else 0.2,
            detail="ZIP central directory not present in the sniff window; used extension",
        )
    if any(name.startswith("word/") for name in names):
        return SniffResult(MimeFamily.WORD, _EXTENSION_MIME[".docx"], "ooxml_word", 0.99)
    if any(name.startswith("xl/") for name in names):
        return SniffResult(MimeFamily.SPREADSHEET, _EXTENSION_MIME[".xlsx"], "ooxml_excel", 0.99)
    if any(name.startswith("ppt/") for name in names):
        return SniffResult(
            MimeFamily.UNKNOWN,
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "ooxml_powerpoint",
            0.99,
            detail="PowerPoint is not a supported INDRA format",
        )
    if "mimetype" in names:
        return SniffResult(MimeFamily.UNKNOWN, "application/zip", "opendocument", 0.6,
                           detail="OpenDocument container is not a supported INDRA format")
    family = EXTENSION_FAMILIES.get(extension, MimeFamily.UNKNOWN)
    return SniffResult(family, "application/zip", "zip_container", 0.4,
                       detail="ZIP with no recognised Office part")


def _sniff_ole2(extension: str) -> SniffResult:
    """OLE2 compound files are ``.doc``/``.xls``/``.msg`` — only the name can tell them apart."""
    family = EXTENSION_FAMILIES.get(extension)
    if family in (MimeFamily.WORD, MimeFamily.SPREADSHEET, MimeFamily.EMAIL):
        return SniffResult(
            family,
            _EXTENSION_MIME.get(extension, "application/x-ole-storage"),
            "ole2_compound",
            0.7,
            detail="Legacy OLE2 container; family taken from the extension",
        )
    return SniffResult(MimeFamily.UNKNOWN, "application/x-ole-storage", "ole2_compound", 0.3,
                       detail="Legacy OLE2 container with an unrecognised extension")


def sniff(head: bytes, *, filename: str = "") -> SniffResult:
    """Classify a byte head into a :class:`MimeFamily`.

    Args:
        head: The first :data:`HEAD_BYTES` of the file (more is fine, less may weaken the result).
        filename: Used only to disambiguate container formats and text sub-types.

    Returns:
        A :class:`SniffResult`. ``MimeFamily.UNKNOWN`` means no parser can be selected.
    """
    extension = Path(filename).suffix.lower()

    if not head:
        return SniffResult(MimeFamily.UNKNOWN, "application/octet-stream", "empty", 1.0,
                           detail="File is empty")

    pdf_at = head[:_PDF_SEARCH_WINDOW].find(_PDF_MAGIC)
    if pdf_at >= 0:
        return SniffResult(MimeFamily.PDF, "application/pdf", "pdf", 0.99,
                           detail="" if pdf_at == 0 else f"%PDF- header at offset {pdf_at}")

    if head.startswith(_RIFF_MAGIC) and head[8:12] == b"WEBP":
        return SniffResult(MimeFamily.IMAGE, "image/webp", "webp", 0.99)

    for signature in MAGIC_SIGNATURES:
        end = signature.offset + len(signature.magic)
        if len(head) >= end and head[signature.offset:end] == signature.magic:
            return SniffResult(signature.family, signature.mime_type, signature.label, 0.98)

    if head.startswith(_ZIP_MAGIC) or head.startswith(_ZIP_EMPTY_MAGIC):
        return _sniff_zip_container(head, extension)

    if head.startswith(_OLE2_MAGIC):
        return _sniff_ole2(extension)

    is_text, text = _looks_like_text(head)
    if is_text:
        stripped = text.lstrip()
        if _looks_like_email(text):
            return SniffResult(MimeFamily.EMAIL, "message/rfc822", "rfc822", 0.9)
        if extension in (".csv", ".tsv"):
            return SniffResult(MimeFamily.SPREADSHEET, _EXTENSION_MIME[extension], "delimited_text", 0.85,
                               detail="Delimited text routed to the spreadsheet parser")
        if stripped.startswith(("{", "[")) or extension == ".json":
            return SniffResult(MimeFamily.TEXT, "application/json", "json", 0.8)
        if _looks_like_delimited(text):
            return SniffResult(MimeFamily.SPREADSHEET, "text/csv", "delimited_text", 0.6,
                               detail="Consistent delimiter across lines")
        return SniffResult(MimeFamily.TEXT, _EXTENSION_MIME.get(extension, "text/plain"),
                           "plain_text", 0.75)

    return SniffResult(MimeFamily.UNKNOWN, "application/octet-stream", "binary", 0.9,
                       detail="No known magic number and the content is not decodable text")


def _looks_like_delimited(text: str) -> bool:
    """Heuristic CSV/TSV detection: the same delimiter count on several consecutive lines.

    Deliberately conservative — a prose paragraph with commas must not be routed to pandas, so we
    require at least three data lines that agree exactly on the field count.
    """
    lines = [line for line in text.splitlines()[:20] if line.strip()]
    if len(lines) < 3:
        return False
    for delimiter in (",", "\t", ";", "|"):
        counts = [line.count(delimiter) for line in lines[:6]]
        if counts[0] >= 1 and len(set(counts)) == 1:
            return True
    return False


# --------------------------------------------------------------------------------------
# Document-type classification
# --------------------------------------------------------------------------------------

_DOC_TYPE_KEYWORDS: Final[tuple[tuple[DocumentType, tuple[str, ...]], ...]] = (
    (DocumentType.ROOT_CAUSE_ANALYSIS, ("root cause analysis", "rca report", "5 why", "fishbone",
                                        "root-cause")),
    (DocumentType.INCIDENT_REPORT, ("incident report", "near miss", "near-miss", "loss of containment",
                                    "safety incident")),
    (DocumentType.WORK_ORDER, ("work order", "workorder", "wo no", "wo#", "job card", "maintenance order")),
    (DocumentType.INSPECTION_REPORT, ("inspection report", "inspection record", "ndt report",
                                      "thickness survey", "vibration survey", "condition monitoring")),
    (DocumentType.SHIFT_LOG, ("shift log", "shift handover", "log book", "logbook", "shift report")),
    (DocumentType.SOP, ("standard operating procedure", "sop no", "work instruction", "procedure no")),
    (DocumentType.REGULATION, ("factory act", "oisd", "dgms", "peso", "gazette", "statutory",
                               "regulation", "compliance requirement")),
    (DocumentType.OEM_MANUAL, ("operation and maintenance manual", "o&m manual", "installation manual",
                               "instruction manual", "oem manual", "service manual", "user manual")),
    (DocumentType.PID_DRAWING, ("piping and instrumentation", "p&id", "process flow diagram",
                                "drawing no", "isometric")),
)

_DOC_TYPE_FILENAME_HINTS: Final[tuple[tuple[DocumentType, tuple[str, ...]], ...]] = (
    (DocumentType.WORK_ORDER, ("wo_", "work_order", "workorder", "jobcard")),
    (DocumentType.INSPECTION_REPORT, ("inspection", "insp_", "ndt", "vibration")),
    (DocumentType.SHIFT_LOG, ("shift", "log_", "logbook")),
    (DocumentType.INCIDENT_REPORT, ("incident", "nearmiss", "near_miss")),
    (DocumentType.ROOT_CAUSE_ANALYSIS, ("rca", "root_cause")),
    (DocumentType.SOP, ("sop", "procedure", "wi_")),
    (DocumentType.OEM_MANUAL, ("manual", "oem", "datasheet")),
    (DocumentType.PID_DRAWING, ("pid", "p&id", "p_id", "drawing", "dwg", "isometric")),
    (DocumentType.REGULATION, ("oisd", "dgms", "peso", "factory_act", "regulation")),
)


def guess_document_type(
    *,
    filename: str,
    family: MimeFamily,
    text_head: str = "",
) -> DocumentType:
    """Classify what a document *is*, which decides how the Copilot weighs it as evidence.

    Content keywords win over filename hints, because filenames in a plant document dump are
    routinely wrong. Family is the last resort so that nothing is left ``UNKNOWN`` when a
    structurally obvious answer exists (an ``.eml`` is an email even with an empty body).
    """
    haystack = text_head[:4000].lower()
    for doc_type, keywords in _DOC_TYPE_KEYWORDS:
        if any(keyword in haystack for keyword in keywords):
            return doc_type

    stem = Path(filename).name.lower()
    for doc_type, hints in _DOC_TYPE_FILENAME_HINTS:
        if any(hint in stem for hint in hints):
            return doc_type

    if family is MimeFamily.EMAIL:
        return DocumentType.EMAIL
    if family is MimeFamily.SPREADSHEET:
        return DocumentType.SPREADSHEET
    if family is MimeFamily.IMAGE:
        return DocumentType.PID_DRAWING if "drawing" in stem or "pid" in stem else DocumentType.UNKNOWN
    return DocumentType.UNKNOWN


# --------------------------------------------------------------------------------------
# Validation entry points
# --------------------------------------------------------------------------------------


def validate_bytes(
    content: bytes,
    *,
    filename: str,
    settings: Settings | None = None,
) -> ValidationReport:
    """Validate an upload and return everything the pipeline needs to route it.

    Args:
        content: Complete file bytes.
        filename: Client-supplied name. Never trusted for typing, only for tie-breaks.
        settings: Injected for tests; defaults to the process settings singleton.

    Returns:
        A :class:`ValidationReport` with the sniffed family, content hash and any integrity warnings.

    Raises:
        FileValidationError: Empty file, oversized file, or a disallowed extension.
        UnsupportedFileTypeError: The bytes match no format INDRA can parse.
    """
    cfg = settings or get_settings()
    warnings: list[str] = []

    safe_name = Path(filename).name.strip() or "unnamed"
    extension = Path(safe_name).suffix.lower()
    size = len(content)

    if size == 0:
        raise FileValidationError(
            "Uploaded file is empty. Re-export the document and upload it again.",
            context={"filename": safe_name},
        )
    if size > cfg.max_upload_bytes:
        raise FileValidationError(
            f"File is {size / 1_048_576:.1f} MB which exceeds the {cfg.max_upload_mb} MB limit. "
            f"Split the document or raise INDRA_MAX_UPLOAD_MB.",
            context={"filename": safe_name, "size_bytes": size, "limit_bytes": cfg.max_upload_bytes},
        )

    allowed = {ext.lower() for ext in cfg.allowed_extensions}
    if extension not in allowed:
        raise FileValidationError(
            f"Extension {extension or '(none)'} is not accepted. "
            f"Allowed: {', '.join(sorted(allowed))}.",
            context={"filename": safe_name, "extension": extension},
        )

    head = content[:HEAD_BYTES]
    result = sniff(head, filename=safe_name)

    if result.family is MimeFamily.UNKNOWN:
        raise UnsupportedFileTypeError(
            f"Could not identify {safe_name} from its contents ({result.label}). "
            f"{result.detail or 'The file may be corrupt or password-protected.'} "
            f"Convert it to PDF, DOCX, XLSX, EML, an image, or plain text.",
            context={"filename": safe_name, "extension": extension, "sniffed": result.label},
        )

    expected = EXTENSION_FAMILIES.get(extension)
    if expected is not None and expected is not result.family:
        warnings.append(
            f"Extension {extension} suggests {expected.value} but the contents are "
            f"{result.family.value} ({result.label}); trusting the contents."
        )
        logger.warning(
            "extension/content mismatch",
            extra={"document_filename": safe_name, "declared": expected.value, "detected": result.family.value},
        )

    if result.confidence < 0.6:
        warnings.append(
            f"Low-confidence format detection ({result.label}, {result.confidence:.2f}); "
            f"parsing may be incomplete."
        )

    text_head = ""
    if result.family in (MimeFamily.TEXT, MimeFamily.EMAIL, MimeFamily.SPREADSHEET):
        _, text_head = _looks_like_text(head)

    document_type = guess_document_type(
        filename=safe_name, family=result.family, text_head=text_head
    )

    digest = content_hash(content)
    logger.info(
        "file validated",
        extra={
            "document_filename": safe_name,
            "mime_family": result.family.value,
            "document_type": document_type.value,
            "size_bytes": size,
            "content_hash": digest[:12],
        },
    )
    return ValidationReport(
        filename=safe_name,
        extension=extension,
        size_bytes=size,
        content_hash=digest,
        mime_family=result.family,
        mime_type=result.mime_type,
        document_type=document_type,
        sniff_label=result.label,
        sniff_confidence=result.confidence,
        warnings=tuple(warnings),
        head=head,
    )


def read_and_validate(path: Path, *, settings: Settings | None = None) -> tuple[bytes, ValidationReport]:
    """Read a file from disk and validate it. CPU/IO-bound — call through ``asyncio.to_thread``.

    Raises:
        FileValidationError: The path is missing, is not a file, or fails validation.
    """
    try:
        if not path.is_file():
            raise FileValidationError(
                f"{path} is not a readable file. Check the path and permissions.",
                context={"path": str(path)},
            )
        content = path.read_bytes()
    except OSError as exc:
        raise FileValidationError(
            f"Could not read {path}: {exc.strerror or exc}. Check the path and permissions.",
            context={"path": str(path)},
            cause=exc,
        ) from exc
    return content, validate_bytes(content, filename=path.name, settings=settings)


__all__ = [
    "EXTENSION_FAMILIES",
    "HEAD_BYTES",
    "MAGIC_SIGNATURES",
    "Signature",
    "SniffResult",
    "ValidationReport",
    "guess_document_type",
    "read_and_validate",
    "sniff",
    "validate_bytes",
]
