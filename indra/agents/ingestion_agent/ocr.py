"""Optical character recognition with per-word confidences.

Tesseract is the primary engine, EasyOCR the optional upgrade. Both are *optional at every level*:

* ``pytesseract`` may not be installed — module-level guard.
* ``pytesseract`` may be installed while the **tesseract binary** is not — probed once, cached.
* ``easyocr`` pulls in torch and is almost never present — imported lazily inside the call.

When none of that is available this module returns an empty :class:`OCRResult` carrying a warning.
It **never raises into the pipeline**: a scanned page that cannot be read is a document with a
warning attached, not a failed ingestion. That is CLAUDE.md rule 6 applied to the one capability
most likely to be missing on a laptop.

Per-word confidence is the point of this module. ``Chunk.ocr_confidence`` flows from here into
``SourceRef.extraction_confidence`` and finally into the "Explain How I Know This" panel, which is
what lets INDRA say *"bearing wear 87% — OCR confidence 0.72, verify with supervisor"* instead of
quietly asserting a number it half-read.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

from indra.core.config import Settings, get_settings
from indra.core.logging import get_logger

try:
    import pytesseract
    from pytesseract import Output as _TesseractOutput

    _HAS_PYTESSERACT = True
except ImportError:  # pragma: no cover - optional dependency
    pytesseract = None  # type: ignore[assignment]
    _TesseractOutput = None  # type: ignore[assignment]
    _HAS_PYTESSERACT = False

try:
    import cv2

    _HAS_CV2 = True
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore[assignment]
    _HAS_CV2 = False

logger = get_logger(__name__)

ImageLike = Path | str | NDArray[np.uint8] | Image.Image

# --------------------------------------------------------------------------------------
# Engine tunables. These describe how tesseract is *driven*, not what INDRA considers
# trustworthy — the trust threshold is ``settings.ocr_min_confidence``.
# --------------------------------------------------------------------------------------

PSM_AUTO: Final[int] = 3
"""Fully automatic page segmentation — full pages and large regions."""

PSM_SINGLE_BLOCK: Final[int] = 6
"""Assume a single uniform block of text — cropped table cells and drawing title blocks."""

PSM_SINGLE_LINE: Final[int] = 7
"""A single text line — the right mode for a tag label beside a P&ID symbol."""

PSM_SINGLE_WORD: Final[int] = 8
"""A single word — the right mode for a tag inside an instrument bubble."""

_MIN_OCR_HEIGHT_PX: Final[int] = 36
"""Tesseract degrades badly below roughly 30 px of glyph height; small crops are upscaled to this."""

_MAX_UPSCALE: Final[float] = 6.0
"""Cap on the upscale factor, so a 2-pixel artefact does not become a 12-megapixel OCR job."""

_TAG_WHITELIST: Final[str] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-/"
"""Character whitelist for tag-region OCR. Massively reduces glyph confusion on short strings."""


@dataclass(frozen=True, slots=True)
class OCRWord:
    """One recognised word with the confidence the engine assigned it."""

    text: str
    confidence: float
    bbox: tuple[int, int, int, int]
    line_num: int = 0
    block_num: int = 0

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


@dataclass(frozen=True, slots=True)
class OCRResult:
    """Recognised text plus the evidence for how much to trust it."""

    text: str = ""
    mean_confidence: float = 0.0
    words: tuple[OCRWord, ...] = ()
    engine: str = "none"
    warnings: tuple[str, ...] = ()
    duration_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return bool(self.text.strip())

    @property
    def low_confidence_words(self) -> tuple[OCRWord, ...]:
        """Words the caller should surface as uncertain. Threshold applied by the caller."""
        return self.words

    def confident_words(self, minimum: float) -> tuple[OCRWord, ...]:
        return tuple(word for word in self.words if word.confidence >= minimum)

    def with_warning(self, message: str) -> OCRResult:
        return OCRResult(
            text=self.text,
            mean_confidence=self.mean_confidence,
            words=self.words,
            engine=self.engine,
            warnings=self.warnings + (message,),
            duration_ms=self.duration_ms,
        )


EMPTY_RESULT: Final[OCRResult] = OCRResult()


# --------------------------------------------------------------------------------------
# Image plumbing
# --------------------------------------------------------------------------------------


def load_image(path: Path) -> NDArray[np.uint8] | None:
    """Read an image into a BGR array, tolerating non-ASCII Windows paths.

    ``cv2.imread`` silently returns ``None`` for paths containing non-ASCII characters on Windows,
    which is a very common way for an Indian-plant document dump to break a pipeline. Reading the
    bytes ourselves and decoding from memory avoids it entirely.
    """
    try:
        raw = np.fromfile(str(path), dtype=np.uint8)
    except (OSError, ValueError) as exc:
        logger.warning("could not read image bytes", extra={"path": str(path), "error": str(exc)})
        return None
    if raw.size == 0:
        return None
    if _HAS_CV2:
        decoded = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if decoded is not None:
            return decoded.astype(np.uint8)
    try:
        with Image.open(path) as handle:
            return np.asarray(handle.convert("RGB"), dtype=np.uint8)[:, :, ::-1].copy()
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        logger.warning("image decode failed", extra={"path": str(path), "error": str(exc)})
        return None


def to_grayscale(array: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Collapse any array shape to single-channel 8-bit."""
    if array.ndim == 2:
        return array
    if array.shape[2] == 4:
        array = array[:, :, :3]
    if _HAS_CV2:
        return cv2.cvtColor(array, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    weights = np.array([0.114, 0.587, 0.299], dtype=np.float32)  # BGR
    return (array[:, :, :3].astype(np.float32) @ weights).astype(np.uint8)


def prepare_for_ocr(array: NDArray[np.uint8], *, binarize: bool = True) -> NDArray[np.uint8]:
    """Upscale small crops and binarise, which is worth several confidence points on plant scans.

    Kept deliberately simple: aggressive denoising destroys the thin strokes in engineering-drawing
    lettering, which is exactly the text this pipeline most needs to read.
    """
    gray = to_grayscale(array)
    height = gray.shape[0]
    if height < _MIN_OCR_HEIGHT_PX and height > 0:
        factor = min(_MAX_UPSCALE, _MIN_OCR_HEIGHT_PX / float(height))
        new_size = (max(1, int(gray.shape[1] * factor)), max(1, int(height * factor)))
        if _HAS_CV2:
            gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_CUBIC).astype(np.uint8)
        else:
            pil = Image.fromarray(gray).resize(new_size, Image.BICUBIC)
            gray = np.asarray(pil, dtype=np.uint8)
    if binarize and _HAS_CV2:
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gray.astype(np.uint8)


def _coerce(source: ImageLike) -> NDArray[np.uint8] | None:
    if isinstance(source, (str, Path)):
        return load_image(Path(source))
    if isinstance(source, Image.Image):
        return np.asarray(source.convert("RGB"), dtype=np.uint8)[:, :, ::-1].copy()
    if isinstance(source, np.ndarray):
        return source.astype(np.uint8) if source.dtype != np.uint8 else source
    return None


# --------------------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------------------


class OCREngine:
    """Async OCR facade. Every recognition call runs on a worker thread.

    One instance is shared by the whole ingestion agent; the availability probe runs once and is
    cached, so a missing tesseract binary costs one log line rather than one per page.
    """

    name: str = "ocr_engine"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._probe_lock = asyncio.Lock()
        self._available: bool | None = None
        self._tesseract_ready: bool = False
        self._tesseract_version: str = ""
        self._easyocr_reader: object | None = None
        self._warned_unavailable = False

    # -- availability -----------------------------------------------------------------
    @property
    def language(self) -> str:
        return "+".join(self._settings.ocr_languages) or "eng"

    async def is_available(self) -> bool:
        """True when at least one OCR engine can actually run. Probed once, then cached."""
        if self._available is None:
            async with self._probe_lock:
                if self._available is None:
                    self._available = await asyncio.to_thread(self._probe)
        return self._available

    def _probe(self) -> bool:
        """Resolve the engine chain once. Runs on a worker thread — tesseract shells out."""
        self._tesseract_ready = self._probe_tesseract()
        if self._tesseract_ready:
            return True
        if self._settings.ocr_engine in ("easyocr", "auto") and self._load_easyocr():
            return True
        return False

    def _probe_tesseract(self) -> bool:
        """Check for the tesseract *binary*, not just the Python wrapper."""
        if not _HAS_PYTESSERACT or self._settings.ocr_engine == "easyocr":
            return False
        try:
            self._tesseract_version = str(pytesseract.get_tesseract_version())
        except Exception as exc:  # noqa: BLE001 - pytesseract raises several unrelated types
            logger.warning(
                "tesseract binary not found; OCR is disabled and scanned pages will yield no text. "
                "Install tesseract-ocr and put it on PATH to enable it.",
                extra={"error": type(exc).__name__},
            )
            return False
        logger.info("tesseract available", extra={"version": self._tesseract_version,
                                                  "languages": self.language})
        return True

    def _load_easyocr(self) -> bool:
        """Lazily construct an EasyOCR reader. Heavy (torch) and usually absent."""
        if self._easyocr_reader is not None:
            return True
        try:
            import easyocr  # noqa: PLC0415 - deliberately lazy, pulls in torch
        except ImportError:  # pragma: no cover - optional dependency
            return False
        try:
            langs = [lang[:2] for lang in self._settings.ocr_languages] or ["en"]
            self._easyocr_reader = easyocr.Reader(langs, gpu=False, verbose=False)
        except Exception as exc:  # noqa: BLE001 - easyocr surfaces model-download failures broadly
            logger.warning("easyocr initialisation failed; skipping that engine",
                           extra={"error": str(exc)})
            return False
        return True

    async def describe(self) -> dict[str, str | bool]:
        """Health-panel summary. Never raises."""
        available = await self.is_available()
        return {
            "available": available,
            "engine": "tesseract" if self._tesseract_ready else
                      ("easyocr" if self._easyocr_reader is not None else "none"),
            "version": self._tesseract_version,
            "languages": self.language,
        }

    # -- recognition ------------------------------------------------------------------
    async def recognize(
        self,
        source: ImageLike,
        *,
        psm: int = PSM_AUTO,
        whitelist: str | None = None,
        preprocess: bool = True,
    ) -> OCRResult:
        """Recognise text in a whole image. Returns an empty result rather than raising.

        Args:
            source: A path, a BGR/grayscale numpy array, or a PIL image.
            psm: Tesseract page segmentation mode; see the ``PSM_*`` constants.
            whitelist: Restrict recognised characters (use :data:`_TAG_WHITELIST` for tags).
            preprocess: Upscale and binarise first. Disable for already-clean renders.
        """
        array = _coerce(source)
        if array is None or array.size == 0:
            return EMPTY_RESULT.with_warning("Image could not be decoded for OCR")

        if not await self.is_available():
            if not self._warned_unavailable:
                logger.warning("OCR requested but no engine is available; returning empty text")
                self._warned_unavailable = True
            return EMPTY_RESULT.with_warning(
                "No OCR engine available (tesseract binary missing); "
                "scanned content was not read and is absent from the index"
            )

        if self._tesseract_ready:
            return await asyncio.to_thread(
                self._run_tesseract, array, psm, whitelist, preprocess
            )
        return await asyncio.to_thread(self._run_easyocr, array, preprocess)

    async def recognize_region(
        self,
        source: ImageLike,
        bbox: tuple[int, int, int, int],
        *,
        margin: int = 0,
        psm: int = PSM_SINGLE_LINE,
        whitelist: str | None = _TAG_WHITELIST,
    ) -> OCRResult:
        """Recognise text inside a pixel region, with an optional margin.

        This is the P&ID tag path: crop the neighbourhood of a detected symbol, restrict the
        alphabet to tag characters, and read a single line.
        """
        array = _coerce(source)
        if array is None or array.size == 0:
            return EMPTY_RESULT.with_warning("Image could not be decoded for region OCR")
        height, width = array.shape[:2]
        x0, y0, x1, y1 = bbox
        x0 = max(0, min(width - 1, int(x0) - margin))
        y0 = max(0, min(height - 1, int(y0) - margin))
        x1 = max(x0 + 1, min(width, int(x1) + margin))
        y1 = max(y0 + 1, min(height, int(y1) + margin))
        crop = array[y0:y1, x0:x1]
        if crop.size == 0:
            return EMPTY_RESULT.with_warning(f"Empty crop for region {bbox}")
        result = await self.recognize(crop, psm=psm, whitelist=whitelist)
        # Translate word boxes back into whole-image coordinates.
        if result.words:
            shifted = tuple(
                OCRWord(
                    text=word.text,
                    confidence=word.confidence,
                    bbox=(word.bbox[0] + x0, word.bbox[1] + y0, word.bbox[2] + x0, word.bbox[3] + y0),
                    line_num=word.line_num,
                    block_num=word.block_num,
                )
                for word in result.words
            )
            return OCRResult(result.text, result.mean_confidence, shifted, result.engine,
                             result.warnings, result.duration_ms)
        return result

    # -- engine implementations (worker thread) ---------------------------------------
    def _run_tesseract(
        self,
        array: NDArray[np.uint8],
        psm: int,
        whitelist: str | None,
        preprocess: bool,
    ) -> OCRResult:
        import time

        started = time.perf_counter()
        prepared = prepare_for_ocr(array) if preprocess else to_grayscale(array)
        config = f"--oem 3 --psm {psm}"
        if whitelist:
            config += f" -c tessedit_char_whitelist={whitelist}"
        try:
            data = pytesseract.image_to_data(
                Image.fromarray(prepared),
                lang=self.language,
                config=config,
                output_type=_TesseractOutput.DICT,
            )
        except Exception as exc:  # noqa: BLE001 - tesseract failure must never break ingestion
            logger.warning("tesseract recognition failed", extra={"error": str(exc)})
            return EMPTY_RESULT.with_warning(f"OCR failed: {exc}")

        words: list[OCRWord] = []
        confidences: list[float] = []
        lines: dict[tuple[int, int, int], list[str]] = {}
        count = len(data.get("text", []))
        for index in range(count):
            text = str(data["text"][index]).strip()
            if not text:
                continue
            try:
                raw_conf = float(data["conf"][index])
            except (TypeError, ValueError):
                raw_conf = -1.0
            if raw_conf < 0:
                continue
            confidence = max(0.0, min(1.0, raw_conf / 100.0))
            left, top = int(data["left"][index]), int(data["top"][index])
            width, height = int(data["width"][index]), int(data["height"][index])
            word = OCRWord(
                text=text,
                confidence=confidence,
                bbox=(left, top, left + width, top + height),
                line_num=int(data.get("line_num", [0] * count)[index]),
                block_num=int(data.get("block_num", [0] * count)[index]),
            )
            words.append(word)
            confidences.append(confidence)
            key = (word.block_num, int(data.get("par_num", [0] * count)[index]), word.line_num)
            lines.setdefault(key, []).append(text)

        text_out = "\n".join(" ".join(parts) for _, parts in sorted(lines.items()))
        mean = sum(confidences) / len(confidences) if confidences else 0.0
        warnings: tuple[str, ...] = ()
        if confidences and mean < self._settings.ocr_min_confidence:
            warnings = (
                f"Mean OCR confidence {mean:.2f} is below the {self._settings.ocr_min_confidence:.2f} "
                f"threshold; extracted values from this source need verification",
            )
        return OCRResult(
            text=text_out,
            mean_confidence=round(mean, 4),
            words=tuple(words),
            engine="tesseract",
            warnings=warnings,
            duration_ms=round((time.perf_counter() - started) * 1000.0, 2),
        )

    def _run_easyocr(self, array: NDArray[np.uint8], preprocess: bool) -> OCRResult:
        import time

        started = time.perf_counter()
        if not self._load_easyocr() or self._easyocr_reader is None:
            return EMPTY_RESULT.with_warning("EasyOCR is not installed")
        prepared = prepare_for_ocr(array) if preprocess else to_grayscale(array)
        try:
            raw = self._easyocr_reader.readtext(prepared)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 - third-party engine, unknown error surface
            logger.warning("easyocr recognition failed", extra={"error": str(exc)})
            return EMPTY_RESULT.with_warning(f"OCR failed: {exc}")

        words: list[OCRWord] = []
        confidences: list[float] = []
        for entry in raw:
            try:
                box, text, score = entry
            except (TypeError, ValueError):
                continue
            text = str(text).strip()
            if not text:
                continue
            xs = [int(point[0]) for point in box]
            ys = [int(point[1]) for point in box]
            confidence = max(0.0, min(1.0, float(score)))
            words.append(OCRWord(text, confidence, (min(xs), min(ys), max(xs), max(ys))))
            confidences.append(confidence)
        mean = sum(confidences) / len(confidences) if confidences else 0.0
        return OCRResult(
            text="\n".join(word.text for word in words),
            mean_confidence=round(mean, 4),
            words=tuple(words),
            engine="easyocr",
            duration_ms=round((time.perf_counter() - started) * 1000.0, 2),
        )


def group_words_into_lines(words: Sequence[OCRWord], *, tolerance_px: int = 6) -> list[str]:
    """Reassemble word boxes into reading-order lines.

    Used by the P&ID parser, where word order from ``image_to_data`` is per-block rather than
    per-visual-line once a drawing has multiple text orientations.
    """
    if not words:
        return []
    ordered = sorted(words, key=lambda w: (w.bbox[1], w.bbox[0]))
    lines: list[list[OCRWord]] = [[ordered[0]]]
    for word in ordered[1:]:
        anchor = lines[-1][0]
        if abs(word.bbox[1] - anchor.bbox[1]) <= tolerance_px:
            lines[-1].append(word)
        else:
            lines.append([word])
    return [" ".join(w.text for w in sorted(line, key=lambda w: w.bbox[0])) for line in lines]


__all__ = [
    "EMPTY_RESULT",
    "ImageLike",
    "OCREngine",
    "OCRResult",
    "OCRWord",
    "PSM_AUTO",
    "PSM_SINGLE_BLOCK",
    "PSM_SINGLE_LINE",
    "PSM_SINGLE_WORD",
    "group_words_into_lines",
    "load_image",
    "prepare_for_ocr",
    "to_grayscale",
]
