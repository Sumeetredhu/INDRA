"""Photo-to-query: point the camera at a pump, get the asset's full profile back.

Pipeline: decode → MSER text-region detection → geometric/contrast scoring → region OCR → tag
normalisation against the live equipment registry → graph lookup → :class:`PhotoQueryResponse`.

Three things drive the design:

**Latency.** The demo moment is *point camera → card appears*, with a budget under two seconds. All
of OpenCV and OCR runs in one worker thread (one hop, not four), the registry is cached, and the
graph reads fan out concurrently.

**Honesty over confidence.** A confidently wrong tag on a plant floor sends a technician to the wrong
pump. Whenever the top candidate is not clearly ahead of the runner-up, the response carries
``tag_alternatives`` and the UI asks "P-101 or P-107?" instead of asserting.

**No cross-agent imports.** The Ingestion Agent owns the canonical tag normaliser (D5). This module
deliberately re-implements the same semantics — glyph-confusion map, structural plant-tag grammar,
``rapidfuzz`` against the registry, never a silent correction — against its own registry snapshot,
because importing another agent's package is forbidden by CLAUDE.md.
"""

from __future__ import annotations

import asyncio
import io
import math
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Sequence

import cv2
import numpy as np
from PIL import Image
from rapidfuzz import fuzz, process

from indra.core.config import Settings
from indra.core.deps import AgentDeps
from indra.core.exceptions import FileValidationError, IndraError, OCRError
from indra.core.logging import get_logger
from indra.core.models import (
    Alert,
    Equipment,
    MaintenanceRecord,
    PhotoQueryResponse,
    Procedure,
    RecommendedAction,
    Severity,
    SourceRef,
)

try:
    import pytesseract

    _HAS_PYTESSERACT = True
except ImportError:  # pragma: no cover - optional dependency
    pytesseract = None  # type: ignore[assignment]
    _HAS_PYTESSERACT = False

try:
    import easyocr

    _HAS_EASYOCR = True
except ImportError:  # pragma: no cover - optional dependency
    easyocr = None  # type: ignore[assignment]
    _HAS_EASYOCR = False

if TYPE_CHECKING:  # pragma: no cover - typing only
    from indra.core.contracts import KnowledgeGraphService, ProactiveService

logger = get_logger(__name__)


# ======================================================================================
# Heuristic parameters
# ======================================================================================


@dataclass(frozen=True, slots=True)
class TagDetectionParams:
    """Geometry and scoring constants for tag-plate detection.

    These live here rather than in :mod:`indra.core.config` on purpose: they are properties of what a
    stencilled plant tag looks like in a photograph, not deployment tunables an operator would ever
    set through the environment. ``indra/core`` is read-only to this agent. Every value is named and
    documented so the heuristic stays auditable.
    """

    #: MSER stability delta. Lower finds more (noisier) regions.
    mser_delta: int = 5
    #: Glyph candidate area, as a fraction of total image area.
    min_glyph_area_ratio: float = 0.00004
    max_glyph_area_ratio: float = 0.02
    #: A glyph is taller than it is wide; merged ligatures push this up, so the band is generous.
    min_glyph_aspect: float = 0.06
    max_glyph_aspect: float = 3.0
    #: Glyph height as a fraction of image height.
    min_glyph_height_ratio: float = 0.012
    max_glyph_height_ratio: float = 0.45
    #: Two glyphs join a line when their vertical centres are within this multiple of glyph height…
    line_vertical_tolerance: float = 0.7
    #: …and the horizontal gap between them is under this multiple of glyph height.
    line_horizontal_gap: float = 1.6
    #: Minimum glyphs in a line for it to be considered a tag ("P-101" is five).
    min_glyphs_per_line: int = 3
    #: Padding added around a detected line, as a fraction of its height.
    region_padding_ratio: float = 0.25
    #: Ideal tag geometry, used by the Gaussian scoring terms below.
    ideal_glyph_count: float = 5.0
    glyph_count_sigma: float = 3.0
    ideal_aspect: float = 4.0
    aspect_sigma: float = 2.5
    ideal_height_ratio: float = 0.06
    height_sigma: float = 0.07
    ideal_ink_ratio: float = 0.28
    ink_sigma: float = 0.22
    #: Standard deviation of grey levels that counts as full contrast.
    contrast_reference: float = 55.0
    #: Score weights. Must sum to 1.0; asserted at import.
    weight_glyphs: float = 0.22
    weight_aspect: float = 0.22
    weight_height: float = 0.14
    weight_contrast: float = 0.20
    weight_ink: float = 0.12
    weight_centre: float = 0.10
    #: How many candidate regions to OCR. Each costs ~30ms; four keeps us inside the budget.
    max_regions: int = 4
    #: Upscale factor applied to a crop before OCR — tesseract wants ~30px glyph height.
    ocr_upscale: int = 3


DEFAULT_DETECTION_PARAMS: Final[TagDetectionParams] = TagDetectionParams()

_WEIGHT_SUM = (
    DEFAULT_DETECTION_PARAMS.weight_glyphs
    + DEFAULT_DETECTION_PARAMS.weight_aspect
    + DEFAULT_DETECTION_PARAMS.weight_height
    + DEFAULT_DETECTION_PARAMS.weight_contrast
    + DEFAULT_DETECTION_PARAMS.weight_ink
    + DEFAULT_DETECTION_PARAMS.weight_centre
)
if abs(_WEIGHT_SUM - 1.0) > 1e-9:  # pragma: no cover - guards an editing mistake
    raise ValueError(f"TagDetectionParams score weights must sum to 1.0, got {_WEIGHT_SUM}")


@dataclass(frozen=True, slots=True)
class TagRegion:
    """One candidate tag plate located in the photograph."""

    bbox: tuple[int, int, int, int]
    """Pixel ``(x0, y0, x1, y1)`` **in original image coordinates**."""
    score: float
    glyph_count: int
    text: str = ""
    ocr_confidence: float = 0.0

    @property
    def has_text(self) -> bool:
        return bool(self.text.strip())


# ======================================================================================
# Detection
# ======================================================================================


def decode_image(data: bytes) -> np.ndarray:
    """Decode image bytes to a BGR array.

    OpenCV first (fast, handles the phone formats), Pillow second (handles the odd TIFF and
    progressive JPEG OpenCV refuses).

    Raises:
        FileValidationError: the payload is not a decodable image.
    """
    if not data:
        raise FileValidationError(
            "Empty image payload. Capture a photo before submitting.", context={"bytes": 0}
        )
    try:
        buffer = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except Exception as exc:  # external boundary: OpenCV decoder
        image = None
        logger.debug("opencv failed to decode the image", extra={"error": str(exc)})
    if image is not None:
        return image
    try:
        with Image.open(io.BytesIO(data)) as handle:
            rgb = handle.convert("RGB")
            return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    except Exception as exc:  # external boundary: Pillow decoder
        raise FileValidationError(
            "Could not decode the photo. Send a JPEG, PNG, WEBP, BMP, or TIFF image.",
            context={"bytes": len(data)},
            cause=exc,
        ) from exc


def _prepare(image: np.ndarray, max_dimension: int) -> tuple[np.ndarray, float]:
    """Downscale to the working resolution and return ``(grayscale, scale_back_factor)``."""
    height, width = image.shape[:2]
    longest = max(height, width)
    scale = 1.0
    working = image
    if longest > max_dimension > 0:
        scale = max_dimension / float(longest)
        working = cv2.resize(
            image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA
        )
    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    # CLAHE rather than global equalisation: plant photos are half in shadow, and a global stretch
    # blows out the lit side of the plate.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray), (1.0 / scale if scale else 1.0)


def _glyph_boxes(gray: np.ndarray, params: TagDetectionParams) -> list[tuple[int, int, int, int]]:
    """Run MSER and keep the connected components shaped like printed characters.

    MSER is run twice — once on the image and once inverted — because plant tags appear both as dark
    text on a light plate and as light engraving on a dark one.
    """
    height, width = gray.shape[:2]
    area = float(height * width)
    mser = cv2.MSER_create()
    for setter, value in (
        ("setDelta", params.mser_delta),
        ("setMinArea", int(area * params.min_glyph_area_ratio)),
        ("setMaxArea", int(area * params.max_glyph_area_ratio)),
    ):
        method = getattr(mser, setter, None)
        if callable(method):
            method(value)

    boxes: list[tuple[int, int, int, int]] = []
    for source in (gray, cv2.bitwise_not(gray)):
        try:
            _, raw = mser.detectRegions(source)
        except cv2.error as exc:  # external boundary: OpenCV
            logger.debug("MSER detection failed on one polarity", extra={"error": str(exc)})
            continue
        for box in raw:
            x, y, w, h = (int(v) for v in box)
            if h <= 0 or w <= 0:
                continue
            aspect = w / float(h)
            height_ratio = h / float(height)
            if not (params.min_glyph_aspect <= aspect <= params.max_glyph_aspect):
                continue
            if not (params.min_glyph_height_ratio <= height_ratio <= params.max_glyph_height_ratio):
                continue
            boxes.append((x, y, x + w, y + h))
    return _dedupe_boxes(boxes)


def _dedupe_boxes(boxes: Sequence[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Collapse MSER's nested detections of the same glyph into one box each."""
    unique: list[tuple[int, int, int, int]] = []
    for box in sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True):
        if any(_iou(box, kept) > 0.5 for kept in unique):
            continue
        unique.append(box)
    return unique


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    intersection = float((x1 - x0) * (y1 - y0))
    union = float((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])) - intersection
    return intersection / union if union > 0 else 0.0


def _group_lines(
    boxes: Sequence[tuple[int, int, int, int]], params: TagDetectionParams
) -> list[tuple[tuple[int, int, int, int], int]]:
    """Chain glyph boxes into text lines. Returns ``[(line_bbox, glyph_count)]``."""
    ordered = sorted(boxes, key=lambda b: (b[1], b[0]))
    used: set[int] = set()
    lines: list[tuple[tuple[int, int, int, int], int]] = []

    for index, box in enumerate(ordered):
        if index in used:
            continue
        used.add(index)
        members = [box]
        height = box[3] - box[1]
        centre_y = (box[1] + box[3]) / 2.0
        right = box[2]
        changed = True
        while changed:
            changed = False
            for other_index, other in enumerate(ordered):
                if other_index in used:
                    continue
                other_height = other[3] - other[1]
                if min(height, other_height) <= 0:
                    continue
                if max(height, other_height) / min(height, other_height) > 2.5:
                    continue
                other_centre = (other[1] + other[3]) / 2.0
                if abs(other_centre - centre_y) > params.line_vertical_tolerance * height:
                    continue
                gap = other[0] - right
                if gap > params.line_horizontal_gap * height or other[2] < members[0][0]:
                    continue
                members.append(other)
                used.add(other_index)
                right = max(right, other[2])
                centre_y = sum((m[1] + m[3]) / 2.0 for m in members) / len(members)
                height = int(sum(m[3] - m[1] for m in members) / len(members))
                changed = True
        if len(members) < params.min_glyphs_per_line:
            continue
        lines.append(
            (
                (
                    min(m[0] for m in members),
                    min(m[1] for m in members),
                    max(m[2] for m in members),
                    max(m[3] for m in members),
                ),
                len(members),
            )
        )
    return lines


def _gaussian(value: float, ideal: float, sigma: float) -> float:
    """Bell-shaped closeness score in ``[0, 1]``."""
    if sigma <= 0:
        return 1.0 if value == ideal else 0.0
    return math.exp(-((value - ideal) ** 2) / (2.0 * sigma * sigma))


def _score_line(
    gray: np.ndarray, bbox: tuple[int, int, int, int], glyphs: int, params: TagDetectionParams
) -> float:
    """Score a candidate line on how much it looks like a stencilled equipment tag."""
    height, width = gray.shape[:2]
    x0, y0, x1, y1 = bbox
    crop = gray[max(0, y0): min(height, y1), max(0, x0): min(width, x1)]
    if crop.size == 0:
        return 0.0

    box_h = float(max(1, y1 - y0))
    box_w = float(max(1, x1 - x0))
    contrast = min(1.0, float(np.std(crop)) / params.contrast_reference)
    try:
        _, binary = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ink = 1.0 - (float(np.count_nonzero(binary)) / float(binary.size))
    except cv2.error:  # pragma: no cover - degenerate crop
        ink = params.ideal_ink_ratio
    ink = min(ink, 1.0 - ink) * 2.0  # polarity-agnostic: dark-on-light == light-on-dark

    centre_dx = ((x0 + x1) / 2.0 - width / 2.0) / (width / 2.0)
    centre_dy = ((y0 + y1) / 2.0 - height / 2.0) / (height / 2.0)
    centrality = max(0.0, 1.0 - math.hypot(centre_dx, centre_dy) / math.sqrt(2.0))

    return round(
        params.weight_glyphs * _gaussian(float(glyphs), params.ideal_glyph_count, params.glyph_count_sigma)
        + params.weight_aspect * _gaussian(box_w / box_h, params.ideal_aspect, params.aspect_sigma)
        + params.weight_height * _gaussian(box_h / float(height), params.ideal_height_ratio, params.height_sigma)
        + params.weight_contrast * contrast
        + params.weight_ink * _gaussian(ink, params.ideal_ink_ratio, params.ink_sigma)
        + params.weight_centre * centrality,
        4,
    )


def detect_tag_regions(
    image: np.ndarray, *, max_dimension: int, params: TagDetectionParams = DEFAULT_DETECTION_PARAMS
) -> tuple[list[TagRegion], np.ndarray, float]:
    """Locate candidate tag plates.

    Returns ``(regions, working_grayscale, scale_back)``. Regions carry bounding boxes already scaled
    back to the *original* image coordinate system so the frontend can float its card over the photo
    the technician actually took. ``working_grayscale`` is kept for OCR, which runs on the enhanced,
    downscaled image.

    Pure CPU: call through :func:`asyncio.to_thread`.
    """
    gray, scale_back = _prepare(image, max_dimension)
    glyphs = _glyph_boxes(gray, params)
    lines = _group_lines(glyphs, params)

    scored: list[tuple[float, tuple[int, int, int, int], int]] = []
    for bbox, count in lines:
        scored.append((_score_line(gray, bbox, count, params), bbox, count))
    scored.sort(key=lambda item: item[0], reverse=True)

    regions: list[TagRegion] = []
    for score, bbox, count in scored[: params.max_regions]:
        x0, y0, x1, y1 = bbox
        pad = int((y1 - y0) * params.region_padding_ratio)
        padded = (
            max(0, x0 - pad), max(0, y0 - pad),
            min(gray.shape[1], x1 + pad), min(gray.shape[0], y1 + pad),
        )
        regions.append(
            TagRegion(bbox=_scale_bbox(padded, scale_back), score=score, glyph_count=count)
        )
    logger.debug(
        "tag region detection complete",
        extra={"glyphs": len(glyphs), "lines": len(lines), "regions": len(regions)},
    )
    return regions, gray, scale_back


def _scale_bbox(bbox: tuple[int, int, int, int], factor: float) -> tuple[int, int, int, int]:
    return (
        int(round(bbox[0] * factor)),
        int(round(bbox[1] * factor)),
        int(round(bbox[2] * factor)),
        int(round(bbox[3] * factor)),
    )


# ======================================================================================
# Region OCR
# ======================================================================================

_TESSERACT_CONFIG: Final[str] = (
    "--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
)
"""Single text line, LSTM engine, plant-tag alphabet only. Restricting the alphabet is the single
biggest accuracy win on stencilled tags — it removes every punctuation and lowercase confusion."""


class RegionOCR:
    """Reads short text out of a cropped region.

    Backends are probed lazily and only once. ``pytesseract`` is a thin wrapper around a *binary*
    that is frequently absent even when the wheel is installed, so availability is decided by
    actually calling it, not by the import succeeding.
    """

    def __init__(self, settings: Settings, *, params: TagDetectionParams = DEFAULT_DETECTION_PARAMS) -> None:
        self._settings = settings
        self._params = params
        self._backend: str | None = None
        self._probed = False
        self._easyocr_reader: Any | None = None

    @property
    def backend(self) -> str:
        """Name of the active backend: ``tesseract``, ``easyocr``, or ``none``."""
        if not self._probed:
            self._probe()
        return self._backend or "none"

    @property
    def available(self) -> bool:
        return self.backend != "none"

    def _probe(self) -> None:
        """Decide which OCR backend actually works in this process. Runs once."""
        self._probed = True
        engine = self._settings.ocr_engine
        order: list[str] = []
        if engine == "tesseract":
            order = ["tesseract"]
        elif engine == "easyocr":
            order = ["easyocr"]
        else:
            order = ["tesseract", "easyocr"]

        for candidate in order:
            if candidate == "tesseract" and _HAS_PYTESSERACT and self._probe_tesseract():
                self._backend = "tesseract"
                logger.info("region OCR backend selected", extra={"ocr_backend": "tesseract"})
                return
            if candidate == "easyocr" and _HAS_EASYOCR:
                self._backend = "easyocr"
                logger.info("region OCR backend selected", extra={"ocr_backend": "easyocr"})
                return
        self._backend = "none"
        logger.warning(
            "no OCR backend available; photo queries will ask the technician to pick the tag",
            extra={
                "pytesseract_installed": _HAS_PYTESSERACT,
                "easyocr_installed": _HAS_EASYOCR,
                "install_hint": "install the tesseract binary and put it on PATH, or pip install easyocr",
            },
        )

    @staticmethod
    def _probe_tesseract() -> bool:
        try:
            pytesseract.get_tesseract_version()
        except Exception as exc:  # external boundary: subprocess to the tesseract binary
            logger.warning(
                "pytesseract is installed but the tesseract binary is not callable",
                extra={"error": str(exc)},
            )
            return False
        return True

    def read(self, gray: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[str, float]:
        """OCR one region of ``gray`` (working-resolution coordinates).

        Returns ``("", 0.0)`` when no backend is available — an empty read, not an exception, because
        the caller has a useful degraded path (ask the technician to choose).

        Raises:
            OCRError: a backend is available but failed on this crop.
        """
        if not self.available:
            return "", 0.0
        crop = self._crop(gray, bbox)
        if crop is None:
            return "", 0.0
        try:
            if self._backend == "tesseract":
                return self._read_tesseract(crop)
            return self._read_easyocr(crop)
        except Exception as exc:  # external boundary: OCR engine
            raise OCRError(
                "OCR failed on the detected tag region. The photo may be too blurred or too dark; "
                "retake it square-on with the plate filling the frame.",
                context={"ocr_backend": self._backend, "bbox": list(bbox)},
                cause=exc,
            ) from exc

    def _crop(self, gray: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
        height, width = gray.shape[:2]
        x0, y0 = max(0, bbox[0]), max(0, bbox[1])
        x1, y1 = min(width, bbox[2]), min(height, bbox[3])
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        crop = gray[y0:y1, x0:x1]
        scale = self._params.ocr_upscale
        enlarged = cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale), interpolation=cv2.INTER_CUBIC)
        blurred = cv2.bilateralFilter(enlarged, 5, 60, 60)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Tesseract expects dark text on white; flip when the plate is engraved light-on-dark.
        if float(np.mean(binary)) < 127.0:
            binary = cv2.bitwise_not(binary)
        return binary

    def _read_tesseract(self, crop: np.ndarray) -> tuple[str, float]:
        data: dict[str, list[Any]] = pytesseract.image_to_data(
            crop, config=_TESSERACT_CONFIG, output_type=pytesseract.Output.DICT
        )
        words: list[str] = []
        confidences: list[float] = []
        for text, conf in zip(data.get("text", []), data.get("conf", []), strict=False):
            token = str(text).strip()
            if not token:
                continue
            try:
                value = float(conf)
            except (TypeError, ValueError):
                value = -1.0
            if value < 0:
                continue
            words.append(token)
            confidences.append(value / 100.0)
        if not words:
            return "", 0.0
        return " ".join(words), round(sum(confidences) / len(confidences), 4)

    def _read_easyocr(self, crop: np.ndarray) -> tuple[str, float]:
        if self._easyocr_reader is None:
            self._easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        results = self._easyocr_reader.readtext(crop, detail=1, paragraph=False)
        if not results:
            return "", 0.0
        words = [str(item[1]).strip() for item in results if str(item[1]).strip()]
        confidences = [float(item[2]) for item in results if len(item) > 2]
        if not words:
            return "", 0.0
        mean = sum(confidences) / len(confidences) if confidences else 0.0
        return " ".join(words), round(max(0.0, min(1.0, mean)), 4)


# ======================================================================================
# Tag normalisation (D5 semantics, re-implemented locally — see module docstring)
# ======================================================================================

_TAG_STRUCTURE: Final[re.Pattern[str]] = re.compile(
    r"^([A-Z0-9]{1,6})[\s\-‐-―_/.]{0,3}([A-Z0-9]{1,6})([A-Z]{0,2})$"
)
"""``<PREFIX><separator><NUMBER><optional suffix>`` — the plant-tag grammar, read permissively so an
OCR read of ``Pl0l`` or ``P 101`` still parses before glyph correction is applied."""

_TO_LETTER: Final[dict[str, str]] = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"}
"""Digit → letter, applied to the prefix group. Mirrors D5's glyph-confusion map."""

_TO_DIGIT: Final[dict[str, str]] = {
    "O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "S": "5", "Z": "2", "G": "6", "B": "8", "T": "7",
}
"""Letter → digit, applied to the numeric group."""

_JUNK: Final[re.Pattern[str]] = re.compile(r"[^A-Z0-9\-]")

#: Confidence for a tag that matched the registry exactly after glyph correction. Not 1.0 — a
#: correction was applied — but high enough to act on.
_CORRECTED_EXACT_CONFIDENCE: Final[float] = 0.92

#: When the best and second-best fuzzy scores are within this many points, the read is ambiguous.
#: ``P-101`` and ``P-107`` differ by one glyph, which is exactly the case that must never resolve
#: silently.
_AMBIGUITY_MARGIN: Final[float] = 6.0

#: Multiplier applied to confidence when the match is ambiguous, pushing it below the UI's
#: act-on-it band so the technician is asked to choose.
_AMBIGUITY_PENALTY: Final[float] = 0.6

#: How many alternatives to offer. More than four is a menu, not a question.
_MAX_ALTERNATIVES: Final[int] = 4


class MobileTagNormalizer:
    """Implements :class:`indra.core.contracts.TagNormalizer` for the photo path.

    Never corrects silently: the return is ``(tag_or_None, confidence, alternatives)`` and the caller
    is expected to show the alternatives whenever confidence is not high.
    """

    name: Final[str] = "mobile_tag_normalizer"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def normalize(
        self, raw: str, *, registry: Sequence[str] | None = None
    ) -> tuple[str | None, float, list[str]]:
        """Resolve an OCR read to a registry tag."""
        candidates = [tag.strip().upper() for tag in (registry or []) if tag and tag.strip()]
        cleaned = self._clean(raw)
        if not cleaned:
            return None, 0.0, candidates[:_MAX_ALTERNATIVES]

        if cleaned in candidates:
            return cleaned, 1.0, []

        corrected = self._glyph_correct(cleaned)
        if corrected and corrected in candidates:
            logger.info(
                "plant tag resolved after glyph correction",
                extra={"raw": raw, "cleaned": cleaned, "resolved_tag": corrected},
            )
            return corrected, _CORRECTED_EXACT_CONFIDENCE, []

        if not candidates:
            # No registry to check against: structurally valid, but unverified.
            if corrected:
                return None, 0.0, []
            return None, 0.0, []

        return self._fuzzy(corrected or cleaned, candidates, raw=raw)

    # -- internals ---------------------------------------------------------------------
    @staticmethod
    def _clean(raw: str) -> str:
        """Uppercase, drop OCR noise, and normalise unicode dashes to ASCII hyphen."""
        if not raw:
            return ""
        text = raw.strip().upper()
        text = re.sub(r"[‐-―−]", "-", text)
        text = _JUNK.sub("", text.replace(" ", ""))
        return text.strip("-")

    @staticmethod
    def _glyph_correct(cleaned: str) -> str | None:
        """Apply the glyph-confusion map positionally: letters in the prefix, digits in the number."""
        match = _TAG_STRUCTURE.match(cleaned.replace("-", "-"))
        if not match:
            return None
        prefix, number, suffix = match.groups()
        fixed_prefix = "".join(_TO_LETTER.get(ch, ch) for ch in prefix)
        fixed_number = "".join(_TO_DIGIT.get(ch, ch) for ch in number)
        if not fixed_prefix.isalpha() or not fixed_number.isdigit():
            return None
        return f"{fixed_prefix}-{fixed_number}{suffix}"

    def _fuzzy(
        self, needle: str, candidates: list[str], *, raw: str
    ) -> tuple[str | None, float, list[str]]:
        """Rank the registry with rapidfuzz and decide whether the winner is trustworthy."""
        try:
            ranked = process.extract(
                needle, candidates, scorer=fuzz.WRatio, limit=_MAX_ALTERNATIVES + 1
            )
        except Exception as exc:  # external boundary: rapidfuzz C extension
            logger.warning("fuzzy tag matching failed", extra={"needle": needle, "error": str(exc)})
            return None, 0.0, candidates[:_MAX_ALTERNATIVES]
        if not ranked:
            return None, 0.0, []

        best_tag, best_score = str(ranked[0][0]), float(ranked[0][1])
        runner_up = float(ranked[1][1]) if len(ranked) > 1 else 0.0
        alternatives = [str(item[0]) for item in ranked[1 : _MAX_ALTERNATIVES + 1]]
        threshold = float(self._settings.pid_tag_fuzzy_threshold)

        confidence = best_score / 100.0
        ambiguous = (best_score - runner_up) < _AMBIGUITY_MARGIN
        if ambiguous:
            confidence *= _AMBIGUITY_PENALTY

        if best_score < threshold:
            logger.info(
                "no registry tag cleared the fuzzy threshold",
                extra={"raw": raw, "needle": needle, "best": best_tag, "score": best_score,
                       "threshold": threshold},
            )
            return None, round(confidence, 4), [best_tag, *alternatives][:_MAX_ALTERNATIVES]

        if ambiguous:
            logger.info(
                "ambiguous tag match; returning alternatives for the technician to choose",
                extra={"raw": raw, "best": best_tag, "score": best_score, "runner_up": runner_up},
            )
            return best_tag, round(confidence, 4), [best_tag, *alternatives][:_MAX_ALTERNATIVES]

        return best_tag, round(confidence, 4), alternatives[: _MAX_ALTERNATIVES - 1]


# ======================================================================================
# Engine
# ======================================================================================

#: Equipment registry cache lifetime. The registry changes only when documents are ingested, and the
#: photo path must not pay a graph round trip per snapshot.
_REGISTRY_TTL_S: Final[float] = 120.0

#: Upper bound on alerts pulled for the status card.
_ALERT_FETCH_LIMIT: Final[int] = 200

#: Quick documents shown on the AR card. More does not fit on a phone held at arm's length.
_QUICK_DOCUMENT_LIMIT: Final[int] = 4

#: Above this the card asserts the tag; below it the card asks.
_ASSERT_CONFIDENCE: Final[float] = 0.8


@dataclass(slots=True)
class _RegistrySnapshot:
    equipment: list[Equipment] = field(default_factory=list)
    fetched_at: float = 0.0

    @property
    def tags(self) -> list[str]:
        return [item.tag for item in self.equipment]


class PhotoQueryEngine:
    """Turns a photograph of an equipment tag into an AR overlay payload."""

    def __init__(self, deps: AgentDeps, *, params: TagDetectionParams = DEFAULT_DETECTION_PARAMS) -> None:
        self._deps = deps
        self._settings = deps.settings
        self._params = params
        self._ocr = RegionOCR(deps.settings, params=params)
        self._normalizer = MobileTagNormalizer(deps.settings)
        self._registry = _RegistrySnapshot()
        self._registry_lock = asyncio.Lock()
        self._knowledge_graph: KnowledgeGraphService | None = None
        self._proactive: ProactiveService | None = None

    def bind(
        self,
        *,
        knowledge_graph: KnowledgeGraphService | None = None,
        proactive: ProactiveService | None = None,
    ) -> None:
        """Attach sibling services. Called by the agent's ``bind()``."""
        self._knowledge_graph = knowledge_graph
        self._proactive = proactive

    @property
    def ocr_backend(self) -> str:
        return self._ocr.backend

    async def warm(self) -> None:
        """Pre-load the equipment registry so the first photo is as fast as the tenth."""
        await self._equipment_registry()

    async def run(self, image: bytes, *, hint_text: str | None = None) -> PhotoQueryResponse:
        """Resolve the tag in ``image`` and assemble the overlay card.

        Args:
            image: Raw photo bytes from the phone camera.
            hint_text: Text the client already has for the tag — a browser-side OCR result, or what
                the technician typed. It takes priority over on-device OCR, which is what keeps the
                feature usable on a host with no OCR binary.

        Raises:
            FileValidationError: the payload is not a decodable image.
        """
        started = time.perf_counter()
        decoded = await asyncio.to_thread(decode_image, image)

        vision_task = asyncio.to_thread(self._vision_sync, decoded)
        registry_task = self._equipment_registry()
        regions, registry = await asyncio.gather(vision_task, registry_task)

        tag, confidence, alternatives, bbox = self._resolve(regions, registry.tags, hint_text=hint_text)

        response = await self._assemble(
            tag=tag, confidence=confidence, alternatives=alternatives, bbox=bbox, registry=registry
        )
        logger.info(
            "photo query complete",
            extra={
                "equipment_tag": tag,
                "tag_confidence": round(confidence, 4),
                "alternatives": len(response.tag_alternatives),
                "regions": len(regions),
                "ocr_backend": self._ocr.backend,
                "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
            },
        )
        return response

    # -- vision ------------------------------------------------------------------------
    def _vision_sync(self, image: np.ndarray) -> list[TagRegion]:
        """Detect and OCR in a single worker-thread hop. All CPU, no awaits."""
        regions, gray, scale_back = detect_tag_regions(
            image, max_dimension=self._settings.photo_max_dimension_px, params=self._params
        )
        if not regions or not self._ocr.available:
            return regions
        inverse = 1.0 / scale_back if scale_back else 1.0
        read: list[TagRegion] = []
        for region in regions:
            working_bbox = _scale_bbox(region.bbox, inverse)
            try:
                text, ocr_confidence = self._ocr.read(gray, working_bbox)
            except OCRError as exc:
                logger.warning("region OCR failed; keeping the region without text",
                               extra={"error": exc.message})
                read.append(region)
                continue
            read.append(
                TagRegion(
                    bbox=region.bbox,
                    score=region.score,
                    glyph_count=region.glyph_count,
                    text=text,
                    ocr_confidence=ocr_confidence,
                )
            )
        return read

    def _resolve(
        self, regions: Sequence[TagRegion], registry: Sequence[str], *, hint_text: str | None
    ) -> tuple[str | None, float, list[str], tuple[int, int, int, int] | None]:
        """Pick the best tag across the client hint and every OCR'd region."""
        best: tuple[str | None, float, list[str], tuple[int, int, int, int] | None] = (
            None, 0.0, [], regions[0].bbox if regions else None,
        )

        if hint_text and hint_text.strip():
            tag, confidence, alternatives = self._normalizer.normalize(hint_text, registry=registry)
            if tag is not None:
                return tag, confidence, alternatives, best[3]
            best = (None, confidence, alternatives, best[3])

        for region in regions:
            if not region.has_text:
                continue
            tag, confidence, alternatives = self._normalizer.normalize(region.text, registry=registry)
            # An OCR read is never more trustworthy than the OCR itself: fold the character
            # confidence into the tag confidence rather than reporting the fuzzy score alone.
            blended = confidence * max(region.ocr_confidence, 0.5) if tag else confidence
            if tag is not None and blended > best[1]:
                best = (tag, round(blended, 4), alternatives, region.bbox)
            elif tag is None and not best[0] and alternatives and not best[2]:
                best = (None, round(blended, 4), alternatives, region.bbox)

        if best[0] is None and not best[2] and registry:
            # Nothing readable. Offer the assets a technician is most likely standing next to
            # (criticality order) rather than asserting a tag we cannot see.
            best = (None, 0.0, list(registry[:_MAX_ALTERNATIVES]), best[3])
        return best

    # -- assembly ----------------------------------------------------------------------
    async def _assemble(
        self,
        *,
        tag: str | None,
        confidence: float,
        alternatives: list[str],
        bbox: tuple[int, int, int, int] | None,
        registry: _RegistrySnapshot,
    ) -> PhotoQueryResponse:
        if tag is None:
            return PhotoQueryResponse(
                detected_tag=None,
                tag_confidence=_clamp(confidence),
                tag_alternatives=alternatives[:_MAX_ALTERNATIVES],
                status_line=self._unresolved_status_line(alternatives),
                quick_actions=[
                    RecommendedAction(
                        action="Select the correct equipment tag from the suggestions",
                        urgency=Severity.INFO,
                        rationale="The tag could not be read from the photo with enough confidence "
                                  "to act on. Confirming beats guessing on a plant floor.",
                    )
                ],
                bbox=bbox,
            )

        equipment = next((item for item in registry.equipment if item.tag == tag), None)
        maintenance, alerts, documents, procedures = await asyncio.gather(
            self._last_maintenance(tag),
            self._open_alerts(tag),
            self._quick_documents(tag),
            self._procedures(tag),
        )
        if equipment is None:
            equipment = await self._equipment(tag)

        return PhotoQueryResponse(
            detected_tag=tag,
            tag_confidence=_clamp(confidence),
            tag_alternatives=[] if confidence >= _ASSERT_CONFIDENCE else alternatives[:_MAX_ALTERNATIVES],
            equipment=equipment,
            status_line=_status_line(tag, equipment, alerts, maintenance, confidence),
            last_maintenance=maintenance,
            open_alerts=alerts,
            quick_documents=documents,
            quick_actions=_quick_actions(tag, alerts, procedures, maintenance),
            bbox=bbox,
        )

    def _unresolved_status_line(self, alternatives: Sequence[str]) -> str:
        if alternatives:
            options = " or ".join(alternatives[:2]) if len(alternatives) <= 2 else ", ".join(alternatives)
            return f"Tag not read with confidence — did you mean {options}?"
        if not self._ocr.available:
            return (
                "No text recognition is available on this server. Type the tag, or pick the "
                "equipment from the list."
            )
        return "No equipment tag was legible in that photo. Move closer and retake it square-on."

    # -- data access (every call wrapped; a dead store degrades the card, never fails it) --
    async def _equipment_registry(self) -> _RegistrySnapshot:
        now = time.monotonic()
        if self._registry.equipment and (now - self._registry.fetched_at) < _REGISTRY_TTL_S:
            return self._registry
        async with self._registry_lock:
            if self._registry.equipment and (time.monotonic() - self._registry.fetched_at) < _REGISTRY_TTL_S:
                return self._registry
            try:
                equipment = await self._deps.graph.list_equipment()
            except IndraError as exc:
                logger.warning(
                    "equipment registry unavailable; photo queries cannot resolve tags",
                    extra={"error": exc.message},
                )
                return self._registry
            except Exception as exc:  # defensive: store contract violation
                logger.warning("equipment registry raised an untyped error", extra={"error": str(exc)})
                return self._registry
            ordered = sorted(equipment, key=_criticality_sort_key(self._settings))
            self._registry = _RegistrySnapshot(equipment=ordered, fetched_at=time.monotonic())
            logger.debug("equipment registry refreshed", extra={"count": len(ordered)})
            return self._registry

    async def _equipment(self, tag: str) -> Equipment | None:
        try:
            return await self._deps.graph.get_equipment(tag)
        except IndraError as exc:
            logger.warning("equipment lookup failed", extra={"equipment_tag": tag, "error": exc.message})
        except Exception as exc:  # defensive
            logger.warning("equipment lookup raised an untyped error",
                           extra={"equipment_tag": tag, "error": str(exc)})
        return None

    async def _last_maintenance(self, tag: str) -> MaintenanceRecord | None:
        try:
            history = await self._deps.graph.maintenance_history(tag)
        except IndraError as exc:
            logger.warning("maintenance history unavailable",
                           extra={"equipment_tag": tag, "error": exc.message})
            return None
        except Exception as exc:  # defensive
            logger.warning("maintenance history raised an untyped error",
                           extra={"equipment_tag": tag, "error": str(exc)})
            return None
        if not history:
            return None
        return max(history, key=lambda record: record.performed_on)

    async def _open_alerts(self, tag: str) -> list[Alert]:
        alerts: list[Alert] = []
        if self._proactive is not None:
            try:
                alerts = list(await self._proactive.alerts(unresolved_only=True))
            except IndraError as exc:
                logger.warning("proactive alerts unavailable", extra={"error": exc.message})
            except Exception as exc:  # defensive
                logger.warning("proactive alerts raised an untyped error", extra={"error": str(exc)})
        if not alerts:
            try:
                alerts = list(
                    await self._deps.metadata.list_alerts(unresolved_only=True, limit=_ALERT_FETCH_LIMIT)
                )
            except IndraError as exc:
                logger.warning("alert store unavailable", extra={"error": exc.message})
                return []
            except Exception as exc:  # defensive
                logger.warning("alert store raised an untyped error", extra={"error": str(exc)})
                return []
        matching = [alert for alert in alerts if alert.equipment_tag.upper() == tag.upper() and not alert.resolved]
        matching.sort(key=lambda alert: (-alert.severity.rank, alert.raised_at), reverse=False)
        return matching

    async def _quick_documents(self, tag: str) -> list[SourceRef]:
        if self._knowledge_graph is None:
            return []
        try:
            result = await self._knowledge_graph.retrieve(
                f"{tag} maintenance history, procedure, and known failure modes",
                equipment_tag=tag,
                top_k=_QUICK_DOCUMENT_LIMIT,
            )
        except IndraError as exc:
            logger.warning("quick document retrieval failed",
                           extra={"equipment_tag": tag, "error": exc.message})
            return []
        except Exception as exc:  # defensive
            logger.warning("quick document retrieval raised an untyped error",
                           extra={"equipment_tag": tag, "error": str(exc)})
            return []
        seen: set[str] = set()
        documents: list[SourceRef] = []
        for passage in result.passages:
            source = passage.as_source()
            if source.document_id in seen:
                continue
            seen.add(source.document_id)
            documents.append(source)
            if len(documents) >= _QUICK_DOCUMENT_LIMIT:
                break
        return documents

    async def _procedures(self, tag: str) -> list[Procedure]:
        try:
            return list(await self._deps.graph.procedures_for(tag))
        except IndraError as exc:
            logger.warning("procedure lookup failed", extra={"equipment_tag": tag, "error": exc.message})
        except Exception as exc:  # defensive
            logger.warning("procedure lookup raised an untyped error",
                           extra={"equipment_tag": tag, "error": str(exc)})
        return []


def _criticality_sort_key(settings: Settings):  # type: ignore[no-untyped-def]
    """Order equipment by ``settings.offline_priority_order`` then tag, deterministically."""
    order = {value: index for index, value in enumerate(settings.offline_priority_order)}
    fallback = len(order)

    def key(item: Equipment) -> tuple[int, str]:
        return order.get(item.criticality.value, fallback), item.tag

    return key


def _status_line(
    tag: str,
    equipment: Equipment | None,
    alerts: Sequence[Alert],
    maintenance: MaintenanceRecord | None,
    confidence: float,
) -> str:
    """One dense line: what it is, how critical, what is wrong, when it was last touched."""
    parts: list[str] = [tag]
    if equipment is not None:
        if equipment.name:
            parts.append(equipment.name)
        elif equipment.equipment_type and equipment.equipment_type != "unknown":
            parts.append(equipment.equipment_type.replace("_", " ").title())
        parts.append(f"Criticality {equipment.criticality.value}")
        if equipment.location:
            parts.append(equipment.location)
    else:
        parts.append("not found in the knowledge graph")

    if alerts:
        worst = max(alerts, key=lambda alert: alert.severity.rank)
        parts.append(f"{len(alerts)} open alert{'s' if len(alerts) != 1 else ''} (worst {worst.severity.value})")
    else:
        parts.append("no open alerts")

    if maintenance is not None:
        parts.append(f"last maintained {maintenance.performed_on.isoformat()}")
    else:
        parts.append("no maintenance on record")

    if confidence < _ASSERT_CONFIDENCE:
        parts.append(f"tag read at {confidence:.0%} confidence — confirm before acting")
    return " · ".join(parts)


def _quick_actions(
    tag: str,
    alerts: Sequence[Alert],
    procedures: Sequence[Procedure],
    maintenance: MaintenanceRecord | None,
) -> list[RecommendedAction]:
    """Assemble the card's action buttons, most urgent first, deduplicated by action text."""
    actions: list[RecommendedAction] = []
    seen: set[str] = set()

    def add(action: RecommendedAction) -> None:
        key = action.action.strip().lower()
        if key in seen:
            return
        seen.add(key)
        actions.append(action)

    for alert in sorted(alerts, key=lambda item: item.severity.rank, reverse=True):
        for recommended in alert.recommended_actions:
            add(recommended)
        if not alert.recommended_actions:
            add(
                RecommendedAction(
                    action=f"Review open alert: {alert.title}",
                    urgency=alert.severity,
                    rationale=alert.body[:400],
                )
            )

    for procedure in procedures[:2]:
        add(
            RecommendedAction(
                action=f"Open procedure: {procedure.title}",
                urgency=Severity.INFO,
                rationale=f"Applies to {tag}." + (
                    f" {len(procedure.steps)} steps." if procedure.steps else ""
                ),
                procedure_id=procedure.procedure_id,
                estimated_minutes=procedure.estimated_minutes,
            )
        )

    if maintenance is not None and maintenance.status == "open":
        add(
            RecommendedAction(
                action=f"Close out open work order from {maintenance.performed_on.isoformat()}",
                urgency=Severity.WARNING,
                rationale=maintenance.findings[:400] or "Work order is still open.",
            )
        )

    add(
        RecommendedAction(
            action=f"View full history for {tag}",
            urgency=Severity.INFO,
            rationale="Maintenance records, failures, and documents linked to this asset.",
        )
    )
    add(
        RecommendedAction(
            action=f"Log an observation on {tag}",
            urgency=Severity.INFO,
            rationale="Queued locally and synced when connectivity returns.",
        )
    )
    return actions


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


__all__ = [
    "DEFAULT_DETECTION_PARAMS",
    "MobileTagNormalizer",
    "PhotoQueryEngine",
    "RegionOCR",
    "TagDetectionParams",
    "TagRegion",
    "decode_image",
    "detect_tag_regions",
]
