"""P&ID computer vision — INDRA's headline differentiator (``docs/DECISIONS.md`` D4).

Most document pipelines treat an engineering drawing as an image with some words in it. This module
reads it as a *graph*: symbols are nodes, pipe runs are edges, arrowheads are direction, and the
lettering beside each symbol is the plant tag that joins the drawing to every work order ever
written about that asset.

The pipeline, in order:

1. **Binarise** — greyscale, adaptive threshold, a morphological close to bridge the gaps that
   scanning leaves in thin drafting lines.
2. **Circles** — ``HoughCircles`` finds pumps and instrument bubbles. Radius alone cannot separate
   them, so the ring fill and the interior ink ratio are measured too: an instrument bubble is a
   thin ring holding two or three glyphs, a pump is a larger circle with an impeller triangle in it.
3. **Rectangles** — contours, ``approxPolyDP``, then classification by aspect ratio and area:
   wide boxes are heat exchangers, tall boxes are columns and vessels, near-square boxes are tanks.
4. **Tags** — each symbol's neighbourhood is cropped and OCR'd with a tag-only character whitelist,
   then resolved through :class:`~indra.agents.ingestion_agent.tag_normalizer.PlantTagNormalizer`
   (D5). A resolved tag *re-classifies* the symbol, because ``PIC-101`` beside a circle settles the
   pump-versus-instrument question that pixels alone leave open.
5. **Pipe runs** — ``HoughLinesP``, collinear merge, then a union-find over segment endpoints so an
   L-shaped run is one edge rather than two.
6. **Connections** — run ends are associated to the nearest symbol within
   ``settings.pid_connection_max_distance_px``; dash ratio separates instrument signal lines from
   process piping; a triangular arrowhead template match resolves flow direction.

Everything degrades. No OpenCV, no readable image, a drawing that is actually a photograph of a
noticeboard — each yields a :class:`~indra.core.models.PIDParseResult` carrying warnings, never an
exception into the pipeline. Three symbols and one connection is a success; a crash is not.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, Sequence

import numpy as np
from numpy.typing import NDArray

from indra.agents.ingestion_agent.ocr import (
    OCREngine,
    OCRResult,
    PSM_SINGLE_LINE,
    PSM_SINGLE_WORD,
    load_image,
    to_grayscale,
)
from indra.agents.ingestion_agent.tag_normalizer import (
    PlantTagNormalizer,
    TagResolution,
    equipment_type_for,
)
from indra.core.config import Settings, get_settings
from indra.core.ids import new_id
from indra.core.logging import get_logger
from indra.core.models import (
    Confidence,
    DetectedConnection,
    DetectedSymbol,
    DocumentMeta,
    EntityType,
    ExtractedEntity,
    ExtractedRelationship,
    PIDParseResult,
    RelationType,
    SymbolClass,
)

try:
    import cv2

    _HAS_CV2 = True
except ImportError:  # pragma: no cover - optional native dependency
    cv2 = None  # type: ignore[assignment]
    _HAS_CV2 = False

logger = get_logger(__name__)

DetectorName = Literal["rule_based", "yolo", "template"]

# ======================================================================================
# Tunables.
#
# These are *drawing conventions*, not deployment policy: an instrument bubble really is a small
# thin ring, a heat exchanger really is a wide box. The genuine deployment knobs — Hough thresholds,
# the symbol confidence floor, the connection radius, the fuzzy tag cutoff — all come from
# ``Settings`` and are read at call time.
# ======================================================================================

_ADAPTIVE_BLOCK: Final[int] = 25
"""Adaptive-threshold window. Large enough to survive the shading band on a scanned A1 sheet."""

_ADAPTIVE_C: Final[int] = 9
_CLOSE_KERNEL: Final[int] = 2
"""A 2 px close bridges scan dropouts in 1 px drafting lines without welding parallel lines together."""

_MIN_RADIUS_RATIO: Final[float] = 0.008
_MAX_RADIUS_RATIO: Final[float] = 0.055
"""Circle radii as a fraction of the short edge. Below the floor is a bullet point; above the
ceiling is a border decoration, not a symbol."""

_INSTRUMENT_RADIUS_RATIO: Final[float] = 0.022
"""Circles at or below this fraction of the short edge are instrument bubbles rather than pumps."""

_PUMP_INK_RATIO: Final[float] = 0.20
"""Interior ink fraction above which a circle is holding an impeller triangle, not two letters."""

_MIN_RECT_AREA_RATIO: Final[float] = 0.00035
_MAX_RECT_AREA_RATIO: Final[float] = 0.16
"""Rectangle area bounds as a fraction of the canvas. The ceiling excludes the drawing frame."""

_EXCHANGER_ASPECT: Final[float] = 2.1
"""Width/height at or above which a box is a shell-and-tube exchanger rather than a vessel."""

_COLUMN_ASPECT: Final[float] = 0.55
"""Width/height at or below which a box is a column or tall vessel."""

_TANK_MIN_AREA_RATIO: Final[float] = 0.012
"""A near-square box this large is a storage tank; smaller ones are equipment boxes."""

_RECTANGULARITY_FLOOR: Final[float] = 0.62
"""Contour area / bounding-box area. Below this the quadrilateral fit is not describing a box."""

_NMS_IOU: Final[float] = 0.35
"""Two detections overlapping more than this are the same symbol seen twice."""

_ARROW_MATCH_THRESHOLD: Final[float] = 0.42
"""Normalised cross-correlation floor for accepting a triangular arrowhead."""

_ARROW_SIZE_RATIO: Final[float] = 0.010
"""Arrowhead template edge as a fraction of the short edge, floored at :data:`_MIN_ARROW_PX`."""

_MIN_ARROW_PX: Final[int] = 7
_MAX_ARROW_PX: Final[int] = 21

_DASH_SOLID_RATIO: Final[float] = 0.72
"""Ink coverage along a run below which the line is dashed — i.e. an instrument signal, not pipe."""

_ANGLE_TOLERANCE_DEG: Final[float] = 7.0
"""Deviation from horizontal/vertical still treated as an orthogonal pipe run."""

_MIN_RUN_LENGTH_RATIO: Final[float] = 0.02
"""Runs shorter than this fraction of the diagonal are hatching or lettering, not piping."""

_MAX_OCR_SYMBOLS: Final[int] = 120
"""Cap on region-OCR calls per drawing. A sheet with more symbols than this is a plot plan."""

_MAX_PIPE_SPEC_RUNS: Final[int] = 12
"""Only the longest runs get a pipe-spec OCR pass; short jumpers never carry a line number."""

_LABEL_BAND_RATIO: Final[float] = 1.15
"""Height of the label band searched under a symbol, as a multiple of the symbol's height."""

_DRAWING_MIN_PIXELS: Final[int] = 420_000
"""A drawing is a large canvas. Below ~0.4 MP this is a photo or a scanned form."""

_DRAWING_MIN_LINE_DENSITY: Final[float] = 55.0
"""Long straight segments per megapixel. Prose scans sit near zero; P&IDs run into the hundreds."""

_DRAWING_MAX_TEXT_DENSITY: Final[float] = 0.055
"""Fraction of the canvas covered by text-shaped components. A page of prose is an order higher."""

_TEXT_COMPONENT_MAX_HEIGHT: Final[int] = 46
_TEXT_COMPONENT_MIN_HEIGHT: Final[int] = 5
_TEXT_COMPONENT_MAX_ASPECT: Final[float] = 8.0

_PIPE_SPEC_RE_SOURCE: Final[str] = r'\b\d{1,2}\s*"?\s*[-/]\s*[A-Z]{1,4}\s*[-/]\s*\d{2,5}\b'

_TYPE_TO_CLASS: Final[dict[str, SymbolClass]] = {
    "pump": SymbolClass.PUMP,
    "vessel": SymbolClass.VESSEL,
    "tank": SymbolClass.TANK,
    "heat_exchanger": SymbolClass.HEAT_EXCHANGER,
    "compressor": SymbolClass.COMPRESSOR,
    "filter": SymbolClass.FILTER,
    "valve": SymbolClass.VALVE,
    "instrument": SymbolClass.INSTRUMENT,
    "reactor": SymbolClass.VESSEL,
    "motor": SymbolClass.PUMP,
}

_CLASS_KEEPS_EQUIPMENT: Final[frozenset[SymbolClass]] = frozenset(
    {
        SymbolClass.PUMP,
        SymbolClass.VESSEL,
        SymbolClass.HEAT_EXCHANGER,
        SymbolClass.COMPRESSOR,
        SymbolClass.TANK,
        SymbolClass.FILTER,
        SymbolClass.VALVE,
        SymbolClass.INSTRUMENT,
    }
)


# ======================================================================================
# Drawing signature
# ======================================================================================


@dataclass(frozen=True, slots=True)
class DrawingSignature:
    """Evidence for (or against) treating an image as an engineering drawing."""

    is_drawing: bool
    line_density: float
    text_density: float
    canvas_pixels: int
    rationale: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "is_drawing": self.is_drawing,
            "line_density_per_mpx": round(self.line_density, 2),
            "text_density": round(self.text_density, 4),
            "canvas_pixels": self.canvas_pixels,
            "rationale": self.rationale,
        }


def looks_like_drawing(array: NDArray[np.uint8], *, settings: Settings | None = None) -> DrawingSignature:
    """Decide whether ``array`` is an engineering drawing rather than a photo or a scanned page.

    Three independent signals, all cheap: a large canvas, a high density of long straight segments,
    and a *low* density of text-shaped connected components. A scanned SOP has the text density of
    prose and almost no long lines; a P&ID is the mirror image of that.

    Runs synchronously — call it from a worker thread.
    """
    cfg = settings or get_settings()
    if array is None or array.size == 0:
        return DrawingSignature(False, 0.0, 0.0, 0, "Image could not be decoded")

    height, width = array.shape[:2]
    canvas = int(height * width)
    if not _HAS_CV2:
        return DrawingSignature(
            False, 0.0, 0.0, canvas,
            "OpenCV is not installed, so drawing detection cannot run; treated as a plain image",
        )

    megapixels = max(canvas / 1_000_000.0, 1e-6)
    gray = to_grayscale(array)
    try:
        edges = cv2.Canny(gray, 60, 170, apertureSize=3)
        min_length = max(int(_MIN_RUN_LENGTH_RATIO * max(height, width)), cfg.pid_hough_min_line_length)
        raw_lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=math.pi / 180.0,
            threshold=int(cfg.pid_hough_threshold),
            minLineLength=min_length,
            maxLineGap=int(cfg.pid_hough_max_line_gap),
        )
        line_count = 0 if raw_lines is None else int(len(raw_lines))
        text_density = _text_component_density(gray)
    except cv2.error as exc:  # pragma: no cover - malformed input
        logger.warning("drawing signature computation failed", extra={"error": str(exc)})
        return DrawingSignature(False, 0.0, 0.0, canvas, f"OpenCV failed while profiling: {exc}")

    line_density = line_count / megapixels
    big_enough = canvas >= _DRAWING_MIN_PIXELS
    linear = line_density >= _DRAWING_MIN_LINE_DENSITY
    sparse_text = text_density <= _DRAWING_MAX_TEXT_DENSITY
    is_drawing = big_enough and linear and sparse_text

    rationale = (
        f"canvas {width}x{height}px ({'>=' if big_enough else '<'} {_DRAWING_MIN_PIXELS} threshold), "
        f"{line_density:.0f} long segments/MP ({'>=' if linear else '<'} {_DRAWING_MIN_LINE_DENSITY:.0f}), "
        f"text coverage {text_density:.3f} ({'<=' if sparse_text else '>'} {_DRAWING_MAX_TEXT_DENSITY})"
    )
    return DrawingSignature(is_drawing, line_density, text_density, canvas, rationale)


def _text_component_density(gray: NDArray[np.uint8]) -> float:
    """Fraction of the canvas covered by connected components shaped like lettering."""
    binary = _binarise(gray)
    if binary is None:
        return 0.0
    try:
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    except cv2.error:  # pragma: no cover - defensive
        return 0.0
    total = float(gray.shape[0] * gray.shape[1]) or 1.0
    covered = 0
    for index in range(1, min(count, 20_000)):
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        if not (_TEXT_COMPONENT_MIN_HEIGHT <= height <= _TEXT_COMPONENT_MAX_HEIGHT):
            continue
        if height <= 0 or width / height > _TEXT_COMPONENT_MAX_ASPECT:
            continue
        covered += width * height
    return covered / total


# ======================================================================================
# Low-level image operations
# ======================================================================================


def _binarise(gray: NDArray[np.uint8]) -> NDArray[np.uint8] | None:
    """Adaptive threshold + morphological close. Ink becomes 255, paper becomes 0."""
    if not _HAS_CV2:
        return None
    try:
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            _ADAPTIVE_BLOCK if _ADAPTIVE_BLOCK % 2 else _ADAPTIVE_BLOCK + 1,
            _ADAPTIVE_C,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (_CLOSE_KERNEL + 1, _CLOSE_KERNEL + 1))
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel).astype(np.uint8)
    except cv2.error as exc:  # pragma: no cover - defensive
        logger.warning("binarisation failed", extra={"error": str(exc)})
        return None


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    intersection = (ix1 - ix0) * (iy1 - iy0)
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - intersection
    return intersection / union if union > 0 else 0.0


def _point_to_box_distance(point: tuple[float, float], box: tuple[int, int, int, int]) -> float:
    px, py = point
    x0, y0, x1, y1 = box
    dx = max(x0 - px, 0.0, px - x1)
    dy = max(y0 - py, 0.0, py - y1)
    return math.hypot(dx, dy)


# ======================================================================================
# Raw detections
# ======================================================================================


@dataclass(slots=True)
class _RawDetection:
    """A shape found by the rule-based pass, before OCR re-classification."""

    symbol_class: SymbolClass
    bbox: tuple[int, int, int, int]
    confidence: float
    shape: Literal["circle", "rectangle"]
    evidence: str

    @property
    def area(self) -> int:
        x0, y0, x1, y1 = self.bbox
        return max(0, x1 - x0) * max(0, y1 - y0)

    def as_dict(self) -> dict[str, Any]:
        return {"class": self.symbol_class.value, "bbox": list(self.bbox),
                "confidence": round(self.confidence, 4), "shape": self.shape,
                "evidence": self.evidence}


def _detect_circles(
    gray: NDArray[np.uint8],
    binary: NDArray[np.uint8],
    *,
    settings: Settings,
) -> list[_RawDetection]:
    """Pumps and instrument bubbles.

    Hough gives no score, so confidence is earned from the pixels: the fraction of the fitted
    circle's perimeter that is actually inked. A true symbol ring scores near 1.0; a coincidental
    accumulator peak in hatching scores far lower and is filtered out by
    ``settings.pid_min_symbol_confidence``.
    """
    height, width = gray.shape[:2]
    short_edge = min(height, width)
    min_radius = max(5, int(_MIN_RADIUS_RATIO * short_edge))
    max_radius = max(min_radius + 4, int(_MAX_RADIUS_RATIO * short_edge))
    try:
        blurred = cv2.medianBlur(gray, 5)
        found = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=float(max(min_radius * 2, 12)),
            param1=140,
            param2=28,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
    except cv2.error as exc:  # pragma: no cover - defensive
        logger.warning("HoughCircles failed", extra={"error": str(exc)})
        return []
    if found is None:
        return []

    instrument_ceiling = _INSTRUMENT_RADIUS_RATIO * short_edge
    out: list[_RawDetection] = []
    for entry in np.round(found[0]).astype(int):
        cx, cy, radius = int(entry[0]), int(entry[1]), int(entry[2])
        if radius <= 0:
            continue
        ring = _ring_fill(binary, cx, cy, radius)
        ink = _interior_ink(binary, cx, cy, radius)
        confidence = max(0.0, min(1.0, 0.30 + 0.62 * ring))
        if radius <= instrument_ceiling and ink < _PUMP_INK_RATIO:
            symbol_class = SymbolClass.INSTRUMENT
            evidence = (
                f"circle r={radius}px (<= {instrument_ceiling:.0f}px instrument ceiling), "
                f"ring fill {ring:.2f}, interior ink {ink:.2f} consistent with lettering"
            )
        else:
            symbol_class = SymbolClass.PUMP
            evidence = (
                f"circle r={radius}px, ring fill {ring:.2f}, interior ink {ink:.2f} "
                f"(>= {_PUMP_INK_RATIO} suggests an impeller/driver glyph, not lettering)"
            )
        bbox = (
            max(0, cx - radius), max(0, cy - radius),
            min(width, cx + radius), min(height, cy + radius),
        )
        out.append(_RawDetection(symbol_class, bbox, confidence, "circle", evidence))
    return out


def _ring_fill(binary: NDArray[np.uint8], cx: int, cy: int, radius: int) -> float:
    """Fraction of the fitted circle's perimeter that is inked."""
    samples = max(24, min(180, radius * 6))
    height, width = binary.shape[:2]
    hits = 0
    for step in range(samples):
        angle = 2.0 * math.pi * step / samples
        x = int(round(cx + radius * math.cos(angle)))
        y = int(round(cy + radius * math.sin(angle)))
        # A 1 px tolerance band: scanned rings wander by a pixel.
        for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            px, py = x + dx, y + dy
            if 0 <= px < width and 0 <= py < height and binary[py, px] > 0:
                hits += 1
                break
    return hits / samples


def _interior_ink(binary: NDArray[np.uint8], cx: int, cy: int, radius: int) -> float:
    """Fraction of the disc *inside* the ring that is inked."""
    inner = int(radius * 0.75)
    if inner < 2:
        return 0.0
    height, width = binary.shape[:2]
    x0, y0 = max(0, cx - inner), max(0, cy - inner)
    x1, y1 = min(width, cx + inner), min(height, cy + inner)
    patch = binary[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    return float(np.count_nonzero(patch)) / float(patch.size)


def _detect_rectangles(binary: NDArray[np.uint8], *, settings: Settings) -> list[_RawDetection]:
    """Vessels, tanks, exchangers and equipment boxes."""
    height, width = binary.shape[:2]
    canvas = float(height * width) or 1.0
    try:
        contours, _hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    except cv2.error as exc:  # pragma: no cover - defensive
        logger.warning("findContours failed", extra={"error": str(exc)})
        return []

    out: list[_RawDetection] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        ratio = area / canvas
        if ratio < _MIN_RECT_AREA_RATIO or ratio > _MAX_RECT_AREA_RATIO:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            continue
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        x, y, w, h = cv2.boundingRect(approx)
        if w < 6 or h < 6:
            continue
        rectangularity = area / float(w * h)
        if rectangularity < _RECTANGULARITY_FLOOR:
            continue
        aspect = w / float(h)
        box_ratio = (w * h) / canvas

        if aspect >= _EXCHANGER_ASPECT:
            symbol_class = SymbolClass.HEAT_EXCHANGER
            reason = f"aspect {aspect:.2f} >= {_EXCHANGER_ASPECT} (wide shell)"
        elif aspect <= _COLUMN_ASPECT:
            symbol_class = SymbolClass.VESSEL
            reason = f"aspect {aspect:.2f} <= {_COLUMN_ASPECT} (tall column)"
        elif box_ratio >= _TANK_MIN_AREA_RATIO:
            symbol_class = SymbolClass.TANK
            reason = f"near-square and {box_ratio * 100:.1f}% of the canvas (storage tank scale)"
        else:
            symbol_class = SymbolClass.VESSEL
            reason = f"aspect {aspect:.2f}, {box_ratio * 100:.2f}% of the canvas (equipment box)"

        confidence = max(0.0, min(1.0, 0.28 + 0.62 * rectangularity))
        evidence = (
            f"4-vertex convex contour, rectangularity {rectangularity:.2f}, {reason}"
        )
        out.append(_RawDetection(symbol_class, (x, y, x + w, y + h), confidence, "rectangle", evidence))
    return out


def _suppress(detections: list[_RawDetection]) -> list[_RawDetection]:
    """Greedy non-maximum suppression, plus removal of boxes that merely enclose a circle.

    A pump drawn inside a dashed package boundary produces both a circle and a rectangle; the
    circle is the symbol and the rectangle is scope annotation.
    """
    ordered = sorted(detections, key=lambda d: (-d.confidence, d.area))
    kept: list[_RawDetection] = []
    for candidate in ordered:
        redundant = False
        for existing in kept:
            if _iou(candidate.bbox, existing.bbox) > _NMS_IOU:
                redundant = True
                break
            if candidate.shape == "rectangle" and existing.shape == "circle" and _contains(candidate.bbox, existing.bbox):
                redundant = True
                break
        if not redundant:
            kept.append(candidate)
    return kept


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return (
        outer[0] <= inner[0] and outer[1] <= inner[1]
        and outer[2] >= inner[2] and outer[3] >= inner[3]
    )


# ======================================================================================
# Pipe runs
# ======================================================================================


@dataclass(slots=True)
class _Segment:
    """One straight piece of a pipe run."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def length(self) -> float:
        return math.hypot(self.x1 - self.x0, self.y1 - self.y0)

    @property
    def orientation(self) -> Literal["h", "v", "d"]:
        dx, dy = abs(self.x1 - self.x0), abs(self.y1 - self.y0)
        if dy <= dx * math.tan(math.radians(_ANGLE_TOLERANCE_DEG)):
            return "h"
        if dx <= dy * math.tan(math.radians(_ANGLE_TOLERANCE_DEG)):
            return "v"
        return "d"

    @property
    def ends(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return (self.x0, self.y0), (self.x1, self.y1)


@dataclass(slots=True)
class _Run:
    """A chain of segments forming one pipe run between two points."""

    segments: list[_Segment]
    start: tuple[int, int]
    end: tuple[int, int]

    @property
    def total_length(self) -> float:
        return sum(segment.length for segment in self.segments)

    @property
    def polyline(self) -> list[tuple[int, int]]:
        points: list[tuple[int, int]] = [self.start]
        for segment in self.segments:
            for point in segment.ends:
                if point not in points:
                    points.append(point)
        if self.end not in points:
            points.append(self.end)
        return points


class _DisjointSet:
    """Minimal union-find over segment indices."""

    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, item: int) -> int:
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self._parent[b] = a


def _detect_segments(gray: NDArray[np.uint8], *, settings: Settings) -> list[_Segment]:
    """Raw Hough segments, filtered to plausible pipe runs."""
    height, width = gray.shape[:2]
    min_length = max(
        int(_MIN_RUN_LENGTH_RATIO * math.hypot(height, width)),
        int(settings.pid_hough_min_line_length),
    )
    try:
        edges = cv2.Canny(gray, 60, 170, apertureSize=3)
        found = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=math.pi / 180.0,
            threshold=int(settings.pid_hough_threshold),
            minLineLength=min_length,
            maxLineGap=int(settings.pid_hough_max_line_gap),
        )
    except cv2.error as exc:  # pragma: no cover - defensive
        logger.warning("HoughLinesP failed", extra={"error": str(exc)})
        return []
    if found is None:
        return []
    return [_Segment(int(x0), int(y0), int(x1), int(y1)) for x0, y0, x1, y1 in found[:, 0, :]]


def _merge_collinear(segments: Sequence[_Segment], *, gap: int) -> list[_Segment]:
    """Collapse Hough's duplicate detections of the same physical line.

    Hough returns one segment per accumulator peak, so a single 800 px pipe frequently arrives as
    six overlapping pieces. Grouping by the perpendicular coordinate and merging along the parallel
    one turns those back into one edge — which matters, because the connection logic keys on run
    *endpoints*.
    """
    tolerance = max(2, gap)
    buckets: dict[tuple[str, int], list[_Segment]] = {}
    diagonals: list[_Segment] = []
    for segment in segments:
        orientation = segment.orientation
        if orientation == "h":
            key = ("h", int(round(((segment.y0 + segment.y1) / 2.0) / tolerance)))
        elif orientation == "v":
            key = ("v", int(round(((segment.x0 + segment.x1) / 2.0) / tolerance)))
        else:
            diagonals.append(segment)
            continue
        buckets.setdefault(key, []).append(segment)

    merged: list[_Segment] = []
    for (orientation, _slot), group in buckets.items():
        if orientation == "h":
            spans = sorted(
                (min(s.x0, s.x1), max(s.x0, s.x1), (s.y0 + s.y1) // 2) for s in group
            )
            current_start, current_end, current_axis = spans[0]
            for start, end, axis in spans[1:]:
                if start <= current_end + tolerance * 2:
                    current_end = max(current_end, end)
                    current_axis = (current_axis + axis) // 2
                else:
                    merged.append(_Segment(current_start, current_axis, current_end, current_axis))
                    current_start, current_end, current_axis = start, end, axis
            merged.append(_Segment(current_start, current_axis, current_end, current_axis))
        else:
            spans = sorted(
                (min(s.y0, s.y1), max(s.y0, s.y1), (s.x0 + s.x1) // 2) for s in group
            )
            current_start, current_end, current_axis = spans[0]
            for start, end, axis in spans[1:]:
                if start <= current_end + tolerance * 2:
                    current_end = max(current_end, end)
                    current_axis = (current_axis + axis) // 2
                else:
                    merged.append(_Segment(current_axis, current_start, current_axis, current_end))
                    current_start, current_end, current_axis = start, end, axis
            merged.append(_Segment(current_axis, current_start, current_axis, current_end))
    return merged + diagonals


def _build_runs(segments: Sequence[_Segment], *, corner_tolerance: int) -> list[_Run]:
    """Chain segments that meet at a corner into single runs."""
    if not segments:
        return []
    joiner = _DisjointSet(len(segments))
    for i, first in enumerate(segments):
        for j in range(i + 1, len(segments)):
            second = segments[j]
            if any(
                math.hypot(a[0] - b[0], a[1] - b[1]) <= corner_tolerance
                for a in first.ends
                for b in second.ends
            ):
                joiner.union(i, j)

    grouped: dict[int, list[_Segment]] = {}
    for index, segment in enumerate(segments):
        grouped.setdefault(joiner.find(index), []).append(segment)

    runs: list[_Run] = []
    for group in grouped.values():
        points = [point for segment in group for point in segment.ends]
        start, end, best = points[0], points[-1], -1.0
        for i, a in enumerate(points):
            for b in points[i + 1:]:
                distance = math.hypot(a[0] - b[0], a[1] - b[1])
                if distance > best:
                    best, start, end = distance, a, b
        runs.append(_Run(segments=group, start=start, end=end))
    return runs


def _dash_ratio(binary: NDArray[np.uint8], run: _Run) -> float:
    """Ink coverage sampled along a run. Low coverage means a dashed instrument signal line."""
    height, width = binary.shape[:2]
    hits = 0
    total = 0
    for segment in run.segments:
        steps = max(8, int(segment.length // 3))
        for step in range(steps + 1):
            t = step / steps
            x = int(round(segment.x0 + (segment.x1 - segment.x0) * t))
            y = int(round(segment.y0 + (segment.y1 - segment.y0) * t))
            total += 1
            found = False
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    px, py = x + dx, y + dy
                    if 0 <= px < width and 0 <= py < height and binary[py, px] > 0:
                        found = True
                        break
                if found:
                    break
            if found:
                hits += 1
    return hits / total if total else 0.0


# ======================================================================================
# Arrowheads
# ======================================================================================


def _arrow_templates(size: int) -> dict[tuple[int, int], NDArray[np.uint8]]:
    """Filled triangles pointing right, left, down and up, keyed by their direction vector."""
    span = max(3, size)
    templates: dict[tuple[int, int], NDArray[np.uint8]] = {}
    shapes = {
        (1, 0): [(0, 0), (0, span - 1), (span - 1, span // 2)],
        (-1, 0): [(span - 1, 0), (span - 1, span - 1), (0, span // 2)],
        (0, 1): [(0, 0), (span - 1, 0), (span // 2, span - 1)],
        (0, -1): [(0, span - 1), (span - 1, span - 1), (span // 2, 0)],
    }
    for direction, points in shapes.items():
        canvas = np.zeros((span, span), dtype=np.uint8)
        cv2.fillPoly(canvas, [np.array(points, dtype=np.int32)], 255)
        templates[direction] = canvas
    return templates


def _flow_direction(
    binary: NDArray[np.uint8],
    run: _Run,
    templates: dict[tuple[int, int], NDArray[np.uint8]],
    *,
    source_point: tuple[int, int],
    target_point: tuple[int, int],
) -> tuple[Literal["forward", "reverse", "bidirectional", "unknown"], float]:
    """Read flow direction off an arrowhead, returning ``(direction, match_score)``.

    Arrowheads sit at the receiving end of a run and at flow-marker positions along it, so three
    regions of interest are searched: both ends and the midpoint.
    """
    height, width = binary.shape[:2]
    span = next(iter(templates.values())).shape[0]
    half = span * 2
    axis = (target_point[0] - source_point[0], target_point[1] - source_point[1])
    if axis == (0, 0):
        return "unknown", 0.0

    midpoint = ((run.start[0] + run.end[0]) // 2, (run.start[1] + run.end[1]) // 2)
    best_score = 0.0
    best_direction: tuple[int, int] | None = None
    for centre in (target_point, midpoint, source_point):
        x0, y0 = max(0, centre[0] - half), max(0, centre[1] - half)
        x1, y1 = min(width, centre[0] + half), min(height, centre[1] + half)
        patch = binary[y0:y1, x0:x1]
        if patch.shape[0] < span or patch.shape[1] < span:
            continue
        for direction, template in templates.items():
            try:
                scores = cv2.matchTemplate(patch, template, cv2.TM_CCOEFF_NORMED)
            except cv2.error:  # pragma: no cover - shape guard above should prevent this
                continue
            score = float(scores.max()) if scores.size else 0.0
            if score > best_score:
                best_score, best_direction = score, direction

    if best_direction is None or best_score < _ARROW_MATCH_THRESHOLD:
        return "unknown", best_score
    dot = best_direction[0] * axis[0] + best_direction[1] * axis[1]
    if dot > 0:
        return "forward", best_score
    if dot < 0:
        return "reverse", best_score
    return "bidirectional", best_score


# ======================================================================================
# Detectors behind contracts.SymbolDetector
# ======================================================================================


class RuleBasedSymbolDetector:
    """Shape-based detection. Implements :class:`indra.core.contracts.SymbolDetector`.

    The shipped default (D4): it needs no training data, no weights file and no GPU, and it finds
    real pumps and vessels on a real drawing today.
    """

    name: DetectorName = "rule_based"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    async def is_available(self) -> bool:
        return _HAS_CV2

    async def detect(self, image_path: Path) -> list[dict[str, Any]]:
        """Return ``[{"class", "bbox", "confidence"}]`` for ``image_path``."""
        array = await asyncio.to_thread(load_image, image_path)
        if array is None:
            logger.warning("symbol detection skipped; image unreadable", extra={"path": str(image_path)})
            return []
        detections = await asyncio.to_thread(self.detect_array, array)
        return [detection.as_dict() for detection in detections]

    def detect_array(self, array: NDArray[np.uint8]) -> list[_RawDetection]:
        """Synchronous core, shared with :class:`PIDVisionParser`. Call on a worker thread."""
        if not _HAS_CV2:
            return []
        gray = to_grayscale(array)
        binary = _binarise(gray)
        if binary is None:
            return []
        detections = _detect_circles(gray, binary, settings=self._settings)
        detections.extend(_detect_rectangles(binary, settings=self._settings))
        floor = float(self._settings.pid_min_symbol_confidence)
        return _suppress([d for d in detections if d.confidence >= floor])


class YoloSymbolDetector:
    """Optional YOLO upgrade. Implements :class:`indra.core.contracts.SymbolDetector`.

    Selected only when ``settings.pid_detector == "yolo"`` *and* ``settings.pid_yolo_weights``
    points at a file that exists. ``ultralytics`` is imported inside the call, so a machine without
    it (the normal case) pays nothing and the module still imports.
    """

    name: DetectorName = "yolo"

    #: YOLO class names are dataset-specific; anything unmapped becomes ``SymbolClass.UNKNOWN``.
    CLASS_MAP: Final[dict[str, SymbolClass]] = {
        "pump": SymbolClass.PUMP,
        "vessel": SymbolClass.VESSEL,
        "tank": SymbolClass.TANK,
        "heat_exchanger": SymbolClass.HEAT_EXCHANGER,
        "heatexchanger": SymbolClass.HEAT_EXCHANGER,
        "exchanger": SymbolClass.HEAT_EXCHANGER,
        "valve": SymbolClass.VALVE,
        "instrument": SymbolClass.INSTRUMENT,
        "compressor": SymbolClass.COMPRESSOR,
        "filter": SymbolClass.FILTER,
    }

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model: object | None = None
        self._resolved = False
        self._lock = asyncio.Lock()

    @property
    def weights(self) -> Path | None:
        return self._settings.pid_yolo_weights

    async def is_available(self) -> bool:
        """True only when weights exist *and* ultralytics loads them. Probed once."""
        if self._resolved:
            return self._model is not None
        async with self._lock:
            if self._resolved:
                return self._model is not None
            self._resolved = True
            self._model = await asyncio.to_thread(self._load)
        return self._model is not None

    def _load(self) -> object | None:
        weights = self.weights
        if weights is None or not Path(weights).is_file():
            logger.info(
                "YOLO detector requested but no weights file is present; using the rule-based path",
                extra={"weights": str(weights) if weights else None},
            )
            return None
        try:
            from ultralytics import YOLO  # noqa: PLC0415 - heavy optional dependency
        except ImportError:
            logger.warning("ultralytics is not installed; falling back to the rule-based detector")
            return None
        try:
            return YOLO(str(weights))
        except Exception as exc:  # noqa: BLE001 - checkpoint loading surfaces many error types
            logger.warning("YOLO weights could not be loaded; falling back to the rule-based detector",
                           extra={"weights": str(weights), "error": str(exc)})
            return None

    async def detect(self, image_path: Path) -> list[dict[str, Any]]:
        if not await self.is_available():
            return []
        return await asyncio.to_thread(self._predict, image_path)

    def _predict(self, image_path: Path) -> list[dict[str, Any]]:
        model = self._model
        if model is None:  # pragma: no cover - guarded by is_available
            return []
        try:
            results = model.predict(  # type: ignore[attr-defined]
                str(image_path),
                conf=float(self._settings.pid_min_symbol_confidence),
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001 - inference surfaces many error types
            logger.warning("YOLO inference failed; drawing falls back to rule-based detection",
                           extra={"error": str(exc)})
            return []
        out: list[dict[str, Any]] = []
        for result in results:
            names = getattr(result, "names", {}) or {}
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                try:
                    x0, y0, x1, y1 = (int(v) for v in box.xyxy[0].tolist())
                    confidence = float(box.conf[0])
                    label = str(names.get(int(box.cls[0]), "unknown")).lower()
                except (AttributeError, IndexError, TypeError, ValueError):
                    continue
                symbol_class = self.CLASS_MAP.get(label, SymbolClass.UNKNOWN)
                out.append({
                    "class": symbol_class.value,
                    "bbox": [x0, y0, x1, y1],
                    "confidence": round(confidence, 4),
                    "shape": "rectangle",
                    "evidence": f"YOLO detection '{label}' at {confidence:.2f}",
                })
        return out


def select_detector(settings: Settings | None = None) -> RuleBasedSymbolDetector | YoloSymbolDetector:
    """Pick the configured detector. ``auto`` prefers YOLO only when weights are on disk (D4)."""
    cfg = settings or get_settings()
    if cfg.pid_detector == "yolo":
        return YoloSymbolDetector(cfg)
    if cfg.pid_detector == "auto":
        weights = cfg.pid_yolo_weights
        if weights is not None and Path(weights).is_file():
            return YoloSymbolDetector(cfg)
    return RuleBasedSymbolDetector(cfg)


# ======================================================================================
# The parser
# ======================================================================================


@dataclass(slots=True)
class _SymbolWork:
    """A detection plus everything OCR and classification later attach to it."""

    detection: _RawDetection
    symbol: DetectedSymbol
    resolution: TagResolution | None = None


class PIDVisionParser:
    """Turns an engineering drawing into symbols, tags and traced connections."""

    name: str = "pid_vision"

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        ocr: OCREngine | None = None,
        normalizer: PlantTagNormalizer | None = None,
        detector: RuleBasedSymbolDetector | YoloSymbolDetector | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._ocr = ocr or OCREngine(self._settings)
        self._normalizer = normalizer or PlantTagNormalizer(self._settings)
        self._detector = detector or select_detector(self._settings)
        self._rule_based = RuleBasedSymbolDetector(self._settings)

    @property
    def detector_name(self) -> DetectorName:
        return self._detector.name

    async def parse(
        self,
        path: Path,
        *,
        document_id: str,
        registry: Sequence[str] | None = None,
    ) -> PIDParseResult:
        """Parse the drawing at ``path``. Never raises — failures arrive as warnings."""
        array = await asyncio.to_thread(load_image, path)
        if array is None:
            return PIDParseResult(
                document_id=document_id,
                image_width=1,
                image_height=1,
                detector_used=self._fallback_name(),
                warnings=[f"Drawing at {path.name} could not be decoded; no symbols were extracted"],
            )
        return await self.parse_array(array, document_id=document_id, registry=registry, path=path)

    async def parse_array(
        self,
        array: NDArray[np.uint8],
        *,
        document_id: str,
        registry: Sequence[str] | None = None,
        path: Path | None = None,
    ) -> PIDParseResult:
        """Parse an already-decoded image. Used by the PDF path, which renders pages in memory."""
        started = time.perf_counter()
        height, width = array.shape[:2]
        warnings: list[str] = []

        if not _HAS_CV2:
            return PIDParseResult(
                document_id=document_id,
                image_width=max(1, int(width)),
                image_height=max(1, int(height)),
                detector_used="rule_based",
                warnings=[
                    "OpenCV is not installed, so the P&ID vision pipeline could not run. "
                    "Install opencv-python to enable symbol and connection extraction."
                ],
                processing_ms=round((time.perf_counter() - started) * 1000.0, 2),
            )

        detections, detector_used, detector_note = await self._detect(array, path)
        if detector_note:
            warnings.append(detector_note)

        if not detections:
            warnings.append(
                "No equipment symbols passed the "
                f"{self._settings.pid_min_symbol_confidence:.2f} confidence floor on this drawing; "
                "it may be a photograph, a very low-resolution scan, or a non-P&ID diagram"
            )
            return PIDParseResult(
                document_id=document_id,
                image_width=int(width),
                image_height=int(height),
                detector_used=detector_used,
                warnings=warnings,
                processing_ms=round((time.perf_counter() - started) * 1000.0, 2),
            )

        works = [
            _SymbolWork(
                detection=detection,
                symbol=DetectedSymbol(
                    symbol_class=detection.symbol_class,
                    bbox=detection.bbox,
                    detection_confidence=round(min(1.0, max(0.0, detection.confidence)), 4),
                    detector=detector_used,
                ),
            )
            for detection in detections
        ]

        ocr_ready = await self._ocr.is_available()
        if ocr_ready:
            await self._read_tags(array, works, registry=registry)
        else:
            warnings.append(
                "No OCR engine is available, so detected symbols carry no plant tags. "
                "The drawing's topology was still extracted; install tesseract to recover the tags."
            )

        binary = await asyncio.to_thread(self._binary_for, array)
        connections, connection_warnings = await asyncio.to_thread(
            self._trace_connections, array, binary, works,
        )
        warnings.extend(connection_warnings)

        if ocr_ready and connections:
            await self._read_pipe_specs(array, connections, works)

        symbols = [work.symbol for work in works]
        overall = self._overall_confidence(symbols, connections)
        logger.info(
            "P&ID parsed",
            extra={
                "document_id": document_id,
                "detector": detector_used,
                "symbols": len(symbols),
                "tagged": sum(1 for s in symbols if s.tag),
                "connections": len(connections),
                "overall_confidence": overall,
            },
        )
        return PIDParseResult(
            document_id=document_id,
            image_width=int(width),
            image_height=int(height),
            symbols=symbols,
            connections=connections,
            detector_used=detector_used,
            overall_confidence=overall,
            warnings=warnings,
            processing_ms=round((time.perf_counter() - started) * 1000.0, 2),
        )

    # -- detection --------------------------------------------------------------------
    def _fallback_name(self) -> DetectorName:
        return "rule_based"

    async def _detect(
        self,
        array: NDArray[np.uint8],
        path: Path | None,
    ) -> tuple[list[_RawDetection], DetectorName, str]:
        """Run the configured detector, falling back to rules when YOLO cannot serve."""
        detector = self._detector
        if isinstance(detector, YoloSymbolDetector):
            if path is not None and await detector.is_available():
                raw = await detector.detect(path)
                converted = [_raw_from_dict(entry) for entry in raw]
                converted = [entry for entry in converted if entry is not None]
                if converted:
                    return _suppress(converted), "yolo", ""  # type: ignore[arg-type]
                note = "YOLO returned no detections; re-ran the rule-based detector on this drawing"
            else:
                note = (
                    "YOLO detector is configured but unavailable (missing weights or ultralytics); "
                    "used the rule-based pipeline instead"
                )
            detections = await asyncio.to_thread(self._rule_based.detect_array, array)
            return detections, "rule_based", note
        detections = await asyncio.to_thread(self._rule_based.detect_array, array)
        return detections, "rule_based", ""

    @staticmethod
    def _binary_for(array: NDArray[np.uint8]) -> NDArray[np.uint8]:
        binary = _binarise(to_grayscale(array))
        return binary if binary is not None else np.zeros(array.shape[:2], dtype=np.uint8)

    # -- tags -------------------------------------------------------------------------
    async def _read_tags(
        self,
        array: NDArray[np.uint8],
        works: list[_SymbolWork],
        *,
        registry: Sequence[str] | None,
    ) -> None:
        """OCR each symbol's neighbourhood and resolve the plant tag (D5).

        The tag also *re-classifies* the symbol: ``equipment_type_for("PIC-101")`` is authoritative
        where circle geometry is only suggestive. This is the "neighbouring text" input that turns a
        pile of shapes into an asset list.
        """
        height, width = array.shape[:2]
        budget = works[:_MAX_OCR_SYMBOLS]
        if len(works) > _MAX_OCR_SYMBOLS:
            logger.info("capping region OCR", extra={"symbols": len(works), "cap": _MAX_OCR_SYMBOLS})

        for work in budget:
            x0, y0, x1, y1 = work.symbol.bbox
            box_height = max(1, y1 - y0)
            margin = max(4, int(0.12 * max(x1 - x0, box_height)))
            inside = (x0, y0, x1, y1)
            label_band = (
                max(0, x0 - margin * 2),
                min(height - 1, y1),
                min(width, x1 + margin * 2),
                min(height, y1 + int(_LABEL_BAND_RATIO * box_height)),
            )

            candidates: list[tuple[OCRResult, str]] = []
            interior = await self._ocr.recognize_region(array, inside, psm=PSM_SINGLE_WORD)
            if interior.ok:
                candidates.append((interior, "inside the symbol"))
            if label_band[3] > label_band[1] + 2:
                below = await self._ocr.recognize_region(array, label_band, psm=PSM_SINGLE_LINE)
                if below.ok:
                    candidates.append((below, "in the label band beneath the symbol"))

            if not candidates:
                continue
            best, where = max(candidates, key=lambda item: item[0].mean_confidence)
            raw_text = " ".join(best.text.split())
            work.symbol.ocr_text = raw_text[:120]

            resolution = self._normalizer.resolve(raw_text, registry=registry)
            work.resolution = resolution
            if resolution.tag is None:
                continue

            # OCR confidence caps tag confidence: a perfect grammar match on an unreadable glyph is
            # still an unreadable glyph.
            combined = round(min(resolution.confidence, max(best.mean_confidence, 0.1)), 4)
            work.symbol.tag = resolution.tag
            work.symbol.tag_confidence = combined
            work.symbol.tag_alternatives = list(resolution.alternatives)

            equipment_type = equipment_type_for(resolution.tag)
            reclassified = _TYPE_TO_CLASS.get(equipment_type)
            if reclassified is not None and reclassified is not work.symbol.symbol_class:
                logger.debug(
                    "symbol re-classified from its tag",
                    extra={"tag": resolution.tag, "from": work.symbol.symbol_class.value,
                           "to": reclassified.value, "read": where},
                )
                work.symbol.symbol_class = reclassified

    async def _read_pipe_specs(
        self,
        array: NDArray[np.uint8],
        connections: list[DetectedConnection],
        works: list[_SymbolWork],
    ) -> None:
        """OCR the line-number annotation beside the longest runs, e.g. ``6"-CS-1001``."""
        import re

        pattern = re.compile(_PIPE_SPEC_RE_SOURCE)
        height, width = array.shape[:2]
        ranked = sorted(
            connections,
            key=lambda c: -_polyline_length(c.polyline),
        )[:_MAX_PIPE_SPEC_RUNS]
        for connection in ranked:
            if len(connection.polyline) < 2:
                continue
            mid_index = len(connection.polyline) // 2
            cx, cy = connection.polyline[mid_index]
            band = (
                max(0, cx - 90), max(0, cy - 26),
                min(width, cx + 90), min(height, cy + 26),
            )
            result = await self._ocr.recognize_region(array, band, psm=PSM_SINGLE_LINE, whitelist=None)
            if not result.ok:
                continue
            match = pattern.search(" ".join(result.text.split()).upper())
            if match:
                connection.pipe_spec = match.group(0)

    # -- connections ------------------------------------------------------------------
    def _trace_connections(
        self,
        array: NDArray[np.uint8],
        binary: NDArray[np.uint8],
        works: list[_SymbolWork],
    ) -> tuple[list[DetectedConnection], list[str]]:
        """Hough → merge → runs → symbol association. Synchronous; run on a worker thread."""
        warnings: list[str] = []
        gray = to_grayscale(array)
        segments = _detect_segments(gray, settings=self._settings)
        if not segments:
            warnings.append(
                "No pipe runs were detected: the Hough threshold "
                f"({self._settings.pid_hough_threshold}) found no straight segments longer than "
                f"{self._settings.pid_hough_min_line_length}px"
            )
            return [], warnings

        merged = _merge_collinear(segments, gap=int(self._settings.pid_hough_max_line_gap))
        corner_tolerance = max(4, int(self._settings.pid_connection_max_distance_px) // 4)
        runs = _build_runs(merged, corner_tolerance=corner_tolerance)

        height, width = binary.shape[:2]
        diagonal = math.hypot(height, width)
        min_run = _MIN_RUN_LENGTH_RATIO * diagonal
        max_distance = float(self._settings.pid_connection_max_distance_px)
        arrow_span = int(
            max(_MIN_ARROW_PX, min(_MAX_ARROW_PX, _ARROW_SIZE_RATIO * min(height, width)))
        )
        templates = _arrow_templates(arrow_span)

        connections: list[DetectedConnection] = []
        seen: set[tuple[str, str]] = set()
        unattached = 0

        for run in runs:
            if run.total_length < min_run:
                continue
            source = _nearest_symbol(run.start, works, max_distance)
            target = _nearest_symbol(run.end, works, max_distance)
            if source is None or target is None:
                unattached += 1
                continue
            source_work, source_distance = source
            target_work, target_distance = target
            if source_work.symbol.symbol_id == target_work.symbol.symbol_id:
                continue
            key = (source_work.symbol.symbol_id, target_work.symbol.symbol_id)
            if key in seen or (key[1], key[0]) in seen:
                continue
            seen.add(key)

            coverage = _dash_ratio(binary, run)
            line_type: Literal["process", "instrument", "utility", "unknown"] = (
                "process" if coverage >= _DASH_SOLID_RATIO else "instrument"
            )
            if SymbolClass.INSTRUMENT in (
                source_work.symbol.symbol_class, target_work.symbol.symbol_class
            ) and line_type == "process" and coverage < 0.92:
                line_type = "instrument"

            direction, arrow_score = _flow_direction(
                binary, run, templates,
                source_point=run.start, target_point=run.end,
            )

            proximity = 1.0 - min(1.0, (source_distance + target_distance) / (2.0 * max_distance))
            length_support = min(1.0, run.total_length / (0.15 * diagonal))
            confidence = 0.40 + 0.35 * proximity + 0.15 * length_support + 0.10 * min(1.0, arrow_score)
            connections.append(
                DetectedConnection(
                    source_symbol_id=source_work.symbol.symbol_id,
                    target_symbol_id=target_work.symbol.symbol_id,
                    confidence=round(max(0.0, min(1.0, confidence)), 4),
                    flow_direction=direction,
                    line_type=line_type,
                    polyline=[(int(x), int(y)) for x, y in run.polyline],
                )
            )

        if unattached:
            warnings.append(
                f"{unattached} pipe run(s) had at least one end further than "
                f"{max_distance:.0f}px from any detected symbol and were not turned into "
                f"connections; raise INDRA_PID_CONNECTION_MAX_DISTANCE_PX if the drawing scale is large"
            )
        return connections, warnings

    # -- scoring ----------------------------------------------------------------------
    @staticmethod
    def _overall_confidence(
        symbols: Sequence[DetectedSymbol],
        connections: Sequence[DetectedConnection],
    ) -> float:
        """Blend detection strength, tag coverage and connectivity into one auditable number."""
        if not symbols:
            return 0.0
        detection = sum(s.detection_confidence for s in symbols) / len(symbols)
        tagged = sum(1 for s in symbols if s.tag)
        tag_rate = tagged / len(symbols)
        connectivity = (
            sum(c.confidence for c in connections) / len(connections) if connections else 0.0
        )
        blended = 0.50 * detection + 0.30 * tag_rate + 0.20 * connectivity
        return round(max(0.0, min(1.0, blended)), 4)


def _raw_from_dict(entry: dict[str, Any]) -> _RawDetection | None:
    """Convert a :class:`~indra.core.contracts.SymbolDetector` dict back into a raw detection."""
    try:
        bbox = tuple(int(v) for v in entry["bbox"])
        if len(bbox) != 4:
            return None
        symbol_class = SymbolClass(str(entry.get("class", "unknown")))
        confidence = float(entry.get("confidence", 0.0))
    except (KeyError, TypeError, ValueError):
        return None
    shape = "circle" if str(entry.get("shape")) == "circle" else "rectangle"
    return _RawDetection(
        symbol_class=symbol_class,
        bbox=bbox,  # type: ignore[arg-type]
        confidence=confidence,
        shape=shape,  # type: ignore[arg-type]
        evidence=str(entry.get("evidence", "external detector")),
    )


def _nearest_symbol(
    point: tuple[int, int],
    works: Sequence[_SymbolWork],
    max_distance: float,
) -> tuple[_SymbolWork, float] | None:
    best: tuple[_SymbolWork, float] | None = None
    for work in works:
        distance = _point_to_box_distance((float(point[0]), float(point[1])), work.symbol.bbox)
        if distance <= max_distance and (best is None or distance < best[1]):
            best = (work, distance)
    return best


def _polyline_length(points: Sequence[tuple[int, int]]) -> float:
    return sum(
        math.hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1])
        for i in range(len(points) - 1)
    )


# ======================================================================================
# Projection into the knowledge graph
# ======================================================================================


def pid_to_graph(
    result: PIDParseResult,
    meta: DocumentMeta,
) -> tuple[list[ExtractedEntity], list[ExtractedRelationship]]:
    """Project a :class:`PIDParseResult` into graph entities and ``CONNECTED_TO`` edges.

    Only tagged symbols become entities: an untagged rectangle is a real detection but it is not an
    *asset*, and writing anonymous nodes into the graph would make the visualisation unreadable for
    no analytical gain. The untagged detections stay on the ``PIDParseResult`` for the drawing
    overlay, which is where they are actually useful.
    """
    entities: list[ExtractedEntity] = []
    by_symbol: dict[str, ExtractedEntity] = {}

    for symbol in result.symbols:
        if not symbol.tag:
            continue
        normalised = max(0.0, min(1.0, symbol.tag_confidence or symbol.detection_confidence))
        entity = ExtractedEntity(
            entity_id=new_id("entity"),
            type=EntityType.EQUIPMENT,
            name=symbol.ocr_text or symbol.tag,
            canonical_name=symbol.tag,
            confidence=Confidence(
                value=normalised,
                rationale=(
                    f"{symbol.symbol_class.value.replace('_', ' ')} symbol detected on the drawing at "
                    f"{symbol.detection_confidence:.2f} by the {symbol.detector} detector; tag read as "
                    f"{symbol.ocr_text or '(blank)'!r} and resolved to {symbol.tag} at "
                    f"{symbol.tag_confidence:.2f}"
                    + (f"; alternatives {', '.join(symbol.tag_alternatives)}" if symbol.tag_alternatives else "")
                ),
                method="vision",
            ),
            document_id=meta.document_id,
            bbox=(
                float(symbol.bbox[0]) / result.image_width,
                float(symbol.bbox[1]) / result.image_height,
                float(symbol.bbox[2]) / result.image_width,
                float(symbol.bbox[3]) / result.image_height,
            ),
            alternatives=list(symbol.tag_alternatives),
            attributes={
                "symbol_class": symbol.symbol_class.value,
                "equipment_type": equipment_type_for(symbol.tag),
                "detector": symbol.detector,
                "source": "pid_vision",
                "is_equipment": symbol.symbol_class in _CLASS_KEEPS_EQUIPMENT,
            },
        )
        entities.append(entity)
        by_symbol[symbol.symbol_id] = entity

    relationships: list[ExtractedRelationship] = []
    for connection in result.connections:
        source = by_symbol.get(connection.source_symbol_id)
        target = by_symbol.get(connection.target_symbol_id)
        if source is None or target is None:
            continue
        head, tail = (source, target) if connection.flow_direction != "reverse" else (target, source)
        relationships.append(
            ExtractedRelationship(
                type=RelationType.CONNECTED_TO,
                source_key=head.key,
                target_key=tail.key,
                confidence=Confidence(
                    value=max(0.0, min(1.0, connection.confidence)),
                    rationale=(
                        f"Pipe run traced on the drawing between {head.canonical_name} and "
                        f"{tail.canonical_name}: {connection.line_type} line, flow direction "
                        f"{connection.flow_direction} from arrowhead template match"
                        + (f", line number {connection.pipe_spec}" if connection.pipe_spec else "")
                    ),
                    method="vision",
                ),
                evidence_text=(
                    f"{connection.line_type} line traced across "
                    f"{len(connection.polyline)} polyline vertices on {meta.title}"
                ),
                document_id=meta.document_id,
                method="vision",
                properties={
                    "line_type": connection.line_type,
                    "flow_direction": connection.flow_direction,
                    "pipe_spec": connection.pipe_spec,
                    "polyline": connection.polyline[:32],
                },
            )
        )
    return entities, relationships


__all__ = [
    "DetectorName",
    "DrawingSignature",
    "PIDVisionParser",
    "RuleBasedSymbolDetector",
    "YoloSymbolDetector",
    "looks_like_drawing",
    "pid_to_graph",
    "select_detector",
]
