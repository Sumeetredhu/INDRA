"""Raster artefacts for the demo corpus, drawn rather than downloaded.

Four things live here, all deterministic for a given seed:

``render_pid``
    A real ISO-style P&ID of the Unit-2 boiler feed water system, drawn with Pillow. Not a
    screenshot and not clip art — a pump circle with an impeller triangle for P-101 and P-102, a
    vertical vessel with dished heads for V-201, a shell-and-tube exchanger for E-301, bow-tie
    valves, instrument bubbles, process lines with arrowheads, pipe specs, notes and a title block.
    It is drawn to be *parsable*: clean black on white, closed geometry, generous symbol
    separation, and tag text placed beside its symbol rather than on top of it, because that is
    what the rule-based detector of ``docs/DECISIONS.md`` D4 (Hough circles, contours, region OCR)
    can actually resolve.

``degrade_scan``
    The same drawing after a photocopier: a small rotation, uneven illumination, compressed
    contrast, gaussian sensor noise, a little optical blur and a genuine low-quality JPEG round
    trip. The point is that ``P-101`` really does become something like ``P-l0l`` for an OCR
    engine, so the tag-correction path (D5) is exercised against a real image instead of being
    asserted against a hand-written fixture string.

``render_nameplate``
    A synthetic photograph of the P-101 equipment nameplate — engraved text on brushed steel, with
    perspective, glare and JPEG artefacts — for the photo-to-query demo beat.

``render_handwriting``
    Script-like handwriting with per-glyph jitter, slant and stroke variation, rendered on paper
    stock. Used for the ``bearing wear 78%`` margin annotation on the work order, which is the
    genuine trigger for the low-OCR-confidence uncertainty flag: the number the diagnosis turns on
    is the one a human wrote by hand.

Error policy: filesystem failures raise :class:`~indra.core.exceptions.BlobStoreError`; failures
inside the drawing pipeline itself raise :class:`~indra.core.exceptions.VisionError`, the typed
error this repository reserves for the engineering-drawing image path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from indra.core.exceptions import BlobStoreError, VisionError
from indra.core.logging import get_logger

logger = get_logger(__name__)

# ======================================================================================
# Canvas and style constants
# ======================================================================================

PID_WIDTH: Final[int] = 1600
PID_HEIGHT: Final[int] = 1120

_WHITE: Final[tuple[int, int, int]] = (255, 255, 255)
_INK: Final[tuple[int, int, int]] = (17, 17, 17)
_GREY: Final[tuple[int, int, int]] = (90, 90, 90)

_W_FRAME: Final[int] = 3
_W_EQUIP: Final[int] = 3
_W_PIPE: Final[int] = 3
_W_UTILITY: Final[int] = 2
_W_LEADER: Final[int] = 1

#: Font search order. Pillow's bundled face comes first on purpose: it is byte-identical on every
#: machine with the same Pillow, which is what makes the generated drawing reproducible. System
#: faces are only a fallback for very old Pillow builds without a scalable default.
_SYSTEM_FONTS: Final[tuple[str, ...]] = (
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
)
_HAND_FONTS: Final[tuple[str, ...]] = (
    "C:/Windows/Fonts/segoesc.ttf",
    "C:/Windows/Fonts/Inkfree.ttf",
    "C:/Windows/Fonts/comic.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
)

# ----- scan degradation (tuned so text stays *nearly* legible, which is where OCR gets it wrong)
_SCAN_ROTATION_DEG: Final[float] = -1.6
_SCAN_CONTRAST: Final[float] = 0.70
_SCAN_LIFT: Final[int] = 44
_SCAN_NOISE_SIGMA: Final[float] = 7.5
_SCAN_BLUR_RADIUS: Final[float] = 0.7
_SCAN_JPEG_QUALITY: Final[int] = 30
_SCAN_SPECK_COUNT: Final[int] = 90


AnchorMode = Literal["lt", "mt", "rt", "lm", "mm", "rm", "lb", "mb", "rb"]


@dataclass(frozen=True, slots=True)
class PidSymbolSpec:
    """Ground truth for one symbol drawn on the P&ID.

    Published in the manifest so the vision pipeline's output can be scored against what was
    actually drawn, rather than against an eyeball.
    """

    tag: str
    symbol_class: str
    bbox: tuple[int, int, int, int]
    label: str


@dataclass(frozen=True, slots=True)
class PidConnectionSpec:
    """Ground truth for one process connection drawn on the P&ID."""

    source_tag: str
    target_tag: str
    line_type: Literal["process", "utility", "instrument"]
    pipe_spec: str


# ======================================================================================
# Geometry of the drawing — single source of truth for both the renderer and the manifest
# ======================================================================================

_V201_BOX: Final[tuple[int, int, int, int]] = (150, 240, 300, 600)
_P101_CENTRE: Final[tuple[int, int]] = (620, 700)
_P102_CENTRE: Final[tuple[int, int]] = (620, 900)
_PUMP_RADIUS: Final[int] = 58
_LP101A_CENTRE: Final[tuple[int, int]] = (400, 460)
_LP101A_RADIUS: Final[int] = 40
_E301_BOX: Final[tuple[int, int, int, int]] = (880, 420, 1180, 560)
_BUBBLE_RADIUS: Final[int] = 27

PID_SYMBOLS: Final[tuple[PidSymbolSpec, ...]] = (
    PidSymbolSpec("V-201", "vessel", _V201_BOX, "DEAERATOR STORAGE VESSEL"),
    PidSymbolSpec("P-101", "pump",
                  (_P101_CENTRE[0] - _PUMP_RADIUS, _P101_CENTRE[1] - _PUMP_RADIUS,
                   _P101_CENTRE[0] + _PUMP_RADIUS, _P101_CENTRE[1] + _PUMP_RADIUS),
                  "BFW PUMP - DUTY"),
    PidSymbolSpec("P-102", "pump",
                  (_P102_CENTRE[0] - _PUMP_RADIUS, _P102_CENTRE[1] - _PUMP_RADIUS,
                   _P102_CENTRE[0] + _PUMP_RADIUS, _P102_CENTRE[1] + _PUMP_RADIUS),
                  "BFW PUMP - STANDBY"),
    PidSymbolSpec("LP-101A", "pump",
                  (_LP101A_CENTRE[0] - _LP101A_RADIUS, _LP101A_CENTRE[1] - _LP101A_RADIUS,
                   _LP101A_CENTRE[0] + _LP101A_RADIUS, _LP101A_CENTRE[1] + _LP101A_RADIUS),
                  "AUX LUBE OIL PUMP"),
    PidSymbolSpec("E-301", "heat_exchanger", _E301_BOX, "FEED WATER PRE-HEATER"),
    PidSymbolSpec("HV-1012", "valve", (378, 678, 422, 722), "SUCTION ISOLATION"),
    PidSymbolSpec("CV-1013", "valve", (678, 318, 722, 362), "DISCHARGE NON-RETURN"),
    PidSymbolSpec("HV-3014", "valve", (1208, 468, 1252, 512), "E-301 OUTLET ISOLATION"),
    PidSymbolSpec("VT-1011", "instrument",
                  (770 - _BUBBLE_RADIUS, 700 - _BUBBLE_RADIUS, 770 + _BUBBLE_RADIUS,
                   700 + _BUBBLE_RADIUS), "VIBRATION TRANSMITTER"),
    PidSymbolSpec("PI-1015", "instrument",
                  (770 - _BUBBLE_RADIUS, 240 - _BUBBLE_RADIUS, 770 + _BUBBLE_RADIUS,
                   240 + _BUBBLE_RADIUS), "DISCHARGE PRESSURE"),
    PidSymbolSpec("PI-1016", "instrument",
                  (500 - _BUBBLE_RADIUS, 380 - _BUBBLE_RADIUS, 500 + _BUBBLE_RADIUS,
                   380 + _BUBBLE_RADIUS), "LUBE OIL HEADER PRESSURE"),
    PidSymbolSpec("TI-2011", "instrument",
                  (400 - _BUBBLE_RADIUS, 180 - _BUBBLE_RADIUS, 400 + _BUBBLE_RADIUS,
                   180 + _BUBBLE_RADIUS), "V-201 OUTLET TEMPERATURE"),
)

PID_CONNECTIONS: Final[tuple[PidConnectionSpec, ...]] = (
    PidConnectionSpec("V-201", "P-101", "process", '6"-BFW-1001-CS'),
    PidConnectionSpec("V-201", "P-102", "process", '6"-BFW-1001-CS'),
    PidConnectionSpec("P-101", "E-301", "process", '4"-BFW-1002-CS'),
    PidConnectionSpec("P-102", "E-301", "process", '4"-BFW-1002-CS'),
    PidConnectionSpec("E-301", "B-401", "process", '4"-BFW-1003-CS'),
    PidConnectionSpec("LP-101A", "P-101", "utility", '1"-LO-1010-CS'),
)


# ======================================================================================
# Font handling
# ======================================================================================


def load_font(size: int, *, handwriting: bool = False) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Return a scalable font of ``size`` pixels, preferring the Pillow-bundled face.

    Deterministic by construction: the bundled face is tried first so that the same Pillow version
    renders the same pixels on every machine. Falls back to common system faces and finally to
    Pillow's bitmap default, which is ugly but never absent.

    Args:
        size: Nominal pixel size.
        handwriting: Prefer a script-like system face, if one is installed. The per-glyph jitter in
            :func:`render_handwriting` is what actually sells the effect, so an absent script face
            degrades the look, not the function.
    """
    if handwriting:
        for candidate in _HAND_FONTS:
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=size)
    except (TypeError, AttributeError, OSError):  # pragma: no cover - Pillow < 10.1
        logger.debug("pillow has no scalable default font; falling back to system faces")
    for candidate in _SYSTEM_FONTS:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    logger.warning(
        "no scalable font available; drawing with Pillow's bitmap default. Text in the generated "
        "P&ID will be small and may not OCR. Install Pillow>=10.1 or a system TrueType font.",
    )
    return ImageFont.load_default()


def _text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    *,
    fill: tuple[int, int, int] = _INK,
    anchor: AnchorMode = "lt",
    bold: bool = False,
) -> None:
    """Draw text, emulating bold with a stroke and degrading gracefully without anchor support."""
    stroke = 1 if bold else 0
    if isinstance(font, ImageFont.FreeTypeFont):
        draw.text(xy, text, font=font, fill=fill, anchor=anchor, stroke_width=stroke,
                  stroke_fill=fill)
        return
    # Bitmap default font: no anchor support, so place manually from the measured box.
    box = draw.textbbox((0, 0), text, font=font)
    width, height = box[2] - box[0], box[3] - box[1]
    x, y = xy
    if anchor[0] == "m":
        x -= width // 2
    elif anchor[0] == "r":
        x -= width
    if anchor[1] == "m":
        y -= height // 2
    elif anchor[1] == "b":
        y -= height
    draw.text((x, y), text, font=font, fill=fill)


# ======================================================================================
# Symbol primitives
# ======================================================================================


def _arrowhead(
    draw: ImageDraw.ImageDraw,
    tip: tuple[int, int],
    direction: Literal["left", "right", "up", "down"],
    size: int = 12,
) -> None:
    """Draw a solid arrowhead whose point sits exactly on ``tip``."""
    x, y = tip
    half = size // 2
    if direction == "right":
        points = [(x, y), (x - size, y - half), (x - size, y + half)]
    elif direction == "left":
        points = [(x, y), (x + size, y - half), (x + size, y + half)]
    elif direction == "down":
        points = [(x, y), (x - half, y - size), (x + half, y - size)]
    else:
        points = [(x, y), (x - half, y + size), (x + half, y + size)]
    draw.polygon(points, fill=_INK)


def _polyline(draw: ImageDraw.ImageDraw, points: Sequence[tuple[int, int]], width: int = _W_PIPE) -> None:
    """Draw a connected run of straight segments with square joints."""
    draw.line([tuple(p) for p in points], fill=_INK, width=width, joint="curve")


def _dashed(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    dash: int = 9,
    gap: int = 6,
    width: int = _W_LEADER,
) -> None:
    """Draw a dashed straight line — the ISO convention for an instrument signal line."""
    x0, y0 = start
    x1, y1 = end
    length = math.hypot(x1 - x0, y1 - y0)
    if length <= 0:
        return
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    position = 0.0
    while position < length:
        seg_end = min(position + dash, length)
        draw.line(
            [(x0 + ux * position, y0 + uy * position), (x0 + ux * seg_end, y0 + uy * seg_end)],
            fill=_INK,
            width=width,
        )
        position = seg_end + gap


def _pump(
    draw: ImageDraw.ImageDraw,
    centre: tuple[int, int],
    radius: int,
    *,
    with_base: bool = True,
) -> None:
    """ISO centrifugal pump: a circle with the impeller triangle pointing at the discharge."""
    cx, cy = centre
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=_INK, width=_W_EQUIP)
    apex = (cx, cy - radius)
    base_left = (cx - int(radius * 0.72), cy + int(radius * 0.62))
    base_right = (cx + int(radius * 0.72), cy + int(radius * 0.62))
    draw.polygon([apex, base_left, base_right], outline=_INK)
    draw.line([apex, base_left], fill=_INK, width=2)
    draw.line([apex, base_right], fill=_INK, width=2)
    draw.line([base_left, base_right], fill=_INK, width=2)
    if with_base:
        foot = cy + radius + 14
        draw.line([(cx - radius - 6, foot), (cx + radius + 6, foot)], fill=_INK, width=_W_EQUIP)
        draw.line([(cx - 26, cy + radius), (cx - 26, foot)], fill=_INK, width=2)
        draw.line([(cx + 26, cy + radius), (cx + 26, foot)], fill=_INK, width=2)


def _vessel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    """Vertical vessel: a rectangular shell closed by two dished heads."""
    x0, y0, x1, y1 = box
    head = (x1 - x0) // 4
    draw.line([(x0, y0 + head), (x0, y1 - head)], fill=_INK, width=_W_EQUIP)
    draw.line([(x1, y0 + head), (x1, y1 - head)], fill=_INK, width=_W_EQUIP)
    draw.arc((x0, y0, x1, y0 + 2 * head), start=180, end=360, fill=_INK, width=_W_EQUIP)
    draw.arc((x0, y1 - 2 * head, x1, y1), start=0, end=180, fill=_INK, width=_W_EQUIP)
    # Internal spray-tray deck, which is what makes it read as a deaerator rather than a drum.
    tray_y = y0 + head + 40
    for offset in (0, 26):
        draw.line([(x0 + 14, tray_y + offset), (x1 - 14, tray_y + offset)], fill=_GREY, width=1)
    # Liquid level indication.
    level_y = y1 - 2 * head - 26
    draw.line([(x0 + 10, level_y), (x1 - 10, level_y)], fill=_GREY, width=1)


def _exchanger(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    """Shell-and-tube exchanger: shell rectangle, two tube sheets, tube passes."""
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=_INK, width=_W_EQUIP)
    sheet_left, sheet_right = x0 + 30, x1 - 30
    draw.line([(sheet_left, y0), (sheet_left, y1)], fill=_INK, width=2)
    draw.line([(sheet_right, y0), (sheet_right, y1)], fill=_INK, width=2)
    mid = (y0 + y1) // 2
    for offset in (-22, 22):
        draw.line([(sheet_left, mid + offset), (sheet_right, mid + offset)], fill=_INK, width=2)


def _valve(
    draw: ImageDraw.ImageDraw,
    centre: tuple[int, int],
    *,
    size: int = 22,
    orientation: Literal["horizontal", "vertical"] = "horizontal",
    hand_wheel: bool = True,
) -> None:
    """Bow-tie valve body, with an optional hand-wheel stem."""
    cx, cy = centre
    half = size // 2
    if orientation == "horizontal":
        left = [(cx - size, cy - half), (cx - size, cy + half), (cx, cy)]
        right = [(cx + size, cy - half), (cx + size, cy + half), (cx, cy)]
    else:
        left = [(cx - half, cy - size), (cx + half, cy - size), (cx, cy)]
        right = [(cx - half, cy + size), (cx + half, cy + size), (cx, cy)]
    for triangle in (left, right):
        draw.polygon(triangle, outline=_INK)
        draw.line([*triangle, triangle[0]], fill=_INK, width=2)
    if hand_wheel:
        stem_top = cy - size - 16
        draw.line([(cx, cy - half), (cx, stem_top)], fill=_INK, width=2)
        draw.line([(cx - 14, stem_top), (cx + 14, stem_top)], fill=_INK, width=2)


def _bubble(
    draw: ImageDraw.ImageDraw,
    centre: tuple[int, int],
    tag: str,
    font_small: ImageFont.ImageFont | ImageFont.FreeTypeFont,
) -> None:
    """Instrument bubble: circle, horizontal diameter, function letters over loop number."""
    cx, cy = centre
    r = _BUBBLE_RADIUS
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=_WHITE, outline=_INK, width=2)
    draw.line([(cx - r, cy), (cx + r, cy)], fill=_INK, width=2)
    letters, _, number = tag.partition("-")
    _text(draw, (cx, cy - 8), letters, font_small, anchor="mm")
    _text(draw, (cx, cy + 10), number, font_small, anchor="mm")


# ======================================================================================
# The P&ID
# ======================================================================================


def render_pid() -> Image.Image:
    """Draw the Unit-2 boiler feed water P&ID and return it as an RGB image.

    Raises:
        VisionError: If Pillow cannot compose the drawing.
    """
    try:
        return _render_pid_unguarded()
    except Exception as exc:  # noqa: BLE001 - broad third-party surface, re-raised typed
        raise VisionError(
            "Failed to render the P-101 P&ID. Check that Pillow is installed correctly "
            "(`python -c \"from PIL import Image, ImageDraw, ImageFont\"`) and that a scalable "
            "font is available.",
            context={"canvas": f"{PID_WIDTH}x{PID_HEIGHT}"},
            cause=exc,
        ) from exc


def _render_pid_unguarded() -> Image.Image:
    image = Image.new("RGB", (PID_WIDTH, PID_HEIGHT), _WHITE)
    draw = ImageDraw.Draw(image)

    f_tag = load_font(26)
    f_tag_big = load_font(30)
    f_body = load_font(18)
    f_small = load_font(15)
    f_tiny = load_font(13)

    # ---- sheet frame and title block -------------------------------------------------
    draw.rectangle((24, 24, PID_WIDTH - 24, PID_HEIGHT - 24), outline=_INK, width=_W_FRAME)
    draw.rectangle((40, 40, PID_WIDTH - 40, PID_HEIGHT - 40), outline=_INK, width=1)

    tb_x0, tb_y0, tb_x1, tb_y1 = 1120, 930, 1560, 1080
    draw.rectangle((tb_x0, tb_y0, tb_x1, tb_y1), outline=_INK, width=2)
    for y in (tb_y0 + 34, tb_y0 + 68, tb_y0 + 100):
        draw.line([(tb_x0, y), (tb_x1, y)], fill=_INK, width=1)
    draw.line([(tb_x0 + 220, tb_y0 + 100), (tb_x0 + 220, tb_y1)], fill=_INK, width=1)
    _text(draw, (tb_x0 + 10, tb_y0 + 9), "BHARAT VINDHYA PETROCHEMICALS LTD.", f_small, bold=True)
    _text(draw, (tb_x0 + 10, tb_y0 + 43), "P&ID - BOILER FEED WATER SYSTEM", f_small, bold=True)
    _text(draw, (tb_x0 + 10, tb_y0 + 75), "UNIT-2 UTILITIES BLOCK", f_small)
    _text(draw, (tb_x0 + 10, tb_y0 + 108), "DWG No.  PID-U2-1010", f_tiny)
    _text(draw, (tb_x0 + 10, tb_y0 + 128), "DATE  2016-01-15", f_tiny)
    _text(draw, (tb_x0 + 230, tb_y0 + 108), "REV  3", f_tiny)
    _text(draw, (tb_x0 + 230, tb_y0 + 128), "SCALE  NTS", f_tiny)

    _text(draw, (60, 58), "PROCESS & INSTRUMENTATION DIAGRAM - BOILER FEED WATER SYSTEM",
          f_tag, bold=True)
    _text(draw, (60, 92), "UNIT-2 UTILITIES BLOCK  /  SHEET 1 OF 1  /  DWG PID-U2-1010 REV 3", f_body)

    # ---- V-201 deaerator storage vessel ----------------------------------------------
    _vessel(draw, _V201_BOX)
    vx0, vy0, vx1, vy1 = _V201_BOX
    vcx = (vx0 + vx1) // 2
    _text(draw, (vx0, vy0 - 62), "V-201", f_tag_big, bold=True)
    _text(draw, (vx0, vy0 - 30), "DEAERATOR STORAGE VESSEL", f_small)
    _text(draw, (vx0 + 4, vy1 + 12), "24 m3 / 8.5 barg", f_tiny, fill=_GREY)

    # Condensate return into the vessel from off-sheet.
    _polyline(draw, [(vcx, 150), (vcx, vy0 + 6)])
    _arrowhead(draw, (vcx, vy0 + 8), "down")
    _text(draw, (vcx + 14, 146), "CONDENSATE RETURN FROM U-2", f_tiny, fill=_GREY)

    # TI-2011 on the vessel outlet.
    _bubble(draw, (400, 180), "TI-2011", f_small)
    _dashed(draw, (400 - _BUBBLE_RADIUS, 180), (vcx + 40, 180))
    _text(draw, (400 + _BUBBLE_RADIUS + 8, 168), "TI-2011", f_tiny, fill=_GREY)

    # ---- suction header: V-201 -> P-101 (and branch to P-102) ------------------------
    _polyline(draw, [(vcx, vy1), (vcx, 700), (_P101_CENTRE[0] - _PUMP_RADIUS - 4, 700)])
    _arrowhead(draw, (_P101_CENTRE[0] - _PUMP_RADIUS - 2, 700), "right")
    _valve(draw, (400, 700))
    _text(draw, (378, 640), "HV-1012", f_tiny)
    _text(draw, (470, 676), '6"-BFW-1001-CS', f_tiny, fill=_GREY)

    _polyline(draw, [(500, 700), (500, 900), (_P102_CENTRE[0] - _PUMP_RADIUS - 4, 900)])
    _arrowhead(draw, (_P102_CENTRE[0] - _PUMP_RADIUS - 2, 900), "right")
    draw.ellipse((496, 696, 504, 704), fill=_INK)  # tee

    # ---- P-101 and P-102 --------------------------------------------------------------
    _pump(draw, _P101_CENTRE, _PUMP_RADIUS)
    _text(draw, (_P101_CENTRE[0] - 44, _P101_CENTRE[1] + _PUMP_RADIUS + 26), "P-101", f_tag_big,
          bold=True)
    _text(draw, (_P101_CENTRE[0] - 44, _P101_CENTRE[1] + _PUMP_RADIUS + 60),
          "BFW PUMP - DUTY", f_small)
    _text(draw, (_P101_CENTRE[0] - 44, _P101_CENTRE[1] + _PUMP_RADIUS + 84),
          "SULZER CP 150-400  148 m3/h @ 395 m", f_tiny, fill=_GREY)

    _pump(draw, _P102_CENTRE, _PUMP_RADIUS)
    _text(draw, (_P102_CENTRE[0] - 44, _P102_CENTRE[1] + _PUMP_RADIUS + 26), "P-102", f_tag_big,
          bold=True)
    _text(draw, (_P102_CENTRE[0] - 44, _P102_CENTRE[1] + _PUMP_RADIUS + 60),
          "BFW PUMP - STANDBY", f_small)

    # VT-1011 vibration transmitter on the P-101 bearing housing.
    _bubble(draw, (770, 700), "VT-1011", f_small)
    _dashed(draw, (770 - _BUBBLE_RADIUS, 700), (_P101_CENTRE[0] + _PUMP_RADIUS, 700))
    _text(draw, (770 + _BUBBLE_RADIUS + 8, 688), "VT-1011", f_tiny, fill=_GREY)

    # ---- discharge header: P-101 / P-102 -> E-301 -------------------------------------
    _polyline(draw, [(_P101_CENTRE[0], _P101_CENTRE[1] - _PUMP_RADIUS), (_P101_CENTRE[0], 340),
                     (830, 340), (830, 490), (_E301_BOX[0] - 4, 490)])
    _arrowhead(draw, (_E301_BOX[0] - 2, 490), "right")
    _polyline(draw, [(_P102_CENTRE[0] + _PUMP_RADIUS, 900), (960, 900), (960, 620), (830, 620),
                     (830, 500)])
    _valve(draw, (700, 340))
    _text(draw, (676, 282), "CV-1013", f_tiny)
    _text(draw, (880, 300), '4"-BFW-1002-CS', f_tiny, fill=_GREY)

    _bubble(draw, (770, 240), "PI-1015", f_small)
    _dashed(draw, (770, 240 + _BUBBLE_RADIUS), (770, 340))
    _text(draw, (770 + _BUBBLE_RADIUS + 8, 228), "PI-1015", f_tiny, fill=_GREY)

    # ---- E-301 feed water pre-heater ---------------------------------------------------
    _exchanger(draw, _E301_BOX)
    ex0, ey0, ex1, ey1 = _E301_BOX
    _text(draw, (ex0, ey0 - 62), "E-301", f_tag_big, bold=True)
    _text(draw, (ex0, ey0 - 30), "FEED WATER PRE-HEATER", f_small)
    _text(draw, (ex0, ey1 + 12), "TEMA BEM 500-4000 / 1.8 MW", f_tiny, fill=_GREY)

    ecx = (ex0 + ex1) // 2
    _polyline(draw, [(ecx, ey0 - 60), (ecx, ey0)], width=_W_UTILITY)
    _arrowhead(draw, (ecx, ey0 + 2), "down")
    _text(draw, (ecx + 12, ey0 - 78), "LP STEAM 3.5 barg", f_tiny, fill=_GREY)
    _polyline(draw, [(ecx, ey1), (ecx, ey1 + 56)], width=_W_UTILITY)
    _arrowhead(draw, (ecx, ey1 + 58), "down")
    _text(draw, (ecx + 12, ey1 + 40), "CONDENSATE TO U-2", f_tiny, fill=_GREY)

    # ---- E-301 -> boiler off-sheet -----------------------------------------------------
    _polyline(draw, [(ex1, 490), (1300, 490), (1300, 180), (1440, 180)])
    _arrowhead(draw, (1442, 180), "right")
    _valve(draw, (1230, 490))
    _text(draw, (1206, 432), "HV-3014", f_tiny)
    _text(draw, (1250, 146), "TO BOILER B-401", f_body, bold=True)
    _text(draw, (1250, 200), '4"-BFW-1003-CS', f_tiny, fill=_GREY)

    # ---- LP-101A auxiliary lube oil pump ----------------------------------------------
    _pump(draw, _LP101A_CENTRE, _LP101A_RADIUS, with_base=False)
    lx, ly = _LP101A_CENTRE
    _text(draw, (lx - 66, ly + _LP101A_RADIUS + 14), "LP-101A", f_tag, bold=True)
    _text(draw, (lx - 66, ly + _LP101A_RADIUS + 42), "AUX LUBE OIL PUMP", f_tiny, fill=_GREY)
    _polyline(draw, [(lx + _LP101A_RADIUS, ly), (560, ly), (560, 668),
                     (_P101_CENTRE[0] - _PUMP_RADIUS + 6, 668)], width=_W_UTILITY)
    _arrowhead(draw, (_P101_CENTRE[0] - _PUMP_RADIUS + 8, 668), "right")
    _text(draw, (lx + _LP101A_RADIUS + 10, ly - 24), '1"-LO-1010  18 l/min @ 2.1 bar', f_tiny,
          fill=_GREY)
    _bubble(draw, (500, 380), "PI-1016", f_small)
    _dashed(draw, (500, 380 + _BUBBLE_RADIUS), (500, ly))
    _text(draw, (500 + _BUBBLE_RADIUS + 8, 368), "PI-1016", f_tiny, fill=_GREY)

    # ---- notes block -------------------------------------------------------------------
    nx0, ny0 = 60, 986
    draw.rectangle((nx0, ny0, 1080, 1076), outline=_INK, width=1)
    _text(draw, (nx0 + 12, ny0 + 8), "NOTES:", f_small, bold=True)
    notes = (
        '1.  ALL PROCESS LINES CS ASTM A106 Gr.B SCH 40 UNLESS NOTED.',
        "2.  P-101 / P-102 DUTY-STANDBY.  AUTO CHANGEOVER ON LOW DISCHARGE PRESSURE AT PI-1015.",
        "3.  VT-1011 VIBRATION ALARM 7.1 mm/s RMS, TRIP 9.5 mm/s.  PI-1016 LUBE OIL LOW ALARM 1.4 bar.",
    )
    for index, note in enumerate(notes):
        _text(draw, (nx0 + 90, ny0 + 8 + index * 26), note, f_tiny)

    return image


# ======================================================================================
# Scan degradation
# ======================================================================================


def degrade_scan(image: Image.Image, *, seed: int) -> Image.Image:
    """Return ``image`` as if photocopied and scanned on tired office hardware.

    Applies, in order: a small rotation, uneven illumination, contrast compression, gaussian sensor
    noise, dust specks, optical blur, and a genuine low-quality JPEG encode/decode round trip. The
    JPEG step matters — synthetic noise alone does not produce the 8x8 ringing around glyph edges
    that makes an OCR engine read ``P-101`` as ``P-l0l``.

    Args:
        image: The clean drawing.
        seed: Deterministic seed; the same seed always produces the same degraded image.

    Raises:
        VisionError: If OpenCV or Pillow fails during degradation.
    """
    try:
        return _degrade_scan_unguarded(image, seed=seed)
    except Exception as exc:  # noqa: BLE001 - broad third-party surface, re-raised typed
        raise VisionError(
            "Failed to degrade the P&ID into a scanned copy. Verify that opencv-python-headless "
            "and numpy are installed (`python -c \"import cv2, numpy\"`).",
            context={"seed": seed},
            cause=exc,
        ) from exc


def _degrade_scan_unguarded(image: Image.Image, *, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)

    rotated = image.rotate(
        _SCAN_ROTATION_DEG, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=_WHITE
    )
    array = np.asarray(rotated.convert("L"), dtype=np.float32)
    height, width = array.shape

    # Uneven illumination: a scanner lamp is brighter in the middle of the platen.
    ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    shading = 1.0 - 0.16 * (xs**2) - 0.10 * (ys**2) + 0.05 * xs
    array = array * shading

    # Contrast compression: blacks lift towards grey, whites fall away from paper white.
    array = array * _SCAN_CONTRAST + _SCAN_LIFT
    array = array + rng.normal(0.0, _SCAN_NOISE_SIGMA, size=array.shape).astype(np.float32)

    # Dust and toner specks.
    for _ in range(_SCAN_SPECK_COUNT):
        cx = int(rng.integers(0, width))
        cy = int(rng.integers(0, height))
        radius = int(rng.integers(1, 4))
        value = float(rng.integers(20, 90)) if rng.random() < 0.7 else 250.0
        y0, y1 = max(0, cy - radius), min(height, cy + radius + 1)
        x0, x1 = max(0, cx - radius), min(width, cx + radius + 1)
        array[y0:y1, x0:x1] = value

    # Two faint roller streaks, which every real scan seems to have.
    for _ in range(2):
        row = int(rng.integers(0, height))
        array[row : row + 2, :] = np.clip(array[row : row + 2, :] - 18.0, 0.0, 255.0)

    degraded = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="L")
    degraded = degraded.filter(ImageFilter.GaussianBlur(radius=_SCAN_BLUR_RADIUS))

    # A real JPEG round trip, so the artefacts are the encoder's, not our imagination's.
    buffer = np.asarray(degraded, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", buffer, [int(cv2.IMWRITE_JPEG_QUALITY), _SCAN_JPEG_QUALITY])
    if not ok:
        raise VisionError(
            "OpenCV refused to JPEG-encode the degraded P&ID. The scanned copy would not carry "
            "real compression artefacts, so the OCR tag-correction path would not be exercised.",
            context={"quality": _SCAN_JPEG_QUALITY},
        )
    decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if decoded is None:  # pragma: no cover - defensive
        raise VisionError("OpenCV could not decode the JPEG it had just encoded.")
    return Image.fromarray(decoded, mode="L").convert("RGB")


# ======================================================================================
# Handwriting
# ======================================================================================


def render_handwriting(
    text: str,
    *,
    seed: int,
    size: int = 46,
    ink: tuple[int, int, int] = (26, 38, 120),
    paper: tuple[int, int, int] = (252, 251, 246),
    padding: int = 18,
) -> Image.Image:
    """Render ``text`` as a script-like hand on paper stock.

    Every glyph gets its own rotation, baseline offset, size jitter and stroke doubling, and the
    whole line is sheared. That per-glyph variation — not the choice of font — is what defeats a
    clean OCR read, which is precisely the point: the wear percentage the diagnosis hinges on is
    handwritten, so the answer that uses it must carry an uncertainty flag.

    Args:
        text: What was written.
        seed: Deterministic seed.
        size: Nominal glyph height in pixels.
        ink: Pen colour (a blue ballpoint by default).
        paper: Background colour.
        padding: Border around the written line.

    Raises:
        VisionError: If Pillow fails to compose the annotation.
    """
    try:
        return _render_handwriting_unguarded(
            text, seed=seed, size=size, ink=ink, paper=paper, padding=padding
        )
    except Exception as exc:  # noqa: BLE001 - broad third-party surface, re-raised typed
        raise VisionError(
            f"Failed to render the handwritten annotation {text!r}. Without it the work order "
            "carries no low-confidence OCR source and the uncertainty flag has no trigger.",
            context={"seed": seed},
            cause=exc,
        ) from exc


def _render_handwriting_unguarded(
    text: str,
    *,
    seed: int,
    size: int,
    ink: tuple[int, int, int],
    paper: tuple[int, int, int],
    padding: int,
) -> Image.Image:
    rng = np.random.default_rng(seed)
    font = load_font(size, handwriting=True)

    advance = int(size * 0.62)
    width = padding * 2 + max(1, len(text)) * advance + size
    height = padding * 2 + int(size * 1.9)
    layer = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(layer)

    baseline = padding + int(size * 1.15)
    cursor = float(padding + size * 0.2)
    for character in text:
        if character == " ":
            cursor += advance * 0.72
            continue
        jitter_y = float(rng.normal(0.0, size * 0.055))
        angle = float(rng.normal(0.0, 5.5))
        glyph_size = max(8, int(size * float(rng.uniform(0.92, 1.08))))
        glyph_font = load_font(glyph_size, handwriting=True)

        cell = Image.new("RGBA", (glyph_size * 3, glyph_size * 3), (255, 255, 255, 0))
        cell_draw = ImageDraw.Draw(cell)
        # Draw the glyph two or three times with sub-pixel offsets: a ballpoint does not lay down
        # a uniform stroke, and the doubled edge is what reads as ink rather than as type.
        for offset_x, offset_y in ((0, 0), (1, 0), (0, 1)):
            _text(
                cell_draw,
                (glyph_size + offset_x, glyph_size + offset_y),
                character,
                glyph_font,
                fill=ink,
                anchor="mm",
            )
        cell = cell.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)
        layer.alpha_composite(
            cell,
            dest=(int(cursor) - glyph_size, int(baseline + jitter_y) - 2 * glyph_size),
        )
        measured = cell_draw.textlength(character, font=glyph_font)
        cursor += max(advance * 0.55, float(measured) * 0.98) + float(rng.uniform(-1.5, 2.5))

    # Slant the whole line the way a right-handed hand does.
    shear = 0.16
    layer = layer.transform(
        (width + int(height * shear), height),
        Image.Transform.AFFINE,
        (1.0, shear, -shear * height, 0.0, 1.0, 0.0),
        resample=Image.Resampling.BICUBIC,
    )
    layer = layer.filter(ImageFilter.GaussianBlur(radius=0.45))

    sheet = Image.new("RGB", layer.size, paper)
    sheet.paste(layer, (0, 0), layer)

    # Light paper grain, so the strip reads as a scanned fragment rather than a rendered one.
    grain = rng.normal(0.0, 3.2, size=(layer.size[1], layer.size[0], 3)).astype(np.float32)
    noisy = np.clip(np.asarray(sheet, dtype=np.float32) + grain, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy, mode="RGB")


# ======================================================================================
# Nameplate photograph
# ======================================================================================


def render_nameplate(
    lines: Sequence[tuple[str, str]],
    *,
    seed: int,
    heading: str,
    width: int = 1000,
    height: int = 640,
) -> Image.Image:
    """Render a synthetic photograph of an engraved equipment nameplate.

    Brushed-steel background, engraved (double-offset) lettering, four fixing holes, a diagonal
    glare band, camera noise and a mild perspective — enough that tag detection and OCR have real
    work to do, and the photo-to-query beat is not a string comparison in disguise.

    Args:
        lines: ``(label, value)`` rows stamped into the plate.
        seed: Deterministic seed.
        heading: Manufacturer/owner line across the top of the plate.
        width: Output width in pixels.
        height: Output height in pixels.

    Raises:
        VisionError: If the photo cannot be composed.
    """
    try:
        return _render_nameplate_unguarded(
            lines, seed=seed, heading=heading, width=width, height=height
        )
    except Exception as exc:  # noqa: BLE001 - broad third-party surface, re-raised typed
        raise VisionError(
            "Failed to render the P-101 nameplate photograph used by the photo-to-query demo beat.",
            context={"seed": seed},
            cause=exc,
        ) from exc


def _render_nameplate_unguarded(
    lines: Sequence[tuple[str, str]],
    *,
    seed: int,
    heading: str,
    width: int,
    height: int,
) -> Image.Image:
    rng = np.random.default_rng(seed)

    base = np.full((height, width, 3), 168.0, dtype=np.float32)
    # Brushed finish: horizontal streaks of slightly different luminance.
    streaks = rng.normal(0.0, 7.0, size=(height, 1, 1)).astype(np.float32)
    base += streaks
    base += rng.normal(0.0, 2.5, size=base.shape).astype(np.float32)
    # Cool steel tint.
    base[:, :, 2] += 8.0
    base[:, :, 0] -= 4.0
    plate = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(plate)

    margin = 34
    draw.rounded_rectangle(
        (margin, margin, width - margin, height - margin), radius=18, outline=(96, 100, 108), width=4
    )
    draw.rounded_rectangle(
        (margin + 8, margin + 8, width - margin - 8, height - margin - 8),
        radius=12, outline=(206, 210, 216), width=1,
    )
    for hx, hy in ((margin + 34, margin + 34), (width - margin - 34, margin + 34),
                   (margin + 34, height - margin - 34), (width - margin - 34, height - margin - 34)):
        draw.ellipse((hx - 13, hy - 13, hx + 13, hy + 13), fill=(58, 60, 66),
                     outline=(210, 212, 218), width=2)

    f_heading = load_font(34)
    f_label = load_font(26)
    f_value = load_font(34)

    def engrave(xy: tuple[int, int], text: str, font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
                *, bold: bool = False) -> None:
        """Engraved lettering: a dark bottom-right edge, a light top-left edge, mid-grey face."""
        x, y = xy
        _text(draw, (x + 2, y + 2), text, font, fill=(214, 218, 224), bold=bold)
        _text(draw, (x - 1, y - 1), text, font, fill=(58, 60, 66), bold=bold)
        _text(draw, (x, y), text, font, fill=(96, 99, 106), bold=bold)

    engrave((margin + 74, margin + 24), heading, f_heading, bold=True)
    draw.line([(margin + 74, margin + 72), (width - margin - 74, margin + 72)], fill=(112, 116, 124),
              width=2)

    row_y = margin + 96
    for label, value in lines:
        engrave((margin + 74, row_y), label, f_label)
        engrave((margin + 300, row_y - 4), value, f_value, bold=True)
        row_y += 62

    # Mild perspective, as if photographed from slightly off-axis and below.
    source = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    target = np.float32([
        [width * 0.045, height * 0.030],
        [width * 0.972, height * 0.008],
        [width * 0.955, height * 0.982],
        [width * 0.020, height * 0.945],
    ])
    matrix = cv2.getPerspectiveTransform(source, target)
    warped = cv2.warpPerspective(
        np.asarray(plate), matrix, (width, height), borderMode=cv2.BORDER_REPLICATE
    ).astype(np.float32)

    # Diagonal glare band from the workshop lighting.
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    band = np.exp(-(((xx * 0.55 + yy * 0.45) - width * 0.42) ** 2) / (2.0 * (width * 0.16) ** 2))
    warped += (band * 62.0)[:, :, None]
    warped += rng.normal(0.0, 4.5, size=warped.shape).astype(np.float32)

    photo = np.clip(warped, 0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(photo, cv2.COLOR_RGB2BGR),
                               [int(cv2.IMWRITE_JPEG_QUALITY), 62])
    if not ok:  # pragma: no cover - defensive
        raise VisionError("OpenCV refused to JPEG-encode the nameplate photograph.")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:  # pragma: no cover - defensive
        raise VisionError("OpenCV could not decode the nameplate JPEG it had just encoded.")
    return Image.fromarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB), mode="RGB")


# ======================================================================================
# Persistence
# ======================================================================================


def save_image(image: Image.Image, path: Path, *, jpeg_quality: int | None = None) -> int:
    """Write ``image`` to ``path`` and return the byte size.

    PNG is written without a ``tIME`` chunk and JPEG without EXIF, so repeated generation produces
    byte-identical files and content-addressed ingestion (D6) stays idempotent.

    Raises:
        BlobStoreError: If the file cannot be written.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() in {".jpg", ".jpeg"}:
            image.save(path, format="JPEG", quality=jpeg_quality or 88, optimize=True)
        else:
            image.save(path, format="PNG", optimize=True)
        return path.stat().st_size
    except (OSError, ValueError) as exc:
        raise BlobStoreError(
            f"Could not write the generated image to {path}. Check that the directory is writable "
            "and that there is free disk space.",
            context={"path": str(path)},
            cause=exc,
        ) from exc


__all__ = [
    "PID_CONNECTIONS", "PID_HEIGHT", "PID_SYMBOLS", "PID_WIDTH", "PidConnectionSpec",
    "PidSymbolSpec", "degrade_scan", "load_font", "render_handwriting", "render_nameplate",
    "render_pid", "save_image",
]
