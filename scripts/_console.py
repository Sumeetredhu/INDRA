"""Terminal report writer for the INDRA scripts.

Why this exists rather than ``print()`` or ``logger.info()``
------------------------------------------------------------
``CLAUDE.md`` rule 3 bans ``print()`` and mandates :func:`indra.core.logging.get_logger` — in
library and agent code, where the structured log *is* the observability channel. The four scripts
in this package are different: their product is a report a human reads in a terminal. A PASS/FAIL
table rendered through the logging formatter carries a timestamp, level, agent and correlation id
on every row, which destroys the alignment that makes the table readable, and under
``settings.log_json`` it is not a table at all.

So the *report* goes to ``sys.stdout`` through this one small, auditable writer, and every
diagnostic, warning and error raised along the way still goes through ``get_logger``. One place
writes to stdout in this repository, and it is this module.

The writer degrades safely: no colour when the stream is not a TTY or ``NO_COLOR`` is set, and
ASCII box characters when the stream encoding cannot represent the Unicode ones (a real concern in
a legacy Windows console).
"""

from __future__ import annotations

import os
import sys
from typing import Final, Literal, Sequence, TextIO

Align = Literal["left", "right", "center"]
State = Literal["pass", "fail", "warn", "skip", "info"]

_RESET: Final[str] = "\033[0m"
_COLOURS: Final[dict[str, str]] = {
    "dim": "\033[90m",
    "bold": "\033[1m",
    "green": "\033[32m",
    "red": "\033[31m",
    "yellow": "\033[33m",
    "blue": "\033[36m",
    "magenta": "\033[35m",
}

_STATE_STYLE: Final[dict[State, tuple[str, str]]] = {
    "pass": ("PASS", "green"),
    "fail": ("FAIL", "red"),
    "warn": ("WARN", "yellow"),
    "skip": ("SKIP", "dim"),
    "info": ("INFO", "blue"),
}

#: Unicode glyph -> ASCII substitute, applied when the output encoding cannot carry the glyph.
_ASCII_FALLBACK: Final[dict[str, str]] = {
    "─": "-", "│": "|", "┌": "+", "┐": "+", "└": "+", "┘": "+",
    "├": "+", "┤": "+", "┬": "+", "┴": "+", "┼": "+", "═": "=",
    "•": "*", "→": "->", "✓": "ok", "✗": "x",
}


class Console:
    """Writes aligned, optionally coloured report output to a text stream.

    Args:
        stream: Destination. Defaults to ``sys.stdout``.
        colour: Force colour on or off. ``None`` auto-detects (TTY and no ``NO_COLOR``).
        width: Total width used by :meth:`rule` and by table truncation.
    """

    def __init__(
        self,
        stream: TextIO | None = None,
        *,
        colour: bool | None = None,
        width: int = 100,
    ) -> None:
        self._stream: TextIO = stream if stream is not None else sys.stdout
        self.width = width
        self._colour: bool = self._detect_colour() if colour is None else colour
        self._unicode: bool = self._detect_unicode()

    # ------------------------------------------------------------------ capability probes
    def _detect_colour(self) -> bool:
        if os.environ.get("NO_COLOR"):
            return False
        isatty = getattr(self._stream, "isatty", None)
        try:
            return bool(isatty and isatty())
        except (OSError, ValueError):  # pragma: no cover - exotic stream
            return False

    def _detect_unicode(self) -> bool:
        encoding = getattr(self._stream, "encoding", None) or "ascii"
        try:
            "─┌┐└┘├┤┬┴┼•→".encode(encoding)
        except (UnicodeEncodeError, LookupError):
            return False
        return True

    # ------------------------------------------------------------------ primitives
    def _sanitise(self, text: str) -> str:
        if self._unicode:
            return text
        for glyph, replacement in _ASCII_FALLBACK.items():
            text = text.replace(glyph, replacement)
        return text.encode("ascii", "replace").decode("ascii")

    def paint(self, text: str, colour: str) -> str:
        """Wrap ``text`` in an ANSI colour, or return it unchanged when colour is disabled."""
        if not self._colour or colour not in _COLOURS:
            return text
        return f"{_COLOURS[colour]}{text}{_RESET}"

    def write(self, text: str = "") -> None:
        """Write one line to the stream and flush, so output interleaves correctly with logs."""
        try:
            self._stream.write(self._sanitise(text) + "\n")
            self._stream.flush()
        except (OSError, ValueError):  # pragma: no cover - closed/broken pipe
            pass

    def blank(self) -> None:
        """Write an empty line."""
        self.write("")

    # ------------------------------------------------------------------ composites
    def banner(self, title: str, subtitle: str = "") -> None:
        """A heavy top-of-report header."""
        bar = "═" * self.width if self._unicode else "=" * self.width
        self.write(self.paint(bar, "dim"))
        self.write(self.paint(title, "bold"))
        if subtitle:
            self.write(self.paint(subtitle, "dim"))
        self.write(self.paint(bar, "dim"))

    def rule(self, title: str = "") -> None:
        """A section divider, optionally labelled."""
        dash = "─" if self._unicode else "-"
        if not title:
            self.write(self.paint(dash * self.width, "dim"))
            return
        label = f"{dash}{dash} {title} "
        self.write(self.paint(label + dash * max(0, self.width - len(label)), "dim"))

    def kv(self, key: str, value: str, *, key_width: int = 26) -> None:
        """A left-aligned ``key: value`` line."""
        self.write(f"  {self.paint(key.ljust(key_width), 'dim')}{value}")

    def bullet(self, text: str, *, indent: int = 2) -> None:
        """A bulleted line."""
        self.write(f"{' ' * indent}{self.paint('•', 'dim')} {text}")

    def status(self, state: State, label: str, detail: str = "") -> None:
        """A single ``[PASS] label — detail`` line."""
        word, colour = _STATE_STYLE[state]
        tail = f"  {self.paint(detail, 'dim')}" if detail else ""
        self.write(f"  [{self.paint(word, colour)}] {label}{tail}")

    def table(
        self,
        headers: Sequence[str],
        rows: Sequence[Sequence[str]],
        *,
        aligns: Sequence[Align] | None = None,
        max_cell: int = 60,
    ) -> None:
        """Render a bordered table.

        Args:
            headers: Column titles.
            rows: Row cells; every row must have ``len(headers)`` entries.
            aligns: Per-column alignment. Defaults to left for every column.
            max_cell: Cells longer than this are truncated with an ellipsis.
        """
        if not headers:
            return
        columns = len(headers)
        alignment: list[Align] = list(aligns) if aligns else ["left"] * columns
        if len(alignment) != columns:
            alignment = (alignment + ["left"] * columns)[:columns]

        def clip(cell: str) -> str:
            flat = " ".join(str(cell).split())
            return flat if len(flat) <= max_cell else flat[: max_cell - 1] + "…"

        body: list[list[str]] = [[clip(c) for c in row] for row in rows]
        widths = [len(str(h)) for h in headers]
        for row in body:
            for i, cell in enumerate(row[:columns]):
                widths[i] = max(widths[i], len(_strip_ansi(cell)))

        h, v = ("─", "│") if self._unicode else ("-", "|")
        corners = ("┌┬┐", "├┼┤", "└┴┘") if self._unicode else ("+++", "+++", "+++")

        def border(spec: str) -> str:
            left, mid, right = spec
            return left + mid.join(h * (w + 2) for w in widths) + right

        def line(cells: Sequence[str]) -> str:
            parts: list[str] = []
            for i, cell in enumerate(cells[:columns]):
                visible = _strip_ansi(cell)
                pad = widths[i] - len(visible)
                if alignment[i] == "right":
                    parts.append(" " * pad + cell)
                elif alignment[i] == "center":
                    left_pad = pad // 2
                    parts.append(" " * left_pad + cell + " " * (pad - left_pad))
                else:
                    parts.append(cell + " " * pad)
            return v + v.join(f" {p} " for p in parts) + v

        self.write(self.paint(border(corners[0]), "dim"))
        self.write(line([self.paint(str(x), "bold") for x in headers]))
        self.write(self.paint(border(corners[1]), "dim"))
        for row in body:
            self.write(line(row))
        self.write(self.paint(border(corners[2]), "dim"))

    def state_cell(self, state: State) -> str:
        """A coloured status word sized for use inside :meth:`table`."""
        word, colour = _STATE_STYLE[state]
        return self.paint(word, colour)


def _strip_ansi(text: str) -> str:
    """Return ``text`` without ANSI escape sequences, for width computation."""
    out: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == "\033":
            end = text.find("m", i)
            if end == -1:
                break
            i = end + 1
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


__all__ = ["Align", "Console", "State"]
