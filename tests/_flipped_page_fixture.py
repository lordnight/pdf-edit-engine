"""Shared page-orientation fixture for the reflow orientation probes.

Builds one-page standard-14 Helvetica (WinAnsi) PDFs whose text runs are laid
out in PAGE space (y grows up from the MediaBox bottom) and written through a
chosen CTM + text-matrix orientation, so one layout and one set of assertions
cover an identity page, a Y-flipped page (``cm 1 0 0 -1 0 h``) carrying a
negative-d, unit or positive-d ``Tm``, a font-size-in-Tm run (``Tf 1`` +
``Tm [12 0 0 12 ...]``) and a scaled CTM. Every orientation draws 12pt text on
the page. pdfminer is the layout oracle: it places glyphs independently of the
engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pikepdf
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTChar, LTTextLine

if TYPE_CHECKING:
    from pathlib import Path

PAGE_WIDTH = 595.0
PAGE_HEIGHT = 842.0
PAGE_FONT_SIZE = 12.0

_Matrix = tuple[float, float, float, float, float, float]
IDENTITY_CTM: _Matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
FLIP_CTM: _Matrix = (1.0, 0.0, 0.0, -1.0, 0.0, PAGE_HEIGHT)
HALF_SCALE_CTM: _Matrix = (0.5, 0.0, 0.0, 0.5, 0.0, 0.0)


def _num(value: float) -> str:
    return f"{value:g}"


def _to_user(ctm: _Matrix, x: float, y: float) -> tuple[float, float]:
    a, b, c, d, e, f = ctm
    det = a * d - b * c
    px, py = x - e, y - f
    return ((px * d - py * c) / det, (py * a - px * b) / det)


@dataclass(frozen=True)
class Orientation:
    """How a run laid out in page space is written into the content stream.

    ``upright`` is how the glyphs render on the page: a positive-d text
    rendering matrix. The mirrored orientations are kept as-is by reflow.
    """

    name: str
    ctm: _Matrix
    tf_size: float
    tm_linear: tuple[float, float, float, float] | None
    upright: bool = True

    def prologue(self) -> bytes:
        if self.ctm == IDENTITY_CTM:
            return b""
        return (" ".join(_num(v) for v in self.ctm) + " cm\n").encode()

    def run(self, x: float, y: float, text: str) -> bytes:
        ux, uy = _to_user(self.ctm, x, y)
        if self.tm_linear is None:
            position = f"{_num(ux)} {_num(uy)} Td"
        else:
            position = " ".join(_num(v) for v in (*self.tm_linear, ux, uy)) + " Tm"
        return f"BT /F1 {_num(self.tf_size)} Tf {position} ({text}) Tj ET\n".encode()


IDENTITY = Orientation("identity", IDENTITY_CTM, 12.0, None, upright=True)
FLIP_NEGATIVE_D_TM = Orientation(
    "flip_negative_d_tm", FLIP_CTM, 1.0, (12.0, 0.0, 0.0, -12.0), upright=True
)
FLIP_UNIT_TM = Orientation("flip_unit_tm", FLIP_CTM, 12.0, (1.0, 0.0, 0.0, -1.0), upright=True)
FLIP_POSITIVE_D_TM = Orientation(
    "flip_positive_d_tm", FLIP_CTM, 1.0, (12.0, 0.0, 0.0, 12.0), upright=False
)
FLIP_IDENTITY_TM = Orientation("flip_identity_tm", FLIP_CTM, 12.0, None, upright=False)
TM_SCALED = Orientation("tm_scaled", IDENTITY_CTM, 1.0, (12.0, 0.0, 0.0, 12.0), upright=True)
SCALED_CTM = Orientation("scaled_ctm", HALF_SCALE_CTM, 24.0, None, upright=True)


def build_page_pdf(
    path: Path,
    content: bytes,
    *,
    mediabox: tuple[float, float, float, float] = (0.0, 0.0, PAGE_WIDTH, PAGE_HEIGHT),
) -> Path:
    """Write a one-page PDF whose /F1 is standard-14 Helvetica (WinAnsi)."""
    pdf = pikepdf.new()
    try:
        helvetica = pdf.make_indirect(
            pikepdf.Dictionary(
                Type=pikepdf.Name.Font,
                Subtype=pikepdf.Name.Type1,
                BaseFont=pikepdf.Name.Helvetica,
                Encoding=pikepdf.Name.WinAnsiEncoding,
            )
        )
        page = pikepdf.Dictionary(
            Type=pikepdf.Name.Page,
            MediaBox=list(mediabox),
            Resources=pikepdf.Dictionary(Font=pikepdf.Dictionary(F1=helvetica)),
            Contents=pdf.make_stream(content),
        )
        pdf.pages.append(pikepdf.Page(page))
        pdf.save(str(path), deterministic_id=True)
    finally:
        pdf.close()
    return path


@dataclass(frozen=True)
class Line:
    """One rendered text line as pdfminer lays it out on the page."""

    text: str
    y0: float
    y1: float
    baseline: float
    upright: bool
    sizes: frozenset[int]


def page_lines(path: Path) -> list[Line]:
    """Return the first page's text lines, top of the page first."""
    lines: list[Line] = []
    for page in extract_pages(str(path), page_numbers=[0]):
        stack = list(page)
        while stack:
            item = stack.pop()
            if isinstance(item, LTTextLine):
                chars = [c for c in item if isinstance(c, LTChar)]
                lines.append(
                    Line(
                        text=" ".join(item.get_text().split()),
                        y0=item.y0,
                        y1=item.y1,
                        baseline=chars[0].matrix[5],
                        upright=all(c.matrix[3] > 0 for c in chars),
                        sizes=frozenset(round(c.size) for c in chars),
                    )
                )
            elif hasattr(item, "__iter__"):
                stack.extend(item)
    return sorted(lines, key=lambda line: -line.baseline)
