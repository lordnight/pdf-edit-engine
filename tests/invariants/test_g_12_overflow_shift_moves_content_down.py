"""INV-G-12: content below an overflowing reflow moves DOWN the page, clear of it.

When a longer replacement adds lines to a paragraph, every element below the
paragraph must move down the page by the added lines' height, keep its order
and not overlap the reflowed lines, whatever the CTM and text-matrix
orientation. The shift and its on-page clamp work in page space: the shift
primitive maps each operand through the CTM in effect at its operator, and the
clamp measures what is below by page-space extents of elements that draw
something. Only content wholly below the paragraph moves or limits the shift;
a background, frame, underlay or margin clip that reaches above it stays.

Regression guard — on the pre-fix engine a Y-flip CTM (``cm 1 0 0 -1 0 h``)
turned the shift around (``y - delta`` in flipped user space moves content UP,
into the paragraph), a positive-d ``Tm`` under the flip grew the paragraph
upward, a scaled CTM placed the reflowed lines at the wrong page position, and
any color operator or empty path counted as content at y=0, so the shift was
suppressed with "0pt available" and the paragraph overlapped what followed.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Literal

import pikepdf
import pytest
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTCurve, LTImage

from pdf_edit_engine import add_watermark, find, replace, replace_block, shift_content_below
from tests._flipped_page_fixture import (
    CHROME_SCALED_FLIP,
    FLIP_IDENTITY_TM,
    FLIP_NEGATIVE_D_TM,
    FLIP_POSITIVE_D_TM,
    FLIP_UNIT_TM,
    IDENTITY,
    PAGE_FONT_SIZE,
    PAGE_HEIGHT,
    SCALED_CTM,
    TM_SCALED,
    Orientation,
    build_page_pdf,
    page_lines,
)
from tests.harness.diff_render import render_page, requires_render

if TYPE_CHECKING:
    from pathlib import Path

    from pdf_edit_engine.models import EditResult
    from tests._flipped_page_fixture import Line

LONG_REPLACEMENT = (
    "100 units plus a replacement far too long to fit on the original line of the page"
)
PARAGRAPH = "Total: 100 units"
FOOTER = "Footer line stays"
HEADER = "Header line stays"
LINE_HEIGHT = PAGE_FONT_SIZE * 1.2
_TOL = 0.5

ORIENTATIONS = [
    IDENTITY,
    FLIP_NEGATIVE_D_TM,
    FLIP_UNIT_TM,
    FLIP_POSITIVE_D_TM,
    FLIP_IDENTITY_TM,
    TM_SCALED,
    SCALED_CTM,
    CHROME_SCALED_FLIP,
]


def _reflow(src: Path, out: Path) -> EditResult:
    matches = find(str(src), "100 units")
    assert matches, "fixture must contain the edit target"
    result = replace(str(src), matches[0], LONG_REPLACEMENT, str(out), reflow=True)
    assert result.success, f"reflow must succeed: {result!r}"
    assert result.fidelity_report.reflow_applied
    return result


def _footer(lines: list[Line]) -> Line:
    [footer] = [line for line in lines if line.text == FOOTER]
    return footer


def _paragraph_lines(lines: list[Line]) -> list[Line]:
    return [line for line in lines if line.text != FOOTER]


def _page(
    orientation: Orientation, tmp_path: Path, *, footer_y: float, prologue: bytes = b""
) -> Path:
    return build_page_pdf(
        tmp_path / "in.pdf",
        orientation.prologue()
        + prologue
        + orientation.run(72, 742, PARAGRAPH)
        + orientation.run(72, footer_y, FOOTER),
    )


@pytest.mark.parametrize("footer_y", [670.0, 700.0], ids=["72pt_below", "42pt_below"])
@pytest.mark.parametrize("orientation", ORIENTATIONS, ids=lambda o: o.name)
def test_footer_moves_down_by_the_added_lines_and_clears_the_paragraph(
    orientation: Orientation, footer_y: float, tmp_path: Path
) -> None:
    # 42pt is a separate block on a 12pt page (the paragraph break is 2.5 em
    # of PAGE-space font size); measured in Tf units a 24pt-Tf run under a
    # 0.5 CTM swallowed it into the paragraph.
    src = _page(orientation, tmp_path, footer_y=footer_y)
    before = _footer(page_lines(src))
    out = tmp_path / "out.pdf"

    _reflow(src, out)

    lines = page_lines(out)
    paragraph = _paragraph_lines(lines)
    footer = _footer(lines)
    added = len(paragraph) - 1
    assert added > 0, "the long replacement must add lines"
    assert " ".join(line.text for line in paragraph) == "Total: " + LONG_REPLACEMENT, (
        "reflowed lines must run down the page in reading order"
    )
    assert all(line.upright == orientation.upright for line in lines), (
        "reflow must keep the glyph orientation of the page"
    )
    assert before.baseline - footer.baseline == pytest.approx(added * LINE_HEIGHT, abs=_TOL), (
        f"footer must move DOWN by {added} x {LINE_HEIGHT}pt; "
        f"it went from y={before.baseline:.1f} to y={footer.baseline:.1f}"
    )
    assert lines[-1] == footer, "the footer must stay below every reflowed line"
    assert footer.y1 <= paragraph[-1].y0 + _TOL, (
        f"footer (top y={footer.y1:.1f}) overlaps the last reflowed line "
        f"(bottom y={paragraph[-1].y0:.1f})"
    )


@pytest.mark.parametrize(
    "prologue", [b"0 g\n", b"n\n"], ids=["color_operator", "empty_path"]
)
def test_elements_that_draw_nothing_do_not_suppress_the_shift(
    prologue: bytes, tmp_path: Path
) -> None:
    src = _page(IDENTITY, tmp_path, footer_y=690, prologue=prologue)
    out = tmp_path / "out.pdf"

    result = _reflow(src, out)

    kinds = {d.kind for d in result.fidelity_report.degradations}
    assert "overflow_shift_suppressed" not in kinds, (
        f"a page with room below must not report 0pt available; warnings={result.warnings!r}"
    )
    lines = page_lines(out)
    footer = _footer(lines)
    added = len(_paragraph_lines(lines)) - 1
    assert 690 - footer.baseline == pytest.approx(added * LINE_HEIGHT, abs=_TOL)


@pytest.mark.parametrize(
    ("footer_y", "kind", "moved"),
    [(60.0, "overflow_shift_clamped", 57.0), (3.0, "overflow_shift_suppressed", 0.0)],
    ids=["clamped", "suppressed"],
)
def test_page_bottom_clamp_is_page_space_under_a_flip(
    footer_y: float, kind: str, moved: float, tmp_path: Path
) -> None:
    paragraph_y = footer_y + 60
    src = build_page_pdf(
        tmp_path / "in.pdf",
        FLIP_NEGATIVE_D_TM.prologue()
        + FLIP_NEGATIVE_D_TM.run(72, paragraph_y, PARAGRAPH)
        + FLIP_NEGATIVE_D_TM.run(72, footer_y, FOOTER),
    )
    out = tmp_path / "out.pdf"

    result = _reflow(src, out)

    degradations = [d for d in result.fidelity_report.degradations if d.kind == kind]
    assert degradations, f"expected {kind}; got {result.fidelity_report.degradations!r}"
    if moved:
        assert f"clamped_to={moved:.1f}pt" in degradations[0].detail, (
            "the clamp must leave the footer's glyph box (baseline - 0.25em) on the page"
        )
    footer = _footer(page_lines(out))
    assert footer_y - footer.baseline == pytest.approx(moved, abs=_TOL), (
        f"footer must move down by {moved}pt, never up; "
        f"it went from y={footer_y} to y={footer.baseline:.1f}"
    )
    assert footer.y0 >= -_TOL, "the clamped footer must stay on the page"


def _flipped_drawing(tmp_path: Path) -> Path:
    """Header text above y=600 and text, a filled rect, a line and an image below it."""
    pdf = pikepdf.new()
    try:
        image = pdf.make_stream(
            b"\x00",
            Type=pikepdf.Name.XObject,
            Subtype=pikepdf.Name.Image,
            Width=1,
            Height=1,
            ColorSpace=pikepdf.Name.DeviceGray,
            BitsPerComponent=8,
        )
        helvetica = pdf.make_indirect(
            pikepdf.Dictionary(
                Type=pikepdf.Name.Font,
                Subtype=pikepdf.Name.Type1,
                BaseFont=pikepdf.Name.Helvetica,
                Encoding=pikepdf.Name.WinAnsiEncoding,
            )
        )
        content = (
            FLIP_NEGATIVE_D_TM.prologue()
            + FLIP_NEGATIVE_D_TM.run(72, 742, "Header above")
            + FLIP_NEGATIVE_D_TM.run(72, 500, "Body below")
            + b"72 392 200 20 re f\n"
            + b"72 422 m 272 422 l S\n"
            + b"q 50 0 0 -40 300 492 cm /Im0 Do Q\n"
        )
        page = pikepdf.Dictionary(
            Type=pikepdf.Name.Page,
            MediaBox=[0, 0, 595, 842],
            Resources=pikepdf.Dictionary(
                Font=pikepdf.Dictionary(F1=helvetica),
                XObject=pikepdf.Dictionary(Im0=image),
            ),
            Contents=pdf.make_stream(content),
        )
        pdf.pages.append(pikepdf.Page(page))
        path = tmp_path / "drawing.pdf"
        pdf.save(str(path))
    finally:
        pdf.close()
    return path


def _graphics_bottoms(path: Path) -> dict[str, float]:
    bottoms: dict[str, float] = {}
    for page in extract_pages(str(path)):
        stack = list(page)
        while stack:
            item = stack.pop()
            if isinstance(item, LTImage):
                bottoms["image"] = item.y0
            elif isinstance(item, LTCurve):
                bottoms["line" if item.height < 1 else "rect"] = item.y0
            elif hasattr(item, "__iter__"):
                stack.extend(item)
    return bottoms


def test_shift_primitive_moves_flipped_text_paths_and_images_down(tmp_path: Path) -> None:
    src = _flipped_drawing(tmp_path)
    out = tmp_path / "out.pdf"
    before_text = {line.text: line.baseline for line in page_lines(src)}
    before_graphics = _graphics_bottoms(src)
    assert set(before_graphics) == {"image", "rect", "line"}, before_graphics

    result = shift_content_below(str(src), 0, 600.0, 30.0, str(out))

    assert result.success
    after_text = {line.text: line.baseline for line in page_lines(out)}
    assert after_text["Header above"] == pytest.approx(before_text["Header above"])
    assert before_text["Body below"] - after_text["Body below"] == pytest.approx(30.0)
    after_graphics = _graphics_bottoms(out)
    for name, bottom in before_graphics.items():
        assert bottom - after_graphics[name] == pytest.approx(30.0), (
            f"{name} must move 30pt down the page; went from {bottom} to {after_graphics[name]}"
        )


def test_replace_block_clamp_ignores_elements_that_draw_nothing(tmp_path: Path) -> None:
    src = _page(IDENTITY, tmp_path, footer_y=690, prologue=b"0 g\n")
    out = tmp_path / "out.pdf"

    result = replace_block(
        str(src), 0, (70.0, 738.0, 200.0, 754.0), "Total: " + LONG_REPLACEMENT, str(out)
    )

    assert result.success, result
    kinds = {d.kind for d in result.fidelity_report.degradations}
    assert "overflow_shift_suppressed" not in kinds, (
        f"a page with room below must not report 0pt available; got kinds={kinds}"
    )
    assert _footer(page_lines(out)).baseline < 690 - _TOL, "the footer must move down"


def _content_ops(path: Path) -> list[tuple[str, list[float]]]:
    with pikepdf.open(str(path)) as pdf:
        return [
            (str(operator), [float(v) for v in operands if isinstance(v, (int, float, Decimal))])
            for operands, operator in pikepdf.parse_content_stream(pdf.pages[0])
        ]


def _rects(path: Path) -> list[list[float]]:
    return [operands for op, operands in _content_ops(path) if op == "re"]


def _line_has_ink(path: Path, line: Line) -> bool:
    image = render_page(path, scale=2.0).convert("L")
    top = round((PAGE_HEIGHT - line.y1) * 2)
    bottom = round((PAGE_HEIGHT - line.y0) * 2)
    band = image.crop((144, max(top, 0), 600, max(bottom, top + 1)))
    return band.getextrema()[0] < 128


def _assert_footer_moved_clear(src: Path, out: Path, result: EditResult, footer_y: float) -> None:
    kinds = {d.kind for d in result.fidelity_report.degradations}
    assert not kinds & {"overflow_shift_suppressed", "overflow_shift_clamped"}, (
        f"content spanning the paragraph must not limit the shift; warnings={result.warnings!r}"
    )
    lines = page_lines(out)
    paragraph = [line for line in _paragraph_lines(lines) if line.text != HEADER]
    footer = _footer(lines)
    added = len(paragraph) - 1
    assert added > 0, "the long replacement must add lines"
    assert footer_y - footer.baseline == pytest.approx(added * LINE_HEIGHT, abs=_TOL)
    assert footer.y1 <= paragraph[-1].y0 + _TOL, (
        f"footer (top y={footer.y1:.1f}) overlaps the last reflowed line "
        f"(bottom y={paragraph[-1].y0:.1f})"
    )


@pytest.mark.parametrize(
    ("orientation", "prologue"),
    [
        (IDENTITY, b"q 1 g 0 0 595 842 re f Q\n"),
        (FLIP_NEGATIVE_D_TM, b"q 1 g 0 0 595 842 re f Q\n"),
        (IDENTITY, b"q 0 G 20 20 555 802 re S Q\n"),
        (IDENTITY, b"q 0.85 g 66 737 200 18 re f Q\n"),
    ],
    ids=["background", "flipped_background", "page_frame", "shaded_row_behind_line"],
)
def test_elements_spanning_the_paragraph_stay_and_do_not_limit_the_shift(
    orientation: Orientation, prologue: bytes, tmp_path: Path
) -> None:
    src = _page(orientation, tmp_path, footer_y=690, prologue=prologue)
    out = tmp_path / "out.pdf"

    result = _reflow(src, out)

    _assert_footer_moved_clear(src, out, result, footer_y=690)
    assert _rects(out) == _rects(src), "a rectangle reaching above the paragraph must not move"


@requires_render
@pytest.mark.parametrize("orientation", [IDENTITY, FLIP_NEGATIVE_D_TM], ids=lambda o: o.name)
def test_a_margin_clip_stays_put_and_the_reflowed_lines_stay_visible(
    orientation: Orientation, tmp_path: Path
) -> None:
    # Chrome wraps the page body in a margin clip ("re W* n") that spans the
    # page: the shift must neither move it nor read its bottom edge as content.
    clip = b"q 36 36 523 770 re W* n\n"
    src = build_page_pdf(
        tmp_path / "in.pdf",
        orientation.prologue()
        + clip
        + orientation.run(72, 790, HEADER)
        + orientation.run(72, 742, PARAGRAPH)
        + orientation.run(72, 690, FOOTER)
        + b"Q\n",
    )
    out = tmp_path / "out.pdf"

    result = _reflow(src, out)

    _assert_footer_moved_clear(src, out, result, footer_y=690)
    assert _rects(out) == _rects(src), "the margin clip must not move"
    for line in page_lines(out):
        assert _line_has_ink(out, line), f"line {line.text!r} has no ink in the render"


def test_a_watermark_underlay_does_not_suppress_the_shift(tmp_path: Path) -> None:
    page = _page(IDENTITY, tmp_path, footer_y=690)
    mark = build_page_pdf(
        tmp_path / "mark.pdf", b"BT /F1 48 Tf 0.9 g 100 400 Td (DRAFT) Tj ET\n"
    )
    src = tmp_path / "watermarked.pdf"
    add_watermark(str(page), str(mark), str(src))
    out = tmp_path / "out.pdf"

    result = _reflow(src, out)

    _assert_footer_moved_clear(src, out, result, footer_y=690)
    underlay = [ops for ops in _content_ops(out) if ops[0] == "cm"]
    assert underlay == [ops for ops in _content_ops(src) if ops[0] == "cm"], (
        "the page-sized underlay must not move"
    )


def _replace_block_layout(
    orientation: Orientation, fit: Literal["none", "shrink"], tmp_path: Path
) -> list[tuple[str, float, frozenset[int], bool]]:
    src = _page(orientation, tmp_path, footer_y=690)
    out = tmp_path / f"{orientation.name}-{fit}.pdf"
    result = replace_block(
        str(src),
        0,
        (70.0, 738.0, 200.0, 754.0),
        "Total: " + LONG_REPLACEMENT,
        str(out),
        fit=fit,
    )
    assert result.success, result
    return [
        (line.text, round(line.baseline, 1), line.sizes, line.upright) for line in page_lines(out)
    ]


@pytest.mark.parametrize("fit", ["none", "shrink"])
@pytest.mark.parametrize(
    "orientation",
    [FLIP_NEGATIVE_D_TM, TM_SCALED, SCALED_CTM, CHROME_SCALED_FLIP],
    ids=lambda o: o.name,
)
def test_replace_block_lays_out_like_an_identity_page(
    orientation: Orientation, fit: Literal["none", "shrink"], tmp_path: Path
) -> None:
    expected = _replace_block_layout(IDENTITY, fit, tmp_path)

    assert _replace_block_layout(orientation, fit, tmp_path) == expected, (
        "replace_block must draw the new text at the page-space size and position, "
        "upright, and move the footer as on an identity page"
    )
