"""INV-B-13: reflow keeps an axis-aligned flip or scale of the text matrix.

A Y-flipped page (``cm 1 0 0 -1 0 h``) draws upright text through a negative-d
``Tm``, and a font-size-in-Tm run carries its size in ``Tm`` under ``Tf 1``.
Reflow must re-emit that orientation (12pt glyphs, upright or mirrored as in
the source, whose lines run down the page in reading order from the edited
line's position) instead of flattening it to an identity ``Tm`` or refusing it
as rotated (the v0.2.0 B.12 gate refused every non-identity ``Tm``), under any
axis-aligned CTM: flipped, scaled or identity. Text whose baseline is rotated,
sheared or reversed on the page, through the ``Tm`` or the CTM, is still
refused with ``rotated_text_unsupported`` (INV-B-8).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from pdf_edit_engine import find, replace
from pdf_edit_engine.reflow import _low_confidence_diagnostics, detect_paragraphs
from tests._flipped_page_fixture import (
    FLIP_CTM,
    FLIP_IDENTITY_TM,
    FLIP_NEGATIVE_D_TM,
    FLIP_POSITIVE_D_TM,
    FLIP_UNIT_TM,
    IDENTITY,
    IDENTITY_CTM,
    PAGE_FONT_SIZE,
    PAGE_WIDTH,
    SCALED_CTM,
    TM_SCALED,
    Orientation,
    build_page_pdf,
    page_lines,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pdf_edit_engine.models import EditResult

LONG_REPLACEMENT = (
    "100 units plus a replacement far too long to fit on the original line of the page"
)
_FIRST_BASELINE = 742.0


def _reflow(src: Path, out: Path) -> EditResult:
    matches = find(str(src), "100 units")
    assert matches, "fixture must contain the edit target"
    return replace(str(src), matches[0], LONG_REPLACEMENT, str(out), reflow=True)


@pytest.mark.parametrize(
    "orientation",
    [
        FLIP_NEGATIVE_D_TM,
        FLIP_UNIT_TM,
        TM_SCALED,
        FLIP_POSITIVE_D_TM,
        FLIP_IDENTITY_TM,
        SCALED_CTM,
    ],
    ids=lambda o: o.name,
)
def test_reflow_keeps_orientation_and_reading_order(
    orientation: Orientation, tmp_path: Path
) -> None:
    src = build_page_pdf(
        tmp_path / "in.pdf",
        orientation.prologue() + orientation.run(72, _FIRST_BASELINE, "Total: 100 units"),
    )
    out = tmp_path / "out.pdf"

    result = _reflow(src, out)

    assert result.success, f"{orientation.name} reflow must succeed: {result!r}"
    assert result.fidelity_report.reflow_applied
    lines = page_lines(out)
    assert len(lines) > 1, "the long replacement must wrap onto more than one line"
    assert " ".join(line.text for line in lines) == "Total: " + LONG_REPLACEMENT, (
        "reflowed lines must run down the page in reading order"
    )
    assert lines[0].baseline == pytest.approx(_FIRST_BASELINE), (
        "the first reflowed line must stay where the edited line was"
    )
    assert all(line.upright == orientation.upright for line in lines), (
        "reflowed glyphs must keep the source orientation (upright stays upright, "
        "a mirrored run stays mirrored)"
    )
    assert {size for line in lines for size in line.sizes} == {round(PAGE_FONT_SIZE)}


_WORDS = "Total: 100 units shipped to the main site office this week"


def _word_by_word_line(orientation: Orientation) -> bytes:
    """One line drawn a word per run, spaced by the engine's 0.6 em default glyph width."""
    content = orientation.prologue()
    x = 72.0
    for word in _WORDS.split():
        content += orientation.run(x, _FIRST_BASELINE, word)
        x += (len(word) + 1) * 0.6 * PAGE_FONT_SIZE
    return content


@pytest.mark.parametrize(
    "orientation", [FLIP_NEGATIVE_D_TM, TM_SCALED, SCALED_CTM], ids=lambda o: o.name
)
def test_paragraph_metrics_are_page_space_whatever_the_orientation(
    orientation: Orientation, tmp_path: Path
) -> None:
    [expected] = detect_paragraphs(
        str(build_page_pdf(tmp_path / "identity.pdf", _word_by_word_line(IDENTITY)))
    )
    [paragraph] = detect_paragraphs(
        str(build_page_pdf(tmp_path / "in.pdf", _word_by_word_line(orientation)))
    )

    for name in ("left_margin", "first_line_y", "paragraph_width", "line_height"):
        assert getattr(paragraph, name) == pytest.approx(getattr(expected, name)), (
            f"{name} must be the page-space value an identity page gives"
        )
    assert _low_confidence_diagnostics(paragraph, PAGE_WIDTH) == pytest.approx(
        _low_confidence_diagnostics(expected, PAGE_WIDTH)
    ), "the S5 table-cluster signal must read the same page-space widths"


_COS5 = math.cos(math.radians(5))
_SIN5 = math.sin(math.radians(5))

_REFUSED_ORIENTATIONS = [
    Orientation("tm_rotated_90_under_flip", FLIP_CTM, 1.0, (0.0, 12.0, 12.0, 0.0)),
    Orientation("ctm_rotated_5_degrees", (_COS5, _SIN5, -_SIN5, _COS5, 72.0, 700.0), 12.0, None),
    Orientation("tm_mirrored_baseline", IDENTITY_CTM, 1.0, (-12.0, 0.0, 0.0, 12.0)),
    Orientation("tm_rotated_180", IDENTITY_CTM, 1.0, (-12.0, 0.0, 0.0, -12.0)),
]


@pytest.mark.parametrize("orientation", _REFUSED_ORIENTATIONS, ids=lambda o: o.name)
def test_rotated_or_reversed_baseline_is_still_refused(
    orientation: Orientation, tmp_path: Path
) -> None:
    src = build_page_pdf(
        tmp_path / "in.pdf",
        orientation.prologue() + orientation.run(300, 700, "Total: 100 units"),
    )
    out = tmp_path / "out.pdf"

    result = _reflow(src, out)

    assert result.success is False, f"reflow on a rotated baseline must refuse: {result!r}"
    kinds = {d.kind for d in result.fidelity_report.degradations}
    assert "rotated_text_unsupported" in kinds, f"got kinds={kinds}"
    assert not out.exists(), "a refused reflow must not write output"
