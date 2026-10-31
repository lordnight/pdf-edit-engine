"""INV-B-13: reflow keeps an axis-aligned flip or scale of the text matrix.

A Y-flipped page (``cm 1 0 0 -1 0 h``) draws upright text through a negative-d
``Tm``, and a font-size-in-Tm run carries its size in ``Tm`` under ``Tf 1``.
Reflow must re-emit that orientation (upright 12pt glyphs whose lines run down
the page in reading order) instead of flattening it to an identity ``Tm``
(mirrored text) or refusing it as rotated (the v0.2.0 B.12 gate refused every
non-identity ``Tm``). Text whose baseline is rotated, sheared or reversed on
the page, through the ``Tm`` or the CTM, is still refused with
``rotated_text_unsupported`` (INV-B-8).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from pdf_edit_engine import find, replace
from tests._flipped_page_fixture import (
    FLIP_CTM,
    FLIP_NEGATIVE_D_TM,
    FLIP_UNIT_TM,
    IDENTITY_CTM,
    PAGE_FONT_SIZE,
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
    [FLIP_NEGATIVE_D_TM, FLIP_UNIT_TM, TM_SCALED],
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
    assert all(line.upright for line in lines), "reflowed glyphs must not be mirrored"
    assert {size for line in lines for size in line.sizes} == {round(PAGE_FONT_SIZE)}


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
