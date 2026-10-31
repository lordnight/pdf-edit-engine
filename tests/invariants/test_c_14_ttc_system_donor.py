"""INV-C-14: a TrueType Collection (.ttc) system donor injects from its first face.

macOS ships most system fonts (Helvetica, Times, ...) as .ttc files, and
fontTools refuses to open a collection without ``fontNumber``. Every Tier 1.5
extension whose donor resolved to a .ttc therefore failed with
``font_extension_failed``. The glyf (Identity-H) and CID-keyed CFF injection
paths must read face 0 of a collection donor instead.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

from fontTools.ttLib import TTFont
from fontTools.ttLib.ttCollection import TTCollection

from pdf_edit_engine._pathutil import open_pdf
from pdf_edit_engine.encoding import FontResolverCache
from pdf_edit_engine.fonts import extend_subset
from tests._identity_h_fixture import (
    _build_identity_h_pdf,
    _find_ttf_for_cidfont,
    _no_ttf,
    _ttfont_open,
)
from tests.corpus_builders import build_cff_cid_missing_glyph_pdf, build_cff_donor_bytes

if TYPE_CHECKING:
    from pathlib import Path


def _write_collection(face: TTFont, path: Path) -> str:
    collection = TTCollection()
    collection.fonts = [face]
    collection.save(str(path))
    return str(path)


@_no_ttf
def test_glyf_injection_reads_face_zero_of_a_ttc_donor(tmp_path: Path) -> None:
    src = tmp_path / "in.pdf"
    assert _build_identity_h_pdf(src, title_text="Acme", title_pattern="single_tj", extra_corpus="")
    ttf = _find_ttf_for_cidfont()
    assert ttf is not None
    donor = _write_collection(_ttfont_open(ttf), tmp_path / "donor.ttc")

    pdf = open_pdf(str(src))
    try:
        page = pdf.pages[0]
        assert not FontResolverCache().get_resolver(page, "F1").can_encode("Z")[0]

        tier = extend_subset(pdf, page, "F1", "Z", full_font_path=donor)

        assert tier == "full_extension"
        assert FontResolverCache().get_resolver(page, "F1").can_encode("Z")[0], (
            "the glyph injected from the .ttc donor must be encodable"
        )
    finally:
        pdf.close()


def test_cff_injection_reads_face_zero_of_a_ttc_donor(tmp_path: Path) -> None:
    src = tmp_path / "cff_cid.pdf"
    src.write_bytes(build_cff_cid_missing_glyph_pdf())
    donor = _write_collection(
        TTFont(io.BytesIO(build_cff_donor_bytes(("C",)))), tmp_path / "donor.ttc"
    )

    pdf = open_pdf(str(src))
    try:
        page = pdf.pages[0]

        tier = extend_subset(pdf, page, "F1", "C", full_font_path=donor)

        assert tier == "full_extension"
        assert FontResolverCache().get_resolver(page, "F1").can_encode("C")[0], (
            "the glyph injected from the .ttc donor must be encodable"
        )
    finally:
        pdf.close()
