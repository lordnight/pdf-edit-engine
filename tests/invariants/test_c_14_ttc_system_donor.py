"""INV-C-14: a TrueType Collection (.ttc) system donor injects from the font's face.

macOS ships most system fonts (Helvetica, Times, ...) as .ttc files, and
fontTools refuses to open a collection without ``fontNumber``. Every Tier 1.5
extension whose donor resolved to a .ttc therefore failed with
``font_extension_failed``. The glyf (Identity-H) and CID-keyed CFF injection
paths must read the collection face whose PostScript name is the font's
(Helvetica.ttc face 1 for Helvetica-Bold), and face 0 when no face matches.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import pytest
from fontTools.ttLib import TTFont
from fontTools.ttLib.ttCollection import TTCollection

from pdf_edit_engine._pathutil import open_pdf
from pdf_edit_engine.encoding import FontResolverCache
from pdf_edit_engine.fonts import extend_subset
from pdf_edit_engine.widths import GlyphWidthCache
from tests._identity_h_fixture import (
    _build_identity_h_pdf,
    _find_ttf_for_cidfont,
    _no_ttf,
    _ttfont_open,
)
from tests.corpus_builders import build_cff_cid_missing_glyph_pdf, build_cff_donor_bytes

if TYPE_CHECKING:
    from pathlib import Path

    import pikepdf


def _write_collection(face: TTFont, path: Path) -> str:
    return _write_collection_of([face], path)


def _write_collection_of(faces: list[TTFont], path: Path) -> str:
    collection = TTCollection()
    collection.fonts = faces
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


def _rename(face: TTFont, postscript_name: str) -> TTFont:
    for record in face["name"].names:
        if record.nameID == 6:
            record.string = postscript_name
    return face


def _widen_every_glyph(face: TTFont, extra: int) -> TTFont:
    hmtx = face["hmtx"]
    for glyph_name, (advance, lsb) in list(hmtx.metrics.items()):
        hmtx.metrics[glyph_name] = (advance + extra, lsb)
    return face


def _font_name(pdf_path: Path) -> str:
    pdf = open_pdf(str(pdf_path))
    try:
        font = pdf.pages[0].Resources.Font.F1
        descriptor = font.DescendantFonts[0].FontDescriptor
        return str(descriptor.FontName).lstrip("/").split("+")[-1]
    finally:
        pdf.close()


def _injected_width(pdf: pikepdf.Pdf, char: str) -> float:
    page = pdf.pages[0]
    encoded = FontResolverCache().get_resolver(page, "F1").encode(char)
    return GlyphWidthCache().get_width(page, "F1", int.from_bytes(encoded, "big"))


def _advance_per_mille(face: TTFont, char: str) -> float:
    glyph_name = face.getBestCmap()[ord(char)]
    return face["hmtx"][glyph_name][0] * 1000.0 / face["head"].unitsPerEm


@_no_ttf
def test_glyf_injection_reads_the_ttc_face_named_by_the_font(tmp_path: Path) -> None:
    src = tmp_path / "in.pdf"
    assert _build_identity_h_pdf(src, title_text="Acme", title_pattern="single_tj", extra_corpus="")
    ttf = _find_ttf_for_cidfont()
    assert ttf is not None
    target = _rename(_ttfont_open(ttf), _font_name(src))
    decoy = _widen_every_glyph(_rename(_ttfont_open(ttf), "DecoyFace-Regular"), 300)
    expected = _advance_per_mille(target, "Z")
    donor = _write_collection_of([decoy, target], tmp_path / "family.ttc")

    pdf = open_pdf(str(src))
    try:
        tier = extend_subset(pdf, pdf.pages[0], "F1", "Z", full_font_path=donor)

        assert tier == "full_extension"
        assert _injected_width(pdf, "Z") == pytest.approx(expected, abs=1.0), (
            "the glyph must come from the face named like the embedded font, not face 0"
        )
    finally:
        pdf.close()


def test_cff_injection_reads_the_ttc_face_named_by_the_font(tmp_path: Path) -> None:
    src = tmp_path / "cff_cid.pdf"
    src.write_bytes(build_cff_cid_missing_glyph_pdf())
    target = _rename(TTFont(io.BytesIO(build_cff_donor_bytes(("C",)))), _font_name(src))
    decoy = _widen_every_glyph(
        _rename(TTFont(io.BytesIO(build_cff_donor_bytes(("C",)))), "DecoyFace-Regular"), 300
    )
    expected = _advance_per_mille(target, "C")
    donor = _write_collection_of([decoy, target], tmp_path / "family.ttc")

    pdf = open_pdf(str(src))
    try:
        tier = extend_subset(pdf, pdf.pages[0], "F1", "C", full_font_path=donor)

        assert tier == "full_extension"
        assert _injected_width(pdf, "C") == pytest.approx(expected, abs=1.0), (
            "the glyph must come from the face named like the embedded font, not face 0"
        )
    finally:
        pdf.close()
