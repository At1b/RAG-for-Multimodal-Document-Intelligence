"""Live integration tests for Tesseract 5 OCR.

These tests execute the real Tesseract engine on the local system.
They are tagged with @pytest.mark.ocr_live and automatically skipped
if Tesseract is not installed on the host.
"""

import shutil
from pathlib import Path

import pymupdf
import pytest
from PIL import Image

from rag.ingestion.exceptions import (
    NoExtractableTextError,
    OcrUnavailableError,
)
from rag.ingestion.image_loader import load_image
from rag.ingestion.ocr.engine import TesseractEngine
from rag.ingestion.ocr.processor import OcrProcessor
from rag.ingestion.pdf_loader import load_pdf

pytestmark = pytest.mark.ocr_live

_WINDOWS_TESSERACT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")


def _is_tesseract_installed() -> bool:
    return bool(shutil.which("tesseract")) or _WINDOWS_TESSERACT.is_file()


@pytest.fixture(autouse=True)
def require_tesseract():
    if not _is_tesseract_installed():
        pytest.skip("Tesseract OCR is not installed on this machine")


def _create_scanned_page_pdf(path: Path, text: str) -> None:
    """Create a 1-page PDF where the text is rasterized into an image-only page."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_textbox(
        pymupdf.Rect(50, 50, 500, 400), text, fontsize=12, fontname="helv"
    )
    pix = page.get_pixmap(dpi=200, colorspace=pymupdf.csGRAY)
    doc.close()

    scan_doc = pymupdf.open()
    scan_page = scan_doc.new_page(width=595, height=842)
    scan_page.insert_image(scan_page.rect, stream=pix.tobytes("png"))
    scan_doc.save(str(path))
    scan_doc.close()


def test_live_ocr_scanned_pdf(tmp_path: Path):
    """Test that real Tesseract recognizes a scanned PDF page within measured CER."""
    pdf_path = tmp_path / "live_scan.pdf"
    sentence = (
        "The optical recognition engine extracts content accurately "
        "from document scans."
    )
    _create_scanned_page_pdf(pdf_path, sentence)

    engine = TesseractEngine()
    proc = OcrProcessor(engine=engine, dpi=200)
    doc = load_pdf(pdf_path, ocr=proc)

    assert len(doc.pages) == 1
    extracted = doc.pages[0].content
    assert "optical recognition engine" in extracted
    assert "document scans" in extracted
    assert doc.pages[0].metadata.get("extraction_method") == "ocr"
    assert "tesseract" in doc.metadata.get("ocr_engine", "").lower()
    assert doc.pages[0].metadata.get("ocr_mean_confidence") is not None
    assert doc.pages[0].metadata["ocr_mean_confidence"] > 90.0


def test_live_ocr_png_and_jpeg(tmp_path: Path):
    """Test that real Tesseract extracts text from direct PNG and JPEG files."""
    text = "Direct image file processing via optical character recognition."
    doc = pymupdf.open()
    page = doc.new_page(width=500, height=200)
    page.insert_textbox(
        pymupdf.Rect(30, 30, 470, 170), text, fontsize=12, fontname="helv"
    )
    pix = page.get_pixmap(dpi=200, colorspace=pymupdf.csGRAY)
    doc.close()

    png_path = tmp_path / "test.png"
    pix.save(str(png_path))

    jpg_path = tmp_path / "test.jpg"
    img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
    img.save(str(jpg_path), format="JPEG", quality=85)

    engine = TesseractEngine()
    proc = OcrProcessor(engine=engine, dpi=200)

    # Test PNG
    doc_png = load_image(png_path, ocr=proc)
    assert "Direct image file processing" in doc_png.pages[0].content
    assert doc_png.source_type == "image"
    assert doc_png.pages[0].metadata["ocr_mean_confidence"] > 90.0

    # Test JPEG
    doc_jpg = load_image(jpg_path, ocr=proc)
    assert "Direct image file processing" in doc_jpg.pages[0].content
    assert doc_jpg.source_type == "image"
    assert doc_jpg.pages[0].metadata["ocr_mean_confidence"] > 90.0


def test_live_ocr_mixed_pdf(tmp_path: Path):
    """Test mixed PDF: native page is untouched, scanned page is processed via OCR."""
    pdf_path = tmp_path / "mixed.pdf"
    doc = pymupdf.open()

    # Page 1: Native digital text
    p1 = doc.new_page(width=595, height=842)
    p1.insert_text((50, 80), "This is native digital text on page one.", fontsize=12)

    # Page 2: Scanned image
    p2_raw = pymupdf.open()
    p2_temp = p2_raw.new_page(width=595, height=842)
    p2_temp.insert_text(
        (50, 80), "This is scanned image text on page two.", fontsize=12
    )
    pix = p2_temp.get_pixmap(dpi=200, colorspace=pymupdf.csGRAY)
    p2_raw.close()

    p2 = doc.new_page(width=595, height=842)
    p2.insert_image(p2.rect, stream=pix.tobytes("png"))

    doc.save(str(pdf_path))
    doc.close()

    engine = TesseractEngine()
    proc = OcrProcessor(engine=engine, dpi=200)
    loaded = load_pdf(pdf_path, ocr=proc)

    assert len(loaded.pages) == 2
    # Page 1 native text exact
    assert "native digital text on page one" in loaded.pages[0].content
    assert loaded.pages[0].metadata.get("extraction_method") == "native_text"

    # Page 2 OCR text
    assert "scanned image text on page two" in loaded.pages[1].content
    assert loaded.pages[1].metadata.get("extraction_method") == "ocr"
    assert loaded.metadata.get("ocr_page_count") == 1


def test_live_ocr_invalid_language():
    """Test that an uninstalled/invalid language raises OcrUnavailableError."""
    engine = TesseractEngine()
    # 'xyz' is a valid 3-letter regex match but not installed
    with pytest.raises(
        OcrUnavailableError, match="language data missing|Failed loading language"
    ):
        engine.recognize(b"fake_png_data", language="xyz")


def test_live_ocr_timeout_handling(tmp_path: Path):
    """Test that a ridiculously short timeout triggers timeout handling."""
    pdf_path = tmp_path / "timeout_scan.pdf"
    _create_scanned_page_pdf(pdf_path, "A sentence that takes non-zero time to OCR.")

    engine = TesseractEngine()
    # 1 microsecond timeout will guarantee a timeout
    proc = OcrProcessor(engine=engine, timeout_seconds=0.000001)

    # In load_pdf, a single-page scanned document timing out results in
    # NoExtractableTextError
    with pytest.raises(
        NoExtractableTextError, match="(?i)no readable text could be extracted"
    ):
        load_pdf(pdf_path, ocr=proc)
