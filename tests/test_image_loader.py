"""Unit tests for image document loader (rag/ingestion/image_loader.py)."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from rag.ingestion.exceptions import (
    InvalidDocumentError,
    NoExtractableTextError,
    OcrLimitExceededError,
    OcrPageError,
    OcrUnavailableError,
    UnsupportedFormatError,
)
from rag.ingestion.image_loader import load_image
from rag.ingestion.ocr.engine import OcrPageResult
from rag.ingestion.ocr.processor import OcrProcessor


class FakeOcrEngine:
    """Fake OCR engine for image testing."""

    def __init__(self, response: OcrPageResult | Exception | None = None) -> None:
        self.response = response or OcrPageResult("Recognized Image Text", 94.5)
        self.call_count = 0
        self.last_input: bytes | None = None

    @property
    def version(self) -> str:
        return "FakeOcrEngine 1.0.0"

    def recognize(
        self,
        image_bytes: bytes,
        *,
        language: str = "eng",
        timeout: float = 60.0,
    ) -> OcrPageResult:
        self.call_count += 1
        self.last_input = image_bytes
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _create_png_file(path: Path, width: int = 200, height: int = 100) -> Path:
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    img.save(path, format="PNG")
    return path


def _create_jpeg_file(path: Path, width: int = 200, height: int = 100) -> Path:
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    img.save(path, format="JPEG")
    return path


class TestImageLoader:
    """Verify load_image functionality and error cases."""

    def test_load_png_success(self, tmp_path):
        png_path = _create_png_file(tmp_path / "invoice.png", 300, 150)
        engine = FakeOcrEngine(OcrPageResult("Invoice #12345 Total $500", 96.0))
        processor = OcrProcessor(engine=engine, dpi=300)

        doc = load_image(png_path, document_name="custom_invoice.png", ocr=processor)

        assert doc.source_type == "image"
        assert doc.document_name == "custom_invoice.png"
        assert len(doc.pages) == 1
        page = doc.pages[0]
        assert page.page_number == 1
        assert page.content == "Invoice #12345 Total $500"
        assert page.metadata["extraction_method"] == "ocr"
        assert page.metadata["ocr_mean_confidence"] == 96.0
        assert page.metadata["char_count"] == len("Invoice #12345 Total $500")

        assert doc.metadata["image_format"] == "png"
        assert doc.metadata["image_width"] == 300
        assert doc.metadata["image_height"] == 150
        assert doc.metadata["total_pages"] == 1
        assert doc.metadata["ocr_page_count"] == 1
        assert doc.metadata["ocr_failed_pages"] == []
        assert doc.metadata["ocr_engine"] == "FakeOcrEngine 1.0.0"

    def test_load_jpeg_success(self, tmp_path):
        jpg_path = _create_jpeg_file(tmp_path / "receipt.jpg", 250, 120)
        engine = FakeOcrEngine(OcrPageResult("Receipt Item 1 $20", 90.0))
        processor = OcrProcessor(engine=engine)

        doc = load_image(jpg_path, ocr=processor)
        assert doc.source_type == "image"
        assert doc.document_name == "receipt.jpg"
        assert doc.metadata["image_format"] == "jpeg"
        assert doc.metadata["image_width"] == 250
        assert doc.metadata["image_height"] == 120
        assert doc.pages[0].content == "Receipt Item 1 $20"

    def test_ocr_disabled_raises_ocr_unavailable(self, tmp_path):
        png_path = _create_png_file(tmp_path / "test.png")
        with pytest.raises(OcrUnavailableError, match="OCR is disabled"):
            load_image(png_path, ocr=None)

    def test_unsupported_image_format_raises(self, tmp_path):
        gif_path = tmp_path / "animation.gif"
        img = Image.new("RGB", (50, 50), color=(100, 100, 100))
        img.save(gif_path, format="GIF")

        processor = OcrProcessor(engine=FakeOcrEngine())
        with pytest.raises(UnsupportedFormatError, match="Unsupported image format"):
            load_image(gif_path, ocr=processor)

    def test_corrupt_image_raises_invalid_document_error(self, tmp_path):
        corrupt_path = tmp_path / "corrupt.png"
        # Write valid PNG header but corrupt body
        corrupt_path.write_bytes(b"\x89PNG\r\n\x1a\nGARBAGE_BYTES_TRUNCATED")

        processor = OcrProcessor(engine=FakeOcrEngine())
        with pytest.raises(InvalidDocumentError, match="Failed to read image"):
            load_image(corrupt_path, ocr=processor)

    def test_image_exceeds_max_pixels_raises(self, tmp_path):
        png_path = _create_png_file(tmp_path / "large.png", 200, 200)
        # 200x200 = 40,000 pixels. Set max to 10,000
        processor = OcrProcessor(engine=FakeOcrEngine(), max_image_pixels=10_000)

        with pytest.raises(OcrLimitExceededError, match="exceeds? configured maximum"):
            load_image(png_path, ocr=processor)

    def test_empty_ocr_text_raises_no_extractable_text(self, tmp_path):
        png_path = _create_png_file(tmp_path / "empty.png")
        engine = FakeOcrEngine(OcrPageResult("", None))
        processor = OcrProcessor(engine=engine)

        with pytest.raises(NoExtractableTextError, match="No readable text"):
            load_image(png_path, ocr=processor)

    def test_engine_error_raises_no_extractable_text(self, tmp_path):
        png_path = _create_png_file(tmp_path / "error.png")
        engine = FakeOcrEngine(OcrPageError("Tesseract timed out"))
        processor = OcrProcessor(engine=engine)

        with pytest.raises(NoExtractableTextError, match="OCR processing failed"):
            load_image(png_path, ocr=processor)

    def test_exif_transposition_applied(self, tmp_path):
        # Create a JPEG with EXIF orientation 6 (90 degrees CCW)
        jpg_path = tmp_path / "rotated.jpg"
        img = Image.new("RGB", (100, 200), color=(255, 255, 255))
        exif_dict = img.getexif()
        exif_dict[0x0112] = 6  # Orientation tag
        img.save(jpg_path, format="JPEG", exif=exif_dict)

        engine = FakeOcrEngine(OcrPageResult("Rotated Text", 95.0))
        processor = OcrProcessor(engine=engine)

        doc = load_image(jpg_path, ocr=processor)
        assert doc.source_type == "image"
        # Verify bytes sent to OCR is valid PNG
        assert engine.last_input is not None
        ocr_img = Image.open(io.BytesIO(engine.last_input))
        # After transpose of 100x200 with orientation 6, dimensions become 200x100
        assert ocr_img.size == (200, 100)
        # Verify document metadata captures the transposed dimensions
        assert doc.metadata["image_width"] == 200
        assert doc.metadata["image_height"] == 100
