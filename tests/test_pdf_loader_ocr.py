"""Unit tests for PDF loader OCR integration (rag/ingestion/pdf_loader.py)."""

from __future__ import annotations

import pymupdf
import pytest

from rag.ingestion.exceptions import (
    NoExtractableTextError,
    OcrLimitExceededError,
    OcrPageError,
    OcrUnavailableError,
)
from rag.ingestion.ocr.engine import OcrPageResult
from rag.ingestion.ocr.processor import OcrProcessor
from rag.ingestion.pdf_loader import load_pdf


def _create_sample_png() -> bytes:
    """Create a minimal 100x100 PNG pixmap in memory."""
    pix = pymupdf.Pixmap(pymupdf.csGRAY, (0, 0, 100, 100), 0)
    pix.clear_with(200)
    return pix.tobytes("png")


def _build_test_pdf(tmp_path, page_types: list[str]) -> str:
    """Build a multi-page test PDF where each page is either 'native' or 'scan'.

    Args:
        tmp_path: Pytest temporary directory.
        page_types: List of strings ('native' or 'scan').
    """
    doc = pymupdf.open()
    png_bytes = _create_sample_png()

    for idx, ptype in enumerate(page_types):
        page = doc.new_page(width=500, height=700)
        if ptype == "native":
            page.insert_text(
                (50, 50),
                f"Page {idx + 1} Native embedded searchable text exceeding threshold.",
            )
        elif ptype == "scan":
            rect = pymupdf.Rect(50, 50, 450, 650)
            page.insert_image(rect, stream=png_bytes)
        elif ptype == "scan_with_footer":
            rect = pymupdf.Rect(50, 50, 450, 650)
            page.insert_image(rect, stream=png_bytes)
            page.insert_text((240, 680), "12")  # < 25 native chars

    pdf_path = tmp_path / "test_ocr.pdf"
    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)


class FakeOcrEngine:
    """Deterministic fake OCR engine for unit tests."""

    def __init__(self, responses: list[OcrPageResult] | None = None) -> None:
        self.responses = list(responses) if responses else []
        self.call_count = 0
        self.calls: list[bytes] = []

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
        self.calls.append(image_bytes)
        if self.responses:
            resp = self.responses.pop(0)
            if isinstance(resp, Exception):
                raise resp
            return resp
        return OcrPageResult(
            text=f"Fake OCR text {self.call_count}",
            mean_confidence=95.0,
        )


class TestPdfLoaderOcr:
    """Verify load_pdf with injected OCR processor."""

    def test_legacy_mode_when_ocr_is_none(self, tmp_path):
        pdf_path = _build_test_pdf(tmp_path, ["native", "scan"])
        doc = load_pdf(pdf_path, ocr=None)

        assert doc.source_type == "pdf"
        assert len(doc.pages) == 2
        # Page 1 has native text, page 2 has empty text (no OCR ran)
        assert "Native embedded" in doc.pages[0].content
        assert doc.pages[1].content == ""
        # Legacy metadata: no extraction_method or ocr_* doc keys
        assert "extraction_method" not in doc.pages[0].metadata
        assert "extraction_method" not in doc.pages[1].metadata
        assert "ocr_page_count" not in doc.metadata

    def test_scanned_multipage_pdf_all_pages_ocred(self, tmp_path):
        pdf_path = _build_test_pdf(tmp_path, ["scan", "scan", "scan"])
        fake_engine = FakeOcrEngine(
            [
                OcrPageResult("Recognized Page One", 92.0),
                OcrPageResult("Recognized Page Two", 94.0),
                OcrPageResult("Recognized Page Three", 96.0),
            ]
        )
        processor = OcrProcessor(engine=fake_engine, dpi=200)
        doc = load_pdf(pdf_path, ocr=processor)

        assert len(doc.pages) == 3
        assert fake_engine.call_count == 3
        for idx, page in enumerate(doc.pages):
            assert page.page_number == idx + 1
            assert page.metadata["extraction_method"] == "ocr"
            assert page.metadata["ocr_mean_confidence"] is not None

        assert doc.pages[0].content == "Recognized Page One"
        assert doc.pages[1].content == "Recognized Page Two"
        assert doc.pages[2].content == "Recognized Page Three"

        assert doc.metadata["ocr_page_count"] == 3
        assert doc.metadata["ocr_failed_pages"] == []
        assert doc.metadata["ocr_engine"] == "FakeOcrEngine 1.0.0"
        assert doc.metadata["ocr_dpi"] == 200

    def test_mixed_pdf_invokes_ocr_only_for_scanned_pages(self, tmp_path):
        pdf_path = _build_test_pdf(tmp_path, ["native", "scan", "native", "scan"])
        fake_engine = FakeOcrEngine(
            [
                OcrPageResult("OCR Page Two", 88.0),
                OcrPageResult("OCR Page Four", 91.0),
            ]
        )
        processor = OcrProcessor(engine=fake_engine, dpi=200)
        doc = load_pdf(pdf_path, ocr=processor)

        assert fake_engine.call_count == 2
        assert len(doc.pages) == 4

        # Pages 1 & 3: native
        assert "Page 1 Native" in doc.pages[0].content
        assert doc.pages[0].metadata["extraction_method"] == "native_text"
        assert "ocr_mean_confidence" not in doc.pages[0].metadata

        assert "Page 3 Native" in doc.pages[2].content
        assert doc.pages[2].metadata["extraction_method"] == "native_text"

        # Pages 2 & 4: OCR
        assert doc.pages[1].content == "OCR Page Two"
        assert doc.pages[1].metadata["extraction_method"] == "ocr"
        assert doc.pages[1].metadata["ocr_mean_confidence"] == 88.0

        assert doc.pages[3].content == "OCR Page Four"
        assert doc.pages[3].metadata["extraction_method"] == "ocr"

        assert doc.metadata["ocr_page_count"] == 2
        assert doc.metadata["ocr_failed_pages"] == []

    def test_partial_ocr_failure_falls_back_cleanly(self, tmp_path):
        pdf_path = _build_test_pdf(tmp_path, ["scan", "scan", "scan"])
        fake_engine = FakeOcrEngine(
            [
                OcrPageResult("Valid Page 1", 90.0),
                OcrPageError("Tesseract crashed on page 2"),
                OcrPageResult("Valid Page 3", 95.0),
            ]
        )
        processor = OcrProcessor(engine=fake_engine, dpi=200)
        doc = load_pdf(pdf_path, ocr=processor)

        assert len(doc.pages) == 3
        assert doc.pages[0].content == "Valid Page 1"
        assert doc.pages[1].content == ""  # failed page has empty native fallback
        assert doc.pages[1].metadata["extraction_method"] == "native_text"
        assert doc.pages[2].content == "Valid Page 3"

        assert doc.metadata["ocr_failed_pages"] == [2]
        assert doc.metadata["ocr_page_count"] == 3

    def test_failed_ocr_page_with_stray_native_chars_preserves_native(self, tmp_path):
        pdf_path = _build_test_pdf(tmp_path, ["scan_with_footer", "native"])
        fake_engine = FakeOcrEngine([OcrPageError("Timeout")])
        processor = OcrProcessor(engine=fake_engine, dpi=200)
        doc = load_pdf(pdf_path, ocr=processor)

        # Page 1 failed OCR, fell back to footer text "12"
        assert doc.pages[0].content.strip() == "12"
        assert doc.pages[0].metadata["extraction_method"] == "native_text"
        assert doc.metadata["ocr_failed_pages"] == [1]

    def test_total_ocr_failure_raises_no_extractable_text(self, tmp_path):
        pdf_path = _build_test_pdf(tmp_path, ["scan", "scan"])
        fake_engine = FakeOcrEngine(
            [
                OcrPageError("Failed 1"),
                OcrPageError("Failed 2"),
            ]
        )
        processor = OcrProcessor(engine=fake_engine, dpi=200)
        with pytest.raises(NoExtractableTextError, match="No readable text"):
            load_pdf(pdf_path, ocr=processor)

    def test_ocr_unavailable_raises_immediately(self, tmp_path):
        pdf_path = _build_test_pdf(tmp_path, ["scan", "scan"])
        fake_engine = FakeOcrEngine(
            [
                OcrUnavailableError("Tesseract not installed"),
            ]
        )
        processor = OcrProcessor(engine=fake_engine, dpi=200)
        with pytest.raises(OcrUnavailableError, match="not installed"):
            load_pdf(pdf_path, ocr=processor)
        assert fake_engine.call_count == 1

    def test_ocr_max_pages_exceeded_raises_before_ocr(self, tmp_path):
        pdf_path = _build_test_pdf(tmp_path, ["scan", "scan", "scan"])
        fake_engine = FakeOcrEngine()
        processor = OcrProcessor(engine=fake_engine, max_pages=2)
        with pytest.raises(
            OcrLimitExceededError, match="exceeding the maximum allowed limit"
        ):
            load_pdf(pdf_path, ocr=processor)
        assert fake_engine.call_count == 0

    def test_custom_ocr_min_native_chars_threshold_propagated(self, tmp_path):
        """Verify non-default ocr_min_native_chars alters OCR trigger decision."""
        # PDF page with large image (>30% coverage) and exactly 35 alphanumeric chars
        doc = pymupdf.open()
        page = doc.new_page(width=500, height=700)
        rect = pymupdf.Rect(50, 50, 450, 650)
        page.insert_image(rect, stream=_create_sample_png())
        text = "Intermediate length text with 35 alnum"
        page.insert_text((50, 670), text)
        pdf_path = tmp_path / "threshold_test.pdf"
        doc.save(str(pdf_path))
        doc.close()

        # Case A: Default threshold (25). 35 >= 25, so OCR should NOT trigger.
        fake_engine_a = FakeOcrEngine()
        proc_default = OcrProcessor(engine=fake_engine_a, ocr_min_native_chars=25)
        doc_a = load_pdf(pdf_path, ocr=proc_default)
        assert fake_engine_a.call_count == 0
        assert doc_a.pages[0].metadata["extraction_method"] == "native_text"
        assert text in doc_a.pages[0].content

        # Case B: Elevated threshold (50). 35 < 50, so OCR SHOULD trigger.
        fake_engine_b = FakeOcrEngine(
            [OcrPageResult("Recognized via custom threshold", 95.0)]
        )
        proc_elevated = OcrProcessor(engine=fake_engine_b, ocr_min_native_chars=50)
        doc_b = load_pdf(pdf_path, ocr=proc_elevated)
        assert fake_engine_b.call_count == 1
        assert doc_b.pages[0].metadata["extraction_method"] == "ocr"
        assert doc_b.pages[0].content == "Recognized via custom threshold"

    def test_mixed_pdf_with_native_and_empty_ocr_page_preserves_partial_success(
        self, tmp_path
    ):
        """Mixed PDF with native text on page 1 and empty OCR result on page 2."""
        pdf_path = _build_test_pdf(tmp_path, ["native", "scan"])
        # Page 2 OCR returns empty / non-alphanumeric text
        empty_res = OcrPageResult(text="   \n\t---\n  ", mean_confidence=12.0)
        fake_engine = FakeOcrEngine([empty_res])
        processor = OcrProcessor(engine=fake_engine, dpi=200)

        doc = load_pdf(pdf_path, ocr=processor)
        assert doc.source_type == "pdf"
        assert len(doc.pages) == 2
        # Page 1 has readable native text
        assert "Native embedded" in doc.pages[0].content
        assert doc.pages[0].metadata["extraction_method"] == "native_text"
        # Page 2 OCR failed due to no alphanumeric text, fell back to native (empty)
        assert doc.pages[1].content == ""
        assert doc.pages[1].metadata["extraction_method"] == "native_text"
        # Metadata records failure on page 2 (1-indexed)
        assert doc.metadata["ocr_page_count"] == 1
        assert doc.metadata["ocr_failed_pages"] == [2]

    def test_scanned_pdf_all_empty_ocr_raises_no_extractable_text(self, tmp_path):
        """Scanned PDF where all OCR pages yield non-alphanumeric output."""
        pdf_path = _build_test_pdf(tmp_path, ["scan", "scan"])
        fake_engine = FakeOcrEngine(
            [
                OcrPageResult(text="", mean_confidence=0.0),
                OcrPageResult(text="  ...   \n  ", mean_confidence=10.0),
            ]
        )
        processor = OcrProcessor(engine=fake_engine, dpi=200)

        with pytest.raises(NoExtractableTextError, match="No readable text"):
            load_pdf(pdf_path, ocr=processor)
