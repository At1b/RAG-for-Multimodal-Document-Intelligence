"""Unit tests for scanned page detection (rag/ingestion/ocr/page_analysis.py)."""

from __future__ import annotations

from unittest.mock import patch

import pymupdf
import pytest

from rag.ingestion.ocr.page_analysis import calculate_image_coverage, needs_ocr


def _create_sample_png_bytes(width: int = 100, height: int = 100) -> bytes:
    """Create a minimal PNG pixmap in memory using PyMuPDF."""
    pix = pymupdf.Pixmap(pymupdf.csGRAY, (0, 0, width, height), 0)
    pix.clear_with(200)
    png_bytes = pix.tobytes("png")
    return png_bytes


class TestPageAnalysis:
    """Verify needs_ocr decision matrix on generated PDF documents."""

    @pytest.fixture
    def png_bytes(self) -> bytes:
        return _create_sample_png_bytes(200, 200)

    def test_text_page_does_not_need_ocr(self):
        doc = pymupdf.open()
        page = doc.new_page(width=500, height=700)
        page.insert_text(
            (50, 50),
            "This is a standard text page with plenty of embedded native characters "
            "that clearly exceed any scanned page threshold.",
        )
        assert needs_ocr(page, min_native_chars=25) is False
        doc.close()

    def test_blank_page_does_not_need_ocr(self):
        doc = pymupdf.open()
        page = doc.new_page(width=500, height=700)
        assert needs_ocr(page, min_native_chars=25) is False
        doc.close()

    def test_sparse_page_does_not_need_ocr(self):
        doc = pymupdf.open()
        page = doc.new_page(width=500, height=700)
        page.insert_text((50, 50), "This page intentionally left blank.")
        assert needs_ocr(page, min_native_chars=25) is False
        doc.close()

    def test_full_page_image_needs_ocr(self, png_bytes):
        doc = pymupdf.open()
        page = doc.new_page(width=500, height=700)
        # Cover almost full page (400x600 in 500x700 = ~68% coverage >= 30%)
        rect = pymupdf.Rect(50, 50, 450, 650)
        page.insert_image(rect, stream=png_bytes)
        assert needs_ocr(page, min_native_chars=25) is True
        doc.close()

    def test_full_page_image_with_short_footer_needs_ocr(self, png_bytes):
        doc = pymupdf.open()
        page = doc.new_page(width=500, height=700)
        rect = pymupdf.Rect(50, 50, 450, 650)
        page.insert_image(rect, stream=png_bytes)
        # Short footer with < 25 alphanumeric chars
        page.insert_text((240, 680), "Page 12")
        assert needs_ocr(page, min_native_chars=25) is True
        doc.close()

    def test_image_with_searchable_ocr_text_layer_does_not_need_ocr(self, png_bytes):
        doc = pymupdf.open()
        page = doc.new_page(width=500, height=700)
        rect = pymupdf.Rect(50, 50, 450, 650)
        page.insert_image(rect, stream=png_bytes)
        # Pre-existing OCR text layer with >= 25 alphanumeric chars
        page.insert_text(
            (60, 80),
            "This scanned document already has an embedded hidden text layer "
            "with full character recognition.",
        )
        assert needs_ocr(page, min_native_chars=25) is False
        doc.close()

    def test_small_logo_image_does_not_need_ocr(self, png_bytes):
        doc = pymupdf.open()
        page = doc.new_page(width=500, height=700)
        # Small logo: 50x50 in 500x700 = 2500 / 350000 = ~0.7% coverage (< 30%)
        rect = pymupdf.Rect(20, 20, 70, 70)
        page.insert_image(rect, stream=png_bytes)
        assert needs_ocr(page, min_native_chars=25) is False
        doc.close()

    def test_calculate_image_coverage_deduplicates_duplicate_xrefs(self, png_bytes):
        """Verify duplicate xrefs in get_images() do not inflate coverage."""
        doc = pymupdf.open()
        page = doc.new_page(width=500, height=500)
        # 100x100 image in 500x500 page = 10,000 / 250,000 = 0.04 (4% coverage)
        rect = pymupdf.Rect(50, 50, 150, 150)
        page.insert_image(rect, stream=png_bytes)

        # Baseline single xref coverage
        base_cov = calculate_image_coverage(page)
        assert abs(base_cov - 0.04) < 0.001

        # Simulate PyMuPDF returning duplicate xref tuples for the same image
        raw_images = page.get_images(full=True)
        assert len(raw_images) >= 1
        duplicate_images = raw_images * 5

        with patch.object(page, "get_images", return_value=duplicate_images):
            deduped_cov = calculate_image_coverage(page)
            # Coverage must remain 0.04, not 5 * 0.04 = 0.20
            assert abs(deduped_cov - 0.04) < 0.001
        doc.close()
