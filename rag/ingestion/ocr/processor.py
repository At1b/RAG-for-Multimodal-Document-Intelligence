"""OCR processor coordinating rendering, resource bounds, and engine execution.

Enforces:
- Pixel limits calculated prior to raster rendering (prevents decompression bomb OOM).
- Prompt pixmap cleanup to minimize working set memory.
- Per-page execution timeout delegation to the engine.
- Single extraction point for PDF pages and raw image bytes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pymupdf

from rag.ingestion.exceptions import OcrPageError
from rag.ingestion.ocr.engine import OcrEngine, OcrPageResult, TesseractEngine

if TYPE_CHECKING:
    import pymupdf

logger = logging.getLogger(__name__)


class OcrProcessor:
    """Orchestrates image rasterization and OCR recognition for pages and images."""

    def __init__(
        self,
        engine: OcrEngine | None = None,
        *,
        language: str = "eng",
        dpi: int = 300,
        ocr_min_native_chars: int = 25,
        timeout_seconds: float = 60.0,
        max_image_pixels: int = 40_000_000,
        max_pages: int = 50,
    ) -> None:
        self.engine: OcrEngine = engine or TesseractEngine()
        self.language = language
        self.dpi = dpi
        self.ocr_min_native_chars = ocr_min_native_chars
        self.timeout_seconds = timeout_seconds
        self.max_image_pixels = max_image_pixels
        self.max_pages = max_pages
        self._engine_version: str | None = None

    @property
    def engine_version(self) -> str:
        """Return engine name and version, querying lazily once."""
        if self._engine_version is None:
            self._engine_version = self.engine.version
        return self._engine_version

    def ocr_pdf_page(self, page: pymupdf.Page) -> OcrPageResult:
        """Render a PDF page to grayscale PNG and extract text via OCR.

        Args:
            page: PyMuPDF Page object.

        Returns:
            OcrPageResult containing recognized text and confidence.

        Raises:
            OcrPageError: If calculated raster pixel count exceeds max_image_pixels
                or if engine recognition fails.
            OcrUnavailableError: If OCR engine or language data is unavailable.
        """
        # Pre-check dimensions before raster allocation (points -> pixels at dpi)
        rect = page.rect
        w_px = int(rect.width * (self.dpi / 72.0))
        h_px = int(rect.height * (self.dpi / 72.0))
        total_pixels = w_px * h_px

        if total_pixels > self.max_image_pixels:
            raise OcrPageError(
                f"Page {page.number + 1} dimensions "
                f"({w_px}x{h_px} = {total_pixels:,} px) "
                f"exceed configured maximum limit of {self.max_image_pixels:,} px."
            )

        try:
            pix = page.get_pixmap(
                dpi=self.dpi,
                colorspace=pymupdf.csGRAY,
                alpha=False,
            )
            png_bytes = pix.tobytes("png")
        except Exception as exc:
            raise OcrPageError(
                f"Failed to rasterize page {page.number + 1} at {self.dpi} DPI: {exc}"
            ) from exc
        finally:
            pix = None  # Release unmanaged pixmap memory immediately

        return self.engine.recognize(
            png_bytes,
            language=self.language,
            timeout=self.timeout_seconds,
        )

    def ocr_image_bytes(self, image_bytes: bytes) -> OcrPageResult:
        """Run OCR on pre-processed PNG image bytes.

        Args:
            image_bytes: PNG-encoded image bytes.

        Returns:
            OcrPageResult containing recognized text and confidence.
        """
        return self.engine.recognize(
            image_bytes,
            language=self.language,
            timeout=self.timeout_seconds,
        )
