"""MM-RAG OCR package for image and scanned document text extraction."""

from __future__ import annotations

from rag.ingestion.ocr.engine import OcrEngine, OcrPageResult, TesseractEngine
from rag.ingestion.ocr.page_analysis import needs_ocr
from rag.ingestion.ocr.processor import OcrProcessor

__all__ = [
    "OcrEngine",
    "OcrPageResult",
    "OcrProcessor",
    "TesseractEngine",
    "needs_ocr",
]
