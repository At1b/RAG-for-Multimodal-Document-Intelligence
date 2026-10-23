"""Image loader for PNG and JPEG documents using Pillow and OCR.

Decodes, validates bounds, normalizes EXIF orientation, converts to grayscale,
and extracts text via the configured OcrProcessor.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from rag.ingestion.document_id import generate_document_id
from rag.ingestion.exceptions import (
    InvalidDocumentError,
    NoExtractableTextError,
    OcrLimitExceededError,
    OcrPageError,
    OcrUnavailableError,
    UnsupportedFormatError,
)
from rag.ingestion.models import Document, PageContent
from rag.ingestion.ocr.processor import OcrProcessor

logger = logging.getLogger(__name__)

_SUPPORTED_IMAGE_FORMATS = {"PNG", "JPEG"}


def load_image(
    file_path: Path,
    *,
    document_name: str | None = None,
    ocr: OcrProcessor | None = None,
) -> Document:
    """Load an image file (PNG or JPEG), run OCR, and return a normalized Document.

    Args:
        file_path: Path to the image file.
        document_name: Optional user-facing filename. Falls back to file's basename.
        ocr: OcrProcessor instance. Required for image documents.

    Returns:
        A normalized Document with source_type="image" and a single PageContent.

    Raises:
        OcrUnavailableError: If OCR is not configured or engine is unavailable.
        UnsupportedFormatError: If image format is not PNG or JPEG.
        OcrLimitExceededError: If image pixel count exceeds configured limit.
        InvalidDocumentError: If image file is corrupt or unreadable.
        NoExtractableTextError: If OCR yields no readable alphanumeric text.
    """
    file_path = Path(file_path)
    name = document_name or file_path.name

    if ocr is None:
        raise OcrUnavailableError(
            "OCR is disabled; image documents require OCR_ENABLED=true."
        )

    # Set decompression bomb limit on Pillow
    Image.MAX_IMAGE_PIXELS = ocr.max_image_pixels

    try:
        with Image.open(file_path) as img:
            img_format = img.format
            if not img_format or img_format.upper() not in _SUPPORTED_IMAGE_FORMATS:
                raise UnsupportedFormatError(
                    f"Unsupported image format: '{img_format}'. Supported: PNG, JPEG."
                )

            width, height = img.size
            total_pixels = width * height
            if total_pixels > ocr.max_image_pixels:
                raise OcrLimitExceededError(
                    f"Image '{name}' dimensions "
                    f"({width}x{height} = {total_pixels:,} px) "
                    f"exceed configured maximum limit of {ocr.max_image_pixels:,} px."
                )

            # Force complete decode to verify file integrity
            img.load()

            # Transpose according to EXIF orientation tag (e.g. smartphone photos)
            transposed = ImageOps.exif_transpose(img)
            target = transposed if transposed is not None else img
            target_width, target_height = target.size

            # Convert to 8-bit grayscale for OCR
            gray = target.convert("L")
            buf = io.BytesIO()
            gray.save(buf, format="PNG")
            png_bytes = buf.getvalue()
    except (
        OcrUnavailableError,
        OcrLimitExceededError,
        UnsupportedFormatError,
    ):
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise OcrLimitExceededError(
            f"Image '{name}' exceeds configured maximum pixel limit: {exc}"
        ) from exc
    except Exception as exc:
        raise InvalidDocumentError(f"Failed to read image '{name}': {exc}") from exc

    # Execute OCR on normalized PNG bytes
    try:
        ocr_result = ocr.ocr_image_bytes(png_bytes)
    except OcrUnavailableError:
        raise
    except OcrPageError as exc:
        raise NoExtractableTextError(
            f"OCR processing failed for image '{name}': {exc}"
        ) from exc
    except Exception as exc:
        raise InvalidDocumentError(
            f"Unexpected failure during OCR of image '{name}': {exc}"
        ) from exc

    text = ocr_result.text.strip()
    if not any(c.isalnum() for c in text):
        raise NoExtractableTextError(
            f"No readable text could be extracted from image '{name}'."
        )

    file_size = file_path.stat().st_size
    page_metadata: dict[str, Any] = {
        "char_count": len(text),
        "extraction_method": "ocr",
    }
    if ocr_result.mean_confidence is not None:
        page_metadata["ocr_mean_confidence"] = ocr_result.mean_confidence

    doc_metadata: dict[str, Any] = {
        "total_pages": 1,
        "file_size_bytes": file_size,
        "image_format": img_format.lower(),
        "image_width": target_width,
        "image_height": target_height,
        "ocr_page_count": 1,
        "ocr_failed_pages": [],
        "ocr_engine": ocr.engine_version,
        "ocr_language": ocr.language,
        "ocr_dpi": ocr.dpi,
    }

    return Document(
        document_id=generate_document_id(),
        document_name=name,
        source_type="image",
        pages=[
            PageContent(
                page_number=1,
                content=text,
                metadata=page_metadata,
            )
        ],
        metadata=doc_metadata,
    )
