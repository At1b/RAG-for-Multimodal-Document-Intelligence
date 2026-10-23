"""PDF document loader using PyMuPDF.

Extracts text page-by-page, preserving page numbers and boundaries.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pymupdf

from rag.ingestion.document_id import generate_document_id
from rag.ingestion.exceptions import (
    InvalidDocumentError,
    NoExtractableTextError,
    OcrLimitExceededError,
    OcrPageError,
    OcrUnavailableError,
)
from rag.ingestion.models import Document, PageContent
from rag.ingestion.ocr.page_analysis import needs_ocr
from rag.ingestion.ocr.processor import OcrProcessor

logger = logging.getLogger(__name__)


def load_pdf(
    file_path: Path,
    *,
    document_name: str | None = None,
    ocr: OcrProcessor | None = None,
) -> Document:
    """Parse a PDF file and return a normalized ``Document``.

    Args:
        file_path: Path to a validated PDF file.
        document_name: Display name for the document.  Falls back to
            the file's basename.
        ocr: Optional OcrProcessor for scanned pages. When None (default),
            runs native text extraction with zero OCR overhead.

    Raises:
        InvalidDocumentError: If PyMuPDF cannot open or read the file.
        OcrLimitExceededError: If scanned pages exceed ocr.max_pages.
        OcrUnavailableError: If OCR is required but the engine/language is missing.
        NoExtractableTextError: If OCR was required on pages and no text
            could be extracted.
    """
    file_path = Path(file_path)
    name = document_name or file_path.name

    try:
        with open(file_path, "rb") as fh:
            doc = pymupdf.open(stream=fh.read(), filetype="pdf")
    except Exception as exc:
        raise InvalidDocumentError(f"Failed to open PDF '{name}': {exc}") from exc

    if doc.is_encrypted and doc.needs_pass:
        doc.close()
        raise InvalidDocumentError(f"PDF '{name}' is password-protected or encrypted.")

    num_doc_pages = len(doc)
    if num_doc_pages == 0:
        doc.close()
        raise InvalidDocumentError(f"PDF '{name}' contains no pages or is corrupted.")

    # Pre-scan pages needing OCR if OCR processor is configured
    ocr_min_native = getattr(ocr, "ocr_min_native_chars", 25) if ocr else 25
    if ocr is not None:
        scan_count = sum(
            1
            for p_idx in range(num_doc_pages)
            if needs_ocr(doc[p_idx], min_native_chars=ocr_min_native)
        )
        if scan_count > ocr.max_pages:
            doc.close()
            raise OcrLimitExceededError(
                f"PDF '{name}' contains {scan_count} scanned pages needing OCR, "
                f"exceeding the maximum allowed limit of {ocr.max_pages} pages."
            )

    pages: list[PageContent] = []
    ocr_attempted_pages: list[int] = []
    ocr_failed_pages: list[int] = []
    current_page = 0

    try:
        for page_num in range(num_doc_pages):
            current_page = page_num + 1
            page = doc[page_num]
            native_text = page.get_text("text")

            # Legacy path when ocr is None
            if ocr is None:
                pages.append(
                    PageContent(
                        page_number=current_page,  # 1-indexed
                        content=native_text,
                        metadata={"char_count": len(native_text)},
                    )
                )
                continue

            # OCR-enabled path: evaluate if page requires OCR
            page_needs_ocr = needs_ocr(page, min_native_chars=ocr_min_native)
            if not page_needs_ocr:
                pages.append(
                    PageContent(
                        page_number=current_page,
                        content=native_text,
                        metadata={
                            "char_count": len(native_text),
                            "extraction_method": "native_text",
                        },
                    )
                )
            else:
                ocr_attempted_pages.append(current_page)
                try:
                    ocr_res = ocr.ocr_pdf_page(page)
                    page_text = ocr_res.text
                    if not any(c.isalnum() for c in page_text):
                        raise OcrPageError(
                            f"OCR produced no readable text for page {current_page}."
                        )
                    page_meta: dict[str, Any] = {
                        "char_count": len(page_text),
                        "extraction_method": "ocr",
                    }
                    if ocr_res.mean_confidence is not None:
                        page_meta["ocr_mean_confidence"] = ocr_res.mean_confidence
                    pages.append(
                        PageContent(
                            page_number=current_page,
                            content=page_text,
                            metadata=page_meta,
                        )
                    )
                except OcrUnavailableError:
                    raise
                except (OcrPageError, Exception) as exc:
                    logger.warning(
                        "OCR failed on page %d of '%s': %s. "
                        "Falling back to native text.",
                        current_page,
                        name,
                        exc,
                    )
                    ocr_failed_pages.append(current_page)
                    pages.append(
                        PageContent(
                            page_number=current_page,
                            content=native_text,
                            metadata={
                                "char_count": len(native_text),
                                "extraction_method": "native_text",
                            },
                        )
                    )
    except (OcrUnavailableError, OcrLimitExceededError):
        raise
    except Exception as exc:
        msg = (
            f"Error extracting text from PDF '{name}' on page {current_page}: {exc}"
            if current_page > 0
            else f"Error extracting text from PDF '{name}': {exc}"
        )
        raise InvalidDocumentError(msg) from exc
    finally:
        doc.close()

    if not pages:
        raise InvalidDocumentError(f"PDF '{name}' contains no pages or is corrupted.")

    # Outcome rules when OCR was active
    if ocr is not None and ocr_attempted_pages:
        usable = [p for p in pages if any(c.isalnum() for c in p.content)]
        if not usable:
            raise NoExtractableTextError(
                f"No readable text could be extracted from PDF '{name}' "
                f"({len(ocr_attempted_pages)} page(s) required OCR; "
                f"{len(ocr_failed_pages)} failed)."
            )

    file_size = file_path.stat().st_size
    doc_metadata: dict[str, Any] = {
        "total_pages": len(pages),
        "file_size_bytes": file_size,
    }

    if ocr is not None and ocr_attempted_pages:
        doc_metadata["ocr_page_count"] = len(ocr_attempted_pages)
        doc_metadata["ocr_failed_pages"] = ocr_failed_pages
        doc_metadata["ocr_engine"] = ocr.engine_version
        doc_metadata["ocr_language"] = ocr.language
        doc_metadata["ocr_dpi"] = ocr.dpi

    return Document(
        document_id=generate_document_id(),
        document_name=name,
        source_type="pdf",
        pages=pages,
        metadata=doc_metadata,
    )
