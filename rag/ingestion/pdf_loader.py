"""PDF document loader using PyMuPDF.

Extracts text page-by-page, preserving page numbers and boundaries.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pymupdf

from rag.ingestion.document_id import generate_document_id
from rag.ingestion.exceptions import InvalidDocumentError
from rag.ingestion.models import Document, PageContent

logger = logging.getLogger(__name__)


def load_pdf(file_path: Path, *, document_name: str | None = None) -> Document:
    """Parse a PDF file and return a normalized ``Document``.

    Args:
        file_path: Path to a validated PDF file.
        document_name: Display name for the document.  Falls back to
            the file's basename.

    Raises:
        InvalidDocumentError: If PyMuPDF cannot open or read the file.
    """
    name = document_name or file_path.name

    try:
        with open(file_path, "rb") as fh:
            doc = pymupdf.open(stream=fh.read(), filetype="pdf")
    except Exception as exc:
        raise InvalidDocumentError(f"Failed to open PDF '{name}': {exc}") from exc

    if doc.is_encrypted and doc.needs_pass:
        doc.close()
        raise InvalidDocumentError(f"PDF '{name}' is password-protected or encrypted.")

    pages: list[PageContent] = []
    current_page = 0
    try:
        for page_num in range(len(doc)):
            current_page = page_num + 1
            page = doc[page_num]
            text = page.get_text("text")
            pages.append(
                PageContent(
                    page_number=current_page,  # 1-indexed
                    content=text,
                    metadata={"char_count": len(text)},
                )
            )
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

    file_size = file_path.stat().st_size

    return Document(
        document_id=generate_document_id(),
        document_name=name,
        source_type="pdf",
        pages=pages,
        metadata={
            "total_pages": len(pages),
            "file_size_bytes": file_size,
        },
    )
