"""Tests for the PDF loader.

Test PDF files are generated programmatically using PyMuPDF so no
binary fixtures need to be committed.
"""

from pathlib import Path

import pymupdf
import pytest

from rag.ingestion.exceptions import InvalidDocumentError
from rag.ingestion.pdf_loader import load_pdf


def _create_pdf(path: Path, pages: list[str]) -> Path:
    """Create a minimal PDF with the given page texts."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()
    return path


# --- valid PDF tests ---


def test_load_valid_pdf(tmp_path: Path):
    """A valid single-page PDF loads successfully."""
    pdf = _create_pdf(tmp_path / "single.pdf", ["Hello World"])
    doc = load_pdf(pdf)
    assert doc.source_type == "pdf"
    assert len(doc.pages) == 1
    assert "Hello World" in doc.pages[0].content


def test_page_numbers_preserved(tmp_path: Path):
    """Page numbers are 1-indexed and sequential."""
    pdf = _create_pdf(tmp_path / "multi.pdf", ["Page 1", "Page 2", "Page 3"])
    doc = load_pdf(pdf)
    assert [p.page_number for p in doc.pages] == [1, 2, 3]


def test_content_extracted(tmp_path: Path):
    """Text content is extracted from each page."""
    pdf = _create_pdf(
        tmp_path / "content.pdf",
        ["First page content", "Second page content"],
    )
    doc = load_pdf(pdf)
    assert "First page content" in doc.pages[0].content
    assert "Second page content" in doc.pages[1].content


def test_multipage_preserves_all_pages(tmp_path: Path):
    """All pages are preserved in a multi-page PDF."""
    pages = [f"Page {i}" for i in range(1, 6)]
    pdf = _create_pdf(tmp_path / "five.pdf", pages)
    doc = load_pdf(pdf)
    assert len(doc.pages) == 5


def test_empty_page_handled(tmp_path: Path):
    """An empty page is stored with empty-string content."""
    pdf = _create_pdf(tmp_path / "empty_page.pdf", ["Content", ""])
    doc = load_pdf(pdf)
    assert len(doc.pages) == 2
    assert doc.pages[1].content.strip() == ""


def test_document_id_exists(tmp_path: Path):
    """The returned document has a non-empty document_id."""
    pdf = _create_pdf(tmp_path / "id.pdf", ["text"])
    doc = load_pdf(pdf)
    assert doc.document_id
    assert len(doc.document_id) > 0


def test_document_name_preserved(tmp_path: Path):
    """The document_name matches the provided name."""
    pdf = _create_pdf(tmp_path / "named.pdf", ["text"])
    doc = load_pdf(pdf, document_name="My Report.pdf")
    assert doc.document_name == "My Report.pdf"


def test_document_name_falls_back_to_filename(tmp_path: Path):
    """Without an explicit name, the filename is used."""
    pdf = _create_pdf(tmp_path / "fallback.pdf", ["text"])
    doc = load_pdf(pdf)
    assert doc.document_name == "fallback.pdf"


def test_metadata_includes_total_pages(tmp_path: Path):
    """Document metadata includes total_pages count."""
    pdf = _create_pdf(tmp_path / "meta.pdf", ["a", "b"])
    doc = load_pdf(pdf)
    assert doc.metadata["total_pages"] == 2


def test_metadata_includes_file_size(tmp_path: Path):
    """Document metadata includes file_size_bytes."""
    pdf = _create_pdf(tmp_path / "size.pdf", ["text"])
    doc = load_pdf(pdf)
    assert doc.metadata["file_size_bytes"] > 0


# --- invalid PDF tests ---


def test_invalid_pdf_raises(tmp_path: Path):
    """A corrupted/non-PDF file raises InvalidDocumentError."""
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"this is not a pdf at all")
    with pytest.raises(InvalidDocumentError):
        load_pdf(bad)
