"""Tests for IngestionService orchestrator."""

from pathlib import Path

import docx as python_docx
import pymupdf
import pytest

from rag.ingestion.exceptions import (
    EmptyFileError,
    FileNotFoundError,
    FileTooLargeError,
    InvalidDocumentError,
    UnsupportedFormatError,
)
from rag.ingestion.service import IngestionService


def _create_sample_pdf(path: Path, pages: list[str]) -> Path:
    doc = pymupdf.open()
    for text in pages:
        p = doc.new_page()
        p.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()
    return path


def _create_sample_docx(path: Path, paragraphs: list[str]) -> Path:
    doc = python_docx.Document()
    for text in paragraphs:
        doc.add_paragraph(text)
    doc.save(str(path))
    return path


def test_ingest_valid_pdf(tmp_path: Path):
    """IngestionService correctly ingests a valid PDF document."""
    pdf = _create_sample_pdf(tmp_path / "sample.pdf", ["Page 1 text", "Page 2 text"])
    service = IngestionService()
    doc = service.ingest(pdf)

    assert doc.source_type == "pdf"
    assert doc.document_name == "sample.pdf"
    assert len(doc.pages) == 2
    assert doc.metadata["total_pages"] == 2
    assert doc.metadata["file_size_bytes"] > 0
    assert doc.document_id
    assert "Page 1 text" in doc.pages[0].content
    assert "Page 2 text" in doc.pages[1].content


def test_ingest_valid_docx(tmp_path: Path):
    """IngestionService correctly ingests a valid DOCX document."""
    docx_file = _create_sample_docx(tmp_path / "sample.docx", ["Hello from DOCX"])
    service = IngestionService()
    doc = service.ingest(docx_file)

    assert doc.source_type == "docx"
    assert doc.document_name == "sample.docx"
    assert len(doc.pages) == 1
    assert doc.metadata["total_pages"] == 1
    assert doc.metadata["file_size_bytes"] > 0
    assert doc.document_id
    assert "Hello from DOCX" in doc.pages[0].content


def test_ingest_multiple_documents_independent(tmp_path: Path):
    """Multiple documents ingested sequentially have unique IDs and isolated state."""
    pdf1 = _create_sample_pdf(tmp_path / "first.pdf", ["Doc 1 content"])
    pdf2 = _create_sample_pdf(tmp_path / "second.pdf", ["Doc 2 content", "Extra page"])

    service = IngestionService()
    doc1 = service.ingest(pdf1)
    doc2 = service.ingest(pdf2)

    assert doc1.document_id != doc2.document_id
    assert doc1.document_name == "first.pdf"
    assert doc2.document_name == "second.pdf"
    assert len(doc1.pages) == 1
    assert len(doc2.pages) == 2
    assert "Doc 1 content" in doc1.pages[0].content
    assert "Doc 2 content" in doc2.pages[0].content


def test_ingest_preserves_custom_document_name(tmp_path: Path):
    """Original filename passed as argument overrides file path name."""
    pdf = _create_sample_pdf(tmp_path / "raw_temp.pdf", ["Content"])
    service = IngestionService()
    doc = service.ingest(pdf, original_filename="Custom Report.pdf")

    assert doc.document_name == "Custom Report.pdf"


def test_ingest_non_ascii_filename(tmp_path: Path):
    """Ingesting a file with non-ASCII filename works seamlessly."""
    pdf = _create_sample_pdf(tmp_path / "rapport_évaluation_2024.pdf", ["Évaluation"])
    service = IngestionService()
    doc = service.ingest(pdf)

    assert doc.document_name == "rapport_évaluation_2024.pdf"
    assert "Évaluation" in doc.pages[0].content


def test_ingest_max_size_exceeded_raises(tmp_path: Path):
    """IngestionService raises FileTooLargeError when size exceeds max_size_bytes."""
    pdf = _create_sample_pdf(tmp_path / "large.pdf", ["Some text content"])
    file_size = pdf.stat().st_size
    service = IngestionService(max_size_bytes=file_size - 1)

    with pytest.raises(FileTooLargeError):
        service.ingest(pdf)


def test_ingest_nonexistent_file_raises(tmp_path: Path):
    """IngestionService raises FileNotFoundError for missing file."""
    service = IngestionService()
    with pytest.raises(FileNotFoundError):
        service.ingest(tmp_path / "does_not_exist.pdf")


def test_ingest_empty_file_raises(tmp_path: Path):
    """IngestionService raises EmptyFileError for zero-byte file."""
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")
    service = IngestionService()
    with pytest.raises(EmptyFileError):
        service.ingest(empty)


def test_ingest_unsupported_format_raises(tmp_path: Path):
    """IngestionService raises UnsupportedFormatError for unsupported extension."""
    txt = tmp_path / "notes.txt"
    txt.write_text("plain text")
    service = IngestionService()
    with pytest.raises(UnsupportedFormatError):
        service.ingest(txt)


def test_ingest_corrupted_file_raises(tmp_path: Path):
    """IngestionService raises InvalidDocumentError for corrupted file."""
    bad_pdf = tmp_path / "corrupted.pdf"
    bad_pdf.write_bytes(b"%PDF-1.4 but completely corrupt data here")
    service = IngestionService()
    with pytest.raises(InvalidDocumentError):
        service.ingest(bad_pdf)
