"""Tests for file-format detection."""

from pathlib import Path

import pytest

from rag.ingestion.detection import FileFormat, detect_format
from rag.ingestion.exceptions import UnsupportedFormatError


def test_detect_pdf(tmp_path: Path):
    """A file with PDF magic bytes is detected as PDF."""
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.7 some content")
    assert detect_format(pdf) == FileFormat.PDF


def test_detect_docx(tmp_path: Path):
    """A file with ZIP magic + .docx extension is detected as DOCX."""
    docx = tmp_path / "test.docx"
    docx.write_bytes(b"PK\x03\x04 some content")
    assert detect_format(docx) == FileFormat.DOCX


def test_unsupported_extension_raises(tmp_path: Path):
    """An unknown extension raises UnsupportedFormatError."""
    txt = tmp_path / "notes.txt"
    txt.write_bytes(b"plain text content")
    with pytest.raises(UnsupportedFormatError):
        detect_format(txt)


def test_pdf_extension_wrong_magic_raises(tmp_path: Path):
    """A .pdf file without PDF magic bytes is rejected."""
    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_bytes(b"not a real pdf")
    with pytest.raises(UnsupportedFormatError):
        detect_format(fake_pdf)


def test_docx_extension_wrong_magic_raises(tmp_path: Path):
    """A .docx file without ZIP magic bytes is rejected."""
    fake_docx = tmp_path / "fake.docx"
    fake_docx.write_bytes(b"not a real docx")
    with pytest.raises(UnsupportedFormatError):
        detect_format(fake_docx)


def test_pdf_magic_takes_priority(tmp_path: Path):
    """PDF magic bytes are detected regardless of extension."""
    weird = tmp_path / "report.pdf"
    weird.write_bytes(b"%PDF-2.0 newer pdf")
    assert detect_format(weird) == FileFormat.PDF
