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
    """PDF magic bytes with .pdf extension are detected as PDF."""
    weird = tmp_path / "report.pdf"
    weird.write_bytes(b"%PDF-2.0 newer pdf")
    assert detect_format(weird) == FileFormat.PDF


def test_pdf_content_with_docx_extension_raises(tmp_path: Path):
    """A .docx file containing PDF magic bytes is rejected as format mismatch."""
    spoofed = tmp_path / "spoofed.docx"
    spoofed.write_bytes(b"%PDF-1.4 some pdf data")
    with pytest.raises(UnsupportedFormatError):
        detect_format(spoofed)


def test_docx_content_with_pdf_extension_raises(tmp_path: Path):
    """A .pdf file containing ZIP/DOCX magic bytes is rejected as format mismatch."""
    spoofed = tmp_path / "spoofed.pdf"
    spoofed.write_bytes(b"PK\x03\x04 some zip data")
    with pytest.raises(UnsupportedFormatError):
        detect_format(spoofed)


def test_case_insensitive_extension_detection(tmp_path: Path):
    """Uppercase extensions like .PDF and .DOCX are properly detected."""
    upper_pdf = tmp_path / "UPPER.PDF"
    upper_pdf.write_bytes(b"%PDF-1.7 data")
    assert detect_format(upper_pdf) == FileFormat.PDF

    upper_docx = tmp_path / "UPPER.DOCX"
    upper_docx.write_bytes(b"PK\x03\x04 data")
    assert detect_format(upper_docx) == FileFormat.DOCX


def test_missing_file_raises_filenotfound(tmp_path: Path):
    """detect_format raises FileNotFoundError when the file does not exist."""
    from rag.ingestion.exceptions import FileNotFoundError

    with pytest.raises(FileNotFoundError):
        detect_format(tmp_path / "nonexistent.pdf")
