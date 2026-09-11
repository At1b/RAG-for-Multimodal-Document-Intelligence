"""Tests for file validation."""

from pathlib import Path

import pytest

from rag.ingestion.exceptions import (
    EmptyFileError,
    FileNotFoundError,
    FileTooLargeError,
    UnsupportedFormatError,
)
from rag.ingestion.validation import validate_file


def test_valid_pdf_passes(tmp_path: Path):
    """A valid-looking PDF file passes validation."""
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.4 minimal content")
    validate_file(pdf)  # should not raise


def test_valid_docx_passes(tmp_path: Path):
    """A valid-looking DOCX file passes validation."""
    docx = tmp_path / "test.docx"
    docx.write_bytes(b"PK\x03\x04 some content")
    validate_file(docx)  # should not raise


def test_missing_file_raises(tmp_path: Path):
    """A non-existent file raises FileNotFoundError."""
    missing = tmp_path / "nonexistent.pdf"
    with pytest.raises(FileNotFoundError):
        validate_file(missing)


def test_empty_file_raises(tmp_path: Path):
    """A zero-byte file raises EmptyFileError."""
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")
    with pytest.raises(EmptyFileError):
        validate_file(empty)


def test_unsupported_extension_raises(tmp_path: Path):
    """A file with an unsupported extension raises UnsupportedFormatError."""
    txt = tmp_path / "notes.txt"
    txt.write_text("some text")
    with pytest.raises(UnsupportedFormatError):
        validate_file(txt)


def test_file_too_large_raises(tmp_path: Path):
    """A file exceeding the size limit raises FileTooLargeError."""
    big = tmp_path / "big.pdf"
    big.write_bytes(b"%PDF-" + b"x" * 1000)
    with pytest.raises(FileTooLargeError):
        validate_file(big, max_size_bytes=500)


def test_file_under_size_limit_passes(tmp_path: Path):
    """A file within the size limit passes validation."""
    pdf = tmp_path / "small.pdf"
    pdf.write_bytes(b"%PDF-1.4 content")
    validate_file(pdf, max_size_bytes=10_000_000)  # should not raise


def test_size_check_skipped_when_none(tmp_path: Path):
    """When max_size_bytes is None, no size check is performed."""
    pdf = tmp_path / "any.pdf"
    pdf.write_bytes(b"%PDF-1.4 content")
    validate_file(pdf, max_size_bytes=None)  # should not raise
