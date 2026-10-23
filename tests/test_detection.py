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


def test_detect_png(tmp_path: Path):
    """A file with PNG magic bytes is detected as PNG."""
    png = tmp_path / "test.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n image data")
    assert detect_format(png) == FileFormat.PNG


def test_detect_jpeg(tmp_path: Path):
    """A file with JPEG magic bytes is detected as JPEG for .jpg and .jpeg."""
    jpg = tmp_path / "photo.jpg"
    jpg.write_bytes(b"\xff\xd8\xff\xe0 image data")
    assert detect_format(jpg) == FileFormat.JPEG

    jpeg = tmp_path / "photo.jpeg"
    jpeg.write_bytes(b"\xff\xd8\xff\xe1 image data")
    assert detect_format(jpeg) == FileFormat.JPEG


def test_png_extension_wrong_magic_raises(tmp_path: Path):
    """A .png file without PNG signature is rejected."""
    fake_png = tmp_path / "fake.png"
    fake_png.write_bytes(b"not a real png")
    with pytest.raises(UnsupportedFormatError):
        detect_format(fake_png)


def test_jpeg_extension_wrong_magic_raises(tmp_path: Path):
    """A .jpg file without JPEG signature is rejected."""
    fake_jpg = tmp_path / "fake.jpg"
    fake_jpg.write_bytes(b"not a real jpeg")
    with pytest.raises(UnsupportedFormatError):
        detect_format(fake_jpg)


def test_image_case_insensitive_detection(tmp_path: Path):
    """Uppercase extensions like .PNG and .JPG are properly detected."""
    upper_png = tmp_path / "IMAGE.PNG"
    upper_png.write_bytes(b"\x89PNG\r\n\x1a\n data")
    assert detect_format(upper_png) == FileFormat.PNG

    upper_jpg = tmp_path / "PHOTO.JPG"
    upper_jpg.write_bytes(b"\xff\xd8\xff data")
    assert detect_format(upper_jpg) == FileFormat.JPEG
