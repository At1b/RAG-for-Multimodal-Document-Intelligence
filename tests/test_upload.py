"""Tests for the document-upload API endpoint."""

import io

import docx as python_docx
import pymupdf
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def _make_pdf_bytes(text: str = "Test content") -> bytes:
    """Create a minimal PDF in-memory and return its bytes."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def _make_docx_bytes(text: str = "Test content") -> bytes:
    """Create a minimal DOCX in-memory and return its bytes."""
    doc = python_docx.Document()
    doc.add_paragraph(text)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# --- successful uploads ---


def test_upload_pdf_returns_200():
    """Uploading a valid PDF returns 200 with document metadata."""
    pdf_bytes = _make_pdf_bytes("Hello from PDF")
    response = client.post(
        "/documents/upload",
        files={"file": ("report.pdf", pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["document_name"] == "report.pdf"
    assert data["source_type"] == "pdf"
    assert data["total_pages"] >= 1
    assert "document_id" in data


def test_upload_docx_returns_200():
    """Uploading a valid DOCX returns 200 with document metadata."""
    docx_bytes = _make_docx_bytes("Hello from DOCX")
    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "document.docx",
                docx_bytes,
                "application/vnd.openxmlformats-officedocument"
                ".wordprocessingml.document",
            )
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["document_name"] == "document.docx"
    assert data["source_type"] == "docx"
    assert "document_id" in data


# --- error cases ---


def test_upload_empty_file_returns_400():
    """Uploading an empty file returns 400."""
    response = client.post(
        "/documents/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert response.status_code == 400


def test_upload_unsupported_format_returns_415():
    """Uploading an unsupported file type returns 415."""
    response = client.post(
        "/documents/upload",
        files={"file": ("notes.txt", b"some text", "text/plain")},
    )
    assert response.status_code == 415


def test_upload_corrupted_pdf_returns_422():
    """Uploading a corrupted PDF returns 422."""
    response = client.post(
        "/documents/upload",
        files={
            "file": ("corrupt.pdf", b"%PDF-1.4 invalid truncated", "application/pdf")
        },
    )
    assert response.status_code == 422


def test_upload_corrupted_docx_returns_422():
    """Uploading a corrupted DOCX returns 422."""
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("not_docx.txt", "hello")
    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "corrupt.docx",
                buf.getvalue(),
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )
    assert response.status_code == 422


def test_upload_missing_filename_returns_400():
    """Uploading with a whitespace filename returns 400."""
    response = client.post(
        "/documents/upload",
        files={"file": ("   ", b"%PDF-1.4 content", "application/pdf")},
    )
    assert response.status_code == 400
    assert "Filename is required" in response.json()["detail"]


def test_upload_non_ascii_filename_preserved():
    """Uploading with a non-ASCII filename preserves the document name in response."""
    pdf_bytes = _make_pdf_bytes("Unicode content")
    response = client.post(
        "/documents/upload",
        files={"file": ("über_rapport_évaluation.pdf", pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 200
    assert response.json()["document_name"] == "über_rapport_évaluation.pdf"


def test_upload_file_too_large_returns_413(monkeypatch):
    """Uploading a file exceeding max_upload_size_mb returns 413."""
    from backend.config import Settings

    # Override settings to have a very small limit (1 MB)
    small_settings = Settings(max_upload_size_mb=1)
    monkeypatch.setattr("backend.documents.get_settings", lambda: small_settings)

    # 1.5 MB payload
    large_payload = b"%PDF-1.4 " + (b"A" * (1024 * 1024 + 500_000))
    response = client.post(
        "/documents/upload",
        files={"file": ("too_large.pdf", large_payload, "application/pdf")},
    )
    assert response.status_code == 413
