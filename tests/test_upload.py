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
