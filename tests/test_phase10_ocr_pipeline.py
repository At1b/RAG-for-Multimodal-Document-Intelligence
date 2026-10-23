"""Pipeline and API integration tests for Phase 10 OCR.

Uses real embeddings (all-MiniLM-L6-v2), real Chroma vector store, real chunker,
and real citation formatter with a fake OCR engine to verify end-to-end flow.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pymupdf
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.config import Settings
from backend.main import app
from rag.generation.models import GenerationResult
from rag.ingestion.exceptions import (
    OcrLimitExceededError,
    OcrUnavailableError,
)
from rag.ingestion.ocr.engine import OcrPageResult
from rag.ingestion.ocr.processor import OcrProcessor


class PipelineFakeOcrEngine:
    """Fake OCR engine mapping inputs to predictable outputs."""

    def __init__(self, page_texts: dict[int, str] | None = None) -> None:
        self.page_texts = page_texts or {}
        self.call_count = 0

    @property
    def version(self) -> str:
        return "FakeTesseract 5.5.0"

    def recognize(
        self,
        image_bytes: bytes,
        *,
        language: str = "eng",
        timeout: float = 60.0,
    ) -> OcrPageResult:
        self.call_count += 1
        text = self.page_texts.get(
            self.call_count, f"Recognized content block {self.call_count}."
        )
        return OcrPageResult(text=text, mean_confidence=95.0)


def _make_scanned_pdf_bytes(page_count: int = 3) -> bytes:
    doc = pymupdf.open()
    pix = pymupdf.Pixmap(pymupdf.csGRAY, (0, 0, 100, 100), 0)
    pix.clear_with(200)
    png_bytes = pix.tobytes("png")

    for _ in range(page_count):
        page = doc.new_page(width=500, height=700)
        rect = pymupdf.Rect(50, 50, 450, 650)
        page.insert_image(rect, stream=png_bytes)

    data = doc.tobytes()
    doc.close()
    return data


def _make_mixed_pdf_bytes() -> bytes:
    doc = pymupdf.open()
    pix = pymupdf.Pixmap(pymupdf.csGRAY, (0, 0, 100, 100), 0)
    pix.clear_with(200)
    png_bytes = pix.tobytes("png")

    # Page 1: Native text about quantum algorithms (~1200 chars)
    p1 = doc.new_page(width=500, height=700)
    p1.insert_text(
        (50, 50),
        "Quantum superposition enables qubits to represent multiple states "
        "simultaneously. " * 15,
    )

    # Page 2: Scanned raster image
    p2 = doc.new_page(width=500, height=700)
    p2.insert_image(pymupdf.Rect(50, 50, 450, 650), stream=png_bytes)

    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def ocr_test_env(tmp_path: Path):
    """Provide TestClient with isolated vector store and fake OCR processor."""
    settings = Settings(
        vector_store_path=str(tmp_path / "vectorstore"),
        vector_store_collection="test_ocr_pipeline",
        ocr_enabled=True,
        chunk_size=500,
        chunk_overlap=50,
    )

    fake_engine = PipelineFakeOcrEngine(
        page_texts={
            1: (
                "Header introduction to multimodal intelligence with extensive "
                "background details and foundational principles. " * 15
            ),
            2: (
                "Optical character recognition bridges physical scans and "
                "embeddings directly into semantic vector spaces. " * 15
            ),
            3: (
                "Conclusion and bibliography references providing complete "
                "documentation and analytical evaluations. " * 15
            ),
        }
    )
    fake_processor = OcrProcessor(engine=fake_engine, dpi=200)

    mock_llm_result = GenerationResult(
        answer="Grounded answer based on OCR text evidence.",
        model_name="mock-ollama",
        metadata={"temperature": 0.1},
    )

    with (
        patch("backend.documents.get_settings", return_value=settings),
        patch("backend.query.get_settings", return_value=settings),
        patch(
            "backend.documents._build_ocr_processor",
            return_value=fake_processor,
        ),
        patch("backend.query.OllamaGenerator") as MockGen,
    ):
        gen_instance = MagicMock()
        gen_instance.generate.return_value = mock_llm_result
        MockGen.return_value = gen_instance
        client = TestClient(app)
        yield client, fake_engine, gen_instance


class TestOcrPipelineIntegration:
    """Verify end-to-end ingestion, indexing, and retrieval with OCR documents."""

    def test_scanned_pdf_upload_index_and_retrieval(self, ocr_test_env):
        client, fake_engine, mock_gen = ocr_test_env

        pdf_bytes = _make_scanned_pdf_bytes(page_count=3)
        upload_resp = client.post(
            "/documents/upload",
            files={"file": ("scanned_doc.pdf", pdf_bytes, "application/pdf")},
        )

        assert upload_resp.status_code == 200
        data = upload_resp.json()
        assert data["document_name"] == "scanned_doc.pdf"
        assert data["num_pages"] == 3
        assert data["ocr_pages"] == 3
        assert data["num_chunks"] >= 1
        assert data["ocr_failed_pages"] == []

        # Query specifically for text on scanned page 2
        query_resp = client.post(
            "/query",
            json={"question": "What bridges physical scans and embeddings?"},
        )
        assert query_resp.status_code == 200
        query_data = query_resp.json()
        assert "sources" in query_data
        assert len(query_data["sources"]) >= 1

        top_source = query_data["sources"][0]
        assert top_source["document_name"] == "scanned_doc.pdf"
        assert top_source["page_number"] == 2

    def test_mixed_pdf_correctly_attributes_native_and_ocr_pages(self, ocr_test_env):
        client, fake_engine, mock_gen = ocr_test_env
        fake_engine.page_texts = {
            1: (
                "Elliptic curve cryptography provides robust key exchange security "
                "for secure transport channels and cryptographic validation. " * 15
            )
        }

        mixed_bytes = _make_mixed_pdf_bytes()
        upload_resp = client.post(
            "/documents/upload",
            files={"file": ("mixed_doc.pdf", mixed_bytes, "application/pdf")},
        )
        assert upload_resp.status_code == 200
        upload_data = upload_resp.json()
        assert upload_data["num_pages"] == 2
        assert upload_data["ocr_pages"] == 1

        # Query native page 1
        q1_resp = client.post(
            "/query",
            json={"question": "How does quantum superposition help qubits?"},
        )
        assert q1_resp.status_code == 200
        assert q1_resp.json()["sources"][0]["page_number"] == 1

        # Query scanned page 2
        q2_resp = client.post(
            "/query",
            json={"question": "What provides robust key exchange security?"},
        )
        assert q2_resp.status_code == 200
        assert q2_resp.json()["sources"][0]["page_number"] == 2

    def test_direct_image_upload_and_query(self, ocr_test_env):
        client, fake_engine, mock_gen = ocr_test_env
        fake_engine.page_texts = {
            1: (
                "Invoice #9988 total amount due is $1,250 for consulting services "
                "rendered during the past billing period. " * 10
            )
        }

        # Create valid PNG image
        img = Image.new("RGB", (300, 200), color=(255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        upload_resp = client.post(
            "/documents/upload",
            files={"file": ("invoice.png", png_bytes, "image/png")},
        )
        assert upload_resp.status_code == 200
        data = upload_resp.json()
        assert data["document_name"] == "invoice.png"
        assert data["num_pages"] == 1
        assert data["ocr_pages"] == 1

        # Query text from image
        query_resp = client.post(
            "/query",
            json={"question": "What is the total amount due for consulting?"},
        )
        assert query_resp.status_code == 200
        top_source = query_resp.json()["sources"][0]
        assert top_source["document_name"] == "invoice.png"
        assert top_source["page_number"] == 1

    def test_relevance_gate_triggers_on_unrelated_query(self, ocr_test_env):
        client, fake_engine, mock_gen = ocr_test_env

        pdf_bytes = _make_scanned_pdf_bytes(page_count=1)
        client.post(
            "/documents/upload",
            files={"file": ("finance.pdf", pdf_bytes, "application/pdf")},
        )

        # Unrelated query with zero semantic overlap
        resp = client.post(
            "/query",
            json={
                "question": "What is the boiling point of liquid nitrogen in Kelvin?"
            },
        )
        assert resp.status_code == 404
        assert (
            "cannot generate an answer without supporting context"
            in resp.json()["detail"]
        )
        mock_gen.generate.assert_not_called()

    def test_api_status_mappings(self, tmp_path):
        """Verify API HTTP status codes across all failure modes."""
        settings = Settings(
            vector_store_path=str(tmp_path / "vectorstore"),
            vector_store_collection="test_status_codes",
            ocr_enabled=True,
        )
        client = TestClient(app)

        with (
            patch("backend.documents.get_settings", return_value=settings),
            patch("backend.query.get_settings", return_value=settings),
        ):
            # 1. Unsupported format .gif -> 415
            gif_bytes = b"GIF89a minimal gif"
            resp = client.post(
                "/documents/upload",
                files={"file": ("test.gif", gif_bytes, "image/gif")},
            )
            assert resp.status_code == 415

            # 2. Corrupt image -> 422
            corrupt_png = b"\x89PNG\r\n\x1a\ncorrupt body"
            resp = client.post(
                "/documents/upload",
                files={"file": ("corrupt.png", corrupt_png, "image/png")},
            )
            assert resp.status_code == 422

            # 3. Tesseract missing / OcrUnavailableError -> 503
            with patch(
                "backend.documents._build_ocr_processor",
                side_effect=OcrUnavailableError("Tesseract not installed"),
            ):
                img = Image.new("RGB", (50, 50), color=(255, 255, 255))
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                resp = client.post(
                    "/documents/upload",
                    files={"file": ("test.png", buf.getvalue(), "image/png")},
                )
                assert resp.status_code == 503

            # 4. Too many OCR pages -> 422
            fake_proc = MagicMock()
            fake_proc.ocr_pdf_page.side_effect = OcrLimitExceededError(
                "Page limit exceeded"
            )
            fake_proc.max_pages = 1
            fake_proc.ocr_min_native_chars = 25
            with patch(
                "backend.documents._build_ocr_processor", return_value=fake_proc
            ):
                pdf_bytes = _make_scanned_pdf_bytes(page_count=3)
                resp = client.post(
                    "/documents/upload",
                    files={"file": ("toolarge.pdf", pdf_bytes, "application/pdf")},
                )
                assert resp.status_code == 422
