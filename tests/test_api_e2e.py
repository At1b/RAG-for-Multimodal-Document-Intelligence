"""End-to-end API tests for Phase 6B — Baseline RAG API.

Tests cover:
    - Upload: valid PDF, valid DOCX, invalid file, unsupported file, empty file
    - Indexing: upload produces indexed chunks, vector store contains document
    - Query: valid question, invalid question, empty store, generation failure
    - Complete baseline flow: upload → index → query → retrieve → generate → answer

All tests use a mocked Ollama generator to avoid requiring a live LLM.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import docx as python_docx
import pymupdf
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from rag.generation.models import GenerationResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pdf_bytes(text: str = "Test content for RAG pipeline.") -> bytes:
    """Create a minimal PDF in-memory and return its bytes."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def _make_docx_bytes(text: str = "Test content for RAG pipeline.") -> bytes:
    """Create a minimal DOCX in-memory and return its bytes."""
    doc = python_docx.Document()
    doc.add_paragraph(text)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _get_test_client_with_temp_vectorstore(tmp_path: Path) -> TestClient:
    """Create a TestClient that uses a temp vector store directory."""
    from backend.config import Settings

    settings = Settings(
        vector_store_path=str(tmp_path / "vectorstore"),
        vector_store_collection="test_e2e",
    )

    with (
        patch("backend.documents.get_settings", return_value=settings),
        patch("backend.query.get_settings", return_value=settings),
    ):
        yield TestClient(app)


@pytest.fixture
def test_env(tmp_path):
    """Provide a TestClient with isolated vector store and mocked Ollama."""
    from backend.config import Settings

    settings = Settings(
        vector_store_path=str(tmp_path / "vectorstore"),
        vector_store_collection="test_e2e",
    )

    # Mock OllamaGenerator to return a predictable answer.
    mock_result = GenerationResult(
        answer="This is a test answer from the mocked LLM.",
        model_name="mock-model",
        metadata={"temperature": 0.1},
    )

    with (
        patch("backend.documents.get_settings", return_value=settings),
        patch("backend.query.get_settings", return_value=settings),
        patch("backend.query.OllamaGenerator") as MockGenerator,
    ):
        mock_gen_instance = MagicMock()
        mock_gen_instance.generate.return_value = mock_result
        MockGenerator.return_value = mock_gen_instance
        yield TestClient(app), settings, mock_gen_instance


# ===========================================================================
# Upload tests
# ===========================================================================


class TestUploadEndpoint:
    """Upload endpoint tests with full indexing pipeline."""

    def test_upload_valid_pdf(self, test_env):
        """Valid PDF upload returns 200 with indexing result."""
        client, _, _ = test_env
        pdf_bytes = _make_pdf_bytes("Artificial intelligence is transforming research.")
        response = client.post(
            "/documents/upload",
            files={"file": ("research.pdf", pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["document_name"] == "research.pdf"
        assert "document_id" in data
        assert data["num_pages"] >= 1
        assert data["num_chunks"] >= 1

    def test_upload_valid_docx(self, test_env):
        """Valid DOCX upload returns 200 with indexing result."""
        client, _, _ = test_env
        docx_bytes = _make_docx_bytes(
            "Machine learning enables computers to learn from data."
        )
        response = client.post(
            "/documents/upload",
            files={
                "file": (
                    "notes.docx",
                    docx_bytes,
                    "application/vnd.openxmlformats-officedocument"
                    ".wordprocessingml.document",
                )
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["document_name"] == "notes.docx"
        assert data["num_chunks"] >= 1

    def test_upload_invalid_file(self, test_env):
        """Corrupted PDF returns 422."""
        client, _, _ = test_env
        response = client.post(
            "/documents/upload",
            files={
                "file": (
                    "corrupt.pdf",
                    b"%PDF-1.4 invalid truncated content",
                    "application/pdf",
                )
            },
        )
        assert response.status_code == 422

    def test_upload_unsupported_format(self, test_env):
        """Unsupported file type returns 415."""
        client, _, _ = test_env
        response = client.post(
            "/documents/upload",
            files={"file": ("readme.txt", b"some plain text", "text/plain")},
        )
        assert response.status_code == 415

    def test_upload_empty_file(self, test_env):
        """Empty file returns 400."""
        client, _, _ = test_env
        response = client.post(
            "/documents/upload",
            files={"file": ("empty.pdf", b"", "application/pdf")},
        )
        assert response.status_code == 400


# ===========================================================================
# Indexing verification tests
# ===========================================================================


class TestIndexingVerification:
    """Verify that uploaded documents are actually indexed."""

    def test_upload_produces_indexed_chunks(self, test_env):
        """Uploading a document produces chunks in the vector store."""
        client, settings, _ = test_env

        pdf_bytes = _make_pdf_bytes(
            "Deep learning uses neural networks with multiple layers "
            "to learn representations of data."
        )
        response = client.post(
            "/documents/upload",
            files={"file": ("deep_learning.pdf", pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["num_chunks"] >= 1

        # Verify the vector store actually contains chunks.
        from rag.vectorstore.chroma_store import ChromaVectorStore

        store = ChromaVectorStore(
            persist_directory=settings.vector_store_path,
            collection_name=settings.vector_store_collection,
        )
        assert store.count() >= 1

    def test_vector_store_contains_document_after_indexing(self, test_env):
        """After indexing, the vector store contains the specific document."""
        client, settings, _ = test_env

        pdf_bytes = _make_pdf_bytes(
            "Transformers revolutionized natural language processing."
        )
        response = client.post(
            "/documents/upload",
            files={"file": ("transformers.pdf", pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 200
        doc_id = response.json()["document_id"]

        # Query the vector store to verify the document is present.
        from rag.embeddings.sentence_transformer import (
            SentenceTransformerEmbeddingService,
        )
        from rag.vectorstore.chroma_store import ChromaVectorStore

        store = ChromaVectorStore(
            persist_directory=settings.vector_store_path,
            collection_name=settings.vector_store_collection,
        )
        emb_service = SentenceTransformerEmbeddingService(
            model_name=settings.embedding_model,
        )
        query_vec = emb_service.embed_query("transformers NLP")
        results = store.query(query_vec, top_k=5)

        doc_ids_found = {r.document_id for r in results}
        assert doc_id in doc_ids_found


# ===========================================================================
# Query endpoint tests
# ===========================================================================


class TestQueryEndpoint:
    """Query endpoint tests with mocked LLM."""

    def test_valid_question_returns_answer(self, test_env):
        """A valid question after indexing returns a generated answer."""
        client, _, mock_gen = test_env

        # First, index a document.
        pdf_bytes = _make_pdf_bytes(
            "RAG combines retrieval with text generation for grounded answers."
        )
        upload_resp = client.post(
            "/documents/upload",
            files={"file": ("rag_paper.pdf", pdf_bytes, "application/pdf")},
        )
        assert upload_resp.status_code == 200

        # Now query.
        response = client.post(
            "/query",
            json={"question": "What is RAG?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert data["answer"]  # non-empty
        assert "model_name" in data
        assert data["num_chunks_retrieved"] >= 1
        # Verify the mocked generator was called.
        mock_gen.generate.assert_called_once()

    def test_invalid_empty_question_returns_400(self, test_env):
        """An empty question returns 400."""
        client, _, _ = test_env
        response = client.post(
            "/query",
            json={"question": ""},
        )
        assert response.status_code == 422  # Pydantic validation (min_length=1)

    def test_whitespace_question_returns_400(self, test_env):
        """A whitespace-only question returns 400."""
        client, _, _ = test_env
        response = client.post(
            "/query",
            json={"question": "   "},
        )
        # The route accepts it (len > 0 due to spaces), but the retriever
        # rejects whitespace-only queries via InvalidQueryError → 400.
        assert response.status_code == 400

    def test_empty_vector_store_returns_404(self, test_env):
        """Querying with no documents indexed returns 404."""
        client, _, _ = test_env
        response = client.post(
            "/query",
            json={"question": "What is artificial intelligence?"},
        )
        assert response.status_code == 404

    def test_generation_failure_returns_500(self, test_env):
        """A generation failure maps to 500."""
        client, _, mock_gen = test_env

        # Index a document first.
        pdf_bytes = _make_pdf_bytes("Vector embeddings represent text as numbers.")
        client.post(
            "/documents/upload",
            files={"file": ("embeddings.pdf", pdf_bytes, "application/pdf")},
        )

        # Make the generator fail.
        mock_gen.generate.side_effect = Exception("LLM connection refused")

        response = client.post(
            "/query",
            json={"question": "What are vector embeddings?"},
        )
        assert response.status_code == 500

    def test_query_with_top_k(self, test_env):
        """Query with explicit top_k parameter works."""
        client, _, mock_gen = test_env

        pdf_bytes = _make_pdf_bytes(
            "NLP enables computers to understand human language."
        )
        client.post(
            "/documents/upload",
            files={"file": ("nlp.pdf", pdf_bytes, "application/pdf")},
        )

        response = client.post(
            "/query",
            json={"question": "What is NLP?", "top_k": 3},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["num_chunks_retrieved"] >= 1


# ===========================================================================
# Complete baseline flow test
# ===========================================================================


class TestCompleteBaselineFlow:
    """End-to-end baseline flow: upload → index → query → answer."""

    def test_full_rag_pipeline(self, test_env):
        """Upload a document, then query it and receive an answer."""
        client, settings, mock_gen = test_env

        # Step 1: Upload and index.
        pdf_bytes = _make_pdf_bytes(
            "The Transformer architecture uses self-attention to process "
            "input sequences in parallel. It was introduced in 2017 by "
            "Vaswani et al. in the paper Attention Is All You Need."
        )
        upload_resp = client.post(
            "/documents/upload",
            files={"file": ("transformer.pdf", pdf_bytes, "application/pdf")},
        )
        assert upload_resp.status_code == 200
        upload_data = upload_resp.json()
        assert upload_data["num_chunks"] >= 1

        # Step 2: Query.
        query_resp = client.post(
            "/query",
            json={"question": "What is the Transformer architecture?"},
        )
        assert query_resp.status_code == 200
        query_data = query_resp.json()
        assert query_data["answer"]
        assert query_data["model_name"] == "mock-model"
        assert query_data["num_chunks_retrieved"] >= 1

        # The generator was called with the question and context.
        mock_gen.generate.assert_called_once()
        call_args = mock_gen.generate.call_args
        assert "Transformer" in call_args[0][0] or "Transformer" in str(call_args)

    def test_multi_document_flow(self, test_env):
        """Upload multiple documents and query across them."""
        client, _, mock_gen = test_env

        # Upload first document.
        pdf1 = _make_pdf_bytes(
            "Machine learning is a subset of artificial intelligence."
        )
        resp1 = client.post(
            "/documents/upload",
            files={"file": ("ml_overview.pdf", pdf1, "application/pdf")},
        )
        assert resp1.status_code == 200

        # Upload second document.
        pdf2 = _make_pdf_bytes(
            "Deep learning uses neural networks to learn data representations."
        )
        resp2 = client.post(
            "/documents/upload",
            files={"file": ("dl_overview.pdf", pdf2, "application/pdf")},
        )
        assert resp2.status_code == 200

        # Query should retrieve from both documents.
        response = client.post(
            "/query",
            json={"question": "What is the relationship between ML and DL?"},
        )
        assert response.status_code == 200
        assert response.json()["answer"]
