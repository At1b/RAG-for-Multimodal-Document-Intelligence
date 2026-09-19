"""Integration tests for the Phase 6A orchestration layer.

These tests exercise real logic for cheaper components (ingestion,
chunking) while mocking expensive ones (embeddings, vector store, LLM).

Test 1 — Indexing integration:
    Real IngestionService + real chunk_document + mock IndexingService
    → verifies a real PDF flows through ingestion and chunking correctly.

Test 2 — Query integration:
    Mock Retriever returning real VectorSearchResult objects
    + mock Generator returning a real GenerationResult
    → verifies the query service correctly assembles the result.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rag.chunking.chunker import ChunkingConfig
from rag.generation.models import GenerationResult
from rag.orchestration.exceptions import (
    DocumentIndexingError,
    EmptyRetrievalError,
)
from rag.orchestration.indexing_service import DocumentIndexingService
from rag.orchestration.query_service import RAGQueryService
from rag.vectorstore.models import VectorSearchResult

# ---------------------------------------------------------------------------
# Indexing integration
# ---------------------------------------------------------------------------


class TestIndexingIntegration:
    """Integration test: real ingestion + real chunking + mock indexing."""

    def _create_test_pdf(self, tmp_dir: Path) -> Path:
        """Create a minimal valid PDF for testing."""
        # Import PyMuPDF to create a real PDF.
        pymupdf = pytest.importorskip("fitz", reason="PyMuPDF required")

        pdf_path = tmp_dir / "test_document.pdf"
        doc = pymupdf.open()
        page = doc.new_page()
        text = (
            "This is the first page of a test document for the MM-RAG system. "
            "It contains enough text to produce at least one chunk during the "
            "chunking phase. The content is deliberately long enough to exceed "
            "the minimum chunk size configured for this test."
        )
        page.insert_text((72, 72), text, fontsize=12)
        doc.save(str(pdf_path))
        doc.close()
        return pdf_path

    def test_real_pdf_ingestion_and_chunking(self):
        """A real PDF flows through ingestion and chunking to mock indexing."""
        from rag.ingestion.service import IngestionService

        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = self._create_test_pdf(Path(tmp_dir))

            # Real ingestion service.
            ingestion_service = IngestionService()

            # Small chunk size to guarantee multiple chunks.
            chunking_config = ChunkingConfig(chunk_size=50, chunk_overlap=10)

            # Mock indexing service — we don't want real embeddings/vector store.
            mock_indexing_service = MagicMock()
            mock_indexing_service.index_chunks.return_value = 0  # Will be updated

            # The mock should return the number of chunks passed to it.
            def capture_chunks(chunks):
                return len(chunks)

            mock_indexing_service.index_chunks.side_effect = capture_chunks

            service = DocumentIndexingService(
                ingestion_service=ingestion_service,
                chunking_config=chunking_config,
                indexing_service=mock_indexing_service,
            )

            result = service.index_document(
                pdf_path, document_name="integration_test.pdf"
            )

            # Verify result.
            assert result.document_name == "integration_test.pdf"
            assert result.num_pages >= 1
            assert result.num_chunks >= 1
            assert len(result.document_id) > 0

            # Verify the mock was called with real Chunk objects.
            mock_indexing_service.index_chunks.assert_called_once()
            actual_chunks = mock_indexing_service.index_chunks.call_args[0][0]
            assert len(actual_chunks) >= 1
            for chunk in actual_chunks:
                assert chunk.document_name == "integration_test.pdf"
                assert len(chunk.content) > 0
                assert chunk.document_id == result.document_id

    def test_invalid_file_raises_indexing_error(self):
        """An invalid file raises DocumentIndexingError through the real pipeline."""
        from rag.ingestion.service import IngestionService

        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_file = Path(tmp_dir) / "not_a_pdf.pdf"
            bad_file.write_bytes(b"this is not a real PDF file")

            ingestion_service = IngestionService()
            chunking_config = ChunkingConfig()
            mock_indexing_service = MagicMock()

            service = DocumentIndexingService(
                ingestion_service=ingestion_service,
                chunking_config=chunking_config,
                indexing_service=mock_indexing_service,
            )

            with pytest.raises(DocumentIndexingError, match="Ingestion failed"):
                service.index_document(bad_file)

            mock_indexing_service.index_chunks.assert_not_called()


# ---------------------------------------------------------------------------
# Query integration
# ---------------------------------------------------------------------------


class TestQueryIntegration:
    """Integration test: mock retriever + mock generator with real models."""

    def test_full_query_flow_with_real_models(self):
        """Query service correctly assembles result from mock components."""
        # Create realistic VectorSearchResult objects.
        results = [
            VectorSearchResult(
                chunk_id="chunk-001",
                document_id="doc-001",
                document_name="annual_report_2024.pdf",
                content="Revenue increased by 25% in fiscal year 2024.",
                score=0.92,
                metadata={"page_number": 38, "source_type": "pdf"},
            ),
            VectorSearchResult(
                chunk_id="chunk-002",
                document_id="doc-002",
                document_name="annual_report_2023.pdf",
                content="Revenue was $100 million in fiscal year 2023.",
                score=0.87,
                metadata={"page_number": 42, "source_type": "pdf"},
            ),
        ]

        generation_result = GenerationResult(
            answer="Revenue increased by 25% from $100M in 2023 to $125M in 2024.",
            model_name="tinyllama",
            metadata={"temperature": 0.1},
        )

        # Configure mocks.
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = results

        mock_generator = MagicMock()
        mock_generator.generate.return_value = generation_result

        service = RAGQueryService(
            retriever=mock_retriever,
            generator=mock_generator,
        )

        result = service.query(
            "How did revenue change between 2023 and 2024?",
            top_k=5,
        )

        # Verify result fields.
        assert result.answer == generation_result.answer
        assert result.model_name == "tinyllama"
        assert result.num_chunks_retrieved == 2
        assert result.metadata == {"temperature": 0.1}

        # Verify retriever was called correctly.
        mock_retriever.retrieve.assert_called_once_with(
            "How did revenue change between 2023 and 2024?",
            top_k=5,
        )

        # Verify generator received the question and results.
        mock_generator.generate.assert_called_once_with(
            "How did revenue change between 2023 and 2024?",
            results,
        )

    def test_empty_retrieval_blocks_generation(self):
        """Empty retrieval results prevent generator from being called."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []

        mock_generator = MagicMock()

        service = RAGQueryService(
            retriever=mock_retriever,
            generator=mock_generator,
        )

        with pytest.raises(EmptyRetrievalError):
            service.query("What is revenue?")

        mock_generator.generate.assert_not_called()

    def test_multi_document_retrieval(self):
        """Query service handles results from multiple documents."""
        results = [
            VectorSearchResult(
                chunk_id=f"chunk-{i}",
                document_id=f"doc-{i:03d}",
                document_name=f"document_{i}.pdf",
                content=f"Content from document {i}.",
                score=0.9 - i * 0.05,
            )
            for i in range(5)
        ]

        generation_result = GenerationResult(
            answer="Information synthesized from multiple documents.",
            model_name="tinyllama",
        )

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = results

        mock_generator = MagicMock()
        mock_generator.generate.return_value = generation_result

        service = RAGQueryService(
            retriever=mock_retriever,
            generator=mock_generator,
        )

        result = service.query("Compare across documents")

        assert result.num_chunks_retrieved == 5
        assert result.answer == "Information synthesized from multiple documents."
