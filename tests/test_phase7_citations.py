"""Phase 7 — Multi-Document Support and Citations tests.

Comprehensive test suite covering:
  1. Source/citation model validation.
  2. Citation formatter (format_sources).
  3. Multi-document indexing.
  4. Document identity preservation in chunk metadata.
  5. Cross-document retrieval.
  6. Citation integrity (no fabrication).
  7. QueryResult sources integration.
  8. API response sources.
  9. Conflicting information across documents.
 10. Phase 6 backward compatibility.
 11. Relevance gate still works.
 12. Missing page metadata handling.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from rag.chunking.models import Chunk
from rag.citations.formatter import _extract_page_number, format_sources
from rag.citations.models import Source
from rag.generation.models import GenerationResult
from rag.orchestration.exceptions import InsufficientContextError
from rag.orchestration.query_service import QueryResult, RAGQueryService
from rag.vectorstore.models import VectorSearchResult

# ==========================================================================
# Helpers / Fixtures
# ==========================================================================


def _make_chunk(
    document_id: str = "doc-001",
    document_name: str = "report.pdf",
    page_number: int | None = 1,
    chunk_index: int = 0,
    content: str = "Some chunk content.",
    source_type: str = "pdf",
) -> Chunk:
    """Create a test Chunk with realistic metadata."""
    return Chunk(
        chunk_id=str(uuid.uuid4()),
        document_id=document_id,
        document_name=document_name,
        source_type=source_type,
        content=content,
        page_number=page_number,
        chunk_index=chunk_index,
        metadata={},
    )


def _make_result(
    chunk_id: str = "chunk-001",
    document_id: str = "doc-001",
    document_name: str = "report.pdf",
    page_number: int | None = 1,
    score: float = 0.85,
    content: str = "Some content.",
) -> VectorSearchResult:
    """Create a test VectorSearchResult."""
    meta = {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "document_name": document_name,
        "source_type": "pdf",
        "chunk_index": 0,
    }
    if page_number is not None:
        meta["page_number"] = page_number
    return VectorSearchResult(
        chunk_id=chunk_id,
        document_id=document_id,
        document_name=document_name,
        content=content,
        score=score,
        metadata=meta,
    )


# ==========================================================================
# 1. Source Model Tests
# ==========================================================================


class TestSourceModel:
    """Tests for the Source Pydantic model."""

    def test_source_with_all_fields(self):
        s = Source(
            document_id="doc-001",
            document_name="report.pdf",
            page_number=5,
            chunk_id="chunk-001",
            score=0.85,
        )
        assert s.document_id == "doc-001"
        assert s.document_name == "report.pdf"
        assert s.page_number == 5
        assert s.chunk_id == "chunk-001"
        assert s.score == 0.85

    def test_source_page_number_none(self):
        """page_number should accept None for missing page info."""
        s = Source(
            document_id="doc-002",
            document_name="notes.docx",
            page_number=None,
            chunk_id="chunk-002",
        )
        assert s.page_number is None

    def test_source_defaults(self):
        """score defaults to 0.0, page_number defaults to None."""
        s = Source(
            document_id="doc-001",
            document_name="report.pdf",
            chunk_id="chunk-001",
        )
        assert s.page_number is None
        assert s.score == 0.0
        assert s.metadata == {}

    def test_source_requires_document_id(self):
        with pytest.raises(ValidationError):
            Source(document_name="a.pdf", chunk_id="c-1")

    def test_source_requires_document_name(self):
        with pytest.raises(ValidationError):
            Source(document_id="d-1", chunk_id="c-1")

    def test_source_requires_chunk_id(self):
        with pytest.raises(ValidationError):
            Source(document_id="d-1", document_name="a.pdf")

    def test_source_serialization(self):
        """Source should serialize to dict with all fields."""
        s = Source(
            document_id="doc-001",
            document_name="report.pdf",
            page_number=3,
            chunk_id="chunk-001",
            score=0.92,
        )
        data = s.model_dump()
        assert data["document_id"] == "doc-001"
        assert data["page_number"] == 3
        assert data["score"] == 0.92

    def test_source_with_metadata(self):
        s = Source(
            document_id="doc-001",
            document_name="report.pdf",
            chunk_id="chunk-001",
            metadata={"source_type": "pdf", "chunk_index": 0},
        )
        assert s.metadata["source_type"] == "pdf"


# ==========================================================================
# 2. Citation Formatter Tests
# ==========================================================================


class TestFormatSources:
    """Tests for the format_sources function."""

    def test_empty_results(self):
        """Empty results should produce empty sources list."""
        sources = format_sources([])
        assert sources == []

    def test_single_result(self):
        result = _make_result()
        sources = format_sources([result])
        assert len(sources) == 1
        assert sources[0].document_id == "doc-001"
        assert sources[0].document_name == "report.pdf"
        assert sources[0].chunk_id == "chunk-001"
        assert sources[0].page_number == 1
        assert sources[0].score == 0.85

    def test_multiple_results_from_different_documents(self):
        r1 = _make_result(
            chunk_id="c-1",
            document_id="d-1",
            document_name="doc_a.pdf",
            page_number=5,
            score=0.9,
        )
        r2 = _make_result(
            chunk_id="c-2",
            document_id="d-2",
            document_name="doc_b.pdf",
            page_number=10,
            score=0.7,
        )
        sources = format_sources([r1, r2])
        assert len(sources) == 2
        assert sources[0].document_name == "doc_a.pdf"
        assert sources[1].document_name == "doc_b.pdf"

    def test_preserves_ordering(self):
        """Sources should preserve retrieval ordering (most relevant first)."""
        r1 = _make_result(chunk_id="c-1", score=0.95)
        r2 = _make_result(chunk_id="c-2", score=0.75)
        r3 = _make_result(chunk_id="c-3", score=0.55)
        sources = format_sources([r1, r2, r3])
        scores = [s.score for s in sources]
        assert scores == [0.95, 0.75, 0.55]

    def test_deduplicates_by_chunk_id(self):
        """Duplicate chunk_ids should be deduplicated (first wins)."""
        r1 = _make_result(chunk_id="same-id", score=0.9)
        r2 = _make_result(chunk_id="same-id", score=0.5)
        sources = format_sources([r1, r2])
        assert len(sources) == 1
        assert sources[0].score == 0.9

    def test_missing_page_number(self):
        """page_number should be None when metadata lacks it."""
        result = _make_result(page_number=None)
        sources = format_sources([result])
        assert sources[0].page_number is None

    def test_page_number_zero_excluded(self):
        """page_number=0 in metadata should be treated as None (invalid page)."""
        result = _make_result()
        result.metadata["page_number"] = 0
        sources = format_sources([result])
        assert sources[0].page_number is None

    def test_non_list_input_raises_type_error(self):
        with pytest.raises(TypeError, match="list"):
            format_sources("not a list")

    def test_non_vector_search_result_raises_type_error(self):
        with pytest.raises(TypeError, match="VectorSearchResult"):
            format_sources([{"chunk_id": "c-1"}])

    def test_metadata_passthrough(self):
        """Chunk metadata should be available in Source.metadata."""
        result = _make_result()
        result.metadata["custom_field"] = "custom_value"
        sources = format_sources([result])
        assert sources[0].metadata.get("custom_field") == "custom_value"


class TestExtractPageNumber:
    """Tests for the internal _extract_page_number helper."""

    def test_valid_integer(self):
        assert _extract_page_number({"page_number": 5}) == 5

    def test_none_value(self):
        assert _extract_page_number({"page_number": None}) is None

    def test_missing_key(self):
        assert _extract_page_number({}) is None

    def test_zero(self):
        assert _extract_page_number({"page_number": 0}) is None

    def test_negative(self):
        assert _extract_page_number({"page_number": -1}) is None

    def test_float_integer_value(self):
        """3.0 from JSON should be accepted as page 3."""
        assert _extract_page_number({"page_number": 3.0}) == 3

    def test_float_non_integer_value(self):
        """3.5 is not a valid page number."""
        assert _extract_page_number({"page_number": 3.5}) is None

    def test_boolean_value(self):
        """Boolean should never be treated as page number."""
        assert _extract_page_number({"page_number": True}) is None

    def test_string_value(self):
        """String page number should return None."""
        assert _extract_page_number({"page_number": "5"}) is None

    def test_page_key_fallback(self):
        """Fallback to page key when page_number is not present."""
        assert _extract_page_number({"page": 4}) == 4
        assert _extract_page_number({"page": None}) is None
        assert _extract_page_number({"page": 0}) is None


# ==========================================================================
# 3. Multi-Document Indexing Tests
# ==========================================================================


class TestMultiDocumentChunkMetadata:
    """Tests for document identity preservation in chunk metadata."""

    def test_different_document_ids_remain_distinct(self):
        """Chunks from different documents should have distinct document_ids."""
        chunk_a = _make_chunk(
            document_id="doc-alpha",
            document_name="alpha.pdf",
            page_number=1,
        )
        chunk_b = _make_chunk(
            document_id="doc-beta",
            document_name="beta.pdf",
            page_number=3,
        )
        assert chunk_a.document_id != chunk_b.document_id
        assert chunk_a.document_name != chunk_b.document_name

    def test_chunk_metadata_preserves_document_identity(self):
        """Chunk should preserve all document identity fields."""
        chunk = _make_chunk(
            document_id="doc-test",
            document_name="test.pdf",
            page_number=7,
        )
        assert chunk.document_id == "doc-test"
        assert chunk.document_name == "test.pdf"
        assert chunk.page_number == 7

    def test_chunk_metadata_preserves_source_type(self):
        chunk = _make_chunk(source_type="docx")
        assert chunk.source_type == "docx"

    def test_multiple_chunks_from_same_document(self):
        """Multiple chunks from the same document share document_id."""
        doc_id = "doc-shared"
        c1 = _make_chunk(document_id=doc_id, chunk_index=0)
        c2 = _make_chunk(document_id=doc_id, chunk_index=1)
        assert c1.document_id == c2.document_id
        assert c1.chunk_id != c2.chunk_id


# ==========================================================================
# 4. ChromaDB Metadata Round-Trip Tests
# ==========================================================================


class TestChromaMetadataRoundTrip:
    """Verify ChromaDB preserves document identity through storage."""

    def test_chunk_to_metadata_preserves_document_id(self):
        from rag.vectorstore.chroma_store import ChromaVectorStore

        chunk = _make_chunk(document_id="doc-rt", document_name="roundtrip.pdf")
        meta = ChromaVectorStore._chunk_to_metadata(chunk)
        assert meta["document_id"] == "doc-rt"
        assert meta["document_name"] == "roundtrip.pdf"
        assert meta["chunk_id"] == chunk.chunk_id

    def test_chunk_to_metadata_with_page_number(self):
        from rag.vectorstore.chroma_store import ChromaVectorStore

        chunk = _make_chunk(page_number=42)
        meta = ChromaVectorStore._chunk_to_metadata(chunk)
        assert meta["page_number"] == 42

    def test_chunk_to_metadata_without_page_number(self):
        from rag.vectorstore.chroma_store import ChromaVectorStore

        chunk = _make_chunk(page_number=None)
        meta = ChromaVectorStore._chunk_to_metadata(chunk)
        assert "page_number" not in meta


# ==========================================================================
# 5. Cross-Document Retrieval Tests
# ==========================================================================


class TestCrossDocumentRetrieval:
    """Tests for retrieval returning chunks from multiple documents."""

    def test_retrieval_results_from_multiple_documents(self):
        """VectorSearchResults from different documents are identifiable."""
        r_a = _make_result(
            chunk_id="c-a",
            document_id="doc-a",
            document_name="country_alpha.pdf",
            page_number=1,
        )
        r_b = _make_result(
            chunk_id="c-b",
            document_id="doc-b",
            document_name="country_beta.pdf",
            page_number=2,
        )
        results = [r_a, r_b]
        sources = format_sources(results)
        doc_names = {s.document_name for s in sources}
        assert "country_alpha.pdf" in doc_names
        assert "country_beta.pdf" in doc_names

    def test_retrieved_chunks_preserve_document_name(self):
        result = _make_result(document_name="specific_doc.pdf")
        sources = format_sources([result])
        assert sources[0].document_name == "specific_doc.pdf"

    def test_retrieved_chunks_preserve_page_number(self):
        result = _make_result(page_number=42)
        sources = format_sources([result])
        assert sources[0].page_number == 42

    def test_retrieved_chunks_preserve_chunk_id(self):
        result = _make_result(chunk_id="unique-chunk-id-123")
        sources = format_sources([result])
        assert sources[0].chunk_id == "unique-chunk-id-123"


# ==========================================================================
# 6. Citation Integrity Tests (No Fabrication)
# ==========================================================================


class TestCitationIntegrity:
    """Ensure citations come from actual metadata, never fabricated."""

    def test_source_uses_actual_document_name(self):
        """Source document_name must come from result, not invented."""
        result = _make_result(document_name="actual_file.pdf")
        sources = format_sources([result])
        assert sources[0].document_name == "actual_file.pdf"
        assert "invented" not in sources[0].document_name

    def test_source_page_number_from_metadata_only(self):
        """If metadata has no page_number, source page should be None."""
        result = _make_result(page_number=None)
        sources = format_sources([result])
        assert sources[0].page_number is None

    def test_no_fabrication_possible_through_normal_path(self):
        """format_sources should only use data from the VectorSearchResult."""
        result = _make_result(
            chunk_id="c-real",
            document_id="d-real",
            document_name="real.pdf",
            page_number=7,
            score=0.8,
        )
        sources = format_sources([result])
        s = sources[0]
        # All fields must match the input result exactly.
        assert s.chunk_id == result.chunk_id
        assert s.document_id == result.document_id
        assert s.document_name == result.document_name
        assert s.page_number == 7
        assert s.score == result.score


# ==========================================================================
# 7. QueryResult Sources Integration Tests
# ==========================================================================


class TestQueryResultSources:
    """Tests for QueryResult including sources."""

    def test_query_result_has_sources_field(self):
        qr = QueryResult(
            answer="Test answer",
            model_name="test-model",
            num_chunks_retrieved=1,
            sources=[],
        )
        assert isinstance(qr.sources, list)
        assert len(qr.sources) == 0

    def test_query_result_sources_default_empty(self):
        """Sources should default to empty list (backward compatible)."""
        qr = QueryResult(
            answer="Test answer",
            model_name="test-model",
            num_chunks_retrieved=0,
        )
        assert qr.sources == []

    def test_query_result_with_sources(self):
        source = Source(
            document_id="d-1",
            document_name="file.pdf",
            page_number=3,
            chunk_id="c-1",
            score=0.9,
        )
        qr = QueryResult(
            answer="Test answer",
            model_name="test-model",
            num_chunks_retrieved=1,
            sources=[source],
        )
        assert len(qr.sources) == 1
        assert qr.sources[0].document_name == "file.pdf"

    def test_query_result_serializes_sources(self):
        source = Source(
            document_id="d-1",
            document_name="file.pdf",
            chunk_id="c-1",
            score=0.9,
        )
        qr = QueryResult(
            answer="Test answer",
            model_name="test-model",
            num_chunks_retrieved=1,
            sources=[source],
        )
        data = qr.model_dump()
        assert "sources" in data
        assert data["sources"][0]["document_name"] == "file.pdf"


# ==========================================================================
# 8. RAGQueryService Sources Integration Tests
# ==========================================================================


class TestRAGQueryServiceSources:
    """Tests for RAGQueryService producing sources in QueryResult."""

    def _build_service(self, results, answer="Test answer"):
        """Build a RAGQueryService with mocked retriever and generator."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = results
        mock_generator = MagicMock()
        mock_generator.generate.return_value = GenerationResult(
            answer=answer,
            model_name="test-model",
        )
        return RAGQueryService(
            retriever=mock_retriever,
            generator=mock_generator,
        )

    def test_query_returns_sources(self):
        results = [
            _make_result(
                chunk_id="c-1", document_id="d-1", document_name="a.pdf", page_number=1
            ),
            _make_result(
                chunk_id="c-2", document_id="d-2", document_name="b.pdf", page_number=5
            ),
        ]
        service = self._build_service(results)
        qr = service.query("What is X?")
        assert len(qr.sources) == 2
        assert qr.sources[0].document_name == "a.pdf"
        assert qr.sources[1].document_name == "b.pdf"

    def test_query_sources_from_metadata_not_llm(self):
        """Sources must come from retrieval metadata, not LLM output."""
        results = [_make_result(document_name="real_doc.pdf", page_number=10)]
        service = self._build_service(results, answer="LLM says page 999")
        qr = service.query("test?")
        # Source must have the real page number from metadata.
        assert qr.sources[0].page_number == 10
        assert qr.sources[0].document_name == "real_doc.pdf"

    def test_query_preserves_backward_compatibility(self):
        """Existing Phase 6 fields should still work."""
        results = [_make_result()]
        service = self._build_service(results)
        qr = service.query("test?")
        assert hasattr(qr, "answer")
        assert hasattr(qr, "model_name")
        assert hasattr(qr, "num_chunks_retrieved")

    def test_empty_retrieval_raises_insufficient_context(self):
        """Phase 6 relevance gate behavior must be preserved."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        mock_generator = MagicMock()
        service = RAGQueryService(
            retriever=mock_retriever,
            generator=mock_generator,
        )
        with pytest.raises(InsufficientContextError):
            service.query("irrelevant question?")
        mock_generator.generate.assert_not_called()


# ==========================================================================
# 9. Conflicting Information Tests
# ==========================================================================


class TestConflictingInformation:
    """Tests for conflicting information across documents."""

    def test_conflicting_docs_both_retrievable(self):
        """When two documents conflict, both should be retrievable."""
        r_a = _make_result(
            chunk_id="c-a",
            document_id="doc-a",
            document_name="country_alpha.pdf",
            content="The capital of Freedonia is Alpha City.",
            page_number=1,
            score=0.9,
        )
        r_b = _make_result(
            chunk_id="c-b",
            document_id="doc-b",
            document_name="country_beta.pdf",
            content="The capital of Freedonia is Beta Town.",
            page_number=2,
            score=0.85,
        )
        sources = format_sources([r_a, r_b])
        assert len(sources) == 2
        doc_ids = {s.document_id for s in sources}
        assert "doc-a" in doc_ids
        assert "doc-b" in doc_ids

    def test_conflicting_docs_preserve_document_boundaries(self):
        """Conflicting documents must remain identifiable."""
        r_a = _make_result(
            chunk_id="c-a",
            document_id="doc-alpha",
            document_name="version_1.pdf",
        )
        r_b = _make_result(
            chunk_id="c-b",
            document_id="doc-beta",
            document_name="version_2.pdf",
        )
        sources = format_sources([r_a, r_b])
        assert sources[0].document_id != sources[1].document_id
        assert sources[0].document_name != sources[1].document_name

    def test_conflicting_docs_in_query_service(self):
        """RAGQueryService should return sources from both conflicting docs."""
        results = [
            _make_result(
                chunk_id="c-1",
                document_id="d-old",
                document_name="policy_2023.pdf",
                content="Retirement age is 65.",
            ),
            _make_result(
                chunk_id="c-2",
                document_id="d-new",
                document_name="policy_2024.pdf",
                content="Retirement age is 67.",
            ),
        ]
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = results
        mock_generator = MagicMock()
        mock_generator.generate.return_value = GenerationResult(
            answer="Both documents mention retirement age.",
            model_name="test-model",
        )
        service = RAGQueryService(
            retriever=mock_retriever,
            generator=mock_generator,
        )
        qr = service.query("What is the retirement age?")
        assert len(qr.sources) == 2
        doc_names = {s.document_name for s in qr.sources}
        assert "policy_2023.pdf" in doc_names
        assert "policy_2024.pdf" in doc_names


# ==========================================================================
# 10. API Response Sources Tests
# ==========================================================================


class TestAPIResponseSources:
    """Tests for the /query API response including sources."""

    def test_query_response_model_has_sources(self):
        from backend.query import QueryResponse, SourceResponse

        resp = QueryResponse(
            answer="test",
            model_name="m",
            num_chunks_retrieved=1,
            sources=[
                SourceResponse(
                    document_id="d-1",
                    document_name="file.pdf",
                    page_number=3,
                    chunk_id="c-1",
                    score=0.9,
                )
            ],
        )
        assert len(resp.sources) == 1
        assert resp.sources[0].document_name == "file.pdf"
        assert resp.sources[0].page_number == 3

    def test_query_response_sources_default_empty(self):
        from backend.query import QueryResponse

        resp = QueryResponse(
            answer="test",
            model_name="m",
            num_chunks_retrieved=0,
        )
        assert resp.sources == []

    def test_source_response_page_number_optional(self):
        from backend.query import SourceResponse

        sr = SourceResponse(
            document_id="d-1",
            document_name="file.pdf",
            page_number=None,
            chunk_id="c-1",
            score=0.8,
        )
        assert sr.page_number is None

    def test_source_response_serialization(self):
        from backend.query import SourceResponse

        sr = SourceResponse(
            document_id="d-1",
            document_name="file.pdf",
            page_number=5,
            chunk_id="c-1",
            score=0.85,
        )
        data = sr.model_dump()
        assert data["document_name"] == "file.pdf"
        assert data["page_number"] == 5


# ==========================================================================
# 11. Phase 6 Backward Compatibility Tests
# ==========================================================================


class TestPhase6BackwardCompatibility:
    """Ensure existing Phase 6 behavior is not broken."""

    def test_query_result_without_sources_still_works(self):
        """QueryResult should still work without explicitly passing sources."""
        qr = QueryResult(
            answer="Backward compatible answer",
            model_name="test-model",
            num_chunks_retrieved=1,
        )
        assert qr.answer == "Backward compatible answer"
        assert qr.sources == []

    def test_query_result_metadata_still_works(self):
        qr = QueryResult(
            answer="Answer",
            model_name="model",
            num_chunks_retrieved=1,
            metadata={"temperature": 0.1},
        )
        assert qr.metadata["temperature"] == 0.1

    def test_relevance_gate_still_prevents_unrelated_answers(self):
        """Phase 6 relevance gate should still work."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        mock_generator = MagicMock()
        service = RAGQueryService(
            retriever=mock_retriever,
            generator=mock_generator,
        )
        with pytest.raises(InsufficientContextError):
            service.query("unrelated question")
        mock_generator.generate.assert_not_called()


# ==========================================================================
# 12. Integration Test: Multi-Doc with Real ChromaDB + Embeddings
# ==========================================================================


class TestMultiDocumentIntegration:
    """Integration tests with real ChromaDB and embeddings.

    These tests use isolated temporary vector stores to avoid
    affecting the persistent production data.
    """

    @pytest.fixture
    def embedding_service(self):
        from rag.embeddings.sentence_transformer import (
            SentenceTransformerEmbeddingService,
        )

        return SentenceTransformerEmbeddingService(
            model_name="all-MiniLM-L6-v2",
            batch_size=32,
        )

    @pytest.fixture
    def vector_store(self, tmp_path):
        from rag.vectorstore.chroma_store import ChromaVectorStore

        return ChromaVectorStore(
            persist_directory=str(tmp_path / "test_vectorstore"),
            collection_name="test_phase7",
        )

    def _index_chunks(self, chunks, embedding_service, vector_store):
        """Helper to embed and store chunks."""
        texts = [c.content for c in chunks]
        embeddings = embedding_service.embed_documents(texts)
        vector_store.add_chunks(chunks, embeddings)

    def test_multiple_documents_can_be_indexed(
        self,
        embedding_service,
        vector_store,
    ):
        """Multiple documents should be independently indexable."""
        chunk_a = _make_chunk(
            document_id="doc-alpha",
            document_name="alpha.pdf",
            content="The capital of Country Alpha is City One.",
        )
        chunk_b = _make_chunk(
            document_id="doc-beta",
            document_name="beta.pdf",
            content="The capital of Country Beta is City Two.",
        )
        self._index_chunks([chunk_a, chunk_b], embedding_service, vector_store)
        assert vector_store.count() == 2

    def test_cross_document_retrieval(
        self,
        embedding_service,
        vector_store,
    ):
        """A query should retrieve from multiple documents."""
        chunk_a = _make_chunk(
            document_id="doc-alpha",
            document_name="alpha.pdf",
            content="Machine learning is a subset of artificial intelligence.",
            page_number=1,
        )
        chunk_b = _make_chunk(
            document_id="doc-beta",
            document_name="beta.pdf",
            content="Deep learning uses neural networks for machine learning tasks.",
            page_number=5,
        )
        chunk_c = _make_chunk(
            document_id="doc-gamma",
            document_name="gamma.pdf",
            content="Cooking pasta requires boiling water and salt.",
            page_number=1,
        )
        self._index_chunks(
            [chunk_a, chunk_b, chunk_c],
            embedding_service,
            vector_store,
        )

        from rag.retrieval.semantic import SemanticRetriever

        retriever = SemanticRetriever(
            embedding_service=embedding_service,
            vector_store=vector_store,
            default_top_k=3,
        )
        results = retriever.retrieve("What is machine learning?")
        assert len(results) >= 2

        sources = format_sources(results)
        doc_names = {s.document_name for s in sources}
        # Both ML-related docs should appear in sources.
        assert "alpha.pdf" in doc_names
        assert "beta.pdf" in doc_names

    def test_document_identity_survives_storage_and_retrieval(
        self,
        embedding_service,
        vector_store,
    ):
        """document_id, document_name, page_number should survive storage."""
        chunk = _make_chunk(
            document_id="doc-roundtrip",
            document_name="roundtrip.pdf",
            page_number=42,
            content="This is a round-trip test for metadata preservation.",
        )
        self._index_chunks([chunk], embedding_service, vector_store)

        from rag.retrieval.semantic import SemanticRetriever

        retriever = SemanticRetriever(
            embedding_service=embedding_service,
            vector_store=vector_store,
            default_top_k=1,
        )
        results = retriever.retrieve("round-trip test metadata")
        assert len(results) == 1
        r = results[0]
        assert r.document_id == "doc-roundtrip"
        assert r.document_name == "roundtrip.pdf"
        assert r.metadata.get("page_number") == 42

    def test_conflicting_documents_both_retrieved(
        self,
        embedding_service,
        vector_store,
    ):
        """Two docs with conflicting info should both be retrievable."""
        chunk_a = _make_chunk(
            document_id="doc-old",
            document_name="policy_2023.pdf",
            content="The retirement age in Freedonia is 65 years.",
            page_number=10,
        )
        chunk_b = _make_chunk(
            document_id="doc-new",
            document_name="policy_2024.pdf",
            content="The retirement age in Freedonia is 67 years.",
            page_number=15,
        )
        self._index_chunks(
            [chunk_a, chunk_b],
            embedding_service,
            vector_store,
        )

        from rag.retrieval.semantic import SemanticRetriever

        retriever = SemanticRetriever(
            embedding_service=embedding_service,
            vector_store=vector_store,
            default_top_k=5,
        )
        results = retriever.retrieve("What is the retirement age in Freedonia?")
        assert len(results) == 2

        sources = format_sources(results)
        doc_ids = {s.document_id for s in sources}
        assert "doc-old" in doc_ids
        assert "doc-new" in doc_ids

    def test_sources_never_fabricate_missing_page(
        self,
        embedding_service,
        vector_store,
    ):
        """DOCX chunks without page_number must not fabricate one."""
        chunk = _make_chunk(
            document_id="doc-docx",
            document_name="notes.docx",
            page_number=None,
            source_type="docx",
            content="These are some meeting notes about project Alpha.",
        )
        self._index_chunks([chunk], embedding_service, vector_store)

        from rag.retrieval.semantic import SemanticRetriever

        retriever = SemanticRetriever(
            embedding_service=embedding_service,
            vector_store=vector_store,
            default_top_k=1,
        )
        results = retriever.retrieve("meeting notes project Alpha")
        sources = format_sources(results)
        assert len(sources) == 1
        # page_number must be None, not invented.
        assert sources[0].page_number is None


# ==========================================================================
# 13. Real Document Fixture Tests (country_alpha.pdf, country_beta.pdf)
# ==========================================================================


class TestRealDocumentFixtures:
    """Multi-document and citation tests using real PDF fixtures."""

    @pytest.fixture
    def fixtures_dir(self) -> Path:
        return Path(__file__).parent / "fixtures"

    @pytest.fixture
    def services(self, tmp_path):
        from rag.chunking.chunker import ChunkingConfig
        from rag.embeddings.indexing import IndexingService
        from rag.embeddings.sentence_transformer import (
            SentenceTransformerEmbeddingService,
        )
        from rag.ingestion.service import IngestionService
        from rag.orchestration.indexing_service import DocumentIndexingService
        from rag.retrieval.semantic import SemanticRetriever
        from rag.vectorstore.chroma_store import ChromaVectorStore

        emb = SentenceTransformerEmbeddingService(
            model_name="all-MiniLM-L6-v2",
            batch_size=32,
        )
        store = ChromaVectorStore(
            persist_directory=str(tmp_path / "fixtures_vectorstore"),
            collection_name="test_fixtures_p7",
        )
        indexing = IndexingService(embedding_service=emb, vector_store=store)
        doc_indexing = DocumentIndexingService(
            ingestion_service=IngestionService(),
            chunking_config=ChunkingConfig(chunk_size=500, chunk_overlap=100),
            indexing_service=indexing,
        )
        retriever = SemanticRetriever(
            embedding_service=emb,
            vector_store=store,
            default_top_k=5,
            min_score=0.30,
        )
        return doc_indexing, retriever, store

    def test_real_pdf_fixtures_exist(self, fixtures_dir):
        """Verify the test PDF fixtures are present and readable."""
        assert (fixtures_dir / "country_alpha.pdf").is_file()
        assert (fixtures_dir / "country_beta.pdf").is_file()
        assert (fixtures_dir / "country_alpha_updated.pdf").is_file()

    def test_index_multiple_real_documents(self, fixtures_dir, services):
        """Index multiple real PDF documents and verify metadata."""
        doc_indexing, _, store = services

        result_a = doc_indexing.index_document(
            file_path=fixtures_dir / "country_alpha.pdf",
            document_name="country_alpha.pdf",
        )
        result_b = doc_indexing.index_document(
            file_path=fixtures_dir / "country_beta.pdf",
            document_name="country_beta.pdf",
        )

        assert result_a.document_id != result_b.document_id
        assert result_a.document_name == "country_alpha.pdf"
        assert result_b.document_name == "country_beta.pdf"
        assert result_a.num_pages == 2
        assert result_b.num_pages == 2
        assert result_a.num_chunks >= 1
        assert result_b.num_chunks >= 1
        assert store.count() == result_a.num_chunks + result_b.num_chunks

    def test_cross_document_retrieval_with_real_pdfs(self, fixtures_dir, services):
        """Retrieve across multiple indexed real PDFs and verify sources."""
        doc_indexing, retriever, _ = services

        res_a = doc_indexing.index_document(
            file_path=fixtures_dir / "country_alpha.pdf",
            document_name="country_alpha.pdf",
        )
        res_b = doc_indexing.index_document(
            file_path=fixtures_dir / "country_beta.pdf",
            document_name="country_beta.pdf",
        )

        # Cross-document query touching both countries.
        results = retriever.retrieve(
            "What are the capitals of Country Alpha and Country Beta?",
            top_k=5,
        )
        assert len(results) >= 2

        sources = format_sources(results)
        doc_names = {s.document_name for s in sources}
        assert "country_alpha.pdf" in doc_names
        assert "country_beta.pdf" in doc_names

        doc_ids = {s.document_id for s in sources}
        assert res_a.document_id in doc_ids
        assert res_b.document_id in doc_ids

        # Page numbers must come from actual document metadata (page 1).
        for s in sources:
            assert s.page_number == 1
            assert s.chunk_id
            assert s.score >= 0.30

    def test_conflicting_information_with_real_updated_pdf(
        self, fixtures_dir, services
    ):
        """Both original and updated documents remain retrievable and identifiable."""
        doc_indexing, retriever, _ = services

        res_orig = doc_indexing.index_document(
            file_path=fixtures_dir / "country_alpha.pdf",
            document_name="country_alpha.pdf",
        )
        res_upd = doc_indexing.index_document(
            file_path=fixtures_dir / "country_alpha_updated.pdf",
            document_name="country_alpha_updated.pdf",
        )

        results = retriever.retrieve("capital of Country Alpha", top_k=5)
        assert len(results) >= 2

        sources = format_sources(results)
        source_doc_names = {s.document_name for s in sources}
        assert "country_alpha.pdf" in source_doc_names
        assert "country_alpha_updated.pdf" in source_doc_names

        source_doc_ids = {s.document_id for s in sources}
        assert res_orig.document_id in source_doc_ids
        assert res_upd.document_id in source_doc_ids

    def test_relevance_gate_with_real_multi_document_index(
        self, fixtures_dir, services
    ):
        """Relevance gate rejects unrelated queries on a multi-doc index."""
        doc_indexing, retriever, _ = services

        doc_indexing.index_document(
            file_path=fixtures_dir / "country_alpha.pdf",
            document_name="country_alpha.pdf",
        )
        doc_indexing.index_document(
            file_path=fixtures_dir / "country_beta.pdf",
            document_name="country_beta.pdf",
        )

        results = retriever.retrieve("What is the boiling point of liquid nitrogen?")
        assert len(results) == 0

        mock_gen = MagicMock()
        query_service = RAGQueryService(
            retriever=retriever,
            generator=mock_gen,
            min_score=0.30,
        )

        with pytest.raises(InsufficientContextError):
            query_service.query("What is the boiling point of liquid nitrogen?")

        mock_gen.generate.assert_not_called()

    def test_prevention_of_fabricated_citation_metadata_real_pdf(
        self, fixtures_dir, services
    ):
        """Sources from real documents contain authentic metadata only."""
        doc_indexing, retriever, _ = services

        res = doc_indexing.index_document(
            file_path=fixtures_dir / "country_alpha.pdf",
            document_name="country_alpha.pdf",
        )

        results = retriever.retrieve("population and history of Country Alpha")
        assert len(results) >= 1

        sources = format_sources(results)
        for s in sources:
            assert s.document_id == res.document_id
            assert s.document_name == "country_alpha.pdf"
            assert s.page_number == 1
            assert s.chunk_id
            assert isinstance(s.score, float)


# ==========================================================================
# 14. Phase 7 End-to-End API Tests (TestClient)
# ==========================================================================


class TestPhase7APIEndToEnd:
    """End-to-end API tests for multi-document indexing and citations."""

    @pytest.fixture
    def test_env(self, tmp_path):
        from fastapi.testclient import TestClient

        from backend.config import Settings
        from backend.main import app

        settings = Settings(
            vector_store_path=str(tmp_path / "api_p7_vectorstore"),
            vector_store_collection="test_api_p7",
            retrieval_min_score=0.30,
        )
        mock_result = GenerationResult(
            answer="Country Alpha: City One; Country Beta: City Two.",
            model_name="mock-model",
            metadata={"temperature": 0.1},
        )
        with (
            patch("backend.documents.get_settings", return_value=settings),
            patch("backend.query.get_settings", return_value=settings),
            patch("backend.query.OllamaGenerator") as MockGen,
        ):
            mock_inst = MagicMock()
            mock_inst.generate.return_value = mock_result
            MockGen.return_value = mock_inst
            yield TestClient(app), settings, mock_inst

    def test_multi_document_upload_and_query_citations(self, test_env):
        """Upload multiple real documents and verify sources in /query API response."""
        client, _, _ = test_env
        fixtures_dir = Path(__file__).parent / "fixtures"

        # Upload Country Alpha
        with open(fixtures_dir / "country_alpha.pdf", "rb") as f:
            resp_a = client.post(
                "/documents/upload",
                files={"file": ("country_alpha.pdf", f.read(), "application/pdf")},
            )
        assert resp_a.status_code == 200
        doc_a_id = resp_a.json()["document_id"]

        # Upload Country Beta
        with open(fixtures_dir / "country_beta.pdf", "rb") as f:
            resp_b = client.post(
                "/documents/upload",
                files={"file": ("country_beta.pdf", f.read(), "application/pdf")},
            )
        assert resp_b.status_code == 200
        doc_b_id = resp_b.json()["document_id"]

        # Query across both documents
        q_resp = client.post(
            "/query",
            json={
                "question": "What are the capitals of Country Alpha and Country Beta?",
                "top_k": 5,
            },
        )
        assert q_resp.status_code == 200
        data = q_resp.json()
        assert "sources" in data
        assert len(data["sources"]) >= 2

        # Verify citation fields
        sources = data["sources"]
        found_docs = {s["document_name"] for s in sources}
        assert "country_alpha.pdf" in found_docs
        assert "country_beta.pdf" in found_docs

        for s in sources:
            assert s["document_id"] in (doc_a_id, doc_b_id)
            assert s["page_number"] == 1
            assert s["chunk_id"]
            assert s["score"] >= 0.30

    def test_docx_upload_preserves_single_page_attribution_in_api(self, test_env):
        """DOCX upload preserves single logical page attribution (page_number=1)."""
        import io

        import docx

        client, _, _ = test_env

        doc = docx.Document()
        doc.add_paragraph(
            "Solar energy systems convert sunlight into electrical power."
        )
        buf = io.BytesIO()
        doc.save(buf)

        resp = client.post(
            "/documents/upload",
            files={
                "file": (
                    "solar.docx",
                    buf.getvalue(),
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
        )
        assert resp.status_code == 200

        q_resp = client.post(
            "/query",
            json={"question": "How do solar energy systems work?"},
        )
        assert q_resp.status_code == 200
        data = q_resp.json()
        assert len(data["sources"]) >= 1
        assert data["sources"][0]["page_number"] == 1
        assert data["sources"][0]["document_name"] == "solar.docx"

    def test_missing_page_metadata_returns_null_in_api(self, test_env):
        """When chunk metadata has no page_number, API returns page_number=null."""
        from rag.chunking.models import Chunk
        from rag.embeddings.sentence_transformer import (
            SentenceTransformerEmbeddingService,
        )
        from rag.vectorstore.chroma_store import ChromaVectorStore

        client, settings, _ = test_env

        chunk = Chunk(
            chunk_id=str(uuid.uuid4()),
            document_id="doc-nopage",
            document_name="readme.txt",
            source_type="txt",
            content="FastAPI is a fast web framework for building APIs.",
            page_number=None,
            chunk_index=0,
            metadata={},
        )
        store = ChromaVectorStore(
            persist_directory=settings.vector_store_path,
            collection_name=settings.vector_store_collection,
        )
        emb = SentenceTransformerEmbeddingService(model_name=settings.embedding_model)
        store.add_chunks([chunk], emb.embed_documents([chunk.content]))

        q_resp = client.post(
            "/query",
            json={"question": "What is FastAPI used for?"},
        )
        assert q_resp.status_code == 200
        data = q_resp.json()
        assert len(data["sources"]) >= 1
        # page_number must be None / null (never fabricated)
        assert data["sources"][0]["page_number"] is None
        assert data["sources"][0]["document_name"] == "readme.txt"

    def test_unrelated_query_in_multi_doc_returns_404_without_fabricated_sources(
        self, test_env
    ):
        """Unrelated query on multi-doc index returns 404 without calling generator."""
        client, _, mock_gen = test_env
        fixtures_dir = Path(__file__).parent / "fixtures"

        with open(fixtures_dir / "country_alpha.pdf", "rb") as f:
            client.post(
                "/documents/upload",
                files={"file": ("country_alpha.pdf", f.read(), "application/pdf")},
            )

        mock_gen.generate.reset_mock()
        q_resp = client.post(
            "/query",
            json={"question": "What is the boiling point of liquid nitrogen?"},
        )
        assert q_resp.status_code == 404
        mock_gen.generate.assert_not_called()
