"""Phase 4 — Semantic Retrieval tests.

Covers:
    - Retriever interface contract
    - Query validation (empty, whitespace, non-string, unicode, long)
    - top_k behavior (default, caller-provided, validation, edge cases)
    - Metadata preservation (all fields)
    - Score preservation
    - Result ordering
    - Dependency failure handling (embedding, vector store)
    - Empty vector store
    - Multiple documents
    - Integration test with real EmbeddingService + ChromaDB
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from rag.embeddings.base import EmbeddingService
from rag.retrieval.base import Retriever
from rag.retrieval.exceptions import (
    EmbeddingError,
    InvalidQueryError,
    InvalidTopKError,
    RetrievalError,
    VectorStoreError,
)
from rag.retrieval.semantic import (
    MAX_QUERY_LENGTH,
    MAX_TOP_K,
    SemanticRetriever,
)
from rag.vectorstore.base import VectorStore
from rag.vectorstore.models import VectorSearchResult

# ======================================================================
# Fixtures
# ======================================================================


def _make_mock_embedding_service(dimension: int = 384) -> MagicMock:
    """Create a mock EmbeddingService with spec."""
    mock = MagicMock(spec=EmbeddingService)
    mock.embed_query.return_value = [0.1] * dimension
    mock.embed_documents.return_value = [[0.1] * dimension]
    mock.dimension = dimension
    mock.model_name = "mock-model"
    return mock


def _make_mock_vector_store() -> MagicMock:
    """Create a mock VectorStore with spec."""
    mock = MagicMock(spec=VectorStore)
    mock.query.return_value = []
    mock.count.return_value = 0
    return mock


def _make_search_result(
    chunk_id: str = "chunk-1",
    document_id: str = "doc-1",
    document_name: str = "test.pdf",
    content: str = "test content",
    score: float = 0.95,
    source_type: str = "pdf",
    page_number: int = 1,
    chunk_index: int = 0,
    extra_metadata: dict[str, Any] | None = None,
) -> VectorSearchResult:
    """Build a VectorSearchResult with full metadata."""
    meta: dict[str, Any] = {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "document_name": document_name,
        "source_type": source_type,
        "page_number": page_number,
        "chunk_index": chunk_index,
    }
    if extra_metadata:
        meta.update(extra_metadata)
    return VectorSearchResult(
        chunk_id=chunk_id,
        document_id=document_id,
        document_name=document_name,
        content=content,
        score=score,
        metadata=meta,
    )


@pytest.fixture
def mock_embedding_service() -> MagicMock:
    return _make_mock_embedding_service()


@pytest.fixture
def mock_vector_store() -> MagicMock:
    return _make_mock_vector_store()


@pytest.fixture
def retriever(mock_embedding_service: MagicMock, mock_vector_store: MagicMock):
    return SemanticRetriever(
        embedding_service=mock_embedding_service,
        vector_store=mock_vector_store,
        default_top_k=5,
    )


# ======================================================================
# 1. Interface / ABC
# ======================================================================


class TestRetrieverInterface:
    """Verify Retriever ABC contract."""

    def test_retriever_is_abstract(self):
        """Cannot instantiate the abstract Retriever directly."""
        with pytest.raises(TypeError):
            Retriever()  # type: ignore[abstract]

    def test_semantic_retriever_is_retriever(
        self, mock_embedding_service, mock_vector_store
    ):
        """SemanticRetriever is a Retriever subclass."""
        r = SemanticRetriever(mock_embedding_service, mock_vector_store)
        assert isinstance(r, Retriever)

    def test_retrieve_method_exists(self, retriever):
        """The retrieve method is callable."""
        assert callable(retriever.retrieve)


# ======================================================================
# 2. Constructor Validation
# ======================================================================


class TestConstructorValidation:
    """Validate SemanticRetriever constructor type/value checks."""

    def test_rejects_non_embedding_service(self, mock_vector_store):
        with pytest.raises(TypeError, match="EmbeddingService"):
            SemanticRetriever("not-an-embedding-service", mock_vector_store)

    def test_rejects_non_vector_store(self, mock_embedding_service):
        with pytest.raises(TypeError, match="VectorStore"):
            SemanticRetriever(mock_embedding_service, "not-a-vector-store")

    def test_rejects_zero_default_top_k(
        self, mock_embedding_service, mock_vector_store
    ):
        with pytest.raises(ValueError, match="default_top_k"):
            SemanticRetriever(
                mock_embedding_service, mock_vector_store, default_top_k=0
            )

    def test_rejects_negative_default_top_k(
        self, mock_embedding_service, mock_vector_store
    ):
        with pytest.raises(ValueError, match="default_top_k"):
            SemanticRetriever(
                mock_embedding_service, mock_vector_store, default_top_k=-1
            )

    def test_rejects_bool_default_top_k(
        self, mock_embedding_service, mock_vector_store
    ):
        with pytest.raises(ValueError, match="default_top_k"):
            SemanticRetriever(
                mock_embedding_service, mock_vector_store, default_top_k=True
            )

    def test_rejects_exceeds_max_default_top_k(
        self, mock_embedding_service, mock_vector_store
    ):
        with pytest.raises(ValueError, match="default_top_k"):
            SemanticRetriever(
                mock_embedding_service, mock_vector_store, default_top_k=MAX_TOP_K + 1
            )

    def test_accepts_max_default_top_k(self, mock_embedding_service, mock_vector_store):
        r = SemanticRetriever(
            mock_embedding_service, mock_vector_store, default_top_k=MAX_TOP_K
        )
        assert r._default_top_k == MAX_TOP_K

    def test_valid_construction(self, mock_embedding_service, mock_vector_store):
        r = SemanticRetriever(
            mock_embedding_service, mock_vector_store, default_top_k=20
        )
        assert isinstance(r, SemanticRetriever)


# ======================================================================
# 3. Query Validation
# ======================================================================


class TestQueryValidation:
    """Test query validation edge cases."""

    def test_empty_query_raises(self, retriever):
        with pytest.raises(InvalidQueryError, match="non-empty"):
            retriever.retrieve("")

    def test_whitespace_only_query_raises(self, retriever):
        with pytest.raises(InvalidQueryError, match="non-empty"):
            retriever.retrieve("   \t\n  ")

    def test_non_string_query_int_raises(self, retriever):
        with pytest.raises(InvalidQueryError, match="string"):
            retriever.retrieve(42)  # type: ignore[arg-type]

    def test_non_string_query_none_raises(self, retriever):
        with pytest.raises(InvalidQueryError, match="string"):
            retriever.retrieve(None)  # type: ignore[arg-type]

    def test_non_string_query_list_raises(self, retriever):
        with pytest.raises(InvalidQueryError, match="string"):
            retriever.retrieve(["query"])  # type: ignore[arg-type]

    def test_long_query_raises(self, retriever):
        long_query = "a" * (MAX_QUERY_LENGTH + 1)
        with pytest.raises(InvalidQueryError, match="maximum length"):
            retriever.retrieve(long_query)

    def test_query_at_max_length_succeeds(self, retriever, mock_vector_store):
        """Query exactly at the limit should succeed."""
        mock_vector_store.query.return_value = []
        query = "a" * MAX_QUERY_LENGTH
        results = retriever.retrieve(query)
        assert isinstance(results, list)

    def test_unicode_query_succeeds(self, retriever, mock_vector_store):
        """Unicode queries (CJK, emoji, Arabic, etc.) should work."""
        mock_vector_store.query.return_value = [
            _make_search_result(content="unicode result")
        ]
        results = retriever.retrieve("如何提高收入？ 📊 مرحبا")
        assert len(results) == 1

    def test_valid_query_succeeds(self, retriever, mock_vector_store):
        """Normal English query works."""
        mock_vector_store.query.return_value = [
            _make_search_result(content="Revenue grew by 25%")
        ]
        results = retriever.retrieve("How did revenue change?")
        assert len(results) == 1

    def test_query_is_passed_to_embedding_service(
        self, retriever, mock_embedding_service, mock_vector_store
    ):
        """The exact query string is passed to embed_query."""
        mock_vector_store.query.return_value = []
        retriever.retrieve("test query")
        mock_embedding_service.embed_query.assert_called_once_with("test query")

    def test_null_bytes_only_query_raises(self, retriever):
        with pytest.raises(InvalidQueryError, match="non-empty"):
            retriever.retrieve("\x00\x00\x00")

    def test_zero_width_spaces_only_query_raises(self, retriever):
        with pytest.raises(InvalidQueryError, match="non-empty"):
            retriever.retrieve("\u200b\u200b  \t")

    def test_bom_only_query_raises(self, retriever):
        with pytest.raises(InvalidQueryError, match="non-empty"):
            retriever.retrieve("\ufeff")

    def test_query_null_bytes_sanitized_before_embedding(
        self, retriever, mock_embedding_service, mock_vector_store
    ):
        mock_vector_store.query.return_value = []
        retriever.retrieve("test\x00 query")
        mock_embedding_service.embed_query.assert_called_once_with("test query")


# ======================================================================
# 4. top_k Behavior
# ======================================================================


class TestTopKBehavior:
    """Test top_k resolution, defaults, and validation."""

    def test_default_top_k_used_when_none(
        self, mock_embedding_service, mock_vector_store
    ):
        """When top_k is not provided, default_top_k is used."""
        r = SemanticRetriever(
            mock_embedding_service, mock_vector_store, default_top_k=7
        )
        mock_vector_store.query.return_value = []
        r.retrieve("test")
        mock_vector_store.query.assert_called_once()
        call_args = mock_vector_store.query.call_args
        assert call_args[1]["top_k"] == 7

    def test_caller_top_k_overrides_default(
        self, mock_embedding_service, mock_vector_store
    ):
        """Caller-provided top_k overrides the default."""
        r = SemanticRetriever(
            mock_embedding_service, mock_vector_store, default_top_k=7
        )
        mock_vector_store.query.return_value = []
        r.retrieve("test", top_k=3)
        call_args = mock_vector_store.query.call_args
        assert call_args[1]["top_k"] == 3

    def test_top_k_one(self, retriever, mock_vector_store):
        """top_k=1 should work."""
        mock_vector_store.query.return_value = [_make_search_result()]
        results = retriever.retrieve("test", top_k=1)
        assert len(results) <= 1

    def test_top_k_zero_raises(self, retriever):
        with pytest.raises(InvalidQueryError, match="top_k"):
            retriever.retrieve("test", top_k=0)

    def test_top_k_negative_raises(self, retriever):
        with pytest.raises(InvalidQueryError, match="top_k"):
            retriever.retrieve("test", top_k=-1)

    def test_top_k_bool_raises(self, retriever):
        with pytest.raises(InvalidQueryError, match="top_k"):
            retriever.retrieve("test", top_k=True)  # type: ignore[arg-type]

    def test_top_k_float_raises(self, retriever):
        with pytest.raises(InvalidQueryError, match="top_k"):
            retriever.retrieve("test", top_k=5.0)  # type: ignore[arg-type]

    def test_top_k_exceeds_max_raises(self, retriever):
        with pytest.raises(InvalidQueryError, match="maximum"):
            retriever.retrieve("test", top_k=MAX_TOP_K + 1)

    def test_top_k_at_max_succeeds(self, retriever, mock_vector_store):
        """top_k exactly at MAX_TOP_K should work."""
        mock_vector_store.query.return_value = []
        results = retriever.retrieve("test", top_k=MAX_TOP_K)
        assert isinstance(results, list)

    def test_fewer_results_than_top_k(self, retriever, mock_vector_store):
        """When store has fewer chunks than top_k, return what's available."""
        mock_vector_store.query.return_value = [
            _make_search_result(chunk_id="c1"),
            _make_search_result(chunk_id="c2"),
        ]
        results = retriever.retrieve("test", top_k=100)
        assert len(results) == 2

    def test_invalid_top_k_raises_invalid_top_k_error(self, retriever):
        """Invalid top_k raises InvalidTopKError specifically."""
        with pytest.raises(InvalidTopKError):
            retriever.retrieve("test", top_k=-1)
        with pytest.raises(InvalidTopKError):
            retriever.retrieve("test", top_k=MAX_TOP_K + 1)


# ======================================================================
# 5. Basic Retrieval
# ======================================================================


class TestBasicRetrieval:
    """Test core retrieval behavior."""

    def test_retrieve_returns_list(self, retriever, mock_vector_store):
        mock_vector_store.query.return_value = []
        results = retriever.retrieve("test")
        assert isinstance(results, list)

    def test_retrieve_returns_vector_search_results(self, retriever, mock_vector_store):
        expected = [_make_search_result()]
        mock_vector_store.query.return_value = expected
        results = retriever.retrieve("test")
        assert len(results) == 1
        assert isinstance(results[0], VectorSearchResult)

    def test_retrieve_passes_query_vector_to_store(
        self, retriever, mock_embedding_service, mock_vector_store
    ):
        """The query embedding is passed to vector_store.query."""
        query_vec = [0.5, 0.3, 0.2]
        mock_embedding_service.embed_query.return_value = query_vec
        mock_vector_store.query.return_value = []

        retriever.retrieve("test")

        mock_vector_store.query.assert_called_once_with(query_vec, top_k=5)

    def test_retrieve_multiple_results(self, retriever, mock_vector_store):
        results_data = [
            _make_search_result(chunk_id=f"c{i}", score=1.0 - i * 0.1) for i in range(5)
        ]
        mock_vector_store.query.return_value = results_data
        results = retriever.retrieve("test")
        assert len(results) == 5


# ======================================================================
# 6. Metadata Preservation
# ======================================================================


class TestMetadataPreservation:
    """Verify all metadata fields survive through retrieval."""

    @pytest.fixture
    def full_result(self):
        return _make_search_result(
            chunk_id="chunk-abc",
            document_id="doc-xyz",
            document_name="report.pdf",
            content="Revenue increased by 25%",
            score=0.92,
            source_type="pdf",
            page_number=42,
            chunk_index=7,
            extra_metadata={"custom_key": "custom_value", "char_count": 100},
        )

    def test_chunk_id_preserved(self, retriever, mock_vector_store, full_result):
        mock_vector_store.query.return_value = [full_result]
        results = retriever.retrieve("revenue")
        assert results[0].chunk_id == "chunk-abc"

    def test_document_id_preserved(self, retriever, mock_vector_store, full_result):
        mock_vector_store.query.return_value = [full_result]
        results = retriever.retrieve("revenue")
        assert results[0].document_id == "doc-xyz"

    def test_document_name_preserved(self, retriever, mock_vector_store, full_result):
        mock_vector_store.query.return_value = [full_result]
        results = retriever.retrieve("revenue")
        assert results[0].document_name == "report.pdf"

    def test_content_preserved(self, retriever, mock_vector_store, full_result):
        mock_vector_store.query.return_value = [full_result]
        results = retriever.retrieve("revenue")
        assert results[0].content == "Revenue increased by 25%"

    def test_score_preserved(self, retriever, mock_vector_store, full_result):
        mock_vector_store.query.return_value = [full_result]
        results = retriever.retrieve("revenue")
        assert results[0].score == pytest.approx(0.92)

    def test_source_type_in_metadata(self, retriever, mock_vector_store, full_result):
        mock_vector_store.query.return_value = [full_result]
        results = retriever.retrieve("revenue")
        assert results[0].metadata["source_type"] == "pdf"

    def test_page_number_in_metadata(self, retriever, mock_vector_store, full_result):
        mock_vector_store.query.return_value = [full_result]
        results = retriever.retrieve("revenue")
        assert results[0].metadata["page_number"] == 42

    def test_chunk_index_in_metadata(self, retriever, mock_vector_store, full_result):
        mock_vector_store.query.return_value = [full_result]
        results = retriever.retrieve("revenue")
        assert results[0].metadata["chunk_index"] == 7

    def test_custom_metadata_preserved(self, retriever, mock_vector_store, full_result):
        mock_vector_store.query.return_value = [full_result]
        results = retriever.retrieve("revenue")
        assert results[0].metadata["custom_key"] == "custom_value"
        assert results[0].metadata["char_count"] == 100


# ======================================================================
# 7. Ordering and Scores
# ======================================================================


class TestOrderingAndScores:
    """Verify result ordering follows vector store semantics."""

    def test_results_preserve_store_ordering(self, retriever, mock_vector_store):
        """Results are returned in the same order as the vector store."""
        ordered = [
            _make_search_result(chunk_id="best", score=0.99),
            _make_search_result(chunk_id="good", score=0.85),
            _make_search_result(chunk_id="ok", score=0.70),
        ]
        mock_vector_store.query.return_value = ordered
        results = retriever.retrieve("test")
        assert [r.chunk_id for r in results] == ["best", "good", "ok"]

    def test_scores_are_floats(self, retriever, mock_vector_store):
        mock_vector_store.query.return_value = [
            _make_search_result(score=0.95),
        ]
        results = retriever.retrieve("test")
        assert isinstance(results[0].score, float)

    def test_near_equal_scores_preserved(self, retriever, mock_vector_store):
        """Near-equal similarity scores are returned as-is."""
        results_data = [
            _make_search_result(chunk_id="a", score=0.9001),
            _make_search_result(chunk_id="b", score=0.9000),
            _make_search_result(chunk_id="c", score=0.8999),
        ]
        mock_vector_store.query.return_value = results_data
        results = retriever.retrieve("test")
        assert results[0].score == pytest.approx(0.9001)
        assert results[1].score == pytest.approx(0.9000)
        assert results[2].score == pytest.approx(0.8999)


# ======================================================================
# 8. Empty Vector Store
# ======================================================================


class TestEmptyVectorStore:
    """Test behavior when the vector store has no chunks."""

    def test_empty_store_returns_empty_list(self, retriever, mock_vector_store):
        mock_vector_store.query.return_value = []
        results = retriever.retrieve("test query")
        assert results == []

    def test_empty_store_does_not_fabricate_results(self, retriever, mock_vector_store):
        mock_vector_store.query.return_value = []
        results = retriever.retrieve("What is the revenue?")
        assert len(results) == 0


# ======================================================================
# 9. Dependency Failure Handling
# ======================================================================


class TestDependencyFailures:
    """Test error wrapping for embedding and vector store failures."""

    def test_embedding_failure_raises_embedding_error(
        self, retriever, mock_embedding_service
    ):
        mock_embedding_service.embed_query.side_effect = RuntimeError("Model crashed")
        with pytest.raises(EmbeddingError, match="Failed to embed query"):
            retriever.retrieve("test")

    def test_embedding_error_is_retrieval_error(
        self, retriever, mock_embedding_service
    ):
        mock_embedding_service.embed_query.side_effect = RuntimeError("boom")
        with pytest.raises(RetrievalError):
            retriever.retrieve("test")

    def test_embedding_error_preserves_cause(self, retriever, mock_embedding_service):
        original = RuntimeError("original cause")
        mock_embedding_service.embed_query.side_effect = original
        with pytest.raises(EmbeddingError) as exc_info:
            retriever.retrieve("test")
        assert exc_info.value.__cause__ is original

    def test_vector_store_failure_raises_vector_store_error(
        self, retriever, mock_embedding_service, mock_vector_store
    ):
        mock_embedding_service.embed_query.return_value = [0.1] * 10
        mock_vector_store.query.side_effect = RuntimeError("ChromaDB crashed")
        with pytest.raises(VectorStoreError, match="Vector store search failed"):
            retriever.retrieve("test")

    def test_vector_store_error_is_retrieval_error(
        self, retriever, mock_embedding_service, mock_vector_store
    ):
        mock_embedding_service.embed_query.return_value = [0.1] * 10
        mock_vector_store.query.side_effect = RuntimeError("crash")
        with pytest.raises(RetrievalError):
            retriever.retrieve("test")

    def test_vector_store_error_preserves_cause(
        self, retriever, mock_embedding_service, mock_vector_store
    ):
        original = RuntimeError("vs cause")
        mock_embedding_service.embed_query.return_value = [0.1] * 10
        mock_vector_store.query.side_effect = original
        with pytest.raises(VectorStoreError) as exc_info:
            retriever.retrieve("test")
        assert exc_info.value.__cause__ is original

    def test_embedding_returns_empty_list_raises_embedding_error(
        self, retriever, mock_embedding_service
    ):
        mock_embedding_service.embed_query.return_value = []
        with pytest.raises(EmbeddingError, match="invalid vector"):
            retriever.retrieve("test")

    def test_embedding_returns_none_raises_embedding_error(
        self, retriever, mock_embedding_service
    ):
        mock_embedding_service.embed_query.return_value = None
        with pytest.raises(EmbeddingError, match="invalid vector"):
            retriever.retrieve("test")

    def test_embedding_returns_nan_raises_embedding_error(
        self, retriever, mock_embedding_service
    ):
        mock_embedding_service.embed_query.return_value = [0.1, float("nan"), 0.3]
        with pytest.raises(EmbeddingError, match="non-numeric or non-finite"):
            retriever.retrieve("test")

    def test_embedding_returns_inf_raises_embedding_error(
        self, retriever, mock_embedding_service
    ):
        mock_embedding_service.embed_query.return_value = [0.1, float("inf"), 0.3]
        with pytest.raises(EmbeddingError, match="non-numeric or non-finite"):
            retriever.retrieve("test")

    def test_vector_store_returns_non_list_raises_vector_store_error(
        self, retriever, mock_embedding_service, mock_vector_store
    ):
        mock_embedding_service.embed_query.return_value = [0.1] * 10
        mock_vector_store.query.return_value = "not a list"
        with pytest.raises(VectorStoreError, match="non-list"):
            retriever.retrieve("test")


# ======================================================================
# 10. Multiple Documents
# ======================================================================


class TestMultipleDocuments:
    """Test retrieval across multiple documents."""

    def test_results_from_different_documents(self, retriever, mock_vector_store):
        """Results can span multiple documents."""
        mock_vector_store.query.return_value = [
            _make_search_result(
                chunk_id="c1",
                document_id="doc-A",
                document_name="report_2023.pdf",
                score=0.95,
            ),
            _make_search_result(
                chunk_id="c2",
                document_id="doc-B",
                document_name="report_2024.pdf",
                score=0.90,
            ),
            _make_search_result(
                chunk_id="c3",
                document_id="doc-A",
                document_name="report_2023.pdf",
                score=0.85,
            ),
        ]
        results = retriever.retrieve("revenue comparison")
        doc_ids = {r.document_id for r in results}
        assert doc_ids == {"doc-A", "doc-B"}
        assert len(results) == 3


# ======================================================================
# 11. Exception Hierarchy
# ======================================================================


class TestExceptionHierarchy:
    """Verify exception class relationships."""

    def test_invalid_query_is_retrieval_error(self):
        assert issubclass(InvalidQueryError, RetrievalError)

    def test_invalid_top_k_is_invalid_query_error(self):
        assert issubclass(InvalidTopKError, InvalidQueryError)

    def test_invalid_top_k_is_retrieval_error(self):
        assert issubclass(InvalidTopKError, RetrievalError)

    def test_embedding_error_is_retrieval_error(self):
        assert issubclass(EmbeddingError, RetrievalError)

    def test_vector_store_error_is_retrieval_error(self):
        assert issubclass(VectorStoreError, RetrievalError)

    def test_retrieval_error_is_exception(self):
        assert issubclass(RetrievalError, Exception)


# ======================================================================
# 12. Integration Test — Real EmbeddingService + ChromaDB
# ======================================================================


class TestSemanticRetrievalIntegration:
    """Integration test using real Phase 3 components.

    Uses a real SentenceTransformerEmbeddingService and ChromaVectorStore
    to verify the complete retrieval flow:

        Query → EmbeddingService → VectorStore → Retriever → Results
    """

    @pytest.fixture
    def integration_retriever(self, tmp_path):
        """Set up a real retriever with indexed chunks."""
        from rag.chunking.models import Chunk
        from rag.embeddings.sentence_transformer import (
            SentenceTransformerEmbeddingService,
        )
        from rag.vectorstore.chroma_store import ChromaVectorStore

        embedding_service = SentenceTransformerEmbeddingService(
            model_name="all-MiniLM-L6-v2", batch_size=32
        )
        vector_store = ChromaVectorStore(
            persist_directory=str(tmp_path / "vectorstore"),
            collection_name="test_retrieval",
        )

        # Index test chunks
        chunks = [
            Chunk(
                chunk_id="chunk-revenue-1",
                document_id="doc-annual-2024",
                document_name="annual_report_2024.pdf",
                source_type="pdf",
                content=(
                    "The company's annual revenue increased by 25%"
                    " in fiscal year 2024, reaching $500 million."
                ),
                page_number=38,
                chunk_index=0,
                metadata={"section": "Financial Performance"},
            ),
            Chunk(
                chunk_id="chunk-employees-1",
                document_id="doc-annual-2024",
                document_name="annual_report_2024.pdf",
                source_type="pdf",
                content=(
                    "The company hired 500 new employees"
                    " in 2024, bringing total headcount to 5000."
                ),
                page_number=12,
                chunk_index=1,
                metadata={"section": "Human Resources"},
            ),
            Chunk(
                chunk_id="chunk-product-1",
                document_id="doc-product",
                document_name="product_guide.docx",
                source_type="docx",
                content=(
                    "The new software product features machine"
                    " learning capabilities for automated"
                    " data analysis."
                ),
                page_number=1,
                chunk_index=0,
                metadata={"section": "Product Overview"},
            ),
        ]

        embeddings = embedding_service.embed_documents([c.content for c in chunks])
        vector_store.add_chunks(chunks, embeddings)

        retriever = SemanticRetriever(
            embedding_service=embedding_service,
            vector_store=vector_store,
            default_top_k=5,
        )

        return retriever, vector_store

    def test_relevant_query_returns_relevant_chunks(self, integration_retriever):
        """A revenue question should rank the revenue chunk highest."""
        retriever, _ = integration_retriever
        results = retriever.retrieve("What was the company's revenue in 2024?")
        assert len(results) > 0
        # The revenue chunk should be in the top results
        top_result = results[0]
        assert "revenue" in top_result.content.lower()

    def test_unrelated_query_returns_lower_scores(self, integration_retriever):
        """An unrelated query should produce lower relevance scores."""
        retriever, _ = integration_retriever
        relevant_results = retriever.retrieve("revenue growth in 2024")
        unrelated_results = retriever.retrieve(
            "quantum physics string theory black holes"
        )
        # Both should return results (semantic search returns top-k regardless)
        assert len(relevant_results) > 0
        assert len(unrelated_results) > 0
        # Relevant query should have higher top score
        assert relevant_results[0].score > unrelated_results[0].score

    def test_top_k_limits_results(self, integration_retriever):
        """top_k should limit the number of results."""
        retriever, _ = integration_retriever
        results = retriever.retrieve("revenue", top_k=1)
        assert len(results) == 1

    def test_top_k_larger_than_collection(self, integration_retriever):
        """top_k > stored chunks returns all available chunks."""
        retriever, _ = integration_retriever
        results = retriever.retrieve("company", top_k=100)
        # We indexed 3 chunks
        assert len(results) == 3

    def test_metadata_preserved_in_integration(self, integration_retriever):
        """Metadata survives the full pipeline."""
        retriever, _ = integration_retriever
        results = retriever.retrieve("revenue growth", top_k=1)
        assert len(results) == 1
        result = results[0]
        # Core fields
        assert result.chunk_id in {
            "chunk-revenue-1",
            "chunk-employees-1",
            "chunk-product-1",
        }
        assert result.document_id in {"doc-annual-2024", "doc-product"}
        assert result.document_name in {
            "annual_report_2024.pdf",
            "product_guide.docx",
        }
        assert result.score > 0
        # Metadata fields
        assert "source_type" in result.metadata
        assert "page_number" in result.metadata
        assert "chunk_index" in result.metadata

    def test_results_ordered_by_similarity(self, integration_retriever):
        """Results should be ordered by descending similarity score."""
        retriever, _ = integration_retriever
        results = retriever.retrieve("revenue", top_k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_multiple_documents_in_results(self, integration_retriever):
        """Results can come from multiple documents."""
        retriever, _ = integration_retriever
        results = retriever.retrieve("company information", top_k=3)
        doc_ids = {r.document_id for r in results}
        # We have chunks from 2 different documents
        assert len(doc_ids) >= 1  # At minimum, results exist

    def test_unicode_query_integration(self, integration_retriever):
        """Unicode queries work end-to-end."""
        retriever, _ = integration_retriever
        results = retriever.retrieve("收入增长")  # "revenue growth" in Chinese
        assert isinstance(results, list)

    def test_empty_store_integration(self, tmp_path):
        """Empty vector store returns empty results."""
        from rag.embeddings.sentence_transformer import (
            SentenceTransformerEmbeddingService,
        )
        from rag.vectorstore.chroma_store import ChromaVectorStore

        embedding_service = SentenceTransformerEmbeddingService(
            model_name="all-MiniLM-L6-v2"
        )
        vector_store = ChromaVectorStore(
            persist_directory=str(tmp_path / "empty_store"),
            collection_name="test_empty",
        )
        retriever = SemanticRetriever(
            embedding_service=embedding_service,
            vector_store=vector_store,
        )
        results = retriever.retrieve("test query")
        assert results == []

    def test_score_is_float_in_integration(self, integration_retriever):
        """Scores are floats in real integration."""
        retriever, _ = integration_retriever
        results = retriever.retrieve("revenue")
        for r in results:
            assert isinstance(r.score, float)

    def test_custom_metadata_round_trip(self, integration_retriever):
        """Custom chunk metadata survives the full pipeline."""
        retriever, _ = integration_retriever
        results = retriever.retrieve("revenue growth fiscal year", top_k=1)
        assert len(results) == 1
        # The revenue chunk has section metadata
        result = results[0]
        if result.chunk_id == "chunk-revenue-1":
            assert result.metadata.get("section") == "Financial Performance"

    def test_relevance_gate_with_real_embeddings(self, integration_retriever):
        """Integration test with real embeddings:
        relevant query passes, unrelated query rejected.
        """
        retriever, _ = integration_retriever
        # With min_score=0.30:
        relevant = retriever.retrieve(
            "What was the company revenue in 2024?", min_score=0.30
        )
        assert len(relevant) >= 1
        assert "revenue" in relevant[0].content.lower()

        unrelated = retriever.retrieve("What is the capital of France?", min_score=0.30)
        assert len(unrelated) == 0


# ======================================================================
# 12. Relevance Gate (Phase 6 Grounding)
# ======================================================================


class TestRelevanceGate:
    """Verify SemanticRetriever relevance gate score filtering."""

    def test_min_score_constructor_validation_valid(
        self, mock_embedding_service, mock_vector_store
    ):
        """Valid min_score floats are accepted."""
        r = SemanticRetriever(mock_embedding_service, mock_vector_store, min_score=0.3)
        assert r.min_score == 0.3

    def test_min_score_boundary_values(self, mock_embedding_service, mock_vector_store):
        """Boundary values -1.0 and 1.0 are accepted."""
        r_neg = SemanticRetriever(
            mock_embedding_service, mock_vector_store, min_score=-1.0
        )
        assert r_neg.min_score == -1.0
        r_pos = SemanticRetriever(
            mock_embedding_service, mock_vector_store, min_score=1.0
        )
        assert r_pos.min_score == 1.0

    def test_min_score_out_of_bounds_rejected(
        self, mock_embedding_service, mock_vector_store
    ):
        """Values outside [-1.0, 1.0] are rejected."""
        with pytest.raises(ValueError, match="min_score"):
            SemanticRetriever(
                mock_embedding_service, mock_vector_store, min_score=-1.01
            )
        with pytest.raises(ValueError, match="min_score"):
            SemanticRetriever(mock_embedding_service, mock_vector_store, min_score=1.01)

    def test_min_score_bool_rejected(self, mock_embedding_service, mock_vector_store):
        """Booleans are rejected for min_score."""
        with pytest.raises(ValueError, match="min_score"):
            SemanticRetriever(mock_embedding_service, mock_vector_store, min_score=True)

    def test_min_score_nan_inf_rejected(
        self, mock_embedding_service, mock_vector_store
    ):
        """NaN and inf values are rejected."""
        import math

        with pytest.raises(ValueError, match="min_score"):
            SemanticRetriever(
                mock_embedding_service, mock_vector_store, min_score=math.nan
            )
        with pytest.raises(ValueError, match="min_score"):
            SemanticRetriever(
                mock_embedding_service, mock_vector_store, min_score=math.inf
            )

    def test_relevant_query_passes_relevance_gate(
        self, mock_embedding_service, mock_vector_store
    ):
        """Chunks meeting or exceeding min_score pass through."""
        mock_vector_store.query.return_value = [
            _make_search_result(chunk_id="c1", score=0.65),
            _make_search_result(chunk_id="c2", score=0.45),
        ]
        retriever = SemanticRetriever(
            mock_embedding_service, mock_vector_store, min_score=0.30
        )
        results = retriever.retrieve("What is Machine Learning?")
        assert len(results) == 2
        assert results[0].chunk_id == "c1"
        assert results[1].chunk_id == "c2"

    def test_unrelated_query_rejected(self, mock_embedding_service, mock_vector_store):
        """Chunks below min_score are rejected, returning empty list."""
        mock_vector_store.query.return_value = [
            _make_search_result(chunk_id="c1", score=0.0396),
            _make_search_result(chunk_id="c2", score=-0.0060),
            _make_search_result(chunk_id="c3", score=-0.0093),
        ]
        retriever = SemanticRetriever(
            mock_embedding_service, mock_vector_store, min_score=0.30
        )
        results = retriever.retrieve("What is the capital of France?")
        assert results == []

    def test_empty_results_handled(self, mock_embedding_service, mock_vector_store):
        """Empty vector store results produce empty list without error."""
        mock_vector_store.query.return_value = []
        retriever = SemanticRetriever(
            mock_embedding_service, mock_vector_store, min_score=0.30
        )
        results = retriever.retrieve("Any question")
        assert results == []

    def test_scores_at_threshold_are_kept(
        self, mock_embedding_service, mock_vector_store
    ):
        """Boundary test: score exactly equal to min_score is kept."""
        mock_vector_store.query.return_value = [
            _make_search_result(chunk_id="c_exact", score=0.30),
        ]
        retriever = SemanticRetriever(
            mock_embedding_service, mock_vector_store, min_score=0.30
        )
        results = retriever.retrieve("boundary test")
        assert len(results) == 1
        assert results[0].chunk_id == "c_exact"

    def test_scores_below_threshold_are_discarded(
        self, mock_embedding_service, mock_vector_store
    ):
        """Boundary test: score just below min_score is discarded."""
        mock_vector_store.query.return_value = [
            _make_search_result(chunk_id="c_below", score=0.2999),
        ]
        retriever = SemanticRetriever(
            mock_embedding_service, mock_vector_store, min_score=0.30
        )
        results = retriever.retrieve("boundary test")
        assert results == []

    def test_multiple_results_partial_relevance(
        self, mock_embedding_service, mock_vector_store
    ):
        """Multiple results where only some are relevant:
        relevant kept, low discarded.
        """
        mock_vector_store.query.return_value = [
            _make_search_result(chunk_id="c1", score=0.85),
            _make_search_result(chunk_id="c2", score=0.45),
            _make_search_result(chunk_id="c3", score=0.25),
            _make_search_result(chunk_id="c4", score=0.10),
        ]
        retriever = SemanticRetriever(
            mock_embedding_service, mock_vector_store, min_score=0.30
        )
        results = retriever.retrieve("test query", top_k=4)
        assert len(results) == 2
        assert [r.chunk_id for r in results] == ["c1", "c2"]

    def test_per_call_min_score_override(
        self, mock_embedding_service, mock_vector_store
    ):
        """Per-call min_score overrides instance-configured threshold."""
        mock_vector_store.query.return_value = [
            _make_search_result(chunk_id="c1", score=0.55),
            _make_search_result(chunk_id="c2", score=0.35),
        ]
        # Instance configured with 0.30 (both would pass)
        retriever = SemanticRetriever(
            mock_embedding_service, mock_vector_store, min_score=0.30
        )
        # Call with override 0.50 (only c1 passes)
        results = retriever.retrieve("test", min_score=0.50)
        assert len(results) == 1
        assert results[0].chunk_id == "c1"

    def test_invalid_per_call_min_score_raises(
        self, mock_embedding_service, mock_vector_store
    ):
        """Invalid per-call min_score raises InvalidQueryError."""
        retriever = SemanticRetriever(
            mock_embedding_service, mock_vector_store, min_score=0.30
        )
        with pytest.raises(InvalidQueryError, match="min_score"):
            retriever.retrieve("test", min_score=1.5)
        with pytest.raises(InvalidQueryError, match="min_score"):
            retriever.retrieve("test", min_score=True)
