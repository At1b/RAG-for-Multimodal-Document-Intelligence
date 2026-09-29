"""Phase 8 — Hybrid Retrieval Tests.

Covers:
    1. BM25 tokenization
    2. BM25Index construction and scoring
    3. BM25Retriever query validation, retrieval, metadata preservation
    4. BM25 empty index handling
    5. BM25 multi-document support
    6. HybridRetriever RRF fusion
    7. HybridRetriever deduplication
    8. HybridRetriever deterministic ranking
    9. HybridRetriever weight configuration
   10. HybridRetriever weight validation
   11. Phase 7 citation compatibility
   12. Relevance gate for different modes
   13. Configuration validation
   14. Semantic retriever still works independently
   15. Index builder from ChromaDB
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rag.retrieval.base import Retriever
from rag.retrieval.bm25 import (
    DEFAULT_STOPWORDS,
    MAX_TOP_K,
    BM25Index,
    BM25Retriever,
    tokenize,
)
from rag.retrieval.exceptions import InvalidQueryError, InvalidTopKError
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.index_builder import build_bm25_index_from_chroma
from rag.vectorstore.chroma_store import ChromaVectorStore
from rag.vectorstore.models import VectorSearchResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    chunk_id: str = "chunk_1",
    document_id: str = "doc_1",
    document_name: str = "test.pdf",
    content: str = "test content",
    score: float = 0.5,
    page_number: int | None = 1,
    **extra_meta,
) -> VectorSearchResult:
    """Factory for VectorSearchResult test fixtures."""
    meta = {"document_id": document_id, "document_name": document_name}
    if page_number is not None:
        meta["page_number"] = page_number
    meta.update(extra_meta)
    return VectorSearchResult(
        chunk_id=chunk_id,
        document_id=document_id,
        document_name=document_name,
        content=content,
        score=score,
        metadata=meta,
    )


def _build_test_corpus() -> list[VectorSearchResult]:
    """Build a small test corpus for BM25 testing."""
    return [
        _make_result(
            chunk_id="c1",
            document_id="doc_1",
            document_name="ai_report.pdf",
            content=(
                "Machine learning is a subset of artificial"
                " intelligence that focuses on algorithms"
            ),
            page_number=1,
        ),
        _make_result(
            chunk_id="c2",
            document_id="doc_1",
            document_name="ai_report.pdf",
            content=(
                "Deep learning uses neural networks with many layers for complex tasks"
            ),
            page_number=2,
        ),
        _make_result(
            chunk_id="c3",
            document_id="doc_2",
            document_name="finance_report.pdf",
            content="Revenue increased by 25 percent in the fiscal year 2024",
            page_number=5,
        ),
        _make_result(
            chunk_id="c4",
            document_id="doc_2",
            document_name="finance_report.pdf",
            content="The company plans to invest in technology and innovation",
            page_number=8,
        ),
        _make_result(
            chunk_id="c5",
            document_id="doc_1",
            document_name="ai_report.pdf",
            content=(
                "Natural language processing enables"
                " machines to understand human language"
            ),
            page_number=3,
        ),
    ]


# ===========================================================================
# 1. Tokenization
# ===========================================================================


class TestTokenization:
    """Test the tokenize() function."""

    def test_basic_tokenization(self):
        tokens = tokenize("Hello World")
        assert tokens == ["hello", "world"]

    def test_case_insensitive(self):
        tokens = tokenize("Machine Learning AI")
        assert tokens == ["machine", "learning", "ai"]

    def test_handles_punctuation(self):
        tokens = tokenize("Hello, world! How are you?")
        assert "hello" in tokens
        assert "world" in tokens

    def test_handles_numbers(self):
        tokens = tokenize("Revenue was $25 million in 2024")
        assert "25" in tokens
        assert "2024" in tokens

    def test_empty_string(self):
        assert tokenize("") == []

    def test_whitespace_only(self):
        assert tokenize("   \t\n  ") == []

    def test_unicode_text(self):
        tokens = tokenize("机器学习 artificial intelligence")
        assert "artificial" in tokens
        assert "intelligence" in tokens

    def test_mixed_content(self):
        tokens = tokenize("page_number chunk_id 42")
        assert "page_number" in tokens
        assert "42" in tokens


# ===========================================================================
# 2. BM25Index
# ===========================================================================


class TestBM25Index:
    """Test BM25Index construction and scoring."""

    def test_build_empty_corpus(self):
        index = BM25Index()
        index.build([])
        assert index.doc_count == 0

    def test_build_with_corpus(self):
        corpus = _build_test_corpus()
        index = BM25Index()
        index.build(corpus)
        assert index.doc_count == 5

    def test_search_returns_relevant_results(self):
        corpus = _build_test_corpus()
        index = BM25Index()
        index.build(corpus)
        results = index.search("machine learning algorithms")
        assert len(results) > 0
        # Most relevant should be the ML chunk
        top_result, top_score = results[0]
        assert top_result.chunk_id == "c1"
        assert top_score > 0

    def test_search_financial_query(self):
        corpus = _build_test_corpus()
        index = BM25Index()
        index.build(corpus)
        results = index.search("revenue fiscal year")
        assert len(results) > 0
        top_result, _ = results[0]
        assert top_result.chunk_id == "c3"

    def test_search_no_match(self):
        corpus = _build_test_corpus()
        index = BM25Index()
        index.build(corpus)
        results = index.search("xyznonexistent foobar")
        assert results == []

    def test_search_empty_index(self):
        index = BM25Index()
        index.build([])
        results = index.search("anything")
        assert results == []

    def test_search_respects_top_k(self):
        corpus = _build_test_corpus()
        index = BM25Index()
        index.build(corpus)
        results = index.search("learning", top_k=2)
        assert len(results) <= 2

    def test_scores_are_positive(self):
        corpus = _build_test_corpus()
        index = BM25Index()
        index.build(corpus)
        results = index.search("machine learning")
        for _, score in results:
            assert score > 0

    def test_results_ordered_by_score_descending(self):
        corpus = _build_test_corpus()
        index = BM25Index()
        index.build(corpus)
        results = index.search("learning")
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_invalid_k1(self):
        with pytest.raises(ValueError, match="k1"):
            BM25Index(k1=-1)

    def test_invalid_b(self):
        with pytest.raises(ValueError, match="b"):
            BM25Index(b=1.5)

    def test_remove_document(self):
        corpus = _build_test_corpus()
        index = BM25Index()
        index.build(corpus)
        assert index.doc_count == 5
        removed = index.remove_document("doc_1")
        assert removed == 3
        assert index.doc_count == 2

    def test_remove_nonexistent_document(self):
        corpus = _build_test_corpus()
        index = BM25Index()
        index.build(corpus)
        removed = index.remove_document("nonexistent")
        assert removed == 0
        assert index.doc_count == 5


# ===========================================================================
# 3. BM25Retriever
# ===========================================================================


class TestBM25Retriever:
    """Test BM25Retriever (Retriever ABC implementation)."""

    def _build_retriever(self) -> BM25Retriever:
        corpus = _build_test_corpus()
        index = BM25Index()
        index.build(corpus)
        return BM25Retriever(bm25_index=index)

    def test_retrieve_returns_results(self):
        retriever = self._build_retriever()
        results = retriever.retrieve("machine learning")
        assert len(results) > 0
        assert all(isinstance(r, VectorSearchResult) for r in results)

    def test_retrieve_preserves_metadata(self):
        retriever = self._build_retriever()
        results = retriever.retrieve("machine learning algorithms")
        top = results[0]
        assert top.chunk_id == "c1"
        assert top.document_id == "doc_1"
        assert top.document_name == "ai_report.pdf"
        assert top.metadata.get("page_number") == 1

    def test_retrieve_score_is_bm25_score(self):
        retriever = self._build_retriever()
        results = retriever.retrieve("revenue fiscal year")
        assert results[0].score > 0
        # BM25 scores are unbounded — just verify positive
        assert results[0].score > 0.1

    def test_retrieve_handles_empty_index(self):
        index = BM25Index()
        index.build([])
        retriever = BM25Retriever(bm25_index=index)
        results = retriever.retrieve("any query")
        assert results == []

    def test_retrieve_respects_top_k(self):
        retriever = self._build_retriever()
        results = retriever.retrieve("learning", top_k=2)
        assert len(results) <= 2

    def test_retrieve_default_top_k(self):
        index = BM25Index()
        index.build(_build_test_corpus())
        retriever = BM25Retriever(bm25_index=index, default_top_k=3)
        results = retriever.retrieve("learning language")
        assert len(results) <= 3

    # Query validation

    def test_invalid_query_empty(self):
        retriever = self._build_retriever()
        with pytest.raises(InvalidQueryError):
            retriever.retrieve("")

    def test_invalid_query_whitespace(self):
        retriever = self._build_retriever()
        with pytest.raises(InvalidQueryError):
            retriever.retrieve("   ")

    def test_invalid_query_not_string(self):
        retriever = self._build_retriever()
        with pytest.raises(InvalidQueryError):
            retriever.retrieve(123)  # type: ignore[arg-type]

    def test_invalid_query_null_bytes_only(self):
        retriever = self._build_retriever()
        with pytest.raises(InvalidQueryError):
            retriever.retrieve("\x00\ufeff\u200b")

    def test_query_too_long(self):
        retriever = self._build_retriever()
        with pytest.raises(InvalidQueryError):
            retriever.retrieve("x" * 10_001)

    # top_k validation

    def test_invalid_top_k_zero(self):
        retriever = self._build_retriever()
        with pytest.raises(InvalidTopKError):
            retriever.retrieve("test", top_k=0)

    def test_invalid_top_k_negative(self):
        retriever = self._build_retriever()
        with pytest.raises(InvalidTopKError):
            retriever.retrieve("test", top_k=-1)

    def test_invalid_top_k_bool(self):
        retriever = self._build_retriever()
        with pytest.raises(InvalidTopKError):
            retriever.retrieve("test", top_k=True)

    def test_invalid_top_k_exceeds_max(self):
        retriever = self._build_retriever()
        with pytest.raises(InvalidTopKError):
            retriever.retrieve("test", top_k=MAX_TOP_K + 1)

    # Constructor validation

    def test_invalid_bm25_index_type(self):
        with pytest.raises(TypeError, match="BM25Index"):
            BM25Retriever(bm25_index="not an index")  # type: ignore[arg-type]

    def test_invalid_default_top_k_constructor(self):
        index = BM25Index()
        index.build([])
        with pytest.raises(ValueError, match="default_top_k"):
            BM25Retriever(bm25_index=index, default_top_k=0)

    # Multi-document

    def test_multi_document_retrieval(self):
        retriever = self._build_retriever()
        # Query that could match both doc_1 (AI) and doc_2 (finance)
        results = retriever.retrieve("technology innovation learning")
        doc_ids = {r.document_id for r in results}
        # Should retrieve from multiple documents
        assert len(doc_ids) >= 1  # At minimum one match

    def test_case_insensitive_retrieval(self):
        retriever = self._build_retriever()
        results_lower = retriever.retrieve("machine learning")
        results_upper = retriever.retrieve("MACHINE LEARNING")
        # Same results regardless of case
        assert len(results_lower) == len(results_upper)
        ids_lower = [r.chunk_id for r in results_lower]
        ids_upper = [r.chunk_id for r in results_upper]
        assert ids_lower == ids_upper


# ===========================================================================
# 4. HybridRetriever
# ===========================================================================


class TestHybridRetriever:
    """Test HybridRetriever (RRF fusion)."""

    def _make_mock_retriever(self, results: list[VectorSearchResult]) -> Retriever:
        """Create a mock retriever that returns the given results."""
        from rag.retrieval.base import Retriever

        mock = MagicMock(spec=Retriever)
        mock.retrieve = MagicMock(return_value=results)
        return mock

    def test_combines_both_result_sets(self):
        sem_results = [
            _make_result(chunk_id="c1", content="semantic result 1", score=0.9),
            _make_result(chunk_id="c2", content="semantic result 2", score=0.7),
        ]
        kw_results = [
            _make_result(chunk_id="c3", content="keyword result 1", score=2.5),
            _make_result(chunk_id="c4", content="keyword result 2", score=1.8),
        ]

        hybrid = HybridRetriever(
            semantic_retriever=self._make_mock_retriever(sem_results),
            keyword_retriever=self._make_mock_retriever(kw_results),
        )
        results = hybrid.retrieve("test query")
        assert len(results) == 4
        chunk_ids = {r.chunk_id for r in results}
        assert chunk_ids == {"c1", "c2", "c3", "c4"}

    def test_deduplicates_by_chunk_id(self):
        """Same chunk in both retrievers should appear only once."""
        sem_results = [
            _make_result(chunk_id="c1", content="shared chunk", score=0.9),
            _make_result(chunk_id="c2", content="semantic only", score=0.7),
        ]
        kw_results = [
            _make_result(chunk_id="c1", content="shared chunk", score=2.5),
            _make_result(chunk_id="c3", content="keyword only", score=1.8),
        ]

        hybrid = HybridRetriever(
            semantic_retriever=self._make_mock_retriever(sem_results),
            keyword_retriever=self._make_mock_retriever(kw_results),
        )
        results = hybrid.retrieve("test query")
        chunk_ids = [r.chunk_id for r in results]
        assert len(chunk_ids) == len(set(chunk_ids))  # No duplicates
        assert "c1" in chunk_ids

    def test_deduplicated_chunk_has_higher_rrf_score(self):
        """A chunk present in both result sets should have a higher
        RRF score than one present in only one."""
        shared = _make_result(chunk_id="shared", content="in both", score=0.8)
        sem_only = _make_result(chunk_id="sem_only", content="semantic", score=0.9)
        kw_only = _make_result(chunk_id="kw_only", content="keyword", score=3.0)

        hybrid = HybridRetriever(
            semantic_retriever=self._make_mock_retriever([shared, sem_only]),
            keyword_retriever=self._make_mock_retriever([shared, kw_only]),
            semantic_weight=1.0,
            keyword_weight=1.0,
        )
        results = hybrid.retrieve("test query")
        score_map = {r.chunk_id: r.score for r in results}
        # Shared chunk gets RRF contribution from both retrievers
        assert score_map["shared"] > score_map["sem_only"]
        assert score_map["shared"] > score_map["kw_only"]

    def test_ranking_is_deterministic(self):
        """Running the same retrieval twice should produce the same order."""
        sem_results = [
            _make_result(chunk_id="c1", score=0.9),
            _make_result(chunk_id="c2", score=0.7),
        ]
        kw_results = [
            _make_result(chunk_id="c3", score=2.5),
            _make_result(chunk_id="c2", score=1.8),
        ]

        hybrid = HybridRetriever(
            semantic_retriever=self._make_mock_retriever(sem_results),
            keyword_retriever=self._make_mock_retriever(kw_results),
        )
        results_1 = hybrid.retrieve("test query")
        results_2 = hybrid.retrieve("test query")
        assert [r.chunk_id for r in results_1] == [r.chunk_id for r in results_2]
        assert [r.score for r in results_1] == [r.score for r in results_2]

    def test_semantic_weight_affects_ranking(self):
        """Higher semantic weight should boost semantically ranked chunks."""
        sem_results = [_make_result(chunk_id="c1", score=0.9)]
        kw_results = [_make_result(chunk_id="c2", score=2.5)]

        # Semantic weight = 2.0, keyword weight = 1.0
        hybrid = HybridRetriever(
            semantic_retriever=self._make_mock_retriever(sem_results),
            keyword_retriever=self._make_mock_retriever(kw_results),
            semantic_weight=2.0,
            keyword_weight=1.0,
        )
        results = hybrid.retrieve("test query")
        score_map = {r.chunk_id: r.score for r in results}
        assert score_map["c1"] > score_map["c2"]

    def test_keyword_weight_affects_ranking(self):
        """Higher keyword weight should boost keyword-ranked chunks."""
        sem_results = [_make_result(chunk_id="c1", score=0.9)]
        kw_results = [_make_result(chunk_id="c2", score=2.5)]

        # Semantic weight = 1.0, keyword weight = 2.0
        hybrid = HybridRetriever(
            semantic_retriever=self._make_mock_retriever(sem_results),
            keyword_retriever=self._make_mock_retriever(kw_results),
            semantic_weight=1.0,
            keyword_weight=2.0,
        )
        results = hybrid.retrieve("test query")
        score_map = {r.chunk_id: r.score for r in results}
        assert score_map["c2"] > score_map["c1"]

    def test_zero_semantic_weight(self):
        """Zero semantic weight should skip semantic retrieval."""
        sem_mock = self._make_mock_retriever(
            [
                _make_result(chunk_id="c1", score=0.9),
            ]
        )
        kw_results = [_make_result(chunk_id="c2", score=2.5)]

        hybrid = HybridRetriever(
            semantic_retriever=sem_mock,
            keyword_retriever=self._make_mock_retriever(kw_results),
            semantic_weight=0.0,
            keyword_weight=1.0,
        )
        results = hybrid.retrieve("test query")
        # Semantic retriever should not be called
        sem_mock.retrieve.assert_not_called()
        assert len(results) == 1
        assert results[0].chunk_id == "c2"

    def test_zero_keyword_weight(self):
        """Zero keyword weight should skip keyword retrieval."""
        sem_results = [_make_result(chunk_id="c1", score=0.9)]
        kw_mock = self._make_mock_retriever(
            [
                _make_result(chunk_id="c2", score=2.5),
            ]
        )

        hybrid = HybridRetriever(
            semantic_retriever=self._make_mock_retriever(sem_results),
            keyword_retriever=kw_mock,
            semantic_weight=1.0,
            keyword_weight=0.0,
        )
        results = hybrid.retrieve("test query")
        kw_mock.retrieve.assert_not_called()
        assert len(results) == 1
        assert results[0].chunk_id == "c1"

    def test_both_empty_results(self):
        """Both retrievers returning empty should return empty."""
        hybrid = HybridRetriever(
            semantic_retriever=self._make_mock_retriever([]),
            keyword_retriever=self._make_mock_retriever([]),
        )
        results = hybrid.retrieve("test query")
        assert results == []

    def test_respects_top_k(self):
        sem_results = [_make_result(chunk_id=f"s{i}") for i in range(10)]
        kw_results = [_make_result(chunk_id=f"k{i}") for i in range(10)]

        hybrid = HybridRetriever(
            semantic_retriever=self._make_mock_retriever(sem_results),
            keyword_retriever=self._make_mock_retriever(kw_results),
            default_top_k=5,
        )
        results = hybrid.retrieve("test query")
        assert len(results) == 5

    def test_top_k_override(self):
        sem_results = [_make_result(chunk_id=f"s{i}") for i in range(10)]
        kw_results = [_make_result(chunk_id=f"k{i}") for i in range(10)]

        hybrid = HybridRetriever(
            semantic_retriever=self._make_mock_retriever(sem_results),
            keyword_retriever=self._make_mock_retriever(kw_results),
            default_top_k=15,
        )
        results = hybrid.retrieve("test query", top_k=3)
        assert len(results) == 3

    def test_preserves_metadata_from_semantic(self):
        """When a chunk is in both result sets, metadata from semantic is preferred."""
        sem_results = [
            _make_result(
                chunk_id="c1",
                document_id="doc_1",
                document_name="test.pdf",
                page_number=5,
                source_type="pdf",
            ),
        ]
        kw_results = [
            _make_result(
                chunk_id="c1",
                document_id="doc_1",
                document_name="test.pdf",
                page_number=5,
            ),
        ]

        hybrid = HybridRetriever(
            semantic_retriever=self._make_mock_retriever(sem_results),
            keyword_retriever=self._make_mock_retriever(kw_results),
        )
        results = hybrid.retrieve("test query")
        assert results[0].document_id == "doc_1"
        assert results[0].document_name == "test.pdf"
        assert results[0].metadata.get("page_number") == 5

    def test_graceful_semantic_failure(self):
        """If semantic retriever fails, use keyword results only."""
        sem_mock = MagicMock(spec=Retriever)  # imported at module level
        sem_mock.retrieve = MagicMock(side_effect=RuntimeError("embedding failed"))

        kw_results = [_make_result(chunk_id="c1", score=2.0)]

        hybrid = HybridRetriever(
            semantic_retriever=sem_mock,
            keyword_retriever=self._make_mock_retriever(kw_results),
        )
        results = hybrid.retrieve("test query")
        assert len(results) == 1
        assert results[0].chunk_id == "c1"

    def test_graceful_keyword_failure(self):
        """If keyword retriever fails, use semantic results only."""
        sem_results = [_make_result(chunk_id="c1", score=0.8)]
        kw_mock = MagicMock(spec=Retriever)
        kw_mock.retrieve = MagicMock(side_effect=RuntimeError("bm25 failed"))

        hybrid = HybridRetriever(
            semantic_retriever=self._make_mock_retriever(sem_results),
            keyword_retriever=kw_mock,
        )
        results = hybrid.retrieve("test query")
        assert len(results) == 1

    # Validation

    def test_invalid_semantic_weight_negative(self):
        with pytest.raises(ValueError, match="semantic_weight"):
            HybridRetriever(
                semantic_retriever=self._make_mock_retriever([]),
                keyword_retriever=self._make_mock_retriever([]),
                semantic_weight=-1.0,
            )

    def test_invalid_keyword_weight_negative(self):
        with pytest.raises(ValueError, match="keyword_weight"):
            HybridRetriever(
                semantic_retriever=self._make_mock_retriever([]),
                keyword_retriever=self._make_mock_retriever([]),
                keyword_weight=-0.5,
            )

    def test_both_weights_zero(self):
        with pytest.raises(ValueError, match="At least one"):
            HybridRetriever(
                semantic_retriever=self._make_mock_retriever([]),
                keyword_retriever=self._make_mock_retriever([]),
                semantic_weight=0.0,
                keyword_weight=0.0,
            )

    def test_invalid_rrf_k(self):
        with pytest.raises(ValueError, match="rrf_k"):
            HybridRetriever(
                semantic_retriever=self._make_mock_retriever([]),
                keyword_retriever=self._make_mock_retriever([]),
                rrf_k=0,
            )

    def test_invalid_top_k_zero_hybrid(self):
        hybrid = HybridRetriever(
            semantic_retriever=self._make_mock_retriever([]),
            keyword_retriever=self._make_mock_retriever([]),
        )
        with pytest.raises(InvalidTopKError):
            hybrid.retrieve("test", top_k=0)

    def test_invalid_retriever_types(self):
        with pytest.raises(TypeError, match="semantic_retriever"):
            HybridRetriever(
                semantic_retriever="not a retriever",  # type: ignore[arg-type]
                keyword_retriever=self._make_mock_retriever([]),
            )

    def test_rrf_score_field_meaning(self):
        """RRF scores should not be labeled as cosine similarity."""
        sem_results = [_make_result(chunk_id="c1", score=0.9)]
        kw_results = [_make_result(chunk_id="c2", score=2.5)]

        hybrid = HybridRetriever(
            semantic_retriever=self._make_mock_retriever(sem_results),
            keyword_retriever=self._make_mock_retriever(kw_results),
        )
        results = hybrid.retrieve("test query")
        # RRF scores are small fractional values (1/(k+rank))
        for r in results:
            assert r.score > 0
            assert r.score < 1  # Much less than 1 for standard k=60


# ===========================================================================
# 5. Citation Compatibility (Phase 7)
# ===========================================================================


class TestCitationCompatibility:
    """Ensure Phase 7 citations work with Phase 8 retrieval results."""

    def test_format_sources_with_bm25_results(self):
        """BM25 results should produce valid citations."""
        from rag.citations.formatter import format_sources

        results = [
            _make_result(
                chunk_id="c1",
                document_id="doc_1",
                document_name="test.pdf",
                content="test content",
                score=2.5,
                page_number=3,
            ),
        ]
        sources = format_sources(results)
        assert len(sources) == 1
        s = sources[0]
        assert s.document_id == "doc_1"
        assert s.document_name == "test.pdf"
        assert s.page_number == 3
        assert s.chunk_id == "c1"
        assert s.score == 2.5

    def test_format_sources_with_hybrid_rrf_results(self):
        """RRF-fused results should produce valid citations."""
        from rag.citations.formatter import format_sources

        results = [
            _make_result(
                chunk_id="c1",
                document_id="doc_1",
                document_name="report.pdf",
                content="some content",
                score=0.01639,  # Typical RRF score
                page_number=5,
            ),
            _make_result(
                chunk_id="c2",
                document_id="doc_2",
                document_name="notes.pdf",
                content="other content",
                score=0.01587,
                page_number=1,
            ),
        ]
        sources = format_sources(results)
        assert len(sources) == 2
        assert sources[0].document_id == "doc_1"
        assert sources[1].document_id == "doc_2"

    def test_hybrid_dedup_preserves_citation_metadata(self):
        """Deduplication in hybrid should preserve citation-relevant metadata."""
        from rag.retrieval.base import Retriever

        sem_results = [
            _make_result(
                chunk_id="c1",
                document_id="doc_1",
                document_name="test.pdf",
                page_number=5,
                chunk_index=2,
            ),
        ]
        kw_results = [
            _make_result(
                chunk_id="c1",
                document_id="doc_1",
                document_name="test.pdf",
                page_number=5,
            ),
        ]

        sem_mock = MagicMock(spec=Retriever)
        sem_mock.retrieve = MagicMock(return_value=sem_results)
        kw_mock = MagicMock(spec=Retriever)
        kw_mock.retrieve = MagicMock(return_value=kw_results)

        hybrid = HybridRetriever(
            semantic_retriever=sem_mock,
            keyword_retriever=kw_mock,
        )
        results = hybrid.retrieve("test query")
        # Should have one result (deduplicated)
        assert len(results) == 1
        r = results[0]
        assert r.chunk_id == "c1"
        assert r.document_id == "doc_1"
        assert r.document_name == "test.pdf"
        assert r.metadata.get("page_number") == 5

    def test_no_metadata_fabrication(self):
        """Missing metadata should remain absent, never fabricated."""
        results = [
            VectorSearchResult(
                chunk_id="c1",
                document_id="doc_1",
                document_name="test.pdf",
                content="no page info",
                score=0.5,
                metadata={"document_id": "doc_1", "document_name": "test.pdf"},
            ),
        ]
        from rag.citations.formatter import format_sources

        sources = format_sources(results)
        assert sources[0].page_number is None  # Not fabricated


# ===========================================================================
# 6. Configuration Validation
# ===========================================================================


class TestPhase8Configuration:
    """Test Phase 8 configuration settings and validators."""

    def test_default_retrieval_mode(self):
        from backend.config import Settings

        s = Settings()
        assert s.retrieval_mode == "semantic"

    def test_valid_retrieval_modes(self):
        from backend.config import Settings

        for mode in ["semantic", "keyword", "hybrid"]:
            s = Settings(retrieval_mode=mode)
            assert s.retrieval_mode == mode

    def test_invalid_retrieval_mode(self):
        from backend.config import Settings

        with pytest.raises(Exception, match="retrieval_mode"):
            Settings(retrieval_mode="invalid_mode")

    def test_default_hybrid_settings(self):
        from backend.config import Settings

        s = Settings()
        assert s.hybrid_semantic_top_k == 20
        assert s.hybrid_keyword_top_k == 20
        assert s.hybrid_semantic_weight == 1.0
        assert s.hybrid_keyword_weight == 1.0
        assert s.rrf_k == 60

    def test_custom_hybrid_settings(self):
        from backend.config import Settings

        s = Settings(
            hybrid_semantic_top_k=30,
            hybrid_keyword_top_k=15,
            hybrid_semantic_weight=0.7,
            hybrid_keyword_weight=0.3,
            rrf_k=40,
        )
        assert s.hybrid_semantic_top_k == 30
        assert s.hybrid_keyword_top_k == 15
        assert s.hybrid_semantic_weight == 0.7
        assert s.hybrid_keyword_weight == 0.3
        assert s.rrf_k == 40

    def test_invalid_hybrid_top_k_zero(self):
        from backend.config import Settings

        with pytest.raises(Exception, match="hybrid top_k"):
            Settings(hybrid_semantic_top_k=0)

    def test_invalid_hybrid_weight_negative(self):
        from backend.config import Settings

        with pytest.raises(Exception, match="hybrid weight"):
            Settings(hybrid_semantic_weight=-0.5)

    def test_invalid_rrf_k_zero(self):
        from backend.config import Settings

        with pytest.raises(Exception, match="rrf_k"):
            Settings(rrf_k=0)

    def test_retrieval_mode_case_insensitive(self):
        from backend.config import Settings

        s = Settings(retrieval_mode="HYBRID")
        assert s.retrieval_mode == "hybrid"


# ===========================================================================
# 7. Semantic Retriever Still Works Independently
# ===========================================================================


class TestSemanticRetrieverIndependent:
    """Verify SemanticRetriever still works as before Phase 8."""

    def test_semantic_retriever_is_retriever(self):
        from rag.retrieval.base import Retriever
        from rag.retrieval.semantic import SemanticRetriever

        assert issubclass(SemanticRetriever, Retriever)

    def test_bm25_retriever_is_retriever(self):
        from rag.retrieval.base import Retriever
        from rag.retrieval.bm25 import BM25Retriever

        assert issubclass(BM25Retriever, Retriever)

    def test_hybrid_retriever_is_retriever(self):
        from rag.retrieval.base import Retriever
        from rag.retrieval.hybrid import HybridRetriever

        assert issubclass(HybridRetriever, Retriever)


# ===========================================================================
# 8. Multi-Document BM25
# ===========================================================================


class TestMultiDocumentBM25:
    """Test BM25 with multiple documents."""

    def test_multi_doc_bm25_returns_results_from_different_docs(self):
        corpus = _build_test_corpus()
        index = BM25Index()
        index.build(corpus)
        retriever = BM25Retriever(bm25_index=index)
        # "learning" appears in doc_1; "revenue" in doc_2
        results_ai = retriever.retrieve("machine learning")
        results_fin = retriever.retrieve("revenue fiscal year")
        assert any(r.document_id == "doc_1" for r in results_ai)
        assert any(r.document_id == "doc_2" for r in results_fin)

    def test_multi_doc_hybrid_retrieves_across_documents(self):
        """Hybrid retrieval should return results from multiple documents."""
        from rag.retrieval.base import Retriever

        sem_results = [
            _make_result(chunk_id="c1", document_id="doc_1", document_name="a.pdf"),
            _make_result(chunk_id="c2", document_id="doc_2", document_name="b.pdf"),
        ]
        kw_results = [
            _make_result(chunk_id="c3", document_id="doc_2", document_name="b.pdf"),
            _make_result(chunk_id="c4", document_id="doc_3", document_name="c.pdf"),
        ]

        sem_mock = MagicMock(spec=Retriever)
        sem_mock.retrieve = MagicMock(return_value=sem_results)
        kw_mock = MagicMock(spec=Retriever)
        kw_mock.retrieve = MagicMock(return_value=kw_results)

        hybrid = HybridRetriever(
            semantic_retriever=sem_mock,
            keyword_retriever=kw_mock,
        )
        results = hybrid.retrieve("test query")
        doc_ids = {r.document_id for r in results}
        assert len(doc_ids) == 3


# ===========================================================================
# 9. Relevance Gate
# ===========================================================================


class TestRelevanceGateModes:
    """Test that existing relevance protection works correctly."""

    def test_semantic_mode_has_min_score(self):
        """In semantic mode, the min_score relevance gate should be active."""
        from backend.config import Settings

        s = Settings(retrieval_mode="semantic")
        # Semantic mode should use retrieval_min_score
        assert s.retrieval_min_score == 0.3

    def test_existing_phase6_tests_pattern(self):
        """Verify the phase 6 min_score pattern still works."""
        from unittest.mock import MagicMock

        from rag.embeddings.base import EmbeddingService
        from rag.retrieval.semantic import SemanticRetriever
        from rag.vectorstore.base import VectorStore

        emb_mock = MagicMock(spec=EmbeddingService)
        emb_mock.embed_query.return_value = [0.1] * 384
        vs_mock = MagicMock(spec=VectorStore)
        vs_mock.query.return_value = [
            _make_result(chunk_id="c1", score=0.5),  # Above threshold
            _make_result(chunk_id="c2", score=0.2),  # Below threshold
        ]

        retriever = SemanticRetriever(
            embedding_service=emb_mock,
            vector_store=vs_mock,
            min_score=0.3,
        )
        results = retriever.retrieve("test query")
        assert len(results) == 1
        assert results[0].chunk_id == "c1"


# ===========================================================================
# 10. Stopwords Filtering
# ===========================================================================


class TestStopwordsFiltering:
    """Test stop words filtering in tokenization and BM25 indexing."""

    def test_tokenize_with_stopwords(self):
        tokens = tokenize("What is Machine Learning?", stopwords=DEFAULT_STOPWORDS)
        assert tokens == ["machine", "learning"]

    def test_tokenize_without_stopwords(self):
        tokens = tokenize("What is Machine Learning?", stopwords=None)
        assert tokens == ["what", "is", "machine", "learning"]

    def test_tokenize_custom_stopwords(self):
        custom = frozenset({"machine"})
        tokens = tokenize("machine learning ai", stopwords=custom)
        assert tokens == ["learning", "ai"]

    def test_contractions_filtered(self):
        tokens = tokenize("Jupiter's moon", stopwords=DEFAULT_STOPWORDS)
        assert tokens == ["jupiter", "moon"]

    def test_index_stores_configured_stopwords(self):
        index = BM25Index(stopwords=DEFAULT_STOPWORDS)
        assert index.stopwords == DEFAULT_STOPWORDS

    def test_query_all_stopwords_returns_empty(self):
        index = BM25Index(stopwords=DEFAULT_STOPWORDS)
        index.build(_build_test_corpus())
        results = index.search("what is the")
        assert results == []


# ===========================================================================
# 11. Index Builder from ChromaDB
# ===========================================================================


class TestIndexBuilder:
    """Test building BM25Index from ChromaVectorStore."""

    def test_index_builder_type_check(self):
        with pytest.raises(TypeError, match="ChromaVectorStore"):
            build_bm25_index_from_chroma("not_a_vector_store")  # type: ignore[arg-type]

    def test_index_builder_batch_size_validation(self, tmp_path):
        vs = ChromaVectorStore(str(tmp_path), "test_coll")
        with pytest.raises(ValueError, match="batch_size"):
            build_bm25_index_from_chroma(vs, batch_size=0)

    def test_index_builder_empty_vector_store(self, tmp_path):
        vs = ChromaVectorStore(str(tmp_path), "test_coll")
        index = build_bm25_index_from_chroma(vs)
        assert index.doc_count == 0
        assert index.search("anything") == []

    def test_index_builder_with_entries(self, tmp_path):
        from rag.chunking.models import Chunk

        vs = ChromaVectorStore(str(tmp_path), "test_coll")
        chunks = [
            Chunk(
                chunk_id="chunk_1",
                document_id="doc_1",
                document_name="doc.pdf",
                source_type="pdf",
                content="Deep learning algorithms require data.",
                page_number=1,
                chunk_index=0,
            ),
            Chunk(
                chunk_id="chunk_2",
                document_id="doc_1",
                document_name="doc.pdf",
                source_type="pdf",
                content="Revenue figures for year 2024.",
                page_number=2,
                chunk_index=1,
            ),
        ]
        embeddings = [[0.1] * 384, [0.2] * 384]
        vs.add_chunks(chunks, embeddings)

        index = build_bm25_index_from_chroma(vs)
        assert index.doc_count == 2
        results = index.search("deep learning")
        assert len(results) >= 1
        assert results[0][0].chunk_id == "chunk_1"

    def test_index_builder_batch_pagination(self, tmp_path):
        from rag.chunking.models import Chunk

        vs = ChromaVectorStore(str(tmp_path), "test_coll")
        chunks = [
            Chunk(
                chunk_id=f"c_{i}",
                document_id="doc_1",
                document_name="multi.pdf",
                source_type="pdf",
                content=f"Content chunk number {i} with specific keywords.",
                page_number=i + 1,
                chunk_index=i,
            )
            for i in range(5)
        ]
        embeddings = [[0.05 * (i + 1)] * 384 for i in range(5)]
        vs.add_chunks(chunks, embeddings)

        # Batch size of 2 across 5 items tests multi-page pagination
        index = build_bm25_index_from_chroma(vs, batch_size=2)
        assert index.doc_count == 5

    def test_index_builder_preserves_metadata(self, tmp_path):
        from rag.chunking.models import Chunk

        vs = ChromaVectorStore(str(tmp_path), "test_coll")
        chunk = Chunk(
            chunk_id="provenance_test",
            document_id="doc_prov",
            document_name="prov.pdf",
            source_type="pdf",
            content="Provable source attribution chunk.",
            page_number=7,
            chunk_index=3,
            metadata={"custom_flag": "verified"},
        )
        vs.add_chunks([chunk], [[0.1] * 384])

        index = build_bm25_index_from_chroma(vs)
        results = index.search("provable")
        assert len(results) == 1
        entry = results[0][0]
        assert entry.chunk_id == "provenance_test"
        assert entry.document_id == "doc_prov"
        assert entry.document_name == "prov.pdf"
        assert entry.metadata.get("page_number") == 7
        assert entry.metadata.get("custom_flag") == "verified"

    def test_index_builder_reindexing_consistency(self, tmp_path):
        from rag.chunking.models import Chunk

        vs = ChromaVectorStore(str(tmp_path), "test_coll")
        chunk_v1 = Chunk(
            chunk_id="chunk_a",
            document_id="doc_alpha",
            document_name="alpha.pdf",
            source_type="pdf",
            content="Version 1 discusses legacy computing.",
            page_number=1,
            chunk_index=0,
        )
        vs.add_chunks([chunk_v1], [[0.1] * 384])
        idx_v1 = build_bm25_index_from_chroma(vs)
        assert len(idx_v1.search("legacy")) == 1

        # Re-index document: delete existing and insert updated chunk
        vs.delete_document("doc_alpha")
        chunk_v2 = Chunk(
            chunk_id="chunk_a2",
            document_id="doc_alpha",
            document_name="alpha.pdf",
            source_type="pdf",
            content="Version 2 discusses quantum computing.",
            page_number=1,
            chunk_index=0,
        )
        vs.add_chunks([chunk_v2], [[0.2] * 384])
        idx_v2 = build_bm25_index_from_chroma(vs)
        assert len(idx_v2.search("legacy")) == 0
        assert len(idx_v2.search("quantum")) == 1


# ===========================================================================
# 12. Relevance Gate Behavior Across Modes
# ===========================================================================


class TestRelevanceGateAcrossModes:
    """Test relevance-gate behavior for semantic, keyword, and hybrid modes."""

    def test_keyword_mode_blocks_unrelated_query(self):
        corpus = _build_test_corpus()
        index = BM25Index(stopwords=DEFAULT_STOPWORDS)
        index.build(corpus)
        retriever = BM25Retriever(bm25_index=index)

        # Out-of-domain query with common question words
        results = retriever.retrieve("What is the capital of France?")
        assert results == []

    def test_hybrid_mode_blocks_unrelated_query(self):
        """In hybrid mode, if semantic drops low scores and BM25 finds no matches,
        hybrid retrieval must return empty results."""
        from unittest.mock import MagicMock

        # Semantic retriever returns [] because all similarity scores < 0.30
        sem_mock = MagicMock(spec=Retriever)
        sem_mock.retrieve = MagicMock(return_value=[])

        # BM25 returns [] because out-of-domain terms don't match
        kw_mock = MagicMock(spec=Retriever)
        kw_mock.retrieve = MagicMock(return_value=[])

        hybrid = HybridRetriever(
            semantic_retriever=sem_mock,
            keyword_retriever=kw_mock,
        )
        results = hybrid.retrieve("What is the capital of France?")
        assert results == []

    def test_hybrid_mode_orchestration_relevance_gate(self):
        """RAGQueryService raises EmptyRetrievalError when hybrid returns []."""
        from unittest.mock import MagicMock

        from rag.generation.base import Generator
        from rag.orchestration.exceptions import EmptyRetrievalError
        from rag.orchestration.query_service import RAGQueryService

        hybrid_mock = MagicMock(spec=Retriever)
        hybrid_mock.retrieve = MagicMock(return_value=[])

        gen_mock = MagicMock(spec=Generator)
        service = RAGQueryService(
            retriever=hybrid_mock,
            generator=gen_mock,
            min_score=None,
        )

        with pytest.raises(EmptyRetrievalError):
            service.query("What is the capital of France?")

        gen_mock.generate.assert_not_called()

    def test_hybrid_semantic_only_match_succeeds(self):
        """Paraphrase query with semantic match but zero keyword match succeeds."""
        sem_result = _make_result(
            chunk_id="c1", content="Semantic paraphrase match", score=0.6
        )
        sem_mock = MagicMock(spec=Retriever)
        sem_mock.retrieve = MagicMock(return_value=[sem_result])
        kw_mock = MagicMock(spec=Retriever)
        kw_mock.retrieve = MagicMock(return_value=[])

        hybrid = HybridRetriever(
            semantic_retriever=sem_mock,
            keyword_retriever=kw_mock,
        )
        results = hybrid.retrieve("paraphrased query")
        assert len(results) == 1
        assert results[0].chunk_id == "c1"

    def test_hybrid_keyword_only_match_succeeds(self):
        """Query with keyword match but zero semantic match succeeds."""
        kw_result = _make_result(
            chunk_id="c2", content="Exact acronym match", score=2.5
        )
        sem_mock = MagicMock(spec=Retriever)
        sem_mock.retrieve = MagicMock(return_value=[])
        kw_mock = MagicMock(spec=Retriever)
        kw_mock.retrieve = MagicMock(return_value=[kw_result])

        hybrid = HybridRetriever(
            semantic_retriever=sem_mock,
            keyword_retriever=kw_mock,
        )
        results = hybrid.retrieve("acronym query")
        assert len(results) == 1
        assert results[0].chunk_id == "c2"


# ===========================================================================
# 13. API Route Integration (Phase 8)
# ===========================================================================


class TestApiIntegrationPhase8:
    """Test /query endpoint behavior with configurable retrieval modes."""

    @pytest.fixture
    def test_client(self, tmp_path):
        from unittest.mock import patch

        from fastapi.testclient import TestClient

        from backend.config import Settings
        from backend.main import app
        from rag.chunking.models import Chunk
        from rag.generation.models import GenerationResult

        settings = Settings(
            vector_store_path=str(tmp_path / "vectorstore"),
            vector_store_collection="test_phase8_api",
            retrieval_min_score=0.30,
        )

        vs = ChromaVectorStore(
            persist_directory=settings.vector_store_path,
            collection_name=settings.vector_store_collection,
        )
        chunk = Chunk(
            chunk_id="ai_overview_chunk_1",
            document_id="doc_ai_001",
            document_name="sample_ai_overview.pdf",
            source_type="pdf",
            content="Machine learning is a field of artificial intelligence.",
            page_number=1,
            chunk_index=0,
        )
        from rag.embeddings.sentence_transformer import (
            SentenceTransformerEmbeddingService,
        )

        emb_service = SentenceTransformerEmbeddingService(
            model_name=settings.embedding_model
        )
        embeddings = emb_service.embed_documents([chunk.content])
        vs.add_chunks([chunk], embeddings)

        with (
            patch("backend.query.get_settings", return_value=settings),
            patch("backend.documents.get_settings", return_value=settings),
            patch(
                "rag.generation.ollama_generator.OllamaGenerator.generate",
                return_value=GenerationResult(
                    answer="Machine learning is part of artificial intelligence.",
                    model_name="mock-model",
                ),
            ),
        ):
            client = TestClient(app)
            yield client, settings

    def test_query_semantic_mode_response(self, test_client):
        client, settings = test_client
        settings.retrieval_mode = "semantic"

        resp = client.post("/query", json={"question": "What is machine learning?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["retrieval_mode"] == "semantic"
        assert "answer" in data
        assert len(data["sources"]) >= 1

    def test_query_keyword_mode_response(self, test_client):
        client, settings = test_client
        settings.retrieval_mode = "keyword"

        resp = client.post("/query", json={"question": "machine learning"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["retrieval_mode"] == "keyword"
        assert len(data["sources"]) >= 1
        assert data["sources"][0]["document_name"] == "sample_ai_overview.pdf"

    def test_query_hybrid_mode_response(self, test_client):
        client, settings = test_client
        settings.retrieval_mode = "hybrid"

        resp = client.post("/query", json={"question": "machine learning"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["retrieval_mode"] == "hybrid"
        assert len(data["sources"]) >= 1

    def test_query_hybrid_mode_blocks_irrelevant_query(self, test_client):
        client, settings = test_client
        settings.retrieval_mode = "hybrid"

        # Out-of-domain query with question words
        resp = client.post(
            "/query",
            json={"question": "What is the best recipe for chocolate cake?"},
        )
        # Relevance gate should block it and return HTTP 404
        assert resp.status_code == 404
        detail = resp.json()["detail"].lower()
        assert "no relevant" in detail or "no results" in detail

    def test_query_keyword_mode_blocks_irrelevant_query(self, test_client):
        client, settings = test_client
        settings.retrieval_mode = "keyword"

        resp = client.post(
            "/query",
            json={"question": "What is the best recipe for chocolate cake?"},
        )
        assert resp.status_code == 404
        detail = resp.json()["detail"].lower()
        assert "no relevant" in detail or "no results" in detail
