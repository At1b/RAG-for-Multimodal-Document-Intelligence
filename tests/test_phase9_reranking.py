"""Phase 9 — Reranking test suite.

Comprehensive unit and integration tests covering:
1. Reranker interface contract.
2. CrossEncoderReranker initialization, lazy loading, and scorer/model injection.
3. Input validation: query sanitization, top_k bounds, malformed candidates.
4. Candidate reordering, score semantics, and deterministic tie-breaking.
5. Failure modes: model initialization, inference errors, NaN/Inf scores.
6. RerankingRetriever composition with semantic, keyword, and hybrid retrievers.
7. Metadata and Phase 7 citation preservation (no fabrication, original score kept).
8. Relevance gate preservation: out-of-domain queries rejected before LLM.
9. RAGQueryService orchestration with reranking enabled and disabled.
10. Backend API /query endpoint with reranker configuration options.
11. Real CrossEncoder smoke test against cached model weights.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.config import Settings
from backend.main import app
from rag.citations.formatter import format_sources
from rag.citations.models import Source
from rag.generation.base import Generator
from rag.generation.context_builder import build_context
from rag.generation.models import GenerationResult
from rag.orchestration.exceptions import (
    EmptyRetrievalError,
    InsufficientContextError,
    QueryError,
)
from rag.orchestration.query_service import QueryResult, RAGQueryService
from rag.reranking.base import Reranker
from rag.reranking.cross_encoder import (
    DEFAULT_RERANKER_MODEL,
    MAX_QUERY_LENGTH,
    MAX_TOP_K,
    CrossEncoderReranker,
)
from rag.reranking.exceptions import (
    InvalidCandidateError,
    RerankerInferenceError,
    RerankerModelInitializationError,
)
from rag.retrieval.base import Retriever
from rag.retrieval.bm25 import BM25Index, BM25Retriever
from rag.retrieval.exceptions import InvalidQueryError, InvalidTopKError
from rag.retrieval.reranking import RerankingRetriever
from rag.vectorstore.models import VectorSearchResult

# ==========================================================================
# Helpers / Fixtures
# ==========================================================================


def _make_candidate(
    chunk_id: str,
    content: str,
    score: float = 0.5,
    document_id: str = "doc_1",
    document_name: str = "doc_1.pdf",
    page_number: int | None = 1,
    metadata: dict | None = None,
) -> VectorSearchResult:
    """Create a test candidate chunk."""
    meta = metadata.copy() if metadata else {}
    if "page_number" not in meta and page_number is not None:
        meta["page_number"] = page_number
    meta["chunk_id"] = chunk_id
    meta["document_id"] = document_id
    meta["document_name"] = document_name
    return VectorSearchResult(
        chunk_id=chunk_id,
        document_id=document_id,
        document_name=document_name,
        content=content,
        score=score,
        metadata=meta,
    )


class DummyRetriever(Retriever):
    """Test double for Retriever."""

    def __init__(self, results: list[VectorSearchResult] | None = None) -> None:
        self._results = results if results is not None else []
        self.last_query: str | None = None
        self.last_top_k: int | None = None
        self.last_min_score: float | None = None

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[VectorSearchResult]:
        self.last_query = query
        self.last_top_k = top_k
        self.last_min_score = min_score
        res = self._results
        if min_score is not None:
            res = [r for r in res if r.score >= min_score]
        if top_k is not None:
            res = res[:top_k]
        return res


class DummyGenerator(Generator):
    """Test double for Generator."""

    def __init__(self, answer: str = "Test answer") -> None:
        self.answer = answer
        self.last_question: str | None = None
        self.last_context: list[VectorSearchResult] | None = None

    def generate(
        self, question: str, context: list[VectorSearchResult]
    ) -> GenerationResult:
        self.last_question = question
        self.last_context = context
        return GenerationResult(
            answer=self.answer,
            model_name="dummy-model",
            metadata={"source_count": len(context)},
        )


# ==========================================================================
# 1. Reranker Interface Contract Tests
# ==========================================================================


class TestRerankerInterface:
    """Verify that Reranker ABC defines the required interface."""

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            Reranker()  # type: ignore

    def test_concrete_subclass_implements_rerank(self):
        class ValidReranker(Reranker):
            def rerank(
                self,
                query: str,
                candidates: list[VectorSearchResult],
                top_k: int | None = None,
            ) -> list[VectorSearchResult]:
                return candidates

        instance = ValidReranker()
        assert isinstance(instance, Reranker)
        assert instance.rerank("query", []) == []


# ==========================================================================
# 2. CrossEncoderReranker Initialization & Injection Tests
# ==========================================================================


class TestCrossEncoderRerankerInitAndInjection:
    """Verify initialization, default settings, and scorer injection."""

    def test_defaults(self):
        reranker = CrossEncoderReranker()
        assert reranker.model_name == DEFAULT_RERANKER_MODEL
        assert reranker._model is None  # Lazy loading: not loaded at init

    def test_custom_model_name(self):
        reranker = CrossEncoderReranker(model_name="custom/reranker-model")
        assert reranker.model_name == "custom/reranker-model"

    def test_invalid_model_name(self):
        with pytest.raises(ValueError, match="model_name"):
            CrossEncoderReranker(model_name="")
        with pytest.raises(ValueError, match="model_name"):
            CrossEncoderReranker(model_name="   ")
        with pytest.raises(ValueError, match="model_name"):
            CrossEncoderReranker(model_name=123)  # type: ignore

    def test_invalid_default_top_k(self):
        with pytest.raises(ValueError, match="default_top_k"):
            CrossEncoderReranker(default_top_k=0)
        with pytest.raises(ValueError, match="default_top_k"):
            CrossEncoderReranker(default_top_k=-5)
        with pytest.raises(ValueError, match="default_top_k"):
            CrossEncoderReranker(default_top_k=MAX_TOP_K + 1)
        with pytest.raises(ValueError, match="default_top_k"):
            CrossEncoderReranker(default_top_k=True)  # bool rejected

    def test_injected_scorer_bypasses_model_loading(self):
        """Injected scorer allows testing without sentence-transformers or downloads."""
        scorer_called = False

        def mock_scorer(pairs):
            nonlocal scorer_called
            scorer_called = True
            return [1.0] * len(pairs)

        reranker = CrossEncoderReranker(scorer=mock_scorer)
        candidates = [_make_candidate("c1", "content 1")]
        results = reranker.rerank("test query", candidates)

        assert scorer_called is True
        assert len(results) == 1
        assert reranker._model is None  # Real model was never initialized

    def test_injected_model_object_bypasses_model_loading(self):
        """Injected model object with predict() method is used directly."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [2.5]

        reranker = CrossEncoderReranker(model=mock_model)
        candidates = [_make_candidate("c1", "content 1")]
        results = reranker.rerank("test query", candidates)

        mock_model.predict.assert_called_once_with([("test query", "content 1")])
        assert results[0].metadata["reranker_score"] == 2.5

    def test_model_loading_failure_raises_initialization_error(self):
        reranker = CrossEncoderReranker(model_name="nonexistent-model-12345")
        with (
            patch(
                "sentence_transformers.CrossEncoder",
                side_effect=RuntimeError("Download failed"),
            ),
            pytest.raises(
                RerankerModelInitializationError, match="Failed to load CrossEncoder"
            ),
        ):
            reranker.rerank("query", [_make_candidate("c1", "content 1")])


# ==========================================================================
# 3. Validation Tests: Query, Candidates, Top-K
# ==========================================================================


class TestCrossEncoderRerankerValidation:
    """Verify input validation and edge-case sanitization."""

    @pytest.fixture
    def reranker(self):
        return CrossEncoderReranker(scorer=lambda pairs: [1.0] * len(pairs))

    def test_invalid_query_types(self, reranker):
        with pytest.raises(InvalidQueryError, match="query must be a string"):
            reranker.rerank(None, [_make_candidate("c1", "test")])  # type: ignore
        with pytest.raises(InvalidQueryError, match="query must be a string"):
            reranker.rerank(12345, [_make_candidate("c1", "test")])  # type: ignore

    def test_empty_or_whitespace_query(self, reranker):
        with pytest.raises(InvalidQueryError, match="non-empty string"):
            reranker.rerank("", [_make_candidate("c1", "test")])
        with pytest.raises(InvalidQueryError, match="non-empty string"):
            reranker.rerank("   \t\n  ", [_make_candidate("c1", "test")])

    def test_invisible_characters_only_query(self, reranker):
        with pytest.raises(InvalidQueryError, match="non-empty string"):
            reranker.rerank("\x00\ufeff\u200b", [_make_candidate("c1", "test")])

    def test_query_too_long(self, reranker):
        long_query = "a" * (MAX_QUERY_LENGTH + 1)
        with pytest.raises(InvalidQueryError, match="maximum length"):
            reranker.rerank(long_query, [_make_candidate("c1", "test")])

    def test_null_bytes_in_valid_query_sanitized(self, reranker):
        received_query = None

        def record_query(pairs):
            nonlocal received_query
            received_query = pairs[0][0]
            return [1.0]

        reranker._scorer = record_query
        reranker.rerank("hello\x00 world", [_make_candidate("c1", "test")])
        assert received_query == "hello world"

    def test_invalid_top_k(self, reranker):
        candidates = [_make_candidate("c1", "test")]
        with pytest.raises(InvalidTopKError):
            reranker.rerank("query", candidates, top_k=0)
        with pytest.raises(InvalidTopKError):
            reranker.rerank("query", candidates, top_k=-1)
        with pytest.raises(InvalidTopKError):
            reranker.rerank("query", candidates, top_k=True)  # bool
        with pytest.raises(InvalidTopKError):
            reranker.rerank("query", candidates, top_k=2.5)  # float
        with pytest.raises(InvalidTopKError):
            reranker.rerank("query", candidates, top_k=MAX_TOP_K + 1)

    def test_invalid_candidates_type(self, reranker):
        with pytest.raises(InvalidCandidateError, match="must be a list"):
            reranker.rerank("query", "not a list")  # type: ignore
        with pytest.raises(InvalidCandidateError, match="VectorSearchResult instances"):
            reranker.rerank("query", ["not a chunk"])  # type: ignore

    def test_empty_candidates_returns_empty_without_scorer(self):
        scorer_called = False

        def scorer(pairs):
            nonlocal scorer_called
            scorer_called = True
            return []

        reranker = CrossEncoderReranker(scorer=scorer)
        result = reranker.rerank("query", [])
        assert result == []
        assert scorer_called is False

    def test_fewer_candidates_than_top_k(self, reranker):
        candidates = [_make_candidate("c1", "only chunk")]
        result = reranker.rerank("query", candidates, top_k=5)
        assert len(result) == 1
        assert result[0].chunk_id == "c1"

    def test_repeated_candidates_handled_gracefully(self):
        """Repeated candidates (same chunk_id) are deduplicated without crashing."""
        c1 = _make_candidate("c1", "duplicate content")
        c2 = _make_candidate("c2", "unique content")

        reranker = CrossEncoderReranker(scorer=lambda pairs: [1.0, 2.0][: len(pairs)])
        # Passing [c1, c1, c2]
        result = reranker.rerank("query", [c1, c1, c2], top_k=5)
        # Should deduplicate c1 and return unique chunks
        chunk_ids = [r.chunk_id for r in result]
        assert len(chunk_ids) == len(set(chunk_ids))
        assert "c1" in chunk_ids
        assert "c2" in chunk_ids


# ==========================================================================
# 4. Ordering, Score Semantics & Deterministic Tie-Breaking
# ==========================================================================


class TestCrossEncoderRerankerOrderingAndScores:
    """Verify relevance sorting, score semantics, and tie-breaking."""

    def test_reorders_by_reranker_score_descending(self):
        c1 = _make_candidate("c1", "passage 1", score=0.9)  # Dense score 0.9
        c2 = _make_candidate("c2", "passage 2", score=0.7)  # Dense score 0.7
        c3 = _make_candidate("c3", "passage 3", score=0.5)  # Dense score 0.5

        # Reranker says c2 is best, then c3, then c1
        scores_map = {
            "passage 1": 1.2,
            "passage 2": 9.8,
            "passage 3": 5.4,
        }
        reranker = CrossEncoderReranker(
            scorer=lambda pairs: [scores_map[p[1]] for p in pairs]
        )

        results = reranker.rerank("test query", [c1, c2, c3], top_k=3)

        assert [r.chunk_id for r in results] == ["c2", "c3", "c1"]
        assert results[0].metadata["reranker_score"] == 9.8
        assert results[1].metadata["reranker_score"] == 5.4
        assert results[2].metadata["reranker_score"] == 1.2

    def test_preserves_original_retrieval_score_in_score_field(self):
        """Crucial: result.score must remain original retrieval score."""
        c1 = _make_candidate("c1", "content 1", score=0.88)
        reranker = CrossEncoderReranker(scorer=lambda pairs: [-4.2])

        results = reranker.rerank("query", [c1])

        assert results[0].score == 0.88  # Original score untouched
        assert results[0].metadata["retrieval_score"] == 0.88
        assert results[0].metadata["reranker_score"] == -4.2

    def test_deterministic_tie_breaking_equal_scores(self):
        """Equal reranker scores fall back to original candidate order."""
        c1 = _make_candidate("c1", "passage 1")
        c2 = _make_candidate("c2", "passage 2")
        c3 = _make_candidate("c3", "passage 3")

        # All items receive identical reranker score
        reranker = CrossEncoderReranker(scorer=lambda pairs: [5.0, 5.0, 5.0])

        results = reranker.rerank("query", [c1, c2, c3], top_k=3)
        assert [r.chunk_id for r in results] == ["c1", "c2", "c3"]

    def test_top_k_limiting(self):
        candidates = [_make_candidate(f"c{i}", f"content {i}") for i in range(10)]
        reranker = CrossEncoderReranker(
            scorer=lambda pairs: [float(i) for i in range(len(pairs))]
        )

        results = reranker.rerank("query", candidates, top_k=3)
        assert len(results) == 3
        # Highest scores were at the end of candidates (9, 8, 7)
        assert [r.chunk_id for r in results] == ["c9", "c8", "c7"]


# ==========================================================================
# 5. Failure Handling Tests: Malformed Outputs, NaN/Inf, Inference Errors
# ==========================================================================


class TestCrossEncoderRerankerFailures:
    """Verify robust error detection and wrapping."""

    def test_scorer_exception_wrapped_in_inference_error(self):
        def failing_scorer(pairs):
            raise RuntimeError("CUDA out of memory or driver crash")

        reranker = CrossEncoderReranker(scorer=failing_scorer)
        with pytest.raises(
            RerankerInferenceError, match="Injected scorer failed"
        ) as exc_info:
            reranker.rerank("query", [_make_candidate("c1", "text")])
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    def test_score_count_mismatch_raises_inference_error(self):
        # 2 candidates but scorer returns 1 score
        reranker = CrossEncoderReranker(scorer=lambda pairs: [1.0])
        candidates = [_make_candidate("c1", "text 1"), _make_candidate("c2", "text 2")]
        with pytest.raises(RerankerInferenceError, match="Scorer returned 1 scores"):
            reranker.rerank("query", candidates)

    def test_nan_score_raises_inference_error(self):
        reranker = CrossEncoderReranker(scorer=lambda pairs: [float("nan")])
        with pytest.raises(RerankerInferenceError, match="non-finite"):
            reranker.rerank("query", [_make_candidate("c1", "text")])

    def test_inf_score_raises_inference_error(self):
        reranker = CrossEncoderReranker(scorer=lambda pairs: [float("inf")])
        with pytest.raises(RerankerInferenceError, match="non-finite"):
            reranker.rerank("query", [_make_candidate("c1", "text")])

    def test_non_numeric_score_raises_inference_error(self):
        reranker = CrossEncoderReranker(scorer=lambda pairs: ["high_score"])  # type: ignore
        with pytest.raises(RerankerInferenceError, match="non-numeric"):
            reranker.rerank("query", [_make_candidate("c1", "text")])

    def test_bool_score_raises_inference_error(self):
        reranker = CrossEncoderReranker(scorer=lambda pairs: [True])  # type: ignore
        with pytest.raises(RerankerInferenceError, match="non-numeric"):
            reranker.rerank("query", [_make_candidate("c1", "text")])


# ==========================================================================
# 6. RerankingRetriever Composition Tests
# ==========================================================================


class TestRerankingRetriever:
    """Verify RerankingRetriever composing base retrievers with a reranker."""

    def test_implements_retriever_abc(self):
        base = DummyRetriever()
        reranker = CrossEncoderReranker(scorer=lambda pairs: [])
        rr = RerankingRetriever(base, reranker)
        assert isinstance(rr, Retriever)

    def test_constructor_validation(self):
        base = DummyRetriever()
        reranker = CrossEncoderReranker(scorer=lambda pairs: [])

        with pytest.raises(TypeError, match="base_retriever"):
            RerankingRetriever("not a retriever", reranker)  # type: ignore
        with pytest.raises(TypeError, match="reranker"):
            RerankingRetriever(base, "not a reranker")  # type: ignore
        with pytest.raises(ValueError, match="candidate_top_k"):
            RerankingRetriever(base, reranker, candidate_top_k=0)
        with pytest.raises(ValueError, match="default_top_k"):
            RerankingRetriever(base, reranker, default_top_k=-1)

    def test_candidate_retrieval_and_reranking_flow(self):
        candidates = [
            _make_candidate("c1", "low relevance", score=0.9),
            _make_candidate("c2", "high relevance", score=0.6),
        ]
        base = DummyRetriever(results=candidates)
        # Reranker scores c2 higher than c1
        reranker = CrossEncoderReranker(
            scorer=lambda pairs: [0.1 if "low" in p[1] else 8.5 for p in pairs]
        )

        rr = RerankingRetriever(base, reranker, candidate_top_k=10, default_top_k=2)
        results = rr.retrieve("What is high relevance?")

        # Base retriever was called with candidate_top_k (10)
        assert base.last_top_k == 10
        # Results are reordered by reranker
        assert [r.chunk_id for r in results] == ["c2", "c1"]
        assert results[0].score == 0.6  # Original score preserved
        assert results[0].metadata["reranker_score"] == 8.5

    def test_candidate_depth_expands_if_requested_top_k_exceeds_it(self):
        """candidate set is large enough to return requested results when available."""
        base = DummyRetriever()
        reranker = CrossEncoderReranker(scorer=lambda pairs: [])
        rr = RerankingRetriever(base, reranker, candidate_top_k=10)

        rr.retrieve("query", top_k=25)
        # Candidate depth expanded to 25 to accommodate requested top_k
        assert base.last_top_k == 25

    def test_empty_candidates_from_base_returns_empty_immediately(self):
        base = DummyRetriever(results=[])
        mock_reranker = MagicMock(spec=Reranker)
        rr = RerankingRetriever(base, mock_reranker)

        results = rr.retrieve("query")
        assert results == []
        mock_reranker.rerank.assert_not_called()

    def test_min_score_propagated_to_base_retriever(self):
        """Relevance gate threshold is enforced at candidate retrieval."""
        candidates = [
            _make_candidate("c1", "relevant", score=0.65),
            _make_candidate("c2", "unrelated", score=0.10),
        ]
        base = DummyRetriever(results=candidates)
        reranker = CrossEncoderReranker(scorer=lambda pairs: [5.0] * len(pairs))
        rr = RerankingRetriever(base, reranker)

        # Base retriever filters out c2 because score 0.10 < 0.30
        results = rr.retrieve("query", min_score=0.30)
        assert len(results) == 1
        assert results[0].chunk_id == "c1"

    def test_supports_bm25_base_retriever(self):
        """Verifies RerankingRetriever cleanly wraps BM25Retriever."""
        c1 = _make_candidate("c1", "Machine learning algorithms and deep neural nets")
        c2 = _make_candidate("c2", "Founding history of City One")

        bm25_index = BM25Index()
        bm25_index.build([c1, c2])
        bm25_retriever = BM25Retriever(bm25_index=bm25_index, default_top_k=5)

        reranker = CrossEncoderReranker(scorer=lambda pairs: [10.0] * len(pairs))
        rr = RerankingRetriever(
            base_retriever=bm25_retriever, reranker=reranker, candidate_top_k=5
        )

        results = rr.retrieve("machine learning")
        assert len(results) == 1
        assert results[0].chunk_id == "c1"
        assert results[0].metadata["reranker_score"] == 10.0


# ==========================================================================
# 7. Metadata and Citation Integrity Tests (Phase 7 Integration)
# ==========================================================================


class TestRerankerMetadataAndCitationIntegrity:
    """Ensure Phase 7 citation structures and metadata survive reranking."""

    def test_all_metadata_fields_preserved(self):
        original_meta = {
            "custom_field": "test_val",
            "page_number": 4,
            "section": "Methods",
            "char_count": 120,
        }
        candidate = _make_candidate(
            chunk_id="chunk_42",
            content="Detailed experimental section.",
            score=0.77,
            document_id="doc_xyz",
            document_name="experiment.pdf",
            metadata=original_meta,
        )
        reranker = CrossEncoderReranker(scorer=lambda pairs: [3.14])

        reranked = reranker.rerank("query", [candidate])
        res = reranked[0]

        assert res.chunk_id == "chunk_42"
        assert res.document_id == "doc_xyz"
        assert res.document_name == "experiment.pdf"
        assert res.content == "Detailed experimental section."
        assert res.score == 0.77
        assert res.metadata["custom_field"] == "test_val"
        assert res.metadata["page_number"] == 4
        assert res.metadata["section"] == "Methods"
        assert res.metadata["char_count"] == 120
        assert res.metadata["retrieval_score"] == 0.77
        assert res.metadata["reranker_score"] == 3.14

    def test_format_sources_preserves_reranker_order_and_distinct_scores(self):
        c1 = _make_candidate("c1", "passage A", score=0.9, page_number=2)
        c2 = _make_candidate("c2", "passage B", score=0.4, page_number=5)

        # Reranker reverses the order: c2 is higher scored
        reranker = CrossEncoderReranker(
            scorer=lambda pairs: [1.0 if "passage A" in p[1] else 8.0 for p in pairs]
        )
        reranked_chunks = reranker.rerank("query", [c1, c2])

        sources = format_sources(reranked_chunks)

        assert len(sources) == 2
        assert sources[0].chunk_id == "c2"
        assert sources[0].page_number == 5
        assert sources[0].score == 0.4  # Original retrieval score
        assert sources[0].reranker_score == 8.0  # Distinct reranker score

        assert sources[1].chunk_id == "c1"
        assert sources[1].page_number == 2
        assert sources[1].score == 0.9
        assert sources[1].reranker_score == 1.0

    def test_context_builder_formats_reranked_chunks_in_reordered_sequence(self):
        c1 = _make_candidate("c1", "Second most relevant content.", score=0.95)
        c2 = _make_candidate("c2", "Top most relevant content.", score=0.40)

        reranker = CrossEncoderReranker(
            scorer=lambda pairs: [2.0 if "Second" in p[1] else 9.0 for p in pairs]
        )
        reranked_chunks = reranker.rerank("query", [c1, c2])

        context_str = build_context(reranked_chunks)

        # Chunk 1 in context must be c2
        assert "[Chunk 1] | Document: doc_1.pdf | Page: 1 | Chunk ID: c2" in context_str
        assert "Top most relevant content." in context_str
        # Chunk 2 in context must be c1
        assert "[Chunk 2] | Document: doc_1.pdf | Page: 1 | Chunk ID: c1" in context_str


# ==========================================================================
# 8. RAGQueryService Orchestration Tests
# ==========================================================================


class TestRAGQueryServiceWithReranking:
    """Verify query service orchestration with reranking."""

    def test_query_service_without_reranker_retains_baseline_behavior(self):
        retriever = DummyRetriever(results=[_make_candidate("c1", "test content")])
        generator = DummyGenerator(answer="Baseline answer")

        service = RAGQueryService(retriever=retriever, generator=generator)
        res = service.query("test question")

        assert res.answer == "Baseline answer"
        assert res.num_chunks_retrieved == 1
        assert res.sources[0].chunk_id == "c1"
        assert res.sources[0].reranker_score is None

    def test_query_service_with_reranker_reorders_and_populates_sources(self):
        candidates = [
            _make_candidate("c1", "First retrieved chunk", score=0.8),
            _make_candidate("c2", "Second retrieved chunk", score=0.5),
        ]
        retriever = DummyRetriever(results=candidates)
        generator = DummyGenerator(answer="Reranked answer")
        reranker = CrossEncoderReranker(
            scorer=lambda pairs: [1.0 if "First" in p[1] else 7.5 for p in pairs]
        )

        service = RAGQueryService(
            retriever=retriever,
            generator=generator,
            reranker=reranker,
            candidate_top_k=20,
        )
        res = service.query("What is the answer?", top_k=2)

        assert res.answer == "Reranked answer"
        # Context passed to generator had c2 first
        assert generator.last_context[0].chunk_id == "c2"
        # Sources reflect reranked order
        assert res.sources[0].chunk_id == "c2"
        assert res.sources[0].reranker_score == 7.5
        assert res.sources[1].chunk_id == "c1"
        assert res.sources[1].reranker_score == 1.0

    def test_relevance_gate_rejects_unrelated_query_before_reranker_and_llm(self):
        """Crucial test: Reranking does not turn irrelevant candidates into evidence."""
        # Unrelated query gets low score (0.05 < 0.30 threshold)
        unrelated_candidate = _make_candidate(
            "c_unrelated", "Capital of France is Paris.", score=0.05
        )
        retriever = DummyRetriever(results=[unrelated_candidate])
        generator = MagicMock(spec=Generator)
        mock_reranker = MagicMock(spec=Reranker)

        service = RAGQueryService(
            retriever=retriever,
            generator=generator,
            min_score=0.30,
            reranker=mock_reranker,
        )

        with pytest.raises(InsufficientContextError):
            service.query("What is the capital of France?")

        # Neither reranker nor LLM generator must be called
        mock_reranker.rerank.assert_not_called()
        generator.generate.assert_not_called()

    def test_relevance_gate_with_empty_candidates_prevents_reranker_and_llm(self):
        """When retriever yields empty results, service raises EmptyRetrievalError."""
        retriever = DummyRetriever(results=[])
        generator = MagicMock(spec=Generator)
        mock_reranker = MagicMock(spec=Reranker)

        service = RAGQueryService(
            retriever=retriever,
            generator=generator,
            reranker=mock_reranker,
        )

        with pytest.raises(EmptyRetrievalError):
            service.query("What is the best recipe for baking chocolate cake?")

        mock_reranker.rerank.assert_not_called()
        generator.generate.assert_not_called()

    def test_reranker_error_wrapped_in_query_error(self):
        retriever = DummyRetriever(results=[_make_candidate("c1", "content")])
        generator = DummyGenerator()
        failing_reranker = CrossEncoderReranker(
            scorer=lambda pairs: (_ for _ in ()).throw(RuntimeError("Model error"))
        )

        service = RAGQueryService(
            retriever=retriever,
            generator=generator,
            reranker=failing_reranker,
        )

        with pytest.raises(QueryError, match="Reranking failed"):
            service.query("query")


# ==========================================================================
# 9. API /query Endpoint Integration Tests
# ==========================================================================


class TestAPIQueryEndpointWithReranking:
    """Verify backend API contracts with reranker enabled and disabled."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_api_query_with_reranking_disabled_default(self, client):
        mock_result = QueryResult(
            answer="Grounded answer without reranker",
            model_name="qwen2.5:0.5b-instruct",
            num_chunks_retrieved=1,
            sources=[
                Source(
                    document_id="doc1",
                    document_name="test.pdf",
                    page_number=1,
                    chunk_id="chunk1",
                    score=0.65,
                )
            ],
        )

        with (
            patch("backend.query.get_settings") as mock_settings,
            patch("backend.query.RAGQueryService.query", return_value=mock_result),
        ):
            settings = Settings()
            settings.reranker_enabled = False
            mock_settings.return_value = settings

            response = client.post("/query", json={"question": "What is AI?"})

            assert response.status_code == 200
            data = response.json()
            assert data["answer"] == "Grounded answer without reranker"
            assert data["reranker_applied"] is False
            assert len(data["sources"]) == 1
            assert data["sources"][0]["score"] == 0.65
            assert data["sources"][0]["reranker_score"] is None

    def test_api_query_with_reranking_enabled(self, client):
        mock_result = QueryResult(
            answer="Grounded answer with reranker",
            model_name="qwen2.5:0.5b-instruct",
            num_chunks_retrieved=1,
            sources=[
                Source(
                    document_id="doc1",
                    document_name="test.pdf",
                    page_number=2,
                    chunk_id="chunk2",
                    score=0.55,
                    reranker_score=8.42,
                )
            ],
        )

        with (
            patch("backend.query.get_settings") as mock_settings,
            patch("backend.query.RAGQueryService.query", return_value=mock_result),
        ):
            settings = Settings()
            settings.reranker_enabled = True
            settings.reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            settings.reranker_candidate_top_k = 20
            mock_settings.return_value = settings

            response = client.post("/query", json={"question": "What is AI?"})

            assert response.status_code == 200
            data = response.json()
            assert data["answer"] == "Grounded answer with reranker"
            assert data["reranker_applied"] is True
            assert len(data["sources"]) == 1
            assert data["sources"][0]["score"] == 0.55
            assert data["sources"][0]["reranker_score"] == 8.42

    def test_api_query_insufficient_context_returns_404_with_reranking_enabled(
        self, client
    ):
        with (
            patch("backend.query.get_settings") as mock_settings,
            patch(
                "backend.query.RAGQueryService.query",
                side_effect=InsufficientContextError("Insufficient context"),
            ),
        ):
            settings = Settings()
            settings.reranker_enabled = True
            mock_settings.return_value = settings

            response = client.post(
                "/query", json={"question": "What is the capital of France?"}
            )
            assert response.status_code == 404
            assert "Insufficient context" in response.json()["detail"]


# ==========================================================================
# 10. Real Cross-Encoder Smoke Test
# ==========================================================================


class TestRealCrossEncoderSmoke:
    """Smoke test running live CrossEncoder inference against cached model weights."""

    @pytest.mark.skipif(
        not pytest.importorskip("sentence_transformers"),
        reason="sentence_transformers required for real model smoke test",
    )
    def test_real_cross_encoder_prediction_and_reordering(self):
        """Verifies real CrossEncoder can score and prioritize relevant chunks."""
        try:
            reranker = CrossEncoderReranker(
                model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                default_top_k=2,
            )
            # Ensure model can load
            _ = reranker._get_model()
        except Exception as exc:
            pytest.skip(f"Model unavailable or network/cache issue: {exc}")

        relevant = _make_candidate(
            "rel",
            "Machine learning is a field of AI that uses statistical techniques.",
            score=0.5,
        )
        irrelevant = _make_candidate(
            "irrel",
            "The capital of France is Paris, located on the Seine river.",
            score=0.5,
        )

        results = reranker.rerank(
            "What is machine learning?", [irrelevant, relevant], top_k=2
        )

        assert len(results) == 2
        # Relevant passage must be ranked first
        assert results[0].chunk_id == "rel"
        assert results[1].chunk_id == "irrel"
        # Relevant passage must have distinctly higher score than irrelevant passage
        rel_score = results[0].metadata["reranker_score"]
        irrel_score = results[1].metadata["reranker_score"]
        assert rel_score > irrel_score
        assert rel_score > 0.0  # Logit should be positive for relevant
        assert irrel_score < 0.0  # Logit should be negative for irrelevant


# ==========================================================================
# 11. Model Weight Reuse, Concurrency & Singleton Tests
# ==========================================================================


class TestModelWeightReuseAndConcurrency:
    """Verify that model weights are reused across instances and requests."""

    def test_model_cache_reuses_loaded_weights_across_instances(self):
        CrossEncoderReranker.clear_cache()
        mock_model = MagicMock()
        mock_model.predict.return_value = [1.5]

        with patch(
            "sentence_transformers.CrossEncoder", return_value=mock_model
        ) as mock_cls:
            r1 = CrossEncoderReranker(model_name="test/cached-model")
            _ = r1.rerank("query", [_make_candidate("c1", "content")])
            assert mock_cls.call_count == 1

            # A second instance with the same model_name must reuse cached model
            r2 = CrossEncoderReranker(model_name="test/cached-model")
            _ = r2.rerank("query", [_make_candidate("c1", "content")])
            # Model loader was NOT called again
            assert mock_cls.call_count == 1

        CrossEncoderReranker.clear_cache()

    def test_clear_cache_removes_cached_model(self):
        CrossEncoderReranker.clear_cache()
        mock_model = MagicMock()
        mock_model.predict.return_value = [1.0]

        with patch(
            "sentence_transformers.CrossEncoder", return_value=mock_model
        ) as mock_cls:
            r1 = CrossEncoderReranker(model_name="test/clear-model")
            _ = r1.rerank("query", [_make_candidate("c1", "content")])
            assert mock_cls.call_count == 1

            CrossEncoderReranker.clear_cache()

            r2 = CrossEncoderReranker(model_name="test/clear-model")
            _ = r2.rerank("query", [_make_candidate("c1", "content")])
            assert mock_cls.call_count == 2

        CrossEncoderReranker.clear_cache()

    def test_get_reranker_singleton_in_backend_query(self):
        from backend.query import get_reranker

        settings_disabled = Settings()
        settings_disabled.reranker_enabled = False
        assert get_reranker(settings_disabled) is None

        settings_enabled = Settings()
        settings_enabled.reranker_enabled = True
        r1 = get_reranker(settings_enabled)
        r2 = get_reranker(settings_enabled)
        assert r1 is not None
        assert r1 is r2  # Same instance reused across requests

    def test_concurrent_reranking_execution(self):
        """Verify thread-safety when multiple threads invoke rerank concurrently."""
        from concurrent.futures import ThreadPoolExecutor

        def mock_scorer(pairs):
            return [float(len(p[1])) for p in pairs]

        reranker = CrossEncoderReranker(scorer=mock_scorer)
        candidates = [
            _make_candidate(f"c{i}", f"content {'x' * i}") for i in range(1, 6)
        ]

        def worker(q_idx: int):
            return reranker.rerank(f"query {q_idx}", candidates, top_k=3)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker, i) for i in range(10)]
            results = [f.result() for f in futures]

        for res in results:
            assert len(res) == 3
            assert [r.chunk_id for r in res] == ["c5", "c4", "c3"]


# ==========================================================================
# 12. Score Unpacking & NaN/Inf Hardening Tests
# ==========================================================================


class TestScoreUnpackingAndFormatSourcesHardening:
    """Verify handling of scalar scores and non-finite values."""

    def test_scorer_returning_scalar_float_handled_cleanly(self):
        """Scorer returning a scalar float or 0-d array for 1 candidate works."""
        reranker = CrossEncoderReranker(
            scorer=lambda pairs: 3.14
        )  # returns scalar float
        res = reranker.rerank("query", [_make_candidate("c1", "text")])
        assert len(res) == 1
        assert res[0].metadata["reranker_score"] == 3.14

    def test_format_sources_ignores_nan_and_inf_reranker_score(self):
        """NaN and Inf reranker scores in chunk metadata are safely ignored."""
        c_nan = _make_candidate(
            "c1", "text 1", metadata={"reranker_score": float("nan")}
        )
        c_inf = _make_candidate(
            "c2", "text 2", metadata={"reranker_score": float("inf")}
        )
        c_neg_inf = _make_candidate(
            "c3", "text 3", metadata={"reranker_score": float("-inf")}
        )

        sources = format_sources([c_nan, c_inf, c_neg_inf])
        assert len(sources) == 3
        assert sources[0].reranker_score is None
        assert sources[1].reranker_score is None
        assert sources[2].reranker_score is None


# ==========================================================================
# 13. Duplicate Reranking & Orphan Configuration Protection Tests
# ==========================================================================


class TestQueryServiceDuplicateRerankingProtection:
    """Verify duplicate reranking prevention and orphan config checks."""

    def test_query_service_rejects_reranking_retriever_plus_reranker(self):
        base = DummyRetriever()
        reranker = CrossEncoderReranker(scorer=lambda pairs: [])
        rr = RerankingRetriever(base, reranker)
        generator = DummyGenerator()

        with pytest.raises(ValueError, match="already a RerankingRetriever"):
            RAGQueryService(
                retriever=rr,
                generator=generator,
                reranker=reranker,
            )

    def test_query_service_rejects_candidate_top_k_without_reranker(self):
        retriever = DummyRetriever()
        generator = DummyGenerator()

        with pytest.raises(
            ValueError,
            match="candidate_top_k cannot be specified without a reranker",
        ):
            RAGQueryService(
                retriever=retriever,
                generator=generator,
                candidate_top_k=20,
            )


# ==========================================================================
# 14. RerankingRetriever min_score Validation Tests
# ==========================================================================


class TestRerankingRetrieverMinScoreValidation:
    """Verify that RerankingRetriever validates min_score."""

    def test_invalid_min_score_types(self):
        base = DummyRetriever()
        reranker = CrossEncoderReranker(scorer=lambda pairs: [])
        rr = RerankingRetriever(base, reranker)

        with pytest.raises(
            InvalidQueryError, match="min_score must be a numeric value"
        ):
            rr.retrieve("query", min_score="invalid")  # type: ignore

        with pytest.raises(
            InvalidQueryError, match="min_score must be a numeric value"
        ):
            rr.retrieve("query", min_score=True)  # type: ignore

    def test_invalid_min_score_bounds(self):
        base = DummyRetriever()
        reranker = CrossEncoderReranker(scorer=lambda pairs: [])
        rr = RerankingRetriever(base, reranker)

        with pytest.raises(InvalidQueryError, match="between -1.0 and 1.0"):
            rr.retrieve("query", min_score=-1.5)

        with pytest.raises(InvalidQueryError, match="between -1.0 and 1.0"):
            rr.retrieve("query", min_score=1.5)

    def test_valid_min_score_forwarded(self):
        candidates = [_make_candidate("c1", "valid", score=0.8)]
        base = DummyRetriever(results=candidates)
        reranker = CrossEncoderReranker(scorer=lambda pairs: [1.0])
        rr = RerankingRetriever(base, reranker)

        res = rr.retrieve("query", min_score=0.5)
        assert base.last_min_score == 0.5
        assert len(res) == 1
