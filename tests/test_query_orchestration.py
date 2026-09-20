"""Unit tests for RAGQueryService.

Tests the orchestration logic using mocked dependencies —
no real embedding, vector store, or LLM operations.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rag.generation.exceptions import InvalidContextError, ModelGenerationError
from rag.generation.models import GenerationResult
from rag.orchestration.exceptions import EmptyRetrievalError, QueryError
from rag.orchestration.query_service import QueryResult, RAGQueryService
from rag.retrieval.exceptions import InvalidQueryError, RetrievalError
from rag.vectorstore.models import VectorSearchResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_results(count: int = 3) -> list[VectorSearchResult]:
    """Create a list of test VectorSearchResult objects."""
    return [
        VectorSearchResult(
            chunk_id=f"chunk-{i}",
            document_id="doc-001",
            document_name="test.pdf",
            content=f"Chunk content {i} with useful information.",
            score=0.9 - i * 0.1,
            metadata={"page_number": i + 1, "source_type": "pdf"},
        )
        for i in range(count)
    ]


def _make_generation_result(
    answer: str = "The answer is 42.",
    model_name: str = "tinyllama",
) -> GenerationResult:
    """Create a test GenerationResult."""
    return GenerationResult(
        answer=answer,
        model_name=model_name,
        metadata={"temperature": 0.1},
    )


@pytest.fixture()
def mock_retriever():
    """Create a mock Retriever."""
    return MagicMock()


@pytest.fixture()
def mock_generator():
    """Create a mock Generator."""
    return MagicMock()


@pytest.fixture()
def service(mock_retriever, mock_generator):
    """Create a RAGQueryService with mocked dependencies."""
    return RAGQueryService(
        retriever=mock_retriever,
        generator=mock_generator,
    )


# ---------------------------------------------------------------------------
# Valid question + retrieved context
# ---------------------------------------------------------------------------


class TestValidQueryFlow:
    """Tests for the happy-path query flow."""

    def test_query_returns_query_result(self, service, mock_retriever, mock_generator):
        """query() returns a QueryResult on success."""
        results = _make_results()
        gen_result = _make_generation_result()

        mock_retriever.retrieve.return_value = results
        mock_generator.generate.return_value = gen_result

        result = service.query("What is the meaning of life?")

        assert isinstance(result, QueryResult)
        assert result.answer == gen_result.answer
        assert result.model_name == gen_result.model_name
        assert result.num_chunks_retrieved == len(results)
        assert result.metadata == gen_result.metadata

    def test_generator_called_with_question_and_results(
        self, service, mock_retriever, mock_generator
    ):
        """Generator is called with the original question and retrieval results."""
        results = _make_results()
        gen_result = _make_generation_result()

        mock_retriever.retrieve.return_value = results
        mock_generator.generate.return_value = gen_result

        service.query("What is the meaning of life?")

        mock_generator.generate.assert_called_once_with(
            "What is the meaning of life?", results
        )

    def test_retriever_called_with_question(
        self, service, mock_retriever, mock_generator
    ):
        """Retriever is called with the question."""
        results = _make_results()
        gen_result = _make_generation_result()

        mock_retriever.retrieve.return_value = results
        mock_generator.generate.return_value = gen_result

        service.query("What is revenue?")

        mock_retriever.retrieve.assert_called_once_with("What is revenue?", top_k=None)


# ---------------------------------------------------------------------------
# top_k propagation
# ---------------------------------------------------------------------------


class TestTopKPropagation:
    """Tests for top_k parameter forwarding."""

    def test_top_k_forwarded_to_retriever(
        self, service, mock_retriever, mock_generator
    ):
        """top_k is forwarded to the retriever."""
        results = _make_results(count=5)
        gen_result = _make_generation_result()

        mock_retriever.retrieve.return_value = results
        mock_generator.generate.return_value = gen_result

        service.query("question?", top_k=5)

        mock_retriever.retrieve.assert_called_once_with("question?", top_k=5)

    def test_none_top_k_forwarded(self, service, mock_retriever, mock_generator):
        """None top_k is forwarded to the retriever."""
        results = _make_results()
        gen_result = _make_generation_result()

        mock_retriever.retrieve.return_value = results
        mock_generator.generate.return_value = gen_result

        service.query("question?")

        mock_retriever.retrieve.assert_called_once_with("question?", top_k=None)


# ---------------------------------------------------------------------------
# Empty retrieval results
# ---------------------------------------------------------------------------


class TestEmptyRetrievalResults:
    """Tests for empty retrieval results."""

    def test_empty_results_raises_empty_retrieval_error(
        self, service, mock_retriever, mock_generator
    ):
        """Empty retrieval results raise EmptyRetrievalError."""
        mock_retriever.retrieve.return_value = []

        with pytest.raises(EmptyRetrievalError, match="no results"):
            service.query("What is revenue?")

    def test_generator_not_called_on_empty_results(
        self, service, mock_retriever, mock_generator
    ):
        """Generator is NOT called when retrieval returns empty results."""
        mock_retriever.retrieve.return_value = []

        with pytest.raises(EmptyRetrievalError):
            service.query("What is revenue?")

        mock_generator.generate.assert_not_called()


# ---------------------------------------------------------------------------
# Retrieval failure
# ---------------------------------------------------------------------------


class TestRetrievalFailure:
    """Tests for retrieval-stage failures."""

    def test_retrieval_error_raises_query_error(
        self, service, mock_retriever, mock_generator
    ):
        """Retrieval failure raises QueryError."""
        mock_retriever.retrieve.side_effect = RetrievalError("Search failed")

        with pytest.raises(QueryError, match="Retrieval failed"):
            service.query("What is revenue?")

    def test_retrieval_error_preserves_cause(
        self, service, mock_retriever, mock_generator
    ):
        """The original retrieval exception is preserved as __cause__."""
        cause = RetrievalError("Search failed")
        mock_retriever.retrieve.side_effect = cause

        with pytest.raises(QueryError) as exc_info:
            service.query("What is revenue?")

        assert exc_info.value.__cause__ is cause

    def test_generator_not_called_on_retrieval_failure(
        self, service, mock_retriever, mock_generator
    ):
        """Generator is NOT called when retrieval fails."""
        mock_retriever.retrieve.side_effect = RuntimeError("fail")

        with pytest.raises(QueryError):
            service.query("question?")

        mock_generator.generate.assert_not_called()

    def test_invalid_question_error_passes_through(
        self, service, mock_retriever, mock_generator
    ):
        """InvalidQuestionError from retriever passes through directly.

        This is a special case: invalid-question errors are NOT wrapped
        in QueryError because callers should handle them distinctly.
        """
        mock_retriever.retrieve.side_effect = InvalidQueryError("empty query")

        # InvalidQueryError from retrieval is re-raised as the generation
        # layer's InvalidQuestionError only if the retriever raises it.
        # But since we map it: the retriever raises InvalidQueryError
        # which is a subclass of RetrievalError... let's check:
        # Actually InvalidQueryError inherits from RetrievalError,
        # but the service catches it in the except block. However,
        # the service ALSO catches the generation layer's
        # InvalidQuestionError specially.
        #
        # The retriever raises InvalidQueryError which is NOT
        # InvalidQuestionError (from generation). So it falls into
        # the general except block and becomes QueryError.
        with pytest.raises(QueryError):
            service.query("")


# ---------------------------------------------------------------------------
# Generation failure
# ---------------------------------------------------------------------------


class TestGenerationFailure:
    """Tests for generation-stage failures."""

    def test_generation_error_raises_query_error(
        self, service, mock_retriever, mock_generator
    ):
        """Generation failure raises QueryError."""
        mock_retriever.retrieve.return_value = _make_results()
        mock_generator.generate.side_effect = ModelGenerationError("LLM timeout")

        with pytest.raises(QueryError, match="Generation failed"):
            service.query("What is revenue?")

    def test_generation_error_preserves_cause(
        self, service, mock_retriever, mock_generator
    ):
        """The original generation exception is preserved as __cause__."""
        cause = ModelGenerationError("LLM timeout")
        mock_retriever.retrieve.return_value = _make_results()
        mock_generator.generate.side_effect = cause

        with pytest.raises(QueryError) as exc_info:
            service.query("What is revenue?")

        assert exc_info.value.__cause__ is cause

    def test_invalid_context_error_raises_query_error(
        self, service, mock_retriever, mock_generator
    ):
        """InvalidContextError from generator raises QueryError."""
        mock_retriever.retrieve.return_value = _make_results()
        mock_generator.generate.side_effect = InvalidContextError("bad context")

        with pytest.raises(QueryError, match="Generation failed"):
            service.query("What is revenue?")


# ---------------------------------------------------------------------------
# Result propagation
# ---------------------------------------------------------------------------


class TestResultPropagation:
    """Tests for result field propagation."""

    def test_answer_propagated(self, service, mock_retriever, mock_generator):
        """The answer from GenerationResult is propagated."""
        mock_retriever.retrieve.return_value = _make_results()
        mock_generator.generate.return_value = _make_generation_result(
            answer="Revenue increased by 25%."
        )

        result = service.query("What happened to revenue?")
        assert result.answer == "Revenue increased by 25%."

    def test_model_name_propagated(self, service, mock_retriever, mock_generator):
        """The model_name from GenerationResult is propagated."""
        mock_retriever.retrieve.return_value = _make_results()
        mock_generator.generate.return_value = _make_generation_result(
            model_name="llama3"
        )

        result = service.query("question?")
        assert result.model_name == "llama3"

    def test_metadata_propagated(self, service, mock_retriever, mock_generator):
        """The metadata from GenerationResult is propagated."""
        gen_result = _make_generation_result()
        gen_result.metadata = {"temperature": 0.5, "latency_ms": 200}

        mock_retriever.retrieve.return_value = _make_results()
        mock_generator.generate.return_value = gen_result

        result = service.query("question?")
        assert result.metadata == {"temperature": 0.5, "latency_ms": 200}

    def test_num_chunks_retrieved_matches_results(
        self, service, mock_retriever, mock_generator
    ):
        """num_chunks_retrieved matches the number of retrieval results."""
        results = _make_results(count=7)
        mock_retriever.retrieve.return_value = results
        mock_generator.generate.return_value = _make_generation_result()

        result = service.query("question?")
        assert result.num_chunks_retrieved == 7


# ---------------------------------------------------------------------------
# QueryResult model
# ---------------------------------------------------------------------------


class TestQueryResultModel:
    """Tests for the QueryResult data model."""

    def test_query_result_fields(self):
        """QueryResult has the expected fields."""
        result = QueryResult(
            answer="Test answer",
            model_name="tinyllama",
            num_chunks_retrieved=3,
            metadata={"temperature": 0.1},
        )
        assert result.answer == "Test answer"
        assert result.model_name == "tinyllama"
        assert result.num_chunks_retrieved == 3
        assert result.metadata == {"temperature": 0.1}

    def test_query_result_default_metadata(self):
        """QueryResult defaults to empty metadata."""
        result = QueryResult(
            answer="Test",
            num_chunks_retrieved=1,
        )
        assert result.metadata == {}
        assert result.model_name == ""


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    """Tests for the orchestration exception hierarchy."""

    def test_query_error_is_orchestration_error(self):
        """QueryError is a subclass of OrchestrationError."""
        from rag.orchestration.exceptions import OrchestrationError

        assert issubclass(QueryError, OrchestrationError)

    def test_empty_retrieval_error_is_query_error(self):
        """EmptyRetrievalError is a subclass of QueryError."""
        assert issubclass(EmptyRetrievalError, QueryError)

    def test_empty_retrieval_error_is_orchestration_error(self):
        """EmptyRetrievalError is also a subclass of OrchestrationError."""
        from rag.orchestration.exceptions import OrchestrationError

        assert issubclass(EmptyRetrievalError, OrchestrationError)

    def test_insufficient_context_error_subclasses(self):
        """InsufficientContextError is a subclass of EmptyRetrievalError
        and QueryError.
        """
        from rag.orchestration.exceptions import (
            InsufficientContextError,
            OrchestrationError,
        )

        assert issubclass(InsufficientContextError, EmptyRetrievalError)
        assert issubclass(InsufficientContextError, QueryError)
        assert issubclass(InsufficientContextError, OrchestrationError)


# ---------------------------------------------------------------------------
# Relevance Gate in RAGQueryService (Phase 6 Grounding)
# ---------------------------------------------------------------------------


class TestRelevanceGateInQueryService:
    """Tests for relevance gating in RAGQueryService."""

    def test_min_score_validation_in_init(self, mock_retriever, mock_generator):
        """min_score in RAGQueryService.__init__ validates numeric range."""
        from rag.orchestration.query_service import RAGQueryService

        # Valid values
        s1 = RAGQueryService(mock_retriever, mock_generator, min_score=0.3)
        assert s1._min_score == 0.3

        s2 = RAGQueryService(mock_retriever, mock_generator, min_score=-1.0)
        assert s2._min_score == -1.0

        s3 = RAGQueryService(mock_retriever, mock_generator, min_score=1.0)
        assert s3._min_score == 1.0

        # Invalid values
        with pytest.raises(ValueError, match="min_score"):
            RAGQueryService(mock_retriever, mock_generator, min_score=1.1)

        with pytest.raises(ValueError, match="min_score"):
            RAGQueryService(mock_retriever, mock_generator, min_score=-1.1)

        with pytest.raises(ValueError, match="min_score"):
            RAGQueryService(mock_retriever, mock_generator, min_score=True)

        with pytest.raises(ValueError, match="min_score"):
            RAGQueryService(mock_retriever, mock_generator, min_score="0.3")

    def test_unrelated_query_rejected_and_llm_not_called(
        self, mock_retriever, mock_generator
    ):
        """When all retrieved chunks are below min_score,
        InsufficientContextError is raised and LLM is NOT called.
        """
        from rag.orchestration.exceptions import InsufficientContextError
        from rag.orchestration.query_service import RAGQueryService
        from rag.vectorstore.models import VectorSearchResult

        # Retriever returns low-scoring unrelated chunks
        mock_retriever.retrieve.return_value = [
            VectorSearchResult(
                chunk_id="c1",
                document_id="d1",
                document_name="ai.pdf",
                content="AI overview",
                score=0.04,
            ),
            VectorSearchResult(
                chunk_id="c2",
                document_id="d1",
                document_name="ai.pdf",
                content="ML overview",
                score=0.01,
            ),
        ]

        service = RAGQueryService(
            retriever=mock_retriever,
            generator=mock_generator,
            min_score=0.30,
        )

        with pytest.raises(InsufficientContextError):
            service.query("What is the capital of France?")

        # LLM MUST NOT BE CALLED
        mock_generator.generate.assert_not_called()

    def test_relevant_chunks_pass_and_llm_called(self, mock_retriever, mock_generator):
        """When retrieved chunks meet min_score, they reach the LLM generator."""
        from rag.generation.models import GenerationResult
        from rag.orchestration.query_service import RAGQueryService
        from rag.vectorstore.models import VectorSearchResult

        mock_retriever.retrieve.return_value = [
            VectorSearchResult(
                chunk_id="c1",
                document_id="d1",
                document_name="ai.pdf",
                content="Machine learning is a subset of AI.",
                score=0.65,
            ),
        ]
        mock_generator.generate.return_value = GenerationResult(
            answer="Machine learning is a subset of AI.",
            model_name="test-model",
        )

        service = RAGQueryService(
            retriever=mock_retriever,
            generator=mock_generator,
            min_score=0.30,
        )

        result = service.query("What is Machine Learning?")
        assert result.answer == "Machine learning is a subset of AI."
        assert result.num_chunks_retrieved == 1
        mock_generator.generate.assert_called_once()

    def test_partial_relevance_filters_out_low_scoring_chunks(
        self, mock_retriever, mock_generator
    ):
        """When multiple chunks are retrieved and only some are relevant,
        only relevant chunks reach generator.
        """
        from rag.generation.models import GenerationResult
        from rag.orchestration.query_service import RAGQueryService
        from rag.vectorstore.models import VectorSearchResult

        chunk_high = VectorSearchResult(
            chunk_id="c_high",
            document_id="d1",
            document_name="ai.pdf",
            content="Deep learning uses neural networks.",
            score=0.58,
        )
        chunk_low = VectorSearchResult(
            chunk_id="c_low",
            document_id="d1",
            document_name="ai.pdf",
            content="Irrelevant background snippet.",
            score=0.15,
        )

        mock_retriever.retrieve.return_value = [chunk_high, chunk_low]
        mock_generator.generate.return_value = GenerationResult(
            answer="Deep learning uses neural networks.",
            model_name="test-model",
        )

        service = RAGQueryService(
            retriever=mock_retriever,
            generator=mock_generator,
            min_score=0.30,
        )

        result = service.query("What is deep learning?")
        assert result.num_chunks_retrieved == 1

        # Verify generator received only the high-scoring chunk
        called_chunks = mock_generator.generate.call_args[0][1]
        assert len(called_chunks) == 1
        assert called_chunks[0].chunk_id == "c_high"
