"""RAG query orchestration service.

Connects the completed retrieval and generation modules into a
single application-level query operation.

Flow::

    question
        ↓
    Retriever.retrieve()         →  list[VectorSearchResult]
        ↓
    Generator.generate()         →  GenerationResult
        ↓
    QueryResult

This service does NOT duplicate retrieval, embedding, context building,
prompt construction, or LLM invocation logic.  It delegates entirely
to existing Phase 4–5 abstractions.

The service is independent of FastAPI / HTTP request-response objects.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from rag.generation.base import Generator
from rag.generation.exceptions import InvalidQuestionError
from rag.orchestration.exceptions import (
    InsufficientContextError,
    QueryError,
)
from rag.retrieval.base import Retriever

logger = logging.getLogger(__name__)


class QueryResult(BaseModel):
    """Result of a RAG query operation.

    Thin application-level wrapper that combines retrieval and
    generation results.  Does NOT duplicate ``VectorSearchResult``
    or ``GenerationResult`` — it presents the essential information
    for API consumers.

    Ready for Phase 7 citations: a ``sources`` field can be added
    without breaking the existing interface.

    Attributes:
        answer: The generated answer text.
        model_name: Identifier of the LLM model used.
        num_chunks_retrieved: Number of context chunks retrieved.
        metadata: Generation metadata (e.g. temperature, latency).
    """

    answer: str = Field(..., min_length=1)
    model_name: str = Field(default="")
    num_chunks_retrieved: int = Field(..., ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGQueryService:
    """Orchestrates the full RAG query pipeline.

    Connects Phase 4 retrieval and Phase 5 generation into a single
    ``query`` operation.

    Args:
        retriever: Phase 4 retriever (abstract interface).
        generator: Phase 5 LLM generator (abstract interface).
        min_score: Optional defensive minimum similarity score threshold.
            SemanticRetriever owns primary score filtering; this threshold
            serves as an additional defensive safeguard.
    """

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        min_score: float | None = None,
    ) -> None:
        if min_score is not None:
            if not isinstance(min_score, (int, float)) or isinstance(min_score, bool):
                raise ValueError(
                    f"min_score must be a float or int, got {type(min_score).__name__}"
                )
            if min_score < -1.0 or min_score > 1.0:
                raise ValueError(
                    f"min_score must be between -1.0 and 1.0, got {min_score}"
                )

        self._retriever = retriever
        self._generator = generator
        self._min_score = float(min_score) if min_score is not None else None

    def query(
        self,
        question: str,
        top_k: int | None = None,
    ) -> QueryResult:
        """Retrieve relevant context and generate an answer.

        Args:
            question: The user's natural-language question.
            top_k: Maximum number of context chunks to retrieve.
                When ``None``, the retriever uses its configured default.

        Returns:
            A ``QueryResult`` with the generated answer and metadata.

        Raises:
            InvalidQuestionError: If the question is invalid
                (empty, whitespace-only, wrong type).
            InsufficientContextError: If retrieval returns no usable context
                or all retrieved results fall below the relevance threshold.
                The LLM is NOT called in this case.
            EmptyRetrievalError: Base class of InsufficientContextError.
            QueryError: If retrieval or generation fails.
                The underlying cause is always preserved.
        """
        # Step 1: Retrieve relevant context chunks.
        try:
            results = self._retriever.retrieve(question, top_k=top_k)
        except InvalidQuestionError:
            # Let invalid-question errors pass through directly —
            # callers should handle these distinctly from pipeline failures.
            raise
        except Exception as exc:
            raise QueryError(f"Retrieval failed for question: {exc}") from exc

        # Step 2: Defensive relevance filtering (SemanticRetriever owns
        # primary filtering).
        if self._min_score is not None:
            results = [r for r in results if r.score >= self._min_score]

        # Step 3: Validate retrieval produced usable context.
        if not results:
            raise InsufficientContextError(
                "Retrieval returned no results or usable context for the "
                "question - cannot generate an answer without supporting context."
            )

        logger.info(
            "Retrieved %d context chunks for question.",
            len(results),
        )

        # Step 3: Generate answer from question + retrieved context.
        try:
            generation_result = self._generator.generate(question, results)
        except Exception as exc:
            raise QueryError(f"Generation failed: {exc}") from exc

        logger.info(
            "Generated answer using model '%s'.",
            generation_result.model_name,
        )

        return QueryResult(
            answer=generation_result.answer,
            model_name=generation_result.model_name,
            num_chunks_retrieved=len(results),
            metadata=generation_result.metadata,
        )
