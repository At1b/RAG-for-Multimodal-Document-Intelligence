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
from rag.orchestration.exceptions import EmptyRetrievalError, QueryError
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
    """

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
    ) -> None:
        self._retriever = retriever
        self._generator = generator

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
            EmptyRetrievalError: If retrieval returns no usable context.
                The LLM is NOT called in this case.
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

        # Step 2: Validate retrieval produced usable context.
        if not results:
            raise EmptyRetrievalError(
                "Retrieval returned no results — cannot generate an "
                "answer without supporting context."
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
