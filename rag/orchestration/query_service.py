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
    format_sources()             →  list[Source]
        ↓
    QueryResult (answer + sources)

This service does NOT duplicate retrieval, embedding, context building,
prompt construction, or LLM invocation logic.  It delegates entirely
to existing Phase 4–5 abstractions.

Sources/citations are built from actual retrieval metadata in
application code — never from LLM output.  See Phase 7 design.

The service is independent of FastAPI / HTTP request-response objects.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from rag.citations.formatter import format_sources
from rag.citations.models import Source
from rag.generation.base import Generator
from rag.generation.exceptions import InvalidQuestionError
from rag.orchestration.exceptions import (
    InsufficientContextError,
    QueryError,
)
from rag.reranking.base import Reranker
from rag.retrieval.base import Retriever
from rag.retrieval.reranking import RerankingRetriever

logger = logging.getLogger(__name__)


class QueryResult(BaseModel):
    """Result of a RAG query operation.

    Thin application-level wrapper that combines retrieval and
    generation results.  Does NOT duplicate ``VectorSearchResult``
    or ``GenerationResult`` — it presents the essential information
    for API consumers.

    Phase 7: Includes structured ``sources`` built from actual
    retrieved chunk metadata.  Sources are never fabricated or
    derived from LLM output.

    Attributes:
        answer: The generated answer text.
        model_name: Identifier of the LLM model used.
        num_chunks_retrieved: Number of context chunks retrieved.
        sources: Structured source/citation list from retrieved chunks.
            Built from application-level metadata, not LLM output.
        metadata: Generation metadata (e.g. temperature, latency).
    """

    answer: str = Field(..., min_length=1)
    model_name: str = Field(default="")
    num_chunks_retrieved: int = Field(..., ge=0)
    sources: list[Source] = Field(default_factory=list)
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
        reranker: Optional Phase 9 cross-encoder reranker.
        candidate_top_k: Optional candidate depth to retrieve before reranking.
    """

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        min_score: float | None = None,
        reranker: Reranker | None = None,
        candidate_top_k: int | None = None,
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

        if reranker is not None and not isinstance(reranker, Reranker):
            raise TypeError(
                f"reranker must be a Reranker instance, got {type(reranker).__name__}"
            )

        if reranker is not None and isinstance(retriever, RerankingRetriever):
            raise ValueError(
                "retriever is already a RerankingRetriever; do not provide a separate "
                "reranker to RAGQueryService to prevent duplicate reranking."
            )

        if reranker is None and candidate_top_k is not None:
            raise ValueError("candidate_top_k cannot be specified without a reranker")

        if candidate_top_k is not None:
            if (
                not isinstance(candidate_top_k, int)
                or isinstance(candidate_top_k, bool)
                or candidate_top_k < 1
                or candidate_top_k > 1000
            ):
                raise ValueError(
                    f"candidate_top_k must be an integer between 1 and 1000, "
                    f"got {candidate_top_k}"
                )

        self._retriever = retriever
        self._generator = generator
        self._min_score = float(min_score) if min_score is not None else None
        self._reranker = reranker
        self._candidate_top_k = candidate_top_k

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
            if self._reranker is not None:
                # Determine candidate retrieval depth
                candidate_depth = self._candidate_top_k or 20
                if top_k is not None and top_k > candidate_depth:
                    candidate_depth = top_k
                results = self._retriever.retrieve(question, top_k=candidate_depth)
            else:
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

        # Step 3: Validate retrieval produced usable context before reranking/LLM.
        if not results:
            raise InsufficientContextError(
                "Retrieval returned no results or usable context for the "
                "question - cannot generate an answer without supporting context."
            )

        # Step 4: Apply reranker if configured
        if self._reranker is not None:
            try:
                results = self._reranker.rerank(
                    query=question,
                    candidates=results,
                    top_k=top_k,
                )
            except Exception as exc:
                raise QueryError(f"Reranking failed for question: {exc}") from exc

            if not results:
                raise InsufficientContextError(
                    "Reranker returned no results for the question."
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

        # Step 4: Build structured sources from retrieval metadata.
        # Sources are built from application-level metadata — never
        # from LLM output.  This ensures citation integrity.
        sources = format_sources(results)

        return QueryResult(
            answer=generation_result.answer,
            model_name=generation_result.model_name,
            num_chunks_retrieved=len(results),
            sources=sources,
            metadata=generation_result.metadata,
        )
