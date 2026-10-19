"""Abstract reranker interface — Phase 9.

All reranker implementations must subclass ``Reranker`` so the rest
of the application (retrieval strategies, query orchestration, API endpoints)
depends on the abstraction rather than a specific model implementation.

Replaceable: swap the concrete class — no downstream code changes required.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag.vectorstore.models import VectorSearchResult


class Reranker(ABC):
    """Abstract base class for reranking candidate chunks."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[VectorSearchResult],
        top_k: int | None = None,
    ) -> list[VectorSearchResult]:
        """Rerank candidate chunks by query-specific relevance.

        Args:
            query: The user's natural-language question.
            candidates: List of already-retrieved candidate chunks.
            top_k: Maximum number of reranked results to return.
                When ``None``, implementations should use a sensible
                default from configuration or return all reordered candidates.

        Returns:
            List of ``VectorSearchResult`` ordered from most relevant to
            least relevant.  Original retrieval scores and all chunk
            metadata must be preserved.  The reranker relevance score
            is recorded distinctly in ``metadata['reranker_score']``.

        Raises:
            InvalidQueryError: If the query is invalid.
            InvalidCandidateError: If candidates are malformed.
            InvalidTopKError: If top_k is invalid.
            RerankingError: If model initialization or inference fails.
        """
