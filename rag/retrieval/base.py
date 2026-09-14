"""Abstract retriever interface.

All retriever implementations must subclass ``Retriever`` so the rest
of the application (future query services, API endpoints) depends on
the abstraction rather than a specific retrieval strategy.

Replaceable: swap the concrete class — no downstream code changes required.
Future strategies (hybrid, reranking) can implement this same interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag.vectorstore.models import VectorSearchResult


class Retriever(ABC):
    """Base class for retrieval strategies.

    Subclasses must implement ``retrieve`` to return relevant document
    chunks for a given user query.
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[VectorSearchResult]:
        """Retrieve the most relevant chunks for a user query.

        Args:
            query: The user's natural-language question.
            top_k: Maximum number of results to return.  When ``None``,
                implementations should use a sensible default from
                project configuration.

        Returns:
            List of ``VectorSearchResult`` ordered by relevance
            (most relevant first).  May be shorter than *top_k*
            when fewer matching chunks are available.

        Raises:
            InvalidQueryError: If the query is invalid.
            EmbeddingError: If the embedding service fails.
            VectorStoreError: If the vector store fails.
        """
