"""Semantic retriever — Phase 4 implementation.

Retrieves relevant document chunks using embedding-based similarity
search.  Composes the existing Phase 3 ``EmbeddingService`` and
``VectorStore`` behind the ``Retriever`` interface.

Flow::

    User Query
        ↓
    Query Validation
        ↓
    EmbeddingService.embed_query()
        ↓
    VectorStore.query()
        ↓
    Top-K VectorSearchResult
"""

from __future__ import annotations

import logging

from rag.embeddings.base import EmbeddingService
from rag.retrieval.base import Retriever
from rag.retrieval.exceptions import (
    EmbeddingError,
    InvalidQueryError,
    VectorStoreError,
)
from rag.vectorstore.base import VectorStore
from rag.vectorstore.models import VectorSearchResult

logger = logging.getLogger(__name__)

# Absolute upper bound for top_k to prevent resource abuse.
MAX_TOP_K = 1000

# Maximum query length in characters.
MAX_QUERY_LENGTH = 10_000

# Default top_k when neither the caller nor configuration specifies one.
_FALLBACK_DEFAULT_TOP_K = 10


class SemanticRetriever(Retriever):
    """Retrieve chunks by semantic similarity.

    Thin, composable layer that delegates embedding to
    ``EmbeddingService`` and vector search to ``VectorStore``.

    Args:
        embedding_service: The embedding service for query vectorization.
        vector_store: The vector store to search.
        default_top_k: Default number of results when the caller does
            not specify ``top_k``.  Falls back to 10 if not provided.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        default_top_k: int = _FALLBACK_DEFAULT_TOP_K,
    ) -> None:
        if not isinstance(embedding_service, EmbeddingService):
            raise TypeError(
                f"embedding_service must be an EmbeddingService instance, "
                f"got {type(embedding_service).__name__}"
            )
        if not isinstance(vector_store, VectorStore):
            raise TypeError(
                f"vector_store must be a VectorStore instance, "
                f"got {type(vector_store).__name__}"
            )
        if (
            not isinstance(default_top_k, int)
            or isinstance(default_top_k, bool)
            or default_top_k < 1
        ):
            raise ValueError(
                f"default_top_k must be an integer >= 1, got {default_top_k}"
            )

        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._default_top_k = default_top_k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[VectorSearchResult]:
        """Retrieve the most relevant chunks for a user query.

        Steps:
            1. Validate the query string.
            2. Resolve and validate ``top_k``.
            3. Embed the query via ``EmbeddingService.embed_query``.
            4. Search the ``VectorStore`` with the query vector.
            5. Return results ordered by similarity (most relevant first).

        Args:
            query: The user's natural-language question.
            top_k: Maximum number of results.  Defaults to the
                configured ``default_top_k`` when ``None``.

        Returns:
            List of ``VectorSearchResult`` ordered by descending
            similarity score.  May be shorter than ``top_k`` when
            the store contains fewer chunks.

        Raises:
            InvalidQueryError: If the query is invalid.
            EmbeddingError: If embedding the query fails.
            VectorStoreError: If the vector store search fails.
        """
        self._validate_query(query)
        effective_top_k = self._resolve_top_k(top_k)

        # Step 3: Embed the query.
        query_vector = self._embed_query(query)

        # Step 4: Search the vector store.
        results = self._search(query_vector, effective_top_k)

        logger.info(
            "Retrieved %d results for query (top_k=%d).",
            len(results),
            effective_top_k,
        )

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_query(query: str) -> None:
        """Validate the user query.

        Raises:
            InvalidQueryError: If the query fails validation.
        """
        if not isinstance(query, str):
            raise InvalidQueryError(
                f"query must be a string, got {type(query).__name__}"
            )
        if not query.strip():
            raise InvalidQueryError("query must be a non-empty string")
        if len(query) > MAX_QUERY_LENGTH:
            raise InvalidQueryError(
                f"query exceeds maximum length of {MAX_QUERY_LENGTH} characters "
                f"(got {len(query)})"
            )

    def _resolve_top_k(self, top_k: int | None) -> int:
        """Resolve and validate the effective top_k value.

        Args:
            top_k: Caller-provided value, or ``None`` for default.

        Returns:
            Validated top_k value.

        Raises:
            InvalidQueryError: If top_k is invalid.
        """
        if top_k is None:
            return self._default_top_k

        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise InvalidQueryError(
                f"top_k must be a positive integer, got {type(top_k).__name__}"
            )
        if top_k < 1:
            raise InvalidQueryError(f"top_k must be >= 1, got {top_k}")
        if top_k > MAX_TOP_K:
            raise InvalidQueryError(
                f"top_k exceeds maximum of {MAX_TOP_K}, got {top_k}"
            )

        return top_k

    def _embed_query(self, query: str) -> list[float]:
        """Embed the query string using the embedding service.

        Raises:
            EmbeddingError: If the embedding service fails.
        """
        try:
            return self._embedding_service.embed_query(query)
        except Exception as exc:
            raise EmbeddingError(f"Failed to embed query: {exc}") from exc

    def _search(
        self,
        query_vector: list[float],
        top_k: int,
    ) -> list[VectorSearchResult]:
        """Search the vector store for similar chunks.

        Raises:
            VectorStoreError: If the vector store search fails.
        """
        try:
            return self._vector_store.query(query_vector, top_k=top_k)
        except Exception as exc:
            raise VectorStoreError(f"Vector store search failed: {exc}") from exc
