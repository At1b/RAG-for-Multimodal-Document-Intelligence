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
import math

from rag.embeddings.base import EmbeddingService
from rag.retrieval.base import Retriever
from rag.retrieval.exceptions import (
    EmbeddingError,
    InvalidQueryError,
    InvalidTopKError,
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
        min_score: float | None = None,
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
            or default_top_k > MAX_TOP_K
        ):
            raise ValueError(
                f"default_top_k must be an integer between 1 and {MAX_TOP_K}, "
                f"got {default_top_k}"
            )
        if min_score is not None:
            if (
                not isinstance(min_score, (int, float))
                or isinstance(min_score, bool)
                or math.isnan(min_score)
                or math.isinf(min_score)
            ):
                raise ValueError(
                    f"min_score must be a finite numeric value, "
                    f"got {type(min_score).__name__}"
                )
            if min_score < -1.0 or min_score > 1.0:
                raise ValueError(
                    f"min_score must be between -1.0 and 1.0, got {min_score}"
                )

        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._default_top_k = default_top_k
        self._min_score = float(min_score) if min_score is not None else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def min_score(self) -> float | None:
        """Configured minimum similarity score threshold, or ``None``."""
        return self._min_score

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[VectorSearchResult]:
        """Retrieve the most relevant chunks for a user query.

        Steps:
            1. Validate the query string.
            2. Resolve and validate ``top_k``.
            3. Resolve and validate ``min_score``.
            4. Embed the query via ``EmbeddingService.embed_query``.
            5. Search the ``VectorStore`` with the query vector.
            6. Filter results by ``min_score`` (if configured).
            7. Return results ordered by similarity (most relevant first).

        Args:
            query: The user's natural-language question.
            top_k: Maximum number of results.  Defaults to the
                configured ``default_top_k`` when ``None``.
            min_score: Optional minimum similarity score threshold.
                Overrides the instance-configured ``min_score`` if provided.
                Chunks with similarity score strictly below this threshold
                are excluded from the returned results.

        Returns:
            List of ``VectorSearchResult`` ordered by descending
            similarity score.  May be shorter than ``top_k`` when
            the store contains fewer chunks or when chunks are filtered
            by ``min_score``.

        Raises:
            InvalidQueryError: If the query or min_score is invalid.
            InvalidTopKError: If top_k is invalid.
            EmbeddingError: If embedding the query fails.
            VectorStoreError: If the vector store search fails.
        """
        self._validate_query(query)
        effective_top_k = self._resolve_top_k(top_k)
        effective_min_score = self._resolve_min_score(min_score)

        # Sanitize query text before embedding (strip null bytes and outer whitespace)
        clean_query = query.replace("\x00", "").strip()

        # Step 4: Embed the query.
        query_vector = self._embed_query(clean_query)

        # Step 5: Search the vector store.
        results = self._search(query_vector, effective_top_k)

        # Step 6: Filter by minimum similarity score threshold.
        if effective_min_score is not None:
            results = [r for r in results if r.score >= effective_min_score]

        logger.info(
            "Retrieved %d results for query (top_k=%d, min_score=%s).",
            len(results),
            effective_top_k,
            effective_min_score,
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
        sanitized = (
            query.replace("\x00", "")
            .replace("\ufeff", "")
            .replace("\u200b", "")
            .strip()
        )
        if not sanitized:
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
            InvalidTopKError: If top_k is invalid.
        """
        candidate = self._default_top_k if top_k is None else top_k

        if not isinstance(candidate, int) or isinstance(candidate, bool):
            raise InvalidTopKError(
                f"top_k must be a positive integer, got {type(candidate).__name__}"
            )
        if candidate < 1:
            raise InvalidTopKError(f"top_k must be >= 1, got {candidate}")
        if candidate > MAX_TOP_K:
            raise InvalidTopKError(
                f"top_k exceeds maximum of {MAX_TOP_K}, got {candidate}"
            )

        return candidate

    def _resolve_min_score(self, min_score: float | None) -> float | None:
        """Resolve and validate the effective min_score threshold.

        Args:
            min_score: Caller-provided threshold, or ``None`` for default.

        Returns:
            Validated min_score value or ``None`` if no threshold is active.

        Raises:
            InvalidQueryError: If min_score is not a valid finite float in [-1.0, 1.0].
        """
        candidate = self._min_score if min_score is None else min_score

        if candidate is None:
            return None

        if (
            not isinstance(candidate, (int, float))
            or isinstance(candidate, bool)
            or math.isnan(candidate)
            or math.isinf(candidate)
        ):
            raise InvalidQueryError(
                f"min_score must be a finite numeric value, "
                f"got {type(candidate).__name__}"
            )
        if candidate < -1.0 or candidate > 1.0:
            raise InvalidQueryError(
                f"min_score must be between -1.0 and 1.0, got {candidate}"
            )

        return float(candidate)

    def _embed_query(self, query: str) -> list[float]:
        """Embed the query string using the embedding service.

        Raises:
            EmbeddingError: If the embedding service fails or returns an invalid vector.
        """
        try:
            vector = self._embedding_service.embed_query(query)
        except Exception as exc:
            raise EmbeddingError(f"Failed to embed query: {exc}") from exc

        if not isinstance(vector, list) or len(vector) == 0:
            got_desc = (
                type(vector).__name__ if not isinstance(vector, list) else "empty list"
            )
            raise EmbeddingError(
                f"Embedding service returned invalid vector: expected non-empty "
                f"list of floats, got {got_desc}"
            )

        for val in vector:
            if (
                not isinstance(val, (int, float))
                or isinstance(val, bool)
                or math.isnan(val)
                or math.isinf(val)
            ):
                raise EmbeddingError(
                    "Embedding service returned vector containing non-numeric "
                    "or non-finite values"
                )

        return vector

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
            results = self._vector_store.query(query_vector, top_k=top_k)
        except Exception as exc:
            raise VectorStoreError(f"Vector store search failed: {exc}") from exc

        if not isinstance(results, list):
            raise VectorStoreError(
                f"Vector store returned non-list result: {type(results).__name__}"
            )

        return results
