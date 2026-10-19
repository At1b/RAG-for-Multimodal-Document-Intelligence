"""Reranking retriever — Phase 9.

Combines an underlying base retriever (semantic, keyword, or hybrid) with
a cross-encoder candidate reranker into a unified ``Retriever`` implementation.

Flow::

    query
      ↓
    base_retriever.retrieve(query, top_k=candidate_top_k, min_score=min_score)
      ↓
    candidates (Top-N)
      ↓
    reranker.rerank(query, candidates, top_k=top_k)
      ↓
    final Top-K results
"""

from __future__ import annotations

import logging

from rag.reranking.base import Reranker
from rag.retrieval.base import Retriever
from rag.retrieval.exceptions import InvalidQueryError, InvalidTopKError
from rag.vectorstore.models import VectorSearchResult

logger = logging.getLogger(__name__)

# Absolute upper bound for top_k to prevent resource abuse.
MAX_TOP_K = 1000

# Maximum query length in characters.
MAX_QUERY_LENGTH = 10_000


class RerankingRetriever(Retriever):
    """Retriever that fetches candidates via a base retriever and applies a reranker.

    Enables semantic, keyword, or hybrid candidate retrieval followed by
    cross-encoder reranking while conforming to the standard ``Retriever`` ABC.

    Args:
        base_retriever: Underlying retriever (SemanticRetriever, BM25Retriever,
            or HybridRetriever) used to fetch candidate chunks.
        reranker: Reranker instance used to reorder candidate chunks.
        candidate_top_k: Number of candidate chunks to fetch from the base
            retriever before reranking. Defaults to 20.
        default_top_k: Default number of final results returned to the caller
            when ``top_k`` is not specified. Defaults to 5.
    """

    def __init__(
        self,
        base_retriever: Retriever,
        reranker: Reranker,
        candidate_top_k: int = 20,
        default_top_k: int = 5,
    ) -> None:
        if not isinstance(base_retriever, Retriever):
            raise TypeError(
                f"base_retriever must be a Retriever instance, "
                f"got {type(base_retriever).__name__}"
            )
        if not isinstance(reranker, Reranker):
            raise TypeError(
                f"reranker must be a Reranker instance, got {type(reranker).__name__}"
            )
        if (
            not isinstance(candidate_top_k, int)
            or isinstance(candidate_top_k, bool)
            or candidate_top_k < 1
            or candidate_top_k > MAX_TOP_K
        ):
            raise ValueError(
                f"candidate_top_k must be an integer between 1 and {MAX_TOP_K}, "
                f"got {candidate_top_k}"
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

        self._base_retriever = base_retriever
        self._reranker = reranker
        self._candidate_top_k = candidate_top_k
        self._default_top_k = default_top_k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def base_retriever(self) -> Retriever:
        """The underlying candidate retriever."""
        return self._base_retriever

    @property
    def reranker(self) -> Reranker:
        """The candidate reranker."""
        return self._reranker

    @property
    def candidate_top_k(self) -> int:
        """Configured candidate depth."""
        return self._candidate_top_k

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[VectorSearchResult]:
        """Retrieve candidates and rerank them by relevance.

        Args:
            query: The user's natural-language question.
            top_k: Maximum number of final results to return.
            min_score: Minimum relevance score passed to the base retriever
                for candidate pre-filtering (preserves Phase 6/8 relevance gate).

        Returns:
            List of ``VectorSearchResult`` ordered by descending reranker score.

        Raises:
            InvalidQueryError: If the query is invalid.
            InvalidTopKError: If top_k is invalid.
            RerankingError: If reranking fails.
        """
        self._validate_query(query)
        effective_final_top_k = self._resolve_top_k(top_k)

        if min_score is not None:
            if not isinstance(min_score, (int, float)) or isinstance(min_score, bool):
                raise InvalidQueryError(
                    f"min_score must be a numeric value, got {type(min_score).__name__}"
                )
            if min_score < -1.0 or min_score > 1.0:
                raise InvalidQueryError(
                    f"min_score must be between -1.0 and 1.0, got {min_score}"
                )

        # Ensure candidate set is at least as large as the requested final top-k
        effective_candidate_top_k = max(self._candidate_top_k, effective_final_top_k)

        # Step 1: Retrieve candidate chunks using base retriever
        candidates = self._base_retriever.retrieve(
            query,
            top_k=effective_candidate_top_k,
            min_score=min_score,
        )

        # Step 2: If no candidates were retrieved (e.g. relevance gate rejected
        # out-of-domain query, or empty index), return empty immediately without
        # invoking the reranker.
        if not candidates:
            logger.info("Base retriever returned 0 candidates; skipping reranker.")
            return []

        # Step 3: Rerank retrieved candidates
        reranked = self._reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=effective_final_top_k,
        )

        logger.info(
            "RerankingRetriever: %d candidates → %d final results "
            "(candidate_top_k=%d, final_top_k=%d).",
            len(candidates),
            len(reranked),
            effective_candidate_top_k,
            effective_final_top_k,
        )

        return reranked

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_query(query: str) -> None:
        """Validate query text."""
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
        """Resolve and validate top_k."""
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
