"""Hybrid retriever — Phase 8.

Combines semantic (dense) and BM25 (sparse) retrieval using Reciprocal
Rank Fusion (RRF) for score-agnostic rank combination.

Why RRF over score normalization:
    BM25 scores are unbounded positive values dependent on corpus statistics,
    while cosine similarity is bounded in [-1, 1].  Score normalization
    (e.g. min-max) is unstable: a single high-scoring outlier can compress
    all other scores to near-zero, and the normalization is batch-dependent
    (same chunk gets different normalized scores depending on what else was
    retrieved).  RRF avoids all of these problems by fusing only **ranks**,
    not raw scores.  It is simple, robust, well-studied, and produces
    stable, deterministic rankings.

RRF formula:
    For each document d, the fused score is:

        RRF(d) = Σ_r  weight_r / (k + rank_r(d))

    where:
        - r ranges over the retrievers (semantic, BM25)
        - rank_r(d) is the 1-based rank of d in retriever r's result list
        - k is a constant (default 60) that dampens the influence of high ranks
        - weight_r is the configurable weight for retriever r

    Documents only present in one retriever's results receive a score
    contribution only from that retriever.

Configuration:
    - semantic_weight: Weight for semantic retriever (default 1.0).
    - keyword_weight: Weight for BM25 retriever (default 1.0).
    - rrf_k: RRF constant (default 60).
    - top_k: Final number of results to return.
    - semantic_top_k: Number of candidates from semantic retriever.
    - keyword_top_k: Number of candidates from BM25 retriever.

Score semantics:
    The ``score`` field in returned ``VectorSearchResult`` contains
    the RRF fusion score, NOT a cosine similarity or BM25 score.
    It is a relative ranking score useful for ordering but not directly
    comparable to Phase 6 ``RETRIEVAL_MIN_SCORE``.
"""

from __future__ import annotations

import logging

from rag.retrieval.base import Retriever
from rag.retrieval.exceptions import InvalidTopKError
from rag.vectorstore.models import VectorSearchResult

logger = logging.getLogger(__name__)

# Absolute upper bound for top_k to prevent resource abuse.
MAX_TOP_K = 1000


class HybridRetriever(Retriever):
    """Combines semantic and keyword retrieval via Reciprocal Rank Fusion.

    Both retrievers are run independently, and their results are fused
    using weighted RRF.  Duplicate chunks (same ``chunk_id``) are
    deduplicated, keeping the entry with the richer metadata.

    Args:
        semantic_retriever: Dense embedding-based retriever.
        keyword_retriever: Sparse BM25-based retriever.
        semantic_weight: Weight for semantic retriever in RRF (default 1.0).
        keyword_weight: Weight for keyword retriever in RRF (default 1.0).
        rrf_k: RRF constant that dampens high-rank influence (default 60).
        default_top_k: Default number of final results (default 10).
        semantic_top_k: Number of candidates from semantic retriever (default 20).
        keyword_top_k: Number of candidates from BM25 retriever (default 20).
    """

    def __init__(
        self,
        semantic_retriever: Retriever,
        keyword_retriever: Retriever,
        *,
        semantic_weight: float = 1.0,
        keyword_weight: float = 1.0,
        rrf_k: int = 60,
        default_top_k: int = 10,
        semantic_top_k: int = 20,
        keyword_top_k: int = 20,
    ) -> None:
        # Validate retrievers
        if not isinstance(semantic_retriever, Retriever):
            raise TypeError(
                f"semantic_retriever must be a Retriever instance, "
                f"got {type(semantic_retriever).__name__}"
            )
        if not isinstance(keyword_retriever, Retriever):
            raise TypeError(
                f"keyword_retriever must be a Retriever instance, "
                f"got {type(keyword_retriever).__name__}"
            )

        # Validate weights
        self._validate_weight(semantic_weight, "semantic_weight")
        self._validate_weight(keyword_weight, "keyword_weight")
        if semantic_weight == 0.0 and keyword_weight == 0.0:
            raise ValueError(
                "At least one of semantic_weight or keyword_weight must be > 0"
            )

        # Validate rrf_k
        if not isinstance(rrf_k, int) or isinstance(rrf_k, bool) or rrf_k < 1:
            raise ValueError(f"rrf_k must be a positive integer, got {rrf_k}")

        # Validate top_k values
        self._validate_top_k(default_top_k, "default_top_k")
        self._validate_top_k(semantic_top_k, "semantic_top_k")
        self._validate_top_k(keyword_top_k, "keyword_top_k")

        self._semantic_retriever = semantic_retriever
        self._keyword_retriever = keyword_retriever
        self._semantic_weight = float(semantic_weight)
        self._keyword_weight = float(keyword_weight)
        self._rrf_k = rrf_k
        self._default_top_k = default_top_k
        self._semantic_top_k = semantic_top_k
        self._keyword_top_k = keyword_top_k

    @staticmethod
    def _validate_weight(value: float, name: str) -> None:
        """Validate a retriever weight."""
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(
                f"{name} must be a numeric value, got {type(value).__name__}"
            )
        if value < 0.0:
            raise ValueError(f"{name} must be >= 0, got {value}")

    @staticmethod
    def _validate_top_k(value: int, name: str) -> None:
        """Validate a top_k parameter."""
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(
                f"{name} must be a positive integer, got {type(value).__name__}"
            )
        if value < 1:
            raise ValueError(f"{name} must be >= 1, got {value}")
        if value > MAX_TOP_K:
            raise ValueError(f"{name} must be <= {MAX_TOP_K}, got {value}")

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[VectorSearchResult]:
        """Retrieve and fuse results from semantic and keyword retrievers.

        Both retrievers are called independently.  Their results are
        combined using weighted Reciprocal Rank Fusion.

        Args:
            query: The user's natural-language question.
            top_k: Maximum number of final results.
            min_score: Not applied to hybrid RRF scores (RRF scores are
                relative ranking scores, not similarity measures).
                Accepted for interface compatibility but ignored.
                The semantic retriever internally applies its own min_score.

        Returns:
            List of ``VectorSearchResult`` ordered by descending RRF
            fusion score.  The ``score`` field contains the RRF score.

        Raises:
            InvalidQueryError: If the query is invalid.
            InvalidTopKError: If top_k is invalid.
        """
        effective_top_k = self._resolve_top_k(top_k)

        # Run both retrievers independently.
        # Each retriever handles its own query validation.
        semantic_results: list[VectorSearchResult] = []
        keyword_results: list[VectorSearchResult] = []

        if self._semantic_weight > 0:
            try:
                semantic_results = self._semantic_retriever.retrieve(
                    query, top_k=self._semantic_top_k
                )
            except Exception:
                logger.warning(
                    "Semantic retriever failed; using keyword only.",
                    exc_info=True,
                )

        if self._keyword_weight > 0:
            try:
                keyword_results = self._keyword_retriever.retrieve(
                    query, top_k=self._keyword_top_k
                )
            except Exception:
                logger.warning(
                    "Keyword retriever failed; using semantic only.",
                    exc_info=True,
                )

        # If both retrievers returned nothing, return empty.
        if not semantic_results and not keyword_results:
            return []

        # Fuse results using weighted RRF.
        fused = self._reciprocal_rank_fusion(
            semantic_results,
            keyword_results,
        )

        # Return top-K fused results.
        results = fused[:effective_top_k]

        logger.info(
            "Hybrid retrieval: %d semantic + %d keyword → %d fused results (top_k=%d).",
            len(semantic_results),
            len(keyword_results),
            len(results),
            effective_top_k,
        )

        return results

    def _reciprocal_rank_fusion(
        self,
        semantic_results: list[VectorSearchResult],
        keyword_results: list[VectorSearchResult],
    ) -> list[VectorSearchResult]:
        """Combine results from both retrievers using weighted RRF.

        Deduplicates by ``chunk_id``.  When the same chunk appears in
        both result sets, the entry from the semantic retriever is
        preferred (it typically has richer metadata from ChromaDB),
        and the RRF scores from both retrievers are summed.

        Args:
            semantic_results: Ranked results from semantic retriever.
            keyword_results: Ranked results from keyword retriever.

        Returns:
            List of ``VectorSearchResult`` sorted by descending RRF score.
        """
        # Track RRF score and best entry per chunk_id.
        rrf_scores: dict[str, float] = {}
        entries: dict[str, VectorSearchResult] = {}

        # Score semantic results.
        for rank_0, result in enumerate(semantic_results):
            rank = rank_0 + 1  # 1-based rank
            rrf_contribution = self._semantic_weight / (self._rrf_k + rank)
            chunk_id = result.chunk_id

            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + rrf_contribution
            # Prefer semantic entry (richer metadata from ChromaDB).
            if chunk_id not in entries:
                entries[chunk_id] = result

        # Score keyword results.
        for rank_0, result in enumerate(keyword_results):
            rank = rank_0 + 1  # 1-based rank
            rrf_contribution = self._keyword_weight / (self._rrf_k + rank)
            chunk_id = result.chunk_id

            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + rrf_contribution
            # Only store if not already present from semantic results.
            if chunk_id not in entries:
                entries[chunk_id] = result

        # Sort by RRF score descending, then by chunk_id for determinism.
        sorted_ids = sorted(
            rrf_scores.keys(),
            key=lambda cid: (-rrf_scores[cid], cid),
        )

        # Build final results with RRF scores.
        fused_results: list[VectorSearchResult] = []
        for chunk_id in sorted_ids:
            entry = entries[chunk_id]
            fused_results.append(
                VectorSearchResult(
                    chunk_id=entry.chunk_id,
                    document_id=entry.document_id,
                    document_name=entry.document_name,
                    content=entry.content,
                    score=round(rrf_scores[chunk_id], 8),
                    metadata=entry.metadata.copy(),
                )
            )

        return fused_results

    def _resolve_top_k(self, top_k: int | None) -> int:
        """Resolve and validate the effective top_k value."""
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
