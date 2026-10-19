"""Cross-encoder candidate reranker — Phase 9.

Concrete implementation of ``Reranker`` backed by the
`sentence-transformers <https://sbert.net/>`_ CrossEncoder architecture.

Default model: ``cross-encoder/ms-marco-MiniLM-L-6-v2``
- License: Apache 2.0
- Architecture: 6-layer MiniLM cross-encoder trained on MS MARCO Passage Ranking
- Output: Relevance logits where higher values indicate greater relevance
- Practical for CPU inference (~80 MB, ~10ms per candidate pair)
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from rag.reranking.base import Reranker
from rag.reranking.exceptions import (
    InvalidCandidateError,
    RerankerInferenceError,
    RerankerModelInitializationError,
)
from rag.retrieval.exceptions import InvalidQueryError, InvalidTopKError
from rag.vectorstore.models import VectorSearchResult

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Default model chosen for Phase 9.
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Absolute upper bound for top_k to prevent resource abuse.
MAX_TOP_K = 1000

# Maximum query length in characters (consistent with retrieval components).
MAX_QUERY_LENGTH = 10_000

# Default top-k returned after reranking if unspecified.
_FALLBACK_DEFAULT_TOP_K = 5

# Shared process-level cache for loaded CrossEncoder model weights.
# Ensures model weights are reused across queries and instances rather
# than reloaded from disk on every request.
_MODEL_CACHE: dict[str, CrossEncoder] = {}
_MODEL_CACHE_LOCK = threading.Lock()


class CrossEncoderReranker(Reranker):
    """Reranker using a Cross-Encoder model.

    The model is **lazily loaded** on the first call to ``rerank``
    so that importing this module has no startup or model download cost.
    The loaded model instance is cached and reused across requests.

    For testing and isolation, a pre-initialized ``model`` or a custom
    ``scorer`` callable can be injected to evaluate reranking logic
    without downloading model weights or performing live inference.

    Args:
        model_name: Hugging Face model identifier for the cross-encoder.
            Defaults to ``cross-encoder/ms-marco-MiniLM-L-6-v2``.
        model: Optional pre-initialized CrossEncoder model or mock object
            implementing a ``predict(sentences)`` method.
        scorer: Optional scoring function accepting a list of
            ``(query, document_content)`` tuples and returning a sequence
            of float scores.  Takes precedence over ``model`` if provided.
        default_top_k: Default number of reranked candidates to return
            when ``top_k`` is not specified by the caller.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        model: Any | None = None,
        scorer: (
            Callable[[list[tuple[str, str]]], Sequence[float] | list[float]] | None
        ) = None,
        default_top_k: int = _FALLBACK_DEFAULT_TOP_K,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
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

        self._model_name = model_name.strip()
        self._model: CrossEncoder | None = model
        self._scorer = scorer
        self._default_top_k = default_top_k
        self._lock = threading.Lock()

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the shared process-level model cache (useful in tests)."""
        with _MODEL_CACHE_LOCK:
            _MODEL_CACHE.clear()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """Identifier of the cross-encoder model."""
        return self._model_name

    def rerank(
        self,
        query: str,
        candidates: list[VectorSearchResult],
        top_k: int | None = None,
    ) -> list[VectorSearchResult]:
        """Rerank candidate chunks by relevance to the query.

        Steps:
            1. Validate query string and sanitize formatting.
            2. Validate and resolve ``top_k``.
            3. Validate candidate list and candidate types.
            4. Short-circuit on empty candidates without model invocation.
            5. Deduplicate candidates by chunk_id, preserving first occurrence.
            6. Compute cross-encoder relevance scores for (query, content) pairs.
            7. Validate score count, types, and finiteness (no NaN or Inf).
            8. Sort deterministically by descending reranker score with stable
               tie-breaking (original candidate order, then chunk_id).
            9. Preserve original retrieval score in ``result.score`` and store
               the reranker score in ``metadata['reranker_score']``.
            10. Return the top-K reranked chunks.

        Args:
            query: The user's natural-language question.
            candidates: List of ``VectorSearchResult`` objects.
            top_k: Number of top reranked chunks to return.

        Returns:
            List of ``VectorSearchResult`` ordered by descending reranker score.

        Raises:
            InvalidQueryError: If the query is invalid or too long.
            InvalidTopKError: If top_k is invalid.
            InvalidCandidateError: If candidates is not a list or contains
                invalid items.
            RerankerModelInitializationError: If model loading fails.
            RerankerInferenceError: If scoring fails or outputs are malformed.
        """
        clean_query = self._validate_query(query)
        effective_top_k = self._resolve_top_k(top_k)

        if not isinstance(candidates, list):
            raise InvalidCandidateError(
                f"candidates must be a list of VectorSearchResult, "
                f"got {type(candidates).__name__}"
            )

        if len(candidates) == 0:
            logger.debug(
                "Empty candidates list passed to rerank; returning empty list."
            )
            return []

        for idx, item in enumerate(candidates):
            if not isinstance(item, VectorSearchResult):
                raise InvalidCandidateError(
                    f"All candidate items must be VectorSearchResult instances, "
                    f"got {type(item).__name__} at index {idx}"
                )

        # Deduplicate candidates by chunk_id while preserving original order
        unique_candidates: list[VectorSearchResult] = []
        seen_chunk_ids: set[str] = set()
        for c in candidates:
            if c.chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(c.chunk_id)
                unique_candidates.append(c)

        # Construct (query, passage) pairs for cross-encoder inference
        pairs = [(clean_query, c.content) for c in unique_candidates]

        # Compute relevance scores
        scores = self._score_pairs(pairs)

        # Validate score count and numeric integrity
        if len(scores) != len(unique_candidates):
            raise RerankerInferenceError(
                f"Scorer returned {len(scores)} scores for {len(unique_candidates)} "
                f"candidates"
            )

        # Build scored items with deterministic tie-breaking:
        # (-score, original_index, chunk_id)
        scored_items: list[tuple[float, int, str, VectorSearchResult]] = []
        for idx, (candidate, raw_score) in enumerate(
            zip(unique_candidates, scores, strict=True)
        ):
            if (
                not isinstance(raw_score, (int, float))
                or isinstance(raw_score, bool)
                or math.isnan(raw_score)
                or math.isinf(raw_score)
            ):
                raise RerankerInferenceError(
                    f"Reranker returned non-finite or non-numeric score at "
                    f"index {idx}: {raw_score}"
                )

            score_val = float(raw_score)
            meta = candidate.metadata.copy() if candidate.metadata else {}
            meta["retrieval_score"] = candidate.score
            meta["reranker_score"] = score_val

            reranked_result = VectorSearchResult(
                chunk_id=candidate.chunk_id,
                document_id=candidate.document_id,
                document_name=candidate.document_name,
                content=candidate.content,
                score=candidate.score,  # Preserve original retrieval score
                metadata=meta,
            )
            # Tuple: (-score, original_rank, chunk_id, result)
            scored_items.append((-score_val, idx, candidate.chunk_id, reranked_result))

        # Deterministic sorting
        scored_items.sort(key=lambda x: (x[0], x[1], x[2]))

        final_results = [item[3] for item in scored_items[:effective_top_k]]

        logger.info(
            "Reranked %d candidates to %d final results (top_k=%d).",
            len(unique_candidates),
            len(final_results),
            effective_top_k,
        )

        return final_results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_query(query: str) -> str:
        """Validate and sanitize the user query."""
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
        return sanitized

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

    def _score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Execute cross-encoder inference on (query, passage) pairs."""
        if self._scorer is not None:
            try:
                raw_scores = self._scorer(pairs)
                if hasattr(raw_scores, "tolist"):
                    raw_scores = raw_scores.tolist()
                if isinstance(raw_scores, (int, float)) and not isinstance(
                    raw_scores, bool
                ):
                    return [float(raw_scores)]
                return list(raw_scores)
            except Exception as exc:
                if isinstance(exc, (InvalidQueryError, RerankerInferenceError)):
                    raise
                raise RerankerInferenceError(f"Injected scorer failed: {exc}") from exc

        model = self._get_model()
        try:
            raw_scores = model.predict(pairs)
            if hasattr(raw_scores, "tolist"):
                raw_scores = raw_scores.tolist()
            if isinstance(raw_scores, (int, float)) and not isinstance(
                raw_scores, bool
            ):
                return [float(raw_scores)]
            return list(raw_scores)
        except Exception as exc:
            if isinstance(exc, (InvalidQueryError, RerankerInferenceError)):
                raise
            raise RerankerInferenceError(
                f"Cross-encoder inference failed: {exc}"
            ) from exc

    def _get_model(self) -> CrossEncoder:
        """Return the loaded model, initializing on first call with shared caching."""
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:
                return self._model

            with _MODEL_CACHE_LOCK:
                cached = _MODEL_CACHE.get(self._model_name)
            if cached is not None:
                self._model = cached
                return self._model

            loaded = self._load_model()
            with _MODEL_CACHE_LOCK:
                _MODEL_CACHE[self._model_name] = loaded
            self._model = loaded
            return self._model

    def _load_model(self) -> CrossEncoder:
        """Load the CrossEncoder model lazily."""
        try:
            from sentence_transformers import CrossEncoder

            logger.info("Loading CrossEncoder model '%s'...", self._model_name)
            return CrossEncoder(self._model_name)
        except Exception as exc:
            raise RerankerModelInitializationError(
                f"Failed to load CrossEncoder model '{self._model_name}': {exc}"
            ) from exc
