"""Reranking-specific exceptions — Phase 9.

Each exception maps to a distinct failure mode in the reranking pipeline.
Follows the same pattern as ``rag.retrieval.exceptions`` and
``rag.generation.exceptions``.
"""

from __future__ import annotations


class RerankingError(Exception):
    """Base exception for all reranking failures."""


class InvalidCandidateError(RerankingError, ValueError):
    """Candidate chunks are invalid, malformed, or have invalid type."""


class RerankerModelInitializationError(RerankingError):
    """The reranker model failed to initialize or load."""


class RerankerInferenceError(RerankingError):
    """The reranker model failed during scoring or produced malformed outputs."""
