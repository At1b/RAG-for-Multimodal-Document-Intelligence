"""Reranking module — Phase 9.

Provides interfaces and implementations for cross-encoder reranking of
retrieved candidate document chunks.
"""

from rag.reranking.base import Reranker
from rag.reranking.cross_encoder import CrossEncoderReranker
from rag.reranking.exceptions import (
    InvalidCandidateError,
    RerankerInferenceError,
    RerankerModelInitializationError,
    RerankingError,
)

__all__ = [
    "CrossEncoderReranker",
    "InvalidCandidateError",
    "Reranker",
    "RerankerInferenceError",
    "RerankerModelInitializationError",
    "RerankingError",
]
