"""MM-RAG Retrieval — Phase 4 + Phase 8.

Public API:
    Retriever            — abstract retriever interface.
    SemanticRetriever    — embedding-based semantic retrieval (Phase 4).
    BM25Index            — in-memory BM25 corpus index (Phase 8).
    BM25Retriever        — sparse keyword retrieval via BM25 (Phase 8).
    HybridRetriever      — combined semantic + keyword via RRF (Phase 8).
    InvalidQueryError    — invalid user query.
    InvalidTopKError     — invalid top_k parameter.
    EmbeddingError       — embedding service failure.
    VectorStoreError     — vector store failure.
    MAX_QUERY_LENGTH     — maximum allowed query length in characters.
    MAX_TOP_K            — maximum allowed top_k value.
"""

from rag.retrieval.base import Retriever
from rag.retrieval.bm25 import BM25Index, BM25Retriever
from rag.retrieval.exceptions import (
    EmbeddingError,
    InvalidQueryError,
    InvalidTopKError,
    RetrievalError,
    VectorStoreError,
)
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.semantic import (
    MAX_QUERY_LENGTH,
    MAX_TOP_K,
    SemanticRetriever,
)

__all__ = [
    "BM25Index",
    "BM25Retriever",
    "EmbeddingError",
    "HybridRetriever",
    "InvalidQueryError",
    "InvalidTopKError",
    "MAX_QUERY_LENGTH",
    "MAX_TOP_K",
    "RetrievalError",
    "Retriever",
    "SemanticRetriever",
    "VectorStoreError",
]
