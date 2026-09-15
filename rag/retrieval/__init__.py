"""MM-RAG Retrieval — Phase 4.

Public API:
    Retriever            — abstract retriever interface.
    SemanticRetriever    — embedding-based semantic retrieval.
    InvalidQueryError    — invalid user query.
    InvalidTopKError     — invalid top_k parameter.
    EmbeddingError       — embedding service failure.
    VectorStoreError     — vector store failure.
    MAX_QUERY_LENGTH     — maximum allowed query length in characters.
    MAX_TOP_K            — maximum allowed top_k value.
"""

from rag.retrieval.base import Retriever
from rag.retrieval.exceptions import (
    EmbeddingError,
    InvalidQueryError,
    InvalidTopKError,
    RetrievalError,
    VectorStoreError,
)
from rag.retrieval.semantic import (
    MAX_QUERY_LENGTH,
    MAX_TOP_K,
    SemanticRetriever,
)

__all__ = [
    "EmbeddingError",
    "InvalidQueryError",
    "InvalidTopKError",
    "MAX_QUERY_LENGTH",
    "MAX_TOP_K",
    "RetrievalError",
    "Retriever",
    "SemanticRetriever",
    "VectorStoreError",
]
