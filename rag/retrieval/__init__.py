"""MM-RAG Retrieval — Phase 4.

Public API:
    Retriever            — abstract retriever interface.
    SemanticRetriever    — embedding-based semantic retrieval.
    InvalidQueryError    — invalid user query.
    EmbeddingError       — embedding service failure.
    VectorStoreError     — vector store failure.
"""

from rag.retrieval.base import Retriever
from rag.retrieval.exceptions import (
    EmbeddingError,
    InvalidQueryError,
    RetrievalError,
    VectorStoreError,
)
from rag.retrieval.semantic import SemanticRetriever

__all__ = [
    "EmbeddingError",
    "InvalidQueryError",
    "RetrievalError",
    "Retriever",
    "SemanticRetriever",
    "VectorStoreError",
]
