"""MM-RAG Vector Store — Phase 3.

Public API:
    VectorStore          — abstract vector store interface.
    ChromaVectorStore    — ChromaDB implementation.
    VectorSearchResult   — single result from a similarity search.
"""

from rag.vectorstore.base import VectorStore
from rag.vectorstore.chroma_store import ChromaVectorStore
from rag.vectorstore.models import VectorSearchResult

__all__ = [
    "ChromaVectorStore",
    "VectorSearchResult",
    "VectorStore",
]
