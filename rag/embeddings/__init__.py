"""MM-RAG Embeddings — Phase 3.

Public API:
    EmbeddingService                      — abstract embedding interface.
    SentenceTransformerEmbeddingService   — sentence-transformers implementation.
    IndexingService                       — Chunk → Embed → Store orchestrator.
"""

from rag.embeddings.base import EmbeddingService
from rag.embeddings.indexing import IndexingService
from rag.embeddings.sentence_transformer import SentenceTransformerEmbeddingService

__all__ = [
    "EmbeddingService",
    "IndexingService",
    "SentenceTransformerEmbeddingService",
]
