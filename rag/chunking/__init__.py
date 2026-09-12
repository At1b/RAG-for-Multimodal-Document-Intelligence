"""MM-RAG Document Chunking — Phase 2.

Public API:
    Chunk          — retrieval-ready text chunk with source metadata.
    ChunkingConfig — configurable chunk size and overlap.
    chunk_document — split a Document into Chunks.
    clean_text     — normalize text whitespace before chunking.
"""

from rag.chunking.chunker import ChunkingConfig, chunk_document
from rag.chunking.models import Chunk
from rag.chunking.text_cleaning import clean_text

__all__ = ["Chunk", "ChunkingConfig", "chunk_document", "clean_text"]
