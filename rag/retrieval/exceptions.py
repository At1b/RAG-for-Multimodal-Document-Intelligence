"""Retrieval-specific exceptions.

Each exception maps to a distinct failure mode in the retrieval pipeline.
Follows the same pattern as ``rag.ingestion.exceptions``.
"""


class RetrievalError(Exception):
    """Base exception for all retrieval-related errors."""


class InvalidQueryError(RetrievalError):
    """The user query is invalid (empty, whitespace-only, wrong type)."""


class EmbeddingError(RetrievalError):
    """The embedding service failed to embed the query."""


class VectorStoreError(RetrievalError):
    """The vector store failed during search."""
