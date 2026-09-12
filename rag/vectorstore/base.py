"""Abstract vector store interface.

All vector store implementations must subclass ``VectorStore`` so the
retrieval and indexing layers remain independent of the storage backend.

Replaceable: swap the concrete class and update configuration — no
downstream code changes required.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag.chunking.models import Chunk
from rag.vectorstore.models import VectorSearchResult


class VectorStore(ABC):
    """Base class for vector storage backends.

    Subclasses must implement all abstract methods to provide:
    - Storing embeddings with associated chunk metadata.
    - Similarity-based lookup.
    - Document-level deletion for re-indexing.
    - Count and reset operations.
    """

    @abstractmethod
    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Store chunk embeddings with their metadata.

        Args:
            chunks: List of ``Chunk`` objects to store.
            embeddings: Corresponding embedding vectors (same length
                as *chunks*).

        Raises:
            ValueError: If *chunks* and *embeddings* have different
                lengths, or either is empty.
        """

    @abstractmethod
    def query(
        self,
        query_vector: list[float],
        top_k: int = 10,
    ) -> list[VectorSearchResult]:
        """Find the most similar stored vectors to *query_vector*.

        Args:
            query_vector: The embedding vector to search against.
            top_k: Maximum number of results to return.

        Returns:
            List of ``VectorSearchResult`` ordered by similarity
            (most similar first).  May be shorter than *top_k* if
            the store contains fewer vectors.
        """

    @abstractmethod
    def delete_document(self, document_id: str) -> int:
        """Delete all stored vectors for a given document.

        Used for re-indexing: remove stale chunks before inserting
        the new version.

        Args:
            document_id: The document whose chunks should be removed.

        Returns:
            Number of vectors deleted.
        """

    @abstractmethod
    def count(self) -> int:
        """Return the total number of stored vectors."""

    @abstractmethod
    def reset(self) -> None:
        """Remove all stored vectors.

        Primarily intended for testing.
        """
