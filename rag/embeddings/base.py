"""Abstract embedding service interface.

All embedding implementations must subclass ``EmbeddingService`` so the
rest of the pipeline (indexing, retrieval) remains independent of the
underlying model.

Replaceable: swap the concrete class and ``EMBEDDING_MODEL`` config
value — no downstream code changes required.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingService(ABC):
    """Base class for text embedding services.

    Subclasses must implement:
        - ``embed_documents`` — batch embedding for indexing.
        - ``embed_query`` — single-text embedding for retrieval.
        - ``dimension`` — dimensionality of the embedding vectors.
        - ``model_name`` — identifier of the underlying model.
    """

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document texts.

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            List of embedding vectors, one per input text.
            Each vector has length ``self.dimension``.

        Raises:
            ValueError: If *texts* is empty or contains non-string items.
            RuntimeError: If the model fails to produce embeddings.
        """

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text.

        Args:
            text: The query string to embed.

        Returns:
            Embedding vector of length ``self.dimension``.

        Raises:
            ValueError: If *text* is empty or not a string.
            RuntimeError: If the model fails to produce an embedding.
        """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors produced by this service."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier of the underlying embedding model."""
