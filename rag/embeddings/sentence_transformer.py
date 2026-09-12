"""Sentence-Transformers embedding service.

Concrete implementation of ``EmbeddingService`` backed by the
`sentence-transformers <https://sbert.net/>`_ library.

Default model: ``all-MiniLM-L6-v2`` (384 dimensions, ~80 MB, Apache 2.0).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rag.embeddings.base import EmbeddingService

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Default model chosen for Phase 3.  See implementation_plan.md for the
# full evaluation and rationale.
DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 64


class SentenceTransformerEmbeddingService(EmbeddingService):
    """Embedding service using a Sentence-Transformers model.

    The model is **lazily loaded** on the first call to ``embed_documents``
    or ``embed_query`` so that importing this module has no startup cost.

    Args:
        model_name: Hugging Face model identifier.
            Defaults to ``all-MiniLM-L6-v2``.
        batch_size: Maximum number of texts encoded in a single forward
            pass.  Defaults to 64.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size < 1
        ):
            raise ValueError(f"batch_size must be an integer >= 1, got {batch_size}")

        self._model_name = model_name.strip()
        self._batch_size = batch_size
        self._model: SentenceTransformer | None = None
        self._dimension: int | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document texts.

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            List of embedding vectors (one per input text).

        Raises:
            ValueError: If *texts* is empty, not a list, or contains
                non-string items.
            RuntimeError: If the model fails to produce embeddings.
        """
        self._validate_texts(texts)
        model = self._get_model()

        try:
            embeddings = model.encode(
                texts,
                batch_size=self._batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return embeddings.tolist()
        except Exception as exc:
            raise RuntimeError(
                f"Embedding generation failed for {len(texts)} texts: {exc}"
            ) from exc

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
        if not isinstance(text, str):
            raise ValueError(f"text must be a string, got {type(text).__name__}")
        if not text.strip():
            raise ValueError("text must be a non-empty string")

        result = self.embed_documents([text])
        return result[0]

    @property
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors."""
        if self._dimension is None:
            model = self._get_model()
            dim = model.get_embedding_dimension()
            if dim is None:
                raise RuntimeError(
                    f"Could not determine embedding dimension for "
                    f"model '{self._model_name}'"
                )
            self._dimension = int(dim)
        return self._dimension

    @property
    def model_name(self) -> str:
        """Identifier of the underlying embedding model."""
        return self._model_name

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_model(self) -> SentenceTransformer:
        """Return the loaded model, initializing on first call."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> SentenceTransformer:
        """Load the Sentence-Transformers model.

        Raises:
            RuntimeError: If the model cannot be loaded (e.g. missing
                dependency, invalid model name, no internet).
        """
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model '%s' ...", self._model_name)
            model = SentenceTransformer(self._model_name)
            logger.info(
                "Loaded embedding model '%s' (dimension=%s).",
                self._model_name,
                model.get_embedding_dimension(),
            )
            return model
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load embedding model '{self._model_name}': {exc}"
            ) from exc

    @staticmethod
    def _validate_texts(texts: list[str]) -> None:
        """Validate the input text list.

        Raises:
            ValueError: If validation fails.
        """
        if not isinstance(texts, list):
            raise ValueError(
                f"texts must be a list of strings, got {type(texts).__name__}"
            )
        if len(texts) == 0:
            raise ValueError("texts must be a non-empty list")
        for i, t in enumerate(texts):
            if not isinstance(t, str):
                raise ValueError(f"texts[{i}] must be a string, got {type(t).__name__}")
            if not t.strip():
                raise ValueError(f"texts[{i}] must be a non-empty string")
