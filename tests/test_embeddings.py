"""Tests for the embedding service (Phase 3).

Tests cover:
- Abstract interface contract
- SentenceTransformerEmbeddingService:
    - Model initialization
    - Single text embedding
    - Multiple texts (batch)
    - Query embedding
    - Embedding dimension consistency
    - Empty input handling
    - Unicode text handling
    - Invalid input handling
    - Model name property
    - Lazy loading
"""

from __future__ import annotations

import pytest

from rag.embeddings.base import EmbeddingService
from rag.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingService,
)

# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------


@pytest.fixture(scope="module")
def embedding_service() -> SentenceTransformerEmbeddingService:
    """Shared embedding service for the module.

    Uses module scope so the model is loaded only once across all tests
    (loading takes a few seconds).
    """
    return SentenceTransformerEmbeddingService(
        model_name="all-MiniLM-L6-v2",
        batch_size=32,
    )


# ---------------------------------------------------------------
# Interface contract
# ---------------------------------------------------------------


class TestEmbeddingServiceInterface:
    """Verify the abstract interface cannot be instantiated directly."""

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            EmbeddingService()  # type: ignore[abstract]


# ---------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------


class TestInitialization:
    """Test embedding service initialization."""

    def test_default_model_name(self):
        svc = SentenceTransformerEmbeddingService()
        assert svc.model_name == "all-MiniLM-L6-v2"

    def test_custom_model_name(self):
        svc = SentenceTransformerEmbeddingService(model_name="all-MiniLM-L6-v2")
        assert svc.model_name == "all-MiniLM-L6-v2"

    def test_empty_model_name_raises(self):
        with pytest.raises(ValueError, match="model_name"):
            SentenceTransformerEmbeddingService(model_name="")

    def test_whitespace_model_name_raises(self):
        with pytest.raises(ValueError, match="model_name"):
            SentenceTransformerEmbeddingService(model_name="   ")

    def test_invalid_batch_size_raises(self):
        with pytest.raises(ValueError, match="batch_size"):
            SentenceTransformerEmbeddingService(batch_size=0)

    def test_negative_batch_size_raises(self):
        with pytest.raises(ValueError, match="batch_size"):
            SentenceTransformerEmbeddingService(batch_size=-1)

    def test_lazy_loading(self):
        """Model should NOT be loaded until first use."""
        svc = SentenceTransformerEmbeddingService()
        assert svc._model is None


# ---------------------------------------------------------------
# Embedding dimension
# ---------------------------------------------------------------


class TestDimension:
    """Test embedding dimension property."""

    def test_dimension_is_384(self, embedding_service):
        assert embedding_service.dimension == 384

    def test_dimension_consistent(self, embedding_service):
        """Dimension property should return the same value on repeated calls."""
        d1 = embedding_service.dimension
        d2 = embedding_service.dimension
        assert d1 == d2 == 384


# ---------------------------------------------------------------
# embed_documents
# ---------------------------------------------------------------


class TestEmbedDocuments:
    """Test batch document embedding."""

    def test_single_text(self, embedding_service):
        result = embedding_service.embed_documents(["Hello world"])
        assert len(result) == 1
        assert len(result[0]) == 384

    def test_multiple_texts(self, embedding_service):
        texts = ["Hello", "World", "Test embedding"]
        result = embedding_service.embed_documents(texts)
        assert len(result) == 3
        for vec in result:
            assert len(vec) == 384

    def test_all_floats(self, embedding_service):
        result = embedding_service.embed_documents(["test"])
        assert all(isinstance(v, float) for v in result[0])

    def test_different_texts_produce_different_embeddings(self, embedding_service):
        result = embedding_service.embed_documents(["cat", "quantum physics"])
        assert result[0] != result[1]

    def test_unicode_text(self, embedding_service):
        texts = ["こんにちは世界", "مرحبا بالعالم", "Ελληνικά", "🎉🎊"]
        result = embedding_service.embed_documents(texts)
        assert len(result) == 4
        for vec in result:
            assert len(vec) == 384

    def test_long_text(self, embedding_service):
        long_text = "word " * 1000
        result = embedding_service.embed_documents([long_text])
        assert len(result) == 1
        assert len(result[0]) == 384

    def test_empty_list_raises(self, embedding_service):
        with pytest.raises(ValueError, match="non-empty"):
            embedding_service.embed_documents([])

    def test_non_list_raises(self, embedding_service):
        with pytest.raises(ValueError, match="list"):
            embedding_service.embed_documents("not a list")  # type: ignore[arg-type]

    def test_non_string_in_list_raises(self, embedding_service):
        with pytest.raises(ValueError, match="string"):
            embedding_service.embed_documents(["valid", 123])  # type: ignore[list-item]

    def test_batch_larger_than_batch_size(self):
        """Verify that batches larger than batch_size are handled correctly."""
        svc = SentenceTransformerEmbeddingService(batch_size=2)
        texts = [f"text {i}" for i in range(5)]
        result = svc.embed_documents(texts)
        assert len(result) == 5


# ---------------------------------------------------------------
# embed_query
# ---------------------------------------------------------------


class TestEmbedQuery:
    """Test single query embedding."""

    def test_returns_vector(self, embedding_service):
        result = embedding_service.embed_query("What is machine learning?")
        assert isinstance(result, list)
        assert len(result) == 384

    def test_all_floats(self, embedding_service):
        result = embedding_service.embed_query("test query")
        assert all(isinstance(v, float) for v in result)

    def test_unicode_query(self, embedding_service):
        result = embedding_service.embed_query("日本語のクエリ")
        assert len(result) == 384

    def test_empty_string_raises(self, embedding_service):
        with pytest.raises(ValueError, match="non-empty"):
            embedding_service.embed_query("")

    def test_whitespace_only_raises(self, embedding_service):
        with pytest.raises(ValueError, match="non-empty"):
            embedding_service.embed_query("   ")

    def test_non_string_raises(self, embedding_service):
        with pytest.raises(ValueError, match="string"):
            embedding_service.embed_query(123)  # type: ignore[arg-type]

    def test_none_raises(self, embedding_service):
        with pytest.raises(ValueError, match="string"):
            embedding_service.embed_query(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------
# Model name
# ---------------------------------------------------------------


class TestModelName:
    """Test model_name property."""

    def test_returns_configured_name(self, embedding_service):
        assert embedding_service.model_name == "all-MiniLM-L6-v2"

    def test_model_name_stripped(self):
        svc = SentenceTransformerEmbeddingService(model_name="  all-MiniLM-L6-v2  ")
        assert svc.model_name == "all-MiniLM-L6-v2"


# ---------------------------------------------------------------
# Invalid model
# ---------------------------------------------------------------


class TestInvalidModel:
    """Test behavior with an invalid model name."""

    def test_invalid_model_raises_on_use(self):
        svc = SentenceTransformerEmbeddingService(
            model_name="nonexistent-model-xyz-12345"
        )
        with pytest.raises(RuntimeError, match="Failed to load"):
            svc.embed_query("test")
