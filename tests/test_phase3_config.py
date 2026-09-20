"""Tests for Phase 3 configuration validation (backend/config.py)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.config import Settings


class TestPhase3ConfigDefaults:
    """Verify Phase 3 configuration defaults."""

    def test_phase3_defaults(self):
        s = Settings()
        assert s.embedding_model == "all-MiniLM-L6-v2"
        assert s.embedding_batch_size == 64
        assert s.vector_store_path == "data/vectorstore"
        assert s.vector_store_collection == "mmrag_chunks"
        assert s.vector_search_top_k == 10
        assert s.retrieval_min_score == 0.3

    def test_custom_phase3_settings(self):
        s = Settings(
            embedding_model="custom-model",
            embedding_batch_size=32,
            vector_store_path="custom/path",
            vector_store_collection="custom_collection",
            vector_search_top_k=20,
            retrieval_min_score=0.45,
        )
        assert s.embedding_model == "custom-model"
        assert s.embedding_batch_size == 32
        assert s.vector_store_path == "custom/path"
        assert s.vector_store_collection == "custom_collection"
        assert s.vector_search_top_k == 20
        assert s.retrieval_min_score == 0.45


class TestEmbeddingModelValidation:
    """Verify validation on embedding_model."""

    def test_empty_model_raises(self):
        with pytest.raises(ValidationError, match="embedding_model"):
            Settings(embedding_model="")

    def test_whitespace_model_raises(self):
        with pytest.raises(ValidationError, match="embedding_model"):
            Settings(embedding_model="   ")

    def test_model_name_stripped(self):
        s = Settings(embedding_model="  bge-small-en-v1.5  ")
        assert s.embedding_model == "bge-small-en-v1.5"


class TestEmbeddingBatchSizeValidation:
    """Verify validation on embedding_batch_size."""

    def test_zero_batch_size_raises(self):
        with pytest.raises(ValidationError, match="embedding_batch_size"):
            Settings(embedding_batch_size=0)

    def test_negative_batch_size_raises(self):
        with pytest.raises(ValidationError, match="embedding_batch_size"):
            Settings(embedding_batch_size=-10)


class TestVectorStorePathValidation:
    """Verify validation on vector_store_path."""

    def test_empty_path_raises(self):
        with pytest.raises(ValidationError, match="vector_store_path"):
            Settings(vector_store_path="")

    def test_whitespace_path_raises(self):
        with pytest.raises(ValidationError, match="vector_store_path"):
            Settings(vector_store_path="   ")

    def test_path_stripped(self):
        s = Settings(vector_store_path="  data/custom  ")
        assert s.vector_store_path == "data/custom"


class TestVectorStoreCollectionValidation:
    """Verify validation on vector_store_collection."""

    def test_empty_collection_raises(self):
        with pytest.raises(ValidationError, match="vector_store_collection"):
            Settings(vector_store_collection="")

    def test_whitespace_collection_raises(self):
        with pytest.raises(ValidationError, match="vector_store_collection"):
            Settings(vector_store_collection="   ")

    def test_too_short_collection_raises(self):
        with pytest.raises(ValidationError, match="between 3 and 512"):
            Settings(vector_store_collection="ab")

    def test_starts_with_underscore_raises(self):
        with pytest.raises(ValidationError, match="alphanumeric"):
            Settings(vector_store_collection="_collection")

    def test_ends_with_dash_raises(self):
        with pytest.raises(ValidationError, match="alphanumeric"):
            Settings(vector_store_collection="collection-")

    def test_valid_collection_names(self):
        s1 = Settings(vector_store_collection="coll.name_1-test")
        assert s1.vector_store_collection == "coll.name_1-test"


class TestVectorSearchTopKValidation:
    """Verify validation on vector_search_top_k."""

    def test_zero_top_k_raises(self):
        with pytest.raises(ValidationError, match="vector_search_top_k"):
            Settings(vector_search_top_k=0)

    def test_negative_top_k_raises(self):
        with pytest.raises(ValidationError, match="vector_search_top_k"):
            Settings(vector_search_top_k=-5)

    def test_exceeds_max_top_k_raises(self):
        with pytest.raises(ValidationError, match="vector_search_top_k"):
            Settings(vector_search_top_k=1001)

    def test_at_max_top_k_valid(self):
        s = Settings(vector_search_top_k=1000)
        assert s.vector_search_top_k == 1000


class TestRetrievalMinScoreValidation:
    """Verify validation on retrieval_min_score."""

    def test_default_min_score(self):
        s = Settings()
        assert s.retrieval_min_score == 0.3

    def test_valid_min_score_float(self):
        s = Settings(retrieval_min_score=0.75)
        assert s.retrieval_min_score == 0.75

    def test_boundary_min_score_negative_one(self):
        s = Settings(retrieval_min_score=-1.0)
        assert s.retrieval_min_score == -1.0

    def test_boundary_min_score_positive_one(self):
        s = Settings(retrieval_min_score=1.0)
        assert s.retrieval_min_score == 1.0

    def test_below_minimum_raises(self):
        with pytest.raises(ValidationError, match="retrieval_min_score"):
            Settings(retrieval_min_score=-1.1)

    def test_above_maximum_raises(self):
        with pytest.raises(ValidationError, match="retrieval_min_score"):
            Settings(retrieval_min_score=1.1)

    def test_bool_rejected(self):
        with pytest.raises(ValidationError, match="retrieval_min_score"):
            Settings(retrieval_min_score=True)

    def test_non_numeric_rejected(self):
        with pytest.raises(ValidationError, match="retrieval_min_score"):
            Settings(retrieval_min_score="not-a-number")
