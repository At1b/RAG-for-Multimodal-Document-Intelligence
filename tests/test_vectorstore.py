"""Tests for the vector store (Phase 3).

Tests cover:
- Abstract interface contract
- ChromaVectorStore:
    - Insert and retrieve
    - Top-K results
    - Metadata preservation (all fields)
    - Persistence and reload from disk
    - Duplicate chunk IDs (upsert behavior)
    - Re-indexing (document-level delete + insert)
    - Empty store queries
    - Count
    - Reset
    - Deterministic collection naming
    - Input validation
"""

from __future__ import annotations

import tempfile

import pytest

from rag.chunking.models import Chunk
from rag.vectorstore.base import VectorStore
from rag.vectorstore.chroma_store import ChromaVectorStore
from rag.vectorstore.models import VectorSearchResult

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _make_chunk(
    chunk_id: str = "chunk-001",
    document_id: str = "doc-001",
    document_name: str = "test.pdf",
    source_type: str = "pdf",
    content: str = "Sample chunk content for testing.",
    page_number: int | None = 1,
    chunk_index: int = 0,
    metadata: dict | None = None,
) -> Chunk:
    """Create a test chunk with sensible defaults."""
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        document_name=document_name,
        source_type=source_type,
        content=content,
        page_number=page_number,
        chunk_index=chunk_index,
        metadata=metadata or {},
    )


def _make_embedding(dim: int = 384, value: float = 0.1) -> list[float]:
    """Create a simple embedding vector."""
    return [value] * dim


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory for vector store persistence.

    Uses ``ignore_cleanup_errors=True`` because ChromaDB's PersistentClient
    may hold file handles open on Windows, preventing immediate cleanup.
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        yield d


@pytest.fixture
def store(tmp_dir) -> ChromaVectorStore:
    """Create a fresh ChromaVectorStore for each test."""
    return ChromaVectorStore(
        persist_directory=tmp_dir,
        collection_name="test_collection",
    )


# ---------------------------------------------------------------
# Interface contract
# ---------------------------------------------------------------


class TestVectorStoreInterface:
    """Verify the abstract interface cannot be instantiated directly."""

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            VectorStore()  # type: ignore[abstract]


# ---------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------


class TestInitialization:
    """Test vector store initialization."""

    def test_creates_collection(self, store):
        assert store.count() == 0

    def test_empty_persist_dir_raises(self):
        with pytest.raises(ValueError, match="persist_directory"):
            ChromaVectorStore(persist_directory="", collection_name="test")

    def test_empty_collection_name_raises(self, tmp_dir):
        with pytest.raises(ValueError, match="collection_name"):
            ChromaVectorStore(persist_directory=tmp_dir, collection_name="")


# ---------------------------------------------------------------
# add_chunks
# ---------------------------------------------------------------


class TestAddChunks:
    """Test adding chunks to the vector store."""

    def test_add_single_chunk(self, store):
        chunk = _make_chunk()
        embedding = _make_embedding()
        store.add_chunks([chunk], [embedding])
        assert store.count() == 1

    def test_add_multiple_chunks(self, store):
        chunks = [_make_chunk(chunk_id=f"chunk-{i}", chunk_index=i) for i in range(5)]
        embeddings = [_make_embedding(value=0.1 * (i + 1)) for i in range(5)]
        store.add_chunks(chunks, embeddings)
        assert store.count() == 5

    def test_empty_chunks_raises(self, store):
        with pytest.raises(ValueError, match="non-empty"):
            store.add_chunks([], [])

    def test_mismatched_lengths_raises(self, store):
        chunks = [_make_chunk()]
        embeddings = [_make_embedding(), _make_embedding()]
        with pytest.raises(ValueError, match="same length"):
            store.add_chunks(chunks, embeddings)

    def test_upsert_duplicate_ids(self, store):
        """Duplicate chunk IDs should be overwritten (upsert)."""
        chunk_v1 = _make_chunk(content="version 1")
        chunk_v2 = _make_chunk(content="version 2")

        store.add_chunks([chunk_v1], [_make_embedding(value=0.1)])
        assert store.count() == 1

        store.add_chunks([chunk_v2], [_make_embedding(value=0.2)])
        assert store.count() == 1  # still 1, overwritten


# ---------------------------------------------------------------
# query
# ---------------------------------------------------------------


class TestQuery:
    """Test similarity search queries."""

    def test_query_empty_store(self, store):
        result = store.query(_make_embedding(), top_k=5)
        assert result == []

    def test_query_returns_results(self, store):
        chunk = _make_chunk()
        embedding = _make_embedding()
        store.add_chunks([chunk], [embedding])

        results = store.query(embedding, top_k=5)
        assert len(results) == 1
        assert isinstance(results[0], VectorSearchResult)

    def test_query_top_k_limits_results(self, store):
        chunks = [_make_chunk(chunk_id=f"chunk-{i}", chunk_index=i) for i in range(10)]
        embeddings = [_make_embedding(value=0.1 * (i + 1)) for i in range(10)]
        store.add_chunks(chunks, embeddings)

        results = store.query(_make_embedding(), top_k=3)
        assert len(results) <= 3

    def test_query_top_k_exceeds_store_size(self, store):
        chunk = _make_chunk()
        store.add_chunks([chunk], [_make_embedding()])

        results = store.query(_make_embedding(), top_k=100)
        assert len(results) == 1  # only 1 in store

    def test_query_returns_scores(self, store):
        chunk = _make_chunk()
        store.add_chunks([chunk], [_make_embedding()])

        results = store.query(_make_embedding(), top_k=5)
        assert results[0].score is not None
        assert isinstance(results[0].score, float)

    def test_query_similarity_ordering(self, store):
        """More similar vectors should have higher scores."""
        chunk_a = _make_chunk(chunk_id="a", content="cat", chunk_index=0)
        chunk_b = _make_chunk(chunk_id="b", content="dog", chunk_index=1)

        # Embed a as [1, 0, 0, ...] and b as [0, 1, 0, ...]
        dim = 384
        emb_a = [1.0] + [0.0] * (dim - 1)
        emb_b = [0.0, 1.0] + [0.0] * (dim - 2)

        store.add_chunks([chunk_a, chunk_b], [emb_a, emb_b])

        # Query with vector similar to a
        results = store.query(emb_a, top_k=2)
        assert results[0].chunk_id == "a"

    def test_query_invalid_empty_vector_raises(self, store):
        with pytest.raises(ValueError, match="non-empty"):
            store.query([], top_k=5)

    def test_query_invalid_top_k_raises(self, store):
        with pytest.raises(ValueError, match="top_k"):
            store.query(_make_embedding(), top_k=0)


# ---------------------------------------------------------------
# Metadata preservation
# ---------------------------------------------------------------


class TestMetadataPreservation:
    """Test that chunk metadata is preserved through storage and retrieval."""

    def test_core_metadata_preserved(self, store):
        chunk = _make_chunk(
            chunk_id="meta-chunk",
            document_id="meta-doc",
            document_name="report.pdf",
            source_type="pdf",
            page_number=42,
            chunk_index=7,
        )
        store.add_chunks([chunk], [_make_embedding()])

        results = store.query(_make_embedding(), top_k=1)
        assert len(results) == 1
        r = results[0]

        assert r.chunk_id == "meta-chunk"
        assert r.document_id == "meta-doc"
        assert r.document_name == "report.pdf"
        assert r.metadata["source_type"] == "pdf"
        assert r.metadata["page_number"] == 42
        assert r.metadata["chunk_index"] == 7

    def test_content_preserved(self, store):
        chunk = _make_chunk(content="This is the original content.")
        store.add_chunks([chunk], [_make_embedding()])

        results = store.query(_make_embedding(), top_k=1)
        assert results[0].content == "This is the original content."

    def test_custom_metadata_preserved(self, store):
        chunk = _make_chunk(metadata={"char_count": 42, "custom_key": "custom_value"})
        store.add_chunks([chunk], [_make_embedding()])

        results = store.query(_make_embedding(), top_k=1)
        assert results[0].metadata["char_count"] == 42
        assert results[0].metadata["custom_key"] == "custom_value"

    def test_page_number_none_handled(self, store):
        chunk = _make_chunk(page_number=None)
        store.add_chunks([chunk], [_make_embedding()])

        results = store.query(_make_embedding(), top_k=1)
        assert len(results) == 1
        # page_number should not be in metadata when None
        assert ("page_number" not in results[0].metadata) or (
            results[0].metadata.get("page_number") is None
        )


# ---------------------------------------------------------------
# delete_document (re-indexing)
# ---------------------------------------------------------------


class TestDeleteDocument:
    """Test document-level deletion for re-indexing."""

    def test_delete_removes_all_document_chunks(self, store):
        chunks = [
            _make_chunk(chunk_id=f"chunk-{i}", document_id="doc-A", chunk_index=i)
            for i in range(3)
        ]
        embeddings = [_make_embedding(value=0.1 * (i + 1)) for i in range(3)]
        store.add_chunks(chunks, embeddings)
        assert store.count() == 3

        deleted = store.delete_document("doc-A")
        assert deleted == 3
        assert store.count() == 0

    def test_delete_preserves_other_documents(self, store):
        chunk_a = _make_chunk(chunk_id="a", document_id="doc-A", chunk_index=0)
        chunk_b = _make_chunk(chunk_id="b", document_id="doc-B", chunk_index=0)

        store.add_chunks(
            [chunk_a, chunk_b],
            [_make_embedding(value=0.1), _make_embedding(value=0.2)],
        )
        assert store.count() == 2

        deleted = store.delete_document("doc-A")
        assert deleted == 1
        assert store.count() == 1

        # doc-B should remain
        results = store.query(_make_embedding(value=0.2), top_k=5)
        assert len(results) == 1
        assert results[0].document_id == "doc-B"

    def test_delete_nonexistent_document_returns_zero(self, store):
        deleted = store.delete_document("nonexistent")
        assert deleted == 0

    def test_delete_empty_id_raises(self, store):
        with pytest.raises(ValueError, match="non-empty"):
            store.delete_document("")

    def test_reindex_replaces_chunks(self, store):
        """Full re-indexing flow: delete old + insert new."""
        # Index version 1
        chunks_v1 = [
            _make_chunk(
                chunk_id=f"v1-{i}",
                document_id="doc-A",
                content=f"v1 content {i}",
                chunk_index=i,
            )
            for i in range(3)
        ]
        store.add_chunks(
            chunks_v1,
            [_make_embedding(value=0.1 * (i + 1)) for i in range(3)],
        )
        assert store.count() == 3

        # Re-index: delete old, insert new (2 chunks now)
        store.delete_document("doc-A")
        chunks_v2 = [
            _make_chunk(
                chunk_id=f"v2-{i}",
                document_id="doc-A",
                content=f"v2 content {i}",
                chunk_index=i,
            )
            for i in range(2)
        ]
        store.add_chunks(
            chunks_v2,
            [_make_embedding(value=0.5 + 0.1 * i) for i in range(2)],
        )
        assert store.count() == 2


# ---------------------------------------------------------------
# count
# ---------------------------------------------------------------


class TestCount:
    """Test count operation."""

    def test_empty_store_count(self, store):
        assert store.count() == 0

    def test_count_after_adds(self, store):
        chunks = [_make_chunk(chunk_id=f"c-{i}", chunk_index=i) for i in range(5)]
        store.add_chunks(chunks, [_make_embedding() for _ in range(5)])
        assert store.count() == 5

    def test_count_after_delete(self, store):
        chunks = [_make_chunk(chunk_id=f"c-{i}", chunk_index=i) for i in range(5)]
        store.add_chunks(chunks, [_make_embedding() for _ in range(5)])
        store.delete_document("doc-001")
        assert store.count() == 0


# ---------------------------------------------------------------
# reset
# ---------------------------------------------------------------


class TestReset:
    """Test reset operation."""

    def test_reset_clears_all(self, store):
        chunks = [_make_chunk(chunk_id=f"c-{i}", chunk_index=i) for i in range(5)]
        store.add_chunks(chunks, [_make_embedding() for _ in range(5)])
        assert store.count() == 5

        store.reset()
        assert store.count() == 0

    def test_reset_empty_store(self, store):
        store.reset()
        assert store.count() == 0

    def test_store_usable_after_reset(self, store):
        store.reset()
        chunk = _make_chunk()
        store.add_chunks([chunk], [_make_embedding()])
        assert store.count() == 1


# ---------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------


class TestPersistence:
    """Test that data survives store recreation (persistence)."""

    def test_data_persists_across_instances(self, tmp_dir):
        # Create store and add data
        store1 = ChromaVectorStore(
            persist_directory=tmp_dir,
            collection_name="persist_test",
        )
        chunk = _make_chunk(content="persistent content")
        store1.add_chunks([chunk], [_make_embedding()])
        assert store1.count() == 1

        # Create new store instance pointing to same directory
        store2 = ChromaVectorStore(
            persist_directory=tmp_dir,
            collection_name="persist_test",
        )
        assert store2.count() == 1

        # Verify data is queryable
        results = store2.query(_make_embedding(), top_k=1)
        assert len(results) == 1
        assert results[0].content == "persistent content"

    def test_different_collections_independent(self, tmp_dir):
        store_a = ChromaVectorStore(
            persist_directory=tmp_dir,
            collection_name="collection_a",
        )
        store_b = ChromaVectorStore(
            persist_directory=tmp_dir,
            collection_name="collection_b",
        )

        store_a.add_chunks([_make_chunk()], [_make_embedding()])
        assert store_a.count() == 1
        assert store_b.count() == 0


# ---------------------------------------------------------------
# Deterministic collection naming
# ---------------------------------------------------------------


class TestCollectionNaming:
    """Test deterministic collection naming."""

    def test_same_name_same_collection(self, tmp_dir):
        store1 = ChromaVectorStore(
            persist_directory=tmp_dir,
            collection_name="deterministic",
        )
        store1.add_chunks([_make_chunk()], [_make_embedding()])

        store2 = ChromaVectorStore(
            persist_directory=tmp_dir,
            collection_name="deterministic",
        )
        assert store2.count() == 1  # same collection, same data
