"""Tests for the indexing service (Phase 3).

Tests cover:
- Chunk → Embedding → Vector Store integration
- Re-indexing flow (document-level replacement)
- Batch processing
- Metadata round-trip
- Input validation
"""

from __future__ import annotations

import tempfile
from unittest.mock import patch

import pytest

from rag.chunking.models import Chunk
from rag.embeddings.indexing import IndexingService
from rag.embeddings.sentence_transformer import SentenceTransformerEmbeddingService
from rag.vectorstore.chroma_store import ChromaVectorStore

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _make_chunk(
    chunk_id: str = "chunk-001",
    document_id: str = "doc-001",
    document_name: str = "test.pdf",
    source_type: str = "pdf",
    content: str = "Machine learning is a subset of artificial intelligence.",
    page_number: int | None = 1,
    chunk_index: int = 0,
    metadata: dict | None = None,
) -> Chunk:
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


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------


@pytest.fixture(scope="module")
def embedding_service() -> SentenceTransformerEmbeddingService:
    """Shared embedding service (module-scoped to avoid reloading model)."""
    return SentenceTransformerEmbeddingService(
        model_name="all-MiniLM-L6-v2",
        batch_size=32,
    )


@pytest.fixture
def tmp_dir():
    """Temporary directory for vector store persistence.

    Uses ``ignore_cleanup_errors=True`` because ChromaDB's PersistentClient
    may hold file handles open on Windows, preventing immediate cleanup.
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        yield d


@pytest.fixture
def indexing_service(embedding_service, tmp_dir):
    """Create an IndexingService with a fresh vector store."""
    store = ChromaVectorStore(
        persist_directory=tmp_dir,
        collection_name="test_indexing",
    )
    return IndexingService(
        embedding_service=embedding_service,
        vector_store=store,
    ), store


# ---------------------------------------------------------------
# Basic indexing
# ---------------------------------------------------------------


class TestBasicIndexing:
    """Test basic Chunk → Embed → Store flow."""

    def test_index_single_chunk(self, indexing_service):
        svc, store = indexing_service
        chunk = _make_chunk()
        count = svc.index_chunks([chunk])
        assert count == 1
        assert store.count() == 1

    def test_index_multiple_chunks(self, indexing_service):
        svc, store = indexing_service
        chunks = [
            _make_chunk(
                chunk_id=f"chunk-{i}",
                content=f"Content for chunk number {i}.",
                chunk_index=i,
            )
            for i in range(5)
        ]
        count = svc.index_chunks(chunks)
        assert count == 5
        assert store.count() == 5

    def test_empty_chunks_raises(self, indexing_service):
        svc, _ = indexing_service
        with pytest.raises(ValueError, match="non-empty"):
            svc.index_chunks([])

    def test_non_list_chunks_raises(self, indexing_service):
        svc, _ = indexing_service
        with pytest.raises(ValueError, match="non-empty"):
            svc.index_chunks("not a list")  # type: ignore[arg-type]

    def test_non_chunk_in_chunks_raises(self, indexing_service):
        svc, _ = indexing_service
        with pytest.raises(ValueError, match="Chunk instance"):
            svc.index_chunks([_make_chunk(), "not a chunk"])  # type: ignore[list-item]


# ---------------------------------------------------------------
# Re-indexing
# ---------------------------------------------------------------


class TestReindexing:
    """Test document-level replacement during re-indexing."""

    def test_reindex_replaces_old_chunks(self, indexing_service):
        svc, store = indexing_service

        # Index version 1 (3 chunks)
        chunks_v1 = [
            _make_chunk(
                chunk_id=f"v1-{i}",
                document_id="doc-reindex",
                content=f"Version 1 content {i}.",
                chunk_index=i,
            )
            for i in range(3)
        ]
        svc.index_chunks(chunks_v1)
        assert store.count() == 3

        # Index version 2 (2 chunks) — should replace v1
        chunks_v2 = [
            _make_chunk(
                chunk_id=f"v2-{i}",
                document_id="doc-reindex",
                content=f"Version 2 content {i}.",
                chunk_index=i,
            )
            for i in range(2)
        ]
        svc.index_chunks(chunks_v2)
        assert store.count() == 2

    def test_reindex_preserves_other_documents(self, indexing_service):
        svc, store = indexing_service

        # Index doc-A
        chunks_a = [
            _make_chunk(
                chunk_id="a-0",
                document_id="doc-A",
                content="Document A content.",
                chunk_index=0,
            )
        ]
        svc.index_chunks(chunks_a)

        # Index doc-B
        chunks_b = [
            _make_chunk(
                chunk_id="b-0",
                document_id="doc-B",
                content="Document B content.",
                chunk_index=0,
            )
        ]
        svc.index_chunks(chunks_b)
        assert store.count() == 2

        # Re-index doc-A only
        chunks_a_v2 = [
            _make_chunk(
                chunk_id="a-v2-0",
                document_id="doc-A",
                content="Document A v2 content.",
                chunk_index=0,
            )
        ]
        svc.index_chunks(chunks_a_v2)

        # doc-A replaced, doc-B preserved
        assert store.count() == 2

    def test_index_without_reindex(self, indexing_service):
        svc, store = indexing_service

        chunk1 = _make_chunk(chunk_id="no-reindex-1", content="First.", chunk_index=0)
        svc.index_chunks([chunk1], reindex=False)
        assert store.count() == 1

        chunk2 = _make_chunk(chunk_id="no-reindex-2", content="Second.", chunk_index=1)
        svc.index_chunks([chunk2], reindex=False)
        # Without reindex, both should exist (though same doc_id)
        # Note: reindex=False skips delete, so both chunks remain
        assert store.count() == 2

    def test_reindex_embedding_failure_preserves_existing_data(self, indexing_service):
        """If embedding fails, old chunks in the vector store must NOT be deleted."""
        svc, store = indexing_service

        # Step 1: Successfully index initial document
        original_chunk = _make_chunk(
            chunk_id="orig-chunk",
            document_id="safe-doc",
            content="Original safe content.",
            chunk_index=0,
        )
        svc.index_chunks([original_chunk])
        assert store.count() == 1

        # Step 2: Attempt re-index with new chunk, but mock embedding failure
        new_chunk = _make_chunk(
            chunk_id="new-chunk",
            document_id="safe-doc",
            content="New content that will fail during embedding.",
            chunk_index=0,
        )

        with patch.object(
            svc._embedding_service,
            "embed_documents",
            side_effect=RuntimeError("Simulated embedding failure"),
        ):
            with pytest.raises(RuntimeError, match="Simulated embedding failure"):
                svc.index_chunks([new_chunk], reindex=True)

        # Step 3: Verify the original chunk was NOT deleted
        assert store.count() == 1
        query_vec = svc._embedding_service.embed_query("safe content")
        results = store.query(query_vec, top_k=1)
        assert len(results) == 1
        assert results[0].chunk_id == "orig-chunk"
        assert results[0].content == "Original safe content."


# ---------------------------------------------------------------
# Metadata round-trip
# ---------------------------------------------------------------


class TestMetadataRoundTrip:
    """Test that chunk metadata survives the full pipeline."""

    def test_metadata_preserved_through_pipeline(self, indexing_service):
        svc, store = indexing_service

        chunk = _make_chunk(
            chunk_id="meta-test",
            document_id="meta-doc",
            document_name="report.pdf",
            source_type="pdf",
            content="Revenue increased by 25% in fiscal year 2024.",
            page_number=42,
            chunk_index=3,
            metadata={"char_count": 46, "custom": "value"},
        )
        svc.index_chunks([chunk])

        # Query with embedded text
        query_vec = svc._embedding_service.embed_query("revenue growth fiscal year")
        results = store.query(query_vec, top_k=1)

        assert len(results) == 1
        r = results[0]
        assert r.chunk_id == "meta-test"
        assert r.document_id == "meta-doc"
        assert r.document_name == "report.pdf"
        assert r.content == "Revenue increased by 25% in fiscal year 2024."
        assert r.metadata["source_type"] == "pdf"
        assert r.metadata["page_number"] == 42
        assert r.metadata["chunk_index"] == 3
        assert r.metadata["char_count"] == 46
        assert r.metadata["custom"] == "value"

    def test_query_finds_semantically_similar(self, indexing_service):
        """Verify that semantic similarity actually works end-to-end."""
        svc, store = indexing_service

        chunks = [
            _make_chunk(
                chunk_id="finance",
                content=(
                    "The company reported strong quarterly"
                    " earnings with revenue growth of 15%."
                ),
                chunk_index=0,
            ),
            _make_chunk(
                chunk_id="biology",
                content="Mitochondria are the powerhouse of the cell and produce ATP.",
                chunk_index=1,
            ),
            _make_chunk(
                chunk_id="cooking",
                content="Preheat the oven to 350 degrees and bake for 25 minutes.",
                chunk_index=2,
            ),
        ]
        svc.index_chunks(chunks)

        query_vec = svc._embedding_service.embed_query(
            "How did the company's revenue perform?"
        )
        results = store.query(query_vec, top_k=1)

        assert len(results) == 1
        assert results[0].chunk_id == "finance"
