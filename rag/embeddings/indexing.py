"""Indexing service — Chunk → Embed → Store orchestrator.

Connects the embedding service and vector store to provide a single
entry point for indexing Phase 2 chunks.

Re-indexing strategy:
    **Document-level replacement** — when indexing chunks for a document
    that is already in the store, all existing chunks for that
    ``document_id`` are deleted before the new chunks are inserted.
    This prevents stale chunks from remaining after re-processing.
"""

from __future__ import annotations

import logging

from rag.chunking.models import Chunk
from rag.embeddings.base import EmbeddingService
from rag.vectorstore.base import VectorStore

logger = logging.getLogger(__name__)


class IndexingService:
    """Orchestrates the Chunk → Embedding → Vector Store pipeline.

    Args:
        embedding_service: The embedding service to use for vectorization.
        vector_store: The vector store for persistence.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store

    def index_chunks(
        self,
        chunks: list[Chunk],
        *,
        reindex: bool = True,
    ) -> int:
        """Embed and store a list of chunks.

        Args:
            chunks: The Phase 2 chunks to index.
            reindex: If ``True`` (default), delete existing chunks for
                each document represented in *chunks* before inserting
                the new ones.

        Returns:
            Number of chunks indexed.

        Raises:
            ValueError: If *chunks* is empty.
            RuntimeError: If embedding or storage fails.
        """
        if not chunks:
            raise ValueError("chunks must be a non-empty list")

        # Re-indexing: delete existing chunks per document.
        if reindex:
            doc_ids = {c.document_id for c in chunks}
            for doc_id in doc_ids:
                deleted = self._vector_store.delete_document(doc_id)
                if deleted > 0:
                    logger.info(
                        "Re-indexing: removed %d existing chunks for document_id='%s'.",
                        deleted,
                        doc_id,
                    )

        # Generate embeddings.
        texts = [c.content for c in chunks]
        logger.info("Generating embeddings for %d chunks ...", len(texts))
        embeddings = self._embedding_service.embed_documents(texts)

        # Store in vector store.
        self._vector_store.add_chunks(chunks, embeddings)
        logger.info("Indexed %d chunks.", len(chunks))

        return len(chunks)
