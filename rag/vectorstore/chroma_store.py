"""ChromaDB vector store implementation.

Concrete implementation of ``VectorStore`` backed by
`ChromaDB <https://docs.trychroma.com/>`_.

Features:
    - Local persistence (directory-based).
    - Native metadata storage per embedding.
    - Deterministic collection naming.
    - Document-level deletion for re-indexing.
    - Cosine similarity search.
"""

from __future__ import annotations

import logging

import chromadb

from rag.chunking.models import Chunk
from rag.vectorstore.base import VectorStore
from rag.vectorstore.models import VectorSearchResult

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION_NAME = "mmrag_chunks"
DEFAULT_PERSIST_DIR = "data/vectorstore"


class ChromaVectorStore(VectorStore):
    """Vector store backed by ChromaDB.

    Args:
        persist_directory: Filesystem path for ChromaDB persistence.
            Defaults to ``data/vectorstore``.
        collection_name: Name of the ChromaDB collection.
            Defaults to ``mmrag_chunks``.
    """

    def __init__(
        self,
        persist_directory: str = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        if not persist_directory or not persist_directory.strip():
            raise ValueError("persist_directory must be a non-empty string")
        if not collection_name or not collection_name.strip():
            raise ValueError("collection_name must be a non-empty string")

        self._persist_directory = persist_directory.strip()
        self._collection_name = collection_name.strip()

        self._client = chromadb.PersistentClient(path=self._persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB initialized: collection='%s', persist_dir='%s', "
            "existing_count=%d.",
            self._collection_name,
            self._persist_directory,
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Store chunk embeddings with their metadata.

        Uses ``upsert`` so that duplicate chunk IDs are overwritten
        rather than causing errors.

        Args:
            chunks: List of ``Chunk`` objects to store.
            embeddings: Corresponding embedding vectors.

        Raises:
            ValueError: If inputs are invalid.
        """
        if not chunks:
            raise ValueError("chunks must be a non-empty list")
        if not embeddings:
            raise ValueError("embeddings must be a non-empty list")
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                f"must have the same length"
            )

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for chunk in chunks:
            ids.append(chunk.chunk_id)
            documents.append(chunk.content)
            metadatas.append(self._chunk_to_metadata(chunk))

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info(
            "Upserted %d chunks into collection '%s'.",
            len(chunks),
            self._collection_name,
        )

    def query(
        self,
        query_vector: list[float],
        top_k: int = 10,
    ) -> list[VectorSearchResult]:
        """Find the most similar stored vectors.

        Args:
            query_vector: The embedding vector to search against.
            top_k: Maximum number of results to return.

        Returns:
            List of ``VectorSearchResult`` ordered by similarity.
        """
        if not query_vector:
            raise ValueError("query_vector must be a non-empty list")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        # Clamp top_k to available count to avoid ChromaDB errors.
        available = self._collection.count()
        if available == 0:
            return []
        effective_k = min(top_k, available)

        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=effective_k,
            include=["documents", "metadatas", "distances"],
        )

        search_results: list[VectorSearchResult] = []

        # ChromaDB returns lists-of-lists (one per query).
        result_ids = results.get("ids", [[]])[0]
        result_docs = results.get("documents", [[]])[0]
        result_metas = results.get("metadatas", [[]])[0]
        result_dists = results.get("distances", [[]])[0]

        for i, chunk_id in enumerate(result_ids):
            meta = result_metas[i] if i < len(result_metas) else {}
            doc_text = result_docs[i] if i < len(result_docs) else ""
            distance = result_dists[i] if i < len(result_dists) else 0.0

            # ChromaDB returns cosine *distance* (lower = more similar).
            # Convert to similarity score (higher = more similar).
            similarity = 1.0 - distance

            page_number = meta.get("page_number")
            if page_number is not None:
                page_number = int(page_number)

            search_results.append(
                VectorSearchResult(
                    chunk_id=chunk_id,
                    document_id=meta.get("document_id", ""),
                    document_name=meta.get("document_name", ""),
                    content=doc_text,
                    score=similarity,
                    metadata=meta,
                )
            )

        return search_results

    def delete_document(self, document_id: str) -> int:
        """Delete all stored vectors for a given document.

        Args:
            document_id: The document whose chunks should be removed.

        Returns:
            Number of vectors deleted.
        """
        if not document_id or not document_id.strip():
            raise ValueError("document_id must be a non-empty string")

        # Count existing chunks for this document before deletion.
        existing = self._collection.get(
            where={"document_id": document_id},
            include=[],
        )
        count_before = len(existing["ids"]) if existing["ids"] else 0

        if count_before > 0:
            self._collection.delete(where={"document_id": document_id})
            logger.info(
                "Deleted %d chunks for document_id='%s' from collection '%s'.",
                count_before,
                document_id,
                self._collection_name,
            )

        return count_before

    def count(self) -> int:
        """Return the total number of stored vectors."""
        return self._collection.count()

    def reset(self) -> None:
        """Remove all stored vectors by deleting and recreating the collection."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Reset collection '%s'.",
            self._collection_name,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_to_metadata(chunk: Chunk) -> dict:
        """Convert a Chunk's metadata fields into a flat dict for ChromaDB.

        ChromaDB metadata values must be str, int, float, or bool.
        Complex nested values are excluded.
        """
        meta: dict = {
            "document_id": chunk.document_id,
            "document_name": chunk.document_name,
            "source_type": chunk.source_type,
            "chunk_index": chunk.chunk_index,
        }

        if chunk.page_number is not None:
            meta["page_number"] = chunk.page_number

        # Carry forward flat chunk metadata values that ChromaDB can store.
        for key, value in chunk.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                # Avoid overwriting core fields.
                if key not in meta:
                    meta[key] = value

        return meta
