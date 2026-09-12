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
import math

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
        if not isinstance(persist_directory, str) or not persist_directory.strip():
            raise ValueError("persist_directory must be a non-empty string")
        if not isinstance(collection_name, str) or not collection_name.strip():
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
        if not isinstance(chunks, list) or len(chunks) == 0:
            raise ValueError("chunks must be a non-empty list of Chunk objects")
        if not isinstance(embeddings, list) or len(embeddings) == 0:
            raise ValueError("embeddings must be a non-empty list of vectors")
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                f"must have the same length"
            )

        # Validate elements and check embedding dimension consistency.
        first_dim = len(embeddings[0]) if isinstance(embeddings[0], list) else 0
        if first_dim == 0:
            raise ValueError("embedding vectors must be non-empty lists of floats")

        for i, (chunk, emb) in enumerate(zip(chunks, embeddings, strict=True)):
            if not isinstance(chunk, Chunk):
                raise ValueError(
                    f"chunks[{i}] must be a Chunk instance, got {type(chunk).__name__}"
                )
            if not isinstance(emb, list) or len(emb) == 0:
                raise ValueError(
                    f"embeddings[{i}] must be a non-empty list of floats, "
                    f"got {type(emb).__name__}"
                )
            if len(emb) != first_dim:
                raise ValueError(
                    f"embedding dimension mismatch: embeddings[{i}] has "
                    f"dimension {len(emb)}, expected {first_dim}"
                )
            if not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in emb
            ):
                raise ValueError(f"embeddings[{i}] contains non-numeric values")

        # Handle in-batch duplicate chunk IDs (upsert semantics: latest wins).
        seen_indices: dict[str, int] = {}
        for i, chunk in enumerate(chunks):
            seen_indices[chunk.chunk_id] = i

        if len(seen_indices) < len(chunks):
            logger.warning(
                "Found %d duplicate chunk IDs in batch of %d; "
                "latest occurrences will overwrite earlier ones.",
                len(chunks) - len(seen_indices),
                len(chunks),
            )
            keep_indices = set(seen_indices.values())
            chunks = [c for idx, c in enumerate(chunks) if idx in keep_indices]
            embeddings = [e for idx, e in enumerate(embeddings) if idx in keep_indices]

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
        if not isinstance(query_vector, list) or len(query_vector) == 0:
            raise ValueError("query_vector must be a non-empty list of floats")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError(f"top_k must be an integer >= 1, got {top_k}")
        if not all(
            isinstance(v, (int, float)) and not isinstance(v, bool)
            for v in query_vector
        ):
            raise ValueError("query_vector must contain only numeric values")

        # Clamp top_k to available count to avoid ChromaDB errors.
        available = self._collection.count()
        if available == 0:
            return []
        effective_k = min(top_k, available)

        try:
            results = self._collection.query(
                query_embeddings=[query_vector],
                n_results=effective_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise ValueError(f"Vector search failed: {exc}") from exc

        search_results: list[VectorSearchResult] = []

        # ChromaDB returns lists-of-lists (one per query).
        result_ids = (results.get("ids") or [[]])[0]
        result_docs = (results.get("documents") or [[]])[0]
        result_metas = (results.get("metadatas") or [[]])[0]
        result_dists = (results.get("distances") or [[]])[0]

        for i, chunk_id in enumerate(result_ids):
            meta = result_metas[i] if i < len(result_metas) and result_metas[i] else {}
            doc_text = (
                result_docs[i]
                if i < len(result_docs) and result_docs[i] is not None
                else ""
            )
            distance = (
                result_dists[i]
                if i < len(result_dists) and result_dists[i] is not None
                else 0.0
            )

            # ChromaDB returns cosine *distance* (lower = more similar).
            # Convert to similarity score (higher = more similar).
            raw_similarity = 1.0 - distance
            similarity = max(-1.0, min(1.0, raw_similarity))

            search_results.append(
                VectorSearchResult(
                    chunk_id=chunk_id,
                    document_id=str(meta.get("document_id") or "unknown_doc"),
                    document_name=str(meta.get("document_name") or "unknown_file"),
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
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("document_id must be a non-empty string")

        # Count existing chunks for this document before deletion.
        existing = self._collection.get(
            where={"document_id": document_id.strip()},
            include=[],
        )
        existing_ids = existing.get("ids") or []
        count_before = len(existing_ids)

        if count_before > 0:
            self._collection.delete(ids=existing_ids)
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
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:
            pass
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
        Complex nested values and non-finite floats (NaN, Inf) are excluded.
        """
        meta: dict = {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "document_name": chunk.document_name,
            "source_type": chunk.source_type,
            "chunk_index": chunk.chunk_index,
        }

        if chunk.page_number is not None:
            meta["page_number"] = chunk.page_number

        # Carry forward flat chunk metadata values that ChromaDB can store.
        for key, value in chunk.metadata.items():
            if isinstance(value, (str, int, bool)):
                # Avoid overwriting core fields.
                if key not in meta:
                    meta[key] = value
            elif isinstance(value, float):
                if not math.isnan(value) and not math.isinf(value):
                    if key not in meta:
                        meta[key] = value

        return meta
