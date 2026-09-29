"""BM25 index builder — Phase 8.

Provides utilities to build and maintain a BM25 index that stays
consistent with the ChromaDB vector store.

Design:
    The BM25 index is an in-memory structure that is rebuilt from
    ChromaDB's persisted data.  This avoids introducing a second
    persistence layer while ensuring the sparse index stays
    consistent with the authoritative dense index.

    Trade-off: Rebuilding from ChromaDB adds startup latency
    proportional to the number of indexed chunks.  For the project
    scope (hundreds to low thousands of chunks), this is acceptable.
    If the corpus grows significantly, a dedicated persistence layer
    for the sparse index would be warranted.
"""

from __future__ import annotations

import logging

from rag.retrieval.bm25 import BM25Index
from rag.vectorstore.chroma_store import ChromaVectorStore
from rag.vectorstore.models import VectorSearchResult

logger = logging.getLogger(__name__)


def build_bm25_index_from_chroma(
    vector_store: ChromaVectorStore,
    *,
    k1: float = 1.5,
    b: float = 0.75,
    batch_size: int = 500,
    stopwords: frozenset[str] | set[str] | None = None,
) -> BM25Index:
    """Build a BM25 index from all chunks stored in ChromaDB.

    Reads all stored documents and metadata from ChromaDB in batches and
    constructs a ``BM25Index`` over them.

    Args:
        vector_store: The ChromaDB vector store to read from.
        k1: BM25 term frequency saturation parameter.
        b: BM25 length normalization parameter.
        batch_size: Number of entries to fetch per batch from ChromaDB.
        stopwords: Optional custom stopwords set. If not specified (None),
            the default English stop words are used by BM25Index.

    Returns:
        A built ``BM25Index`` ready for search.

    Raises:
        TypeError: If vector_store is not a ChromaVectorStore instance.
        ValueError: If batch_size < 1.
    """
    if not isinstance(vector_store, ChromaVectorStore):
        raise TypeError(
            f"vector_store must be a ChromaVectorStore instance, "
            f"got {type(vector_store).__name__}"
        )
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size < 1
    ):
        raise ValueError(f"batch_size must be a positive integer, got {batch_size}")

    kwargs: dict = {"k1": k1, "b": b}
    if stopwords is not None:
        kwargs["stopwords"] = stopwords

    total = vector_store.count()
    if total == 0:
        logger.info("ChromaDB is empty; creating empty BM25 index.")
        index = BM25Index(**kwargs)
        index.build([])
        return index

    # Fetch all entries from ChromaDB in batches.
    entries: list[VectorSearchResult] = []
    collection = vector_store._collection  # noqa: SLF001 — internal access justified

    offset = 0
    while offset < total:
        limit = min(batch_size, total - offset)
        result = collection.get(
            include=["documents", "metadatas"],
            limit=limit,
            offset=offset,
        )

        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []

        for i, chunk_id in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
            doc_text = (
                documents[i] if i < len(documents) and documents[i] is not None else ""
            )

            entries.append(
                VectorSearchResult(
                    chunk_id=chunk_id,
                    document_id=str(meta.get("document_id") or "unknown_doc"),
                    document_name=str(meta.get("document_name") or "unknown_file"),
                    content=doc_text,
                    score=0.0,
                    metadata=meta.copy(),
                )
            )

        offset += len(ids)
        if len(ids) == 0:
            # Safety break if no more records returned
            break

    logger.info(
        "Loaded %d entries from ChromaDB for BM25 index construction.",
        len(entries),
    )

    index = BM25Index(**kwargs)
    index.build(entries)
    return index
