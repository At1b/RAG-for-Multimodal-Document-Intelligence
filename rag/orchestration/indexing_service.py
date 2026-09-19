"""Document indexing orchestration service.

Connects the completed ingestion, chunking, and indexing modules
into a single application-level indexing operation.

Flow::

    file_path
        ↓
    IngestionService.ingest()  →  Document
        ↓
    chunk_document()           →  list[Chunk]
        ↓
    IndexingService.index_chunks()  →  stored in vector store

This service does NOT duplicate file parsing, text cleaning, chunking,
embedding, or vector-store logic.  It delegates entirely to existing
Phase 1–3 abstractions.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel, Field

from rag.chunking.chunker import ChunkingConfig, chunk_document
from rag.embeddings.indexing import IndexingService
from rag.ingestion.service import IngestionService
from rag.orchestration.exceptions import DocumentIndexingError

logger = logging.getLogger(__name__)


class IndexingResult(BaseModel):
    """Result of a document indexing operation.

    Attributes:
        document_id: The unique ID assigned to the indexed document.
        document_name: The user-facing filename.
        num_pages: Number of pages extracted from the document.
        num_chunks: Number of chunks stored in the vector store.
    """

    document_id: str = Field(..., min_length=1)
    document_name: str = Field(..., min_length=1)
    num_pages: int = Field(..., ge=0)
    num_chunks: int = Field(..., ge=0)


class DocumentIndexingService:
    """Orchestrates the full document indexing pipeline.

    Connects Phase 1 ingestion, Phase 2 chunking, and Phase 3
    indexing into a single ``index_document`` operation.

    Args:
        ingestion_service: Phase 1 ingestion service.
        chunking_config: Phase 2 chunking configuration.
        indexing_service: Phase 3 chunk → embed → store orchestrator.
    """

    def __init__(
        self,
        ingestion_service: IngestionService,
        chunking_config: ChunkingConfig,
        indexing_service: IndexingService,
    ) -> None:
        self._ingestion_service = ingestion_service
        self._chunking_config = chunking_config
        self._indexing_service = indexing_service

    def index_document(
        self,
        file_path: str | Path,
        document_name: str | None = None,
    ) -> IndexingResult:
        """Ingest, chunk, embed, and store a document.

        Args:
            file_path: Path to the document file.
            document_name: Optional user-facing filename.  Falls back
                to the filename from *file_path* when not provided.

        Returns:
            An ``IndexingResult`` with document metadata and chunk count.

        Raises:
            DocumentIndexingError: If any stage of the pipeline fails.
                The underlying cause is always preserved.
        """
        file_path = Path(file_path)
        display_name = document_name or file_path.name

        # Step 1: Ingest — validate, detect format, parse.
        try:
            document = self._ingestion_service.ingest(
                file_path, original_filename=display_name
            )
        except Exception as exc:
            raise DocumentIndexingError(
                f"Ingestion failed for '{display_name}': {exc}"
            ) from exc

        logger.info(
            "Ingested '%s' (id=%s, pages=%d).",
            document.document_name,
            document.document_id,
            len(document.pages),
        )

        # Step 2: Chunk — clean text, split into retrieval-ready chunks.
        try:
            chunks = chunk_document(document, config=self._chunking_config)
        except Exception as exc:
            raise DocumentIndexingError(
                f"Chunking failed for '{display_name}': {exc}"
            ) from exc

        if not chunks:
            raise DocumentIndexingError(
                f"Document '{display_name}' produced zero chunks after "
                f"cleaning and chunking — nothing to index."
            )

        logger.info(
            "Chunked '%s' into %d chunks.",
            document.document_name,
            len(chunks),
        )

        # Step 3: Index — embed and store in vector store.
        try:
            num_indexed = self._indexing_service.index_chunks(chunks)
        except Exception as exc:
            raise DocumentIndexingError(
                f"Indexing failed for '{display_name}': {exc}"
            ) from exc

        logger.info(
            "Indexed %d chunks for '%s'.",
            num_indexed,
            document.document_name,
        )

        return IndexingResult(
            document_id=document.document_id,
            document_name=document.document_name,
            num_pages=len(document.pages),
            num_chunks=num_indexed,
        )
