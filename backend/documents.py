"""Document upload API router.

Thin route handler: accepts upload, delegates to DocumentIndexingService,
maps exceptions to HTTP responses.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from backend.config import get_settings
from rag.chunking.chunker import ChunkingConfig
from rag.embeddings.indexing import IndexingService
from rag.embeddings.sentence_transformer import SentenceTransformerEmbeddingService
from rag.ingestion.exceptions import (
    EmptyFileError,
    FileTooLargeError,
    InvalidDocumentError,
    UnsupportedFormatError,
)
from rag.ingestion.service import IngestionService
from rag.orchestration.exceptions import DocumentIndexingError
from rag.orchestration.indexing_service import DocumentIndexingService
from rag.vectorstore.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

# Mapping from ingestion exception types to HTTP status codes.
_INGESTION_ERROR_STATUS: dict[type, int] = {
    EmptyFileError: 400,
    FileTooLargeError: 413,
    UnsupportedFormatError: 415,
    InvalidDocumentError: 422,
}


def _resolve_indexing_error(exc: DocumentIndexingError) -> HTTPException:
    """Map a DocumentIndexingError to an appropriate HTTPException.

    The DocumentIndexingService wraps all pipeline failures in
    DocumentIndexingError.  This helper inspects the cause chain to
    find the original ingestion exception and maps it to the correct
    HTTP status code.
    """
    cause = exc.__cause__
    if cause is not None:
        for error_type, status_code in _INGESTION_ERROR_STATUS.items():
            if isinstance(cause, error_type):
                return HTTPException(status_code=status_code, detail=str(cause))

    # Generic indexing failure (chunking, embedding, or storage error).
    logger.error("Document indexing failed: %s", exc, exc_info=True)
    return HTTPException(status_code=500, detail=str(exc))


@router.post("/upload")
async def upload_document(file: UploadFile):
    """Upload a document for ingestion and indexing.

    Accepts PDF and DOCX files.  The document is ingested, chunked,
    embedded, and indexed into the vector store in a single operation.
    Returns metadata about the indexed document.
    """
    if not file.filename or not file.filename.strip():
        raise HTTPException(status_code=400, detail="Filename is required.")

    settings = get_settings()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024

    # Write upload to a temp file so the ingestion pipeline can work
    # with a regular filesystem path (required by PyMuPDF / python-docx).
    suffix = Path(file.filename).suffix
    tmp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = Path(tmp.name)
            chunk_size = 1024 * 1024  # 1 MB chunks
            total_read = 0
            while chunk := await file.read(chunk_size):
                total_read += len(chunk)
                if total_read > max_bytes:
                    raise FileTooLargeError(
                        f"File '{file.filename}' exceeds the "
                        f"{settings.max_upload_size_mb} MB limit."
                    )
                tmp.write(chunk)

        # Construct the full indexing pipeline from settings.
        ingestion_service = IngestionService(max_size_bytes=max_bytes)
        chunking_config = ChunkingConfig(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        embedding_service = SentenceTransformerEmbeddingService(
            model_name=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
        )
        vector_store = ChromaVectorStore(
            persist_directory=settings.vector_store_path,
            collection_name=settings.vector_store_collection,
        )
        indexing_service = IndexingService(
            embedding_service=embedding_service,
            vector_store=vector_store,
        )
        doc_indexing_service = DocumentIndexingService(
            ingestion_service=ingestion_service,
            chunking_config=chunking_config,
            indexing_service=indexing_service,
        )

        result = doc_indexing_service.index_document(
            file_path=tmp_path,
            document_name=file.filename,
        )

        return {
            "document_id": result.document_id,
            "document_name": result.document_name,
            "num_pages": result.num_pages,
            "num_chunks": result.num_chunks,
        }

    except FileTooLargeError as exc:
        # FileTooLargeError raised during streaming (before pipeline).
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except DocumentIndexingError as exc:
        # Unwrap the cause to map to specific HTTP status codes.
        raise _resolve_indexing_error(exc) from exc
    finally:
        # Always clean up the temp file.
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except PermissionError:
                import gc

                gc.collect()
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
