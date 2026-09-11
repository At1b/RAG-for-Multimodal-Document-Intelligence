"""Document upload API router.

Thin route handler: accepts upload, delegates to IngestionService,
maps exceptions to HTTP responses.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from backend.config import get_settings
from rag.ingestion.exceptions import (
    EmptyFileError,
    FileNotFoundError,
    FileTooLargeError,
    IngestionError,
    InvalidDocumentError,
    UnsupportedFormatError,
)
from rag.ingestion.service import IngestionService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload")
async def upload_document(file: UploadFile):
    """Upload a document for ingestion.

    Accepts PDF and DOCX files.  Returns normalized metadata about
    the processed document.
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

        service = IngestionService(max_size_bytes=max_bytes)
        document = service.ingest(tmp_path, original_filename=file.filename)

        return {
            "document_id": document.document_id,
            "document_name": document.document_name,
            "source_type": document.source_type,
            "total_pages": len(document.pages),
            "metadata": document.metadata,
        }

    except EmptyFileError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except UnsupportedFormatError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except InvalidDocumentError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except IngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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
