"""Ingestion service — orchestrates validation, detection, and loading.

This is the single entry-point that the API layer calls.  Business
logic stays here; route handlers remain thin.
"""

from __future__ import annotations

import logging
from pathlib import Path

from rag.ingestion.detection import FileFormat, detect_format
from rag.ingestion.docx_loader import load_docx
from rag.ingestion.models import Document
from rag.ingestion.pdf_loader import load_pdf
from rag.ingestion.validation import validate_file

logger = logging.getLogger(__name__)

# Mapping from detected format to its loader function.
_LOADERS = {
    FileFormat.PDF: load_pdf,
    FileFormat.DOCX: load_docx,
}


class IngestionService:
    """Facade for the document-ingestion pipeline.

    Usage::

        service = IngestionService(max_size_bytes=50 * 1024 * 1024)
        document = service.ingest(Path("report.pdf"), "report.pdf")
    """

    def __init__(self, *, max_size_bytes: int | None = None) -> None:
        self.max_size_bytes = max_size_bytes

    def ingest(
        self,
        file_path: Path,
        original_filename: str | None = None,
    ) -> Document:
        """Validate, detect format, parse, and return a normalized document.

        Args:
            file_path: Local path to the file to ingest.
            original_filename: The user-facing filename.  Falls back to
                ``file_path.name`` when not provided.

        Returns:
            A ``Document`` instance with extracted content and metadata.

        Raises:
            FileNotFoundError: File does not exist.
            EmptyFileError: File is zero bytes.
            FileTooLargeError: File exceeds the configured limit.
            UnsupportedFormatError: Format is not supported.
            InvalidDocumentError: File is corrupted or unreadable.
        """
        file_path = Path(file_path)
        display_name = original_filename or file_path.name

        logger.info("Ingesting document: %s", display_name)

        # 1. Validate
        validate_file(file_path, max_size_bytes=self.max_size_bytes)

        # 2. Detect format
        fmt = detect_format(file_path)
        logger.info("Detected format: %s", fmt.value)

        # 3. Load
        loader = _LOADERS[fmt]
        document = loader(file_path, document_name=display_name)

        logger.info(
            "Ingested '%s': id=%s, pages=%d",
            display_name,
            document.document_id,
            len(document.pages),
        )
        return document
