"""MM-RAG Document Ingestion — Phase 1.

Public API:
    IngestionService  — orchestrates validation, detection, and loading.
    Document          — normalized document representation.
    PageContent       — single page/section within a document.
"""

from rag.ingestion.models import Document, PageContent
from rag.ingestion.service import IngestionService

__all__ = ["Document", "IngestionService", "PageContent"]
