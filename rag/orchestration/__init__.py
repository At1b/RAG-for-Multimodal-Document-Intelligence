"""MM-RAG Orchestration Layer — Phase 6A.

Application-level services that connect ingestion, chunking, embedding,
retrieval, and generation into end-to-end RAG workflows.

Public API:
    DocumentIndexingService  — index a document (ingest → chunk → embed → store).
    RAGQueryService          — answer a question (retrieve → generate).
    IndexingResult           — result of a document indexing operation.
    QueryResult              — result of a RAG query operation.
    OrchestrationError       — base exception for orchestration failures.
    DocumentIndexingError    — indexing pipeline failure.
    QueryError               — query pipeline failure.
    EmptyRetrievalError      — no usable retrieval context.
"""

from rag.orchestration.exceptions import (
    DocumentIndexingError,
    EmptyRetrievalError,
    OrchestrationError,
    QueryError,
)
from rag.orchestration.indexing_service import (
    DocumentIndexingService,
    IndexingResult,
)
from rag.orchestration.query_service import (
    QueryResult,
    RAGQueryService,
)

__all__ = [
    "DocumentIndexingError",
    "DocumentIndexingService",
    "EmptyRetrievalError",
    "IndexingResult",
    "OrchestrationError",
    "QueryError",
    "QueryResult",
    "RAGQueryService",
]
