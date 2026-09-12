"""Chunk representation for the RAG pipeline.

Each ``Chunk`` is a retrieval-ready text segment produced by the
chunking phase.  It preserves all source metadata needed for
downstream embedding, retrieval, and citation.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A single retrieval-ready text chunk with source metadata.

    Attributes:
        chunk_id: Unique identifier (UUID4), never derived from
            filesystem paths or user-controlled filenames.
        document_id: ID of the source document (from Phase 1).
        document_name: Original filename of the source document.
        source_type: Lowercase format identifier (``"pdf"``, ``"docx"``).
        content: Cleaned text content for this chunk.
        page_number: 1-indexed page number where the chunk starts,
            or ``None`` when page information is unavailable.
        chunk_index: 0-based position of this chunk within the document.
            Provides deterministic ordering.
        metadata: Arbitrary metadata carried forward from the source
            document/page plus chunk-specific information.
    """

    chunk_id: str
    document_id: str
    document_name: str
    source_type: str
    content: str
    page_number: int | None = None
    chunk_index: int = Field(..., ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
