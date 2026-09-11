"""Normalized document representation.

All loaders (PDF, DOCX, future formats) produce the same ``Document``
model so downstream phases (chunking, embedding, retrieval) can work
independently of the original file format.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PageContent(BaseModel):
    """A single logical page or section within a document.

    Attributes:
        page_number: 1-indexed page number. For formats without native
            page boundaries (e.g. DOCX), this is set to 1.
        content: Extracted text content for this page.
        metadata: Arbitrary per-page metadata (e.g. char_count).
    """

    page_number: int = Field(..., ge=1)
    content: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    """Format-independent representation of an ingested document.

    Attributes:
        document_id: Unique identifier (UUID4), never derived from
            filesystem paths or user-controlled filenames.
        document_name: Original filename as provided by the user.
        source_type: Lowercase format identifier (``"pdf"``, ``"docx"``).
        pages: Ordered list of page contents.
        metadata: Arbitrary document-level metadata
            (e.g. total_pages, file_size_bytes).
    """

    document_id: str
    document_name: str
    source_type: str
    pages: list[PageContent] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
