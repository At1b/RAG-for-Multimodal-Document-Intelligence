"""Source / citation data model — Phase 7.

``Source`` is a clean, structured representation of a single
document source referenced in a RAG query response.

Design principles:
    - Contains only information that actually exists in retrieved
      chunk metadata.  No field is ever fabricated.
    - ``page_number`` is ``None`` when the metadata does not include
      page information — never invented.
    - ``document_id`` is the stable identity (UUID), not the filename.
    - All fields are validated via Pydantic.

This model does NOT depend on the LLM output.  It is built
entirely from application-level retrieved metadata.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Source(BaseModel):
    """A single source/citation from a retrieved document chunk.

    Attributes:
        document_id: Stable unique identifier for the source document.
        document_name: User-facing filename of the source document.
        page_number: 1-indexed page number where the chunk originates,
            or ``None`` when page information is unavailable.
        chunk_id: Unique identifier of the specific chunk.
        score: Similarity score from retrieval (higher = more similar).
        metadata: Additional metadata carried from the chunk, if any.
    """

    document_id: str = Field(..., min_length=1)
    document_name: str = Field(..., min_length=1)
    page_number: int | None = Field(default=None)
    chunk_id: str = Field(..., min_length=1)
    score: float = Field(default=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
