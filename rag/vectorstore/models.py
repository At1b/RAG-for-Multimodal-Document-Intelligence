"""Data models for vector store results."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class VectorSearchResult(BaseModel):
    """A single result from a vector similarity search.

    Attributes:
        chunk_id: Unique identifier of the matched chunk.
        document_id: ID of the source document.
        document_name: Original filename of the source document.
        content: Text content of the matched chunk.
        score: Similarity/distance score (interpretation depends on
            the metric used by the vector store).
        metadata: Full metadata dictionary stored with the vector.
    """

    chunk_id: str = Field(..., min_length=1)
    document_id: str = Field(..., min_length=1)
    document_name: str = Field(..., min_length=1)
    content: str = Field(default="")
    score: float = Field(default=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
