"""Document chunking — splits normalized documents into retrieval-ready chunks.

The chunker takes a Phase 1 ``Document`` and produces a list of ``Chunk``
objects with metadata preserved for downstream embedding and retrieval.

Strategy: fixed-size character chunking with configurable overlap.

Defaults:
    chunk_size=1000 characters (~200-250 words) — fits typical embedding
        model context windows and produces granular retrieval units.
    chunk_overlap=200 characters — enough to avoid breaking mid-sentence
        at chunk boundaries while keeping redundancy reasonable.
"""

from __future__ import annotations

import logging
from typing import Self

from pydantic import BaseModel, field_validator, model_validator

from rag.chunking.chunk_id import generate_chunk_id
from rag.chunking.models import Chunk
from rag.chunking.text_cleaning import clean_text
from rag.ingestion.models import Document

logger = logging.getLogger(__name__)


class ChunkingConfig(BaseModel):
    """Configuration for the chunking strategy.

    Attributes:
        chunk_size: Maximum number of characters per chunk.  Must be > 0.
        chunk_overlap: Number of overlapping characters between adjacent
            chunks.  Must be >= 0 and < chunk_size.
    """

    chunk_size: int = 1000
    chunk_overlap: int = 200

    @field_validator("chunk_size")
    @classmethod
    def _chunk_size_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"chunk_size must be > 0, got {v}")
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"chunk_overlap must be >= 0, got {v}")
        return v

    @model_validator(mode="after")
    def _validate_overlap(self) -> Self:
        """Validate cross-field constraint: overlap < chunk_size."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be < "
                f"chunk_size ({self.chunk_size})"
            )
        return self


def chunk_document(
    document: Document,
    config: ChunkingConfig | None = None,
) -> list[Chunk]:
    """Split a normalized document into retrieval-ready chunks.

    Args:
        document: A Phase 1 ``Document`` with extracted page content.
        config: Chunking parameters.  Uses defaults when ``None``.

    Returns:
        Ordered list of ``Chunk`` objects with metadata preserved.
        Returns an empty list if the document has no meaningful text.
    """
    if config is None:
        config = ChunkingConfig()

    # Clean each page and record exact page boundaries in cleaned text.
    full_cleaned, page_boundaries = _build_cleaned_document(document)

    if not full_cleaned:
        logger.info(
            "Document '%s' (id=%s) has no text after cleaning — producing zero chunks.",
            document.document_name,
            document.document_id,
        )
        return []

    # Split into text segments with offsets.
    segments = _split_text(full_cleaned, config.chunk_size, config.chunk_overlap)

    # Build Chunk objects with metadata.
    chunks: list[Chunk] = []
    for idx, (start_offset, segment) in enumerate(segments):
        page_num = _page_for_offset(start_offset, page_boundaries)

        # Carry forward document-level metadata.
        chunk_metadata = dict(document.metadata)

        # Carry forward page-level metadata if available.
        if page_num is not None:
            for page in document.pages:
                if page.page_number == page_num:
                    for k, v in page.metadata.items():
                        if k not in chunk_metadata and k != "char_count":
                            chunk_metadata[k] = v
                    break

        # Record chunk-specific information.
        chunk_metadata.update(
            {
                "char_count": len(segment),
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
            }
        )

        chunks.append(
            Chunk(
                chunk_id=generate_chunk_id(),
                document_id=document.document_id,
                document_name=document.document_name,
                source_type=document.source_type,
                content=segment,
                page_number=page_num,
                chunk_index=idx,
                metadata=chunk_metadata,
            )
        )

    logger.info(
        "Chunked document '%s' (id=%s) into %d chunks (chunk_size=%d, overlap=%d).",
        document.document_name,
        document.document_id,
        len(chunks),
        config.chunk_size,
        config.chunk_overlap,
    )
    return chunks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_cleaned_document(
    document: Document,
) -> tuple[str, list[tuple[int, int, int]]]:
    """Clean each page and record exact page boundaries in cleaned text.

    Returns:
        A tuple of (full_cleaned_text, page_boundaries) where page_boundaries
        is a list of (start_char, end_char, page_number) tuples in the
        concatenated cleaned text.
    """
    parts: list[str] = []
    boundaries: list[tuple[int, int, int]] = []
    current = 0

    for page in document.pages:
        cleaned_page = clean_text(page.content)
        if not cleaned_page:
            continue

        if parts:
            # Paragraph separator between distinct pages.
            parts.append("\n\n")
            current += 2

        start = current
        parts.append(cleaned_page)
        current += len(cleaned_page)
        boundaries.append((start, current, page.page_number))

    full_cleaned = "".join(parts)
    return full_cleaned, boundaries


def _split_text(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[tuple[int, str]]:
    """Split *text* into fixed-size segments with overlap.

    Guarantees:
    - Every character in *text* appears in at least one segment.
    - Segments preserve text order.
    - No infinite loop (step is always >= 1).

    Returns:
        List of (start_offset, segment) tuples.
    """
    if not text:
        return []

    text_len = len(text)
    if text_len <= chunk_size:
        return [(0, text)]

    step = chunk_size - overlap
    if step < 1:
        step = 1

    segments: list[tuple[int, str]] = []
    start = 0

    while start < text_len:
        end = start + chunk_size
        segment = text[start:end]
        segments.append((start, segment))

        if end >= text_len:
            break

        start += step

    return segments


def _page_for_offset(
    offset: int,
    page_boundaries: list[tuple[int, int, int]],
) -> int | None:
    """Determine the page number for a character offset in cleaned text.

    Uses exact page boundary lookup in the cleaned text.

    Returns ``None`` if page information is unavailable.
    """
    if not page_boundaries:
        return None

    for _start, end, page_num in page_boundaries:
        if offset < end:
            return page_num

    return page_boundaries[-1][2]
