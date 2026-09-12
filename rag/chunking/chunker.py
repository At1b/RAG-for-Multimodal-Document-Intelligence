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
from typing import Any

from pydantic import BaseModel, field_validator

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

    def model_post_init(self, __context: Any) -> None:
        """Validate cross-field constraint: overlap < chunk_size."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be < "
                f"chunk_size ({self.chunk_size})"
            )


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

    # Concatenate all page text, tracking page boundaries for
    # page-number assignment.
    full_text, page_offsets = _build_full_text(document)

    # Clean the concatenated text.
    cleaned = clean_text(full_text)

    if not cleaned:
        logger.info(
            "Document '%s' (id=%s) has no text after cleaning — producing zero chunks.",
            document.document_name,
            document.document_id,
        )
        return []

    # Split into raw text segments.
    segments = _split_text(cleaned, config.chunk_size, config.chunk_overlap)

    # Build Chunk objects with metadata.
    chunks: list[Chunk] = []
    for idx, segment in enumerate(segments):
        # Determine the page number for this segment.
        # We use the page where the segment's starting offset falls in
        # the *cleaned* text.  Since cleaning may shift offsets slightly,
        # we track against the cleaned full text.
        start_offset = _find_segment_start(cleaned, segment, idx, config)
        page_num = _page_for_offset(start_offset, full_text, cleaned, page_offsets)

        chunks.append(
            Chunk(
                chunk_id=generate_chunk_id(),
                document_id=document.document_id,
                document_name=document.document_name,
                source_type=document.source_type,
                content=segment,
                page_number=page_num,
                chunk_index=idx,
                metadata={
                    "char_count": len(segment),
                    "chunk_size": config.chunk_size,
                    "chunk_overlap": config.chunk_overlap,
                },
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


def _build_full_text(
    document: Document,
) -> tuple[str, list[tuple[int, int, int]]]:
    """Concatenate page texts and record page boundary offsets.

    Returns:
        A tuple of (full_text, page_offsets) where page_offsets is a list
        of (start_char, end_char, page_number) tuples in the raw
        (pre-cleaning) concatenated text.
    """
    parts: list[str] = []
    offsets: list[tuple[int, int, int]] = []
    current = 0

    for page in document.pages:
        text = page.content
        start = current
        parts.append(text)
        current += len(text)
        offsets.append((start, current, page.page_number))

        # Add a newline separator between pages.
        parts.append("\n")
        current += 1

    full = "".join(parts)
    return full, offsets


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split *text* into fixed-size segments with overlap.

    Guarantees:
    - Every character in *text* appears in at least one segment.
    - Segments preserve text order.
    - No infinite loop (step is always >= 1).
    """
    if not text:
        return []

    text_len = len(text)
    if text_len <= chunk_size:
        return [text]

    step = chunk_size - overlap
    # Safety: step must be >= 1 (guaranteed by config validation:
    # overlap < chunk_size), but guard defensively.
    if step < 1:
        step = 1

    segments: list[str] = []
    start = 0

    while start < text_len:
        end = start + chunk_size
        segment = text[start:end]
        segments.append(segment)

        # If we've reached the end, stop.
        if end >= text_len:
            break

        start += step

    return segments


def _find_segment_start(
    cleaned: str,
    segment: str,
    index: int,
    config: ChunkingConfig,
) -> int:
    """Calculate the character offset of *segment* within *cleaned* text."""
    step = config.chunk_size - config.chunk_overlap
    if step < 1:
        step = 1
    return index * step


def _page_for_offset(
    offset: int,
    raw_text: str,
    cleaned_text: str,
    page_offsets: list[tuple[int, int, int]],
) -> int | None:
    """Determine the page number for a character offset in cleaned text.

    Uses a proportional mapping from cleaned-text offset to raw-text
    offset, then looks up the page boundary table.

    Returns ``None`` if page information is unavailable.
    """
    if not page_offsets:
        return None

    # Map cleaned offset to approximate raw offset proportionally.
    cleaned_len = len(cleaned_text)
    raw_len = len(raw_text)

    if cleaned_len == 0:
        return page_offsets[0][2] if page_offsets else None

    # Proportional mapping.
    raw_offset = int(offset * raw_len / cleaned_len) if cleaned_len > 0 else 0
    raw_offset = min(raw_offset, raw_len - 1) if raw_len > 0 else 0

    # Find which page this raw offset falls in.
    for start, end, page_num in page_offsets:
        if start <= raw_offset < end:
            return page_num

    # Fallback: return the last page number.
    return page_offsets[-1][2]
