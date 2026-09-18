"""Context builder — converts retrieval results into generator input.

Transforms a list of ``VectorSearchResult`` objects (from Phase 4)
into a formatted context string suitable for inclusion in an LLM prompt.

Responsibilities:
    - Preserve result order (most relevant first).
    - Include document name, page number, chunk ID, and content.
    - Handle empty results by raising ``InvalidContextError``.
    - Limit context size by omitting later (lower-relevance) *complete*
      chunks rather than truncating mid-chunk.
    - Log when truncation occurs.

This module does NOT implement citation formatting (Phase 7).
"""

from __future__ import annotations

import logging

from rag.generation.exceptions import InvalidContextError
from rag.vectorstore.models import VectorSearchResult

logger = logging.getLogger(__name__)

# Separator between formatted chunks in the context string.
_CHUNK_SEPARATOR = "\n\n---\n\n"


def build_context(
    results: list[VectorSearchResult],
    *,
    max_chars: int = 3000,
) -> str:
    """Build a formatted context string from retrieval results.

    Each chunk is formatted with its source metadata (document name,
    page number, chunk ID) followed by its content.  Chunks are
    included in the order they appear in *results* (most relevant
    first).

    If the accumulated context would exceed *max_chars*, later chunks
    are omitted entirely — no mid-chunk truncation.  A warning is
    logged when truncation occurs.

    Args:
        results: Retrieval results from Phase 4.  Must not be empty.
        max_chars: Maximum total character length for the formatted
            context.  Must be >= 100.

    Returns:
        Formatted context string ready for prompt inclusion.

    Raises:
        InvalidContextError: If *results* is empty, ``None``, or not
            a list, or if no chunks fit within the character limit.
    """
    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if not isinstance(results, list):
        raise InvalidContextError(
            f"context must be a list of VectorSearchResult, "
            f"got {type(results).__name__}"
        )
    if len(results) == 0:
        raise InvalidContextError(
            "retrieval context is empty — cannot generate an answer "
            "without supporting context"
        )
    if max_chars < 100:
        raise InvalidContextError(f"max_chars must be >= 100, got {max_chars}")

    # ------------------------------------------------------------------
    # Format and accumulate chunks
    # ------------------------------------------------------------------
    formatted_chunks: list[str] = []
    total_chars = 0
    truncated_count = 0

    for idx, result in enumerate(results):
        chunk_text = _format_chunk(result, idx + 1)
        separator_cost = len(_CHUNK_SEPARATOR) if formatted_chunks else 0
        new_total = total_chars + separator_cost + len(chunk_text)

        if new_total > max_chars:
            # The first chunk must always be included even if it
            # exceeds max_chars, otherwise we'd return nothing.
            if not formatted_chunks:
                formatted_chunks.append(chunk_text)
                total_chars = len(chunk_text)
                truncated_count = len(results) - 1
                logger.warning(
                    "First chunk (%d chars) exceeds max_chars (%d); "
                    "included anyway.  Remaining %d chunks omitted.",
                    len(chunk_text),
                    max_chars,
                    truncated_count,
                )
                break

            # Omit this and all remaining chunks.
            truncated_count = len(results) - idx
            logger.warning(
                "Context limit reached (%d/%d chars).  Omitting %d remaining chunk(s).",
                total_chars,
                max_chars,
                truncated_count,
            )
            break

        formatted_chunks.append(chunk_text)
        total_chars = new_total

    return _CHUNK_SEPARATOR.join(formatted_chunks)


def _format_chunk(result: VectorSearchResult, position: int) -> str:
    """Format a single retrieval result as a context block.

    Args:
        result: The retrieval result to format.
        position: 1-based position in the result list.

    Returns:
        Formatted string containing metadata header and content.
    """
    header_parts = [f"[Chunk {position}]"]
    header_parts.append(f"Document: {result.document_name}")

    page = result.metadata.get("page_number") or result.metadata.get("page")
    if page is not None:
        header_parts.append(f"Page: {page}")

    header_parts.append(f"Chunk ID: {result.chunk_id}")

    header = " | ".join(header_parts)
    return f"{header}\n{result.content}"
