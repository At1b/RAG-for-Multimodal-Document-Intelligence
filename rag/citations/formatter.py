"""Citation formatter — Phase 7.

Converts a list of ``VectorSearchResult`` objects into structured
``Source`` models using only actual retrieved metadata.

Design principles:
    - NEVER fabricates document names, page numbers, or chunk IDs.
    - Uses ``page_number`` from metadata when available; ``None`` otherwise.
    - Deduplicates sources by ``chunk_id`` (same chunk should not
      appear twice even if retrieval somehow returns duplicates).
    - Preserves retrieval ordering (most relevant first).
    - Independent of LLM output — sources come from application-level
      metadata, not from the generated text.
"""

from __future__ import annotations

import logging

from rag.citations.models import Source
from rag.vectorstore.models import VectorSearchResult

logger = logging.getLogger(__name__)


def format_sources(
    results: list[VectorSearchResult],
) -> list[Source]:
    """Extract structured sources from retrieval results.

    Args:
        results: List of ``VectorSearchResult`` objects from
            the retrieval layer.  May be empty.

    Returns:
        List of ``Source`` objects ordered by retrieval relevance
        (most relevant first).  Duplicate chunk IDs are removed,
        keeping the first (highest-scored) occurrence.

    Raises:
        TypeError: If *results* is not a list or contains non-
            ``VectorSearchResult`` items.
    """
    if not isinstance(results, list):
        raise TypeError(
            f"results must be a list of VectorSearchResult, "
            f"got {type(results).__name__}"
        )

    if not results:
        return []

    sources: list[Source] = []
    seen_chunk_ids: set[str] = set()

    for idx, result in enumerate(results):
        if not isinstance(result, VectorSearchResult):
            raise TypeError(
                f"All items must be VectorSearchResult instances, "
                f"got {type(result).__name__} at index {idx}"
            )

        # Deduplicate by chunk_id — keep the first (highest-scored) occurrence.
        if result.chunk_id in seen_chunk_ids:
            logger.debug(
                "Skipping duplicate chunk_id '%s' at index %d.",
                result.chunk_id,
                idx,
            )
            continue
        seen_chunk_ids.add(result.chunk_id)

        # Extract page_number from metadata — never invent one.
        meta = result.metadata or {}
        page_number = _extract_page_number(meta)

        sources.append(
            Source(
                document_id=result.document_id,
                document_name=result.document_name,
                page_number=page_number,
                chunk_id=result.chunk_id,
                score=result.score,
                metadata=meta,
            )
        )

    logger.info(
        "Formatted %d source(s) from %d retrieval result(s).",
        len(sources),
        len(results),
    )

    return sources


def _extract_page_number(meta: dict) -> int | None:
    """Safely extract page_number from chunk metadata.

    Returns the integer page number if present and valid,
    or ``None`` if missing, non-numeric, or otherwise invalid.
    Never fabricates a page number.
    """
    raw = meta.get("page_number")
    if raw is None:
        raw = meta.get("page")
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw if raw >= 1 else None
    if isinstance(raw, float):
        # Accept integer-valued floats (e.g. 3.0 from JSON).
        int_val = int(raw)
        return int_val if int_val >= 1 and raw == int_val else None
    return None
