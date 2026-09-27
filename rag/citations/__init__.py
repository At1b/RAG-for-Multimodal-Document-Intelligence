"""MM-RAG Citation / Source Attribution — Phase 7.

Provides structured source models and a citation formatter that
extracts provenance information from retrieved chunk metadata.

Public API:
    Source           — structured source/citation model.
    format_sources   — extract sources from VectorSearchResult list.
"""

from rag.citations.formatter import format_sources
from rag.citations.models import Source

__all__ = [
    "Source",
    "format_sources",
]
