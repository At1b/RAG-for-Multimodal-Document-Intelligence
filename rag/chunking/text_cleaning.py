"""Text cleaning / normalization for pre-chunking processing.

This module applies conservative transformations that fix common
whitespace issues in extracted document text without destroying
meaningful content.

What it does:
- Normalize line endings (``\\r\\n``, ``\\r``) → ``\\n``
- Collapse runs of 3+ newlines → 2 newlines (preserve paragraph breaks)
- Strip leading/trailing whitespace per line
- Convert tabs to spaces
- Collapse runs of 2+ horizontal spaces → single space (within lines)
- Strip leading/trailing whitespace from the full text

What it does NOT do (preserves semantic content):
- No lowercasing
- No punctuation removal
- No stop-word removal
- No aggressive sentence splitting
- No Unicode normalization beyond whitespace
"""

from __future__ import annotations

import re

# Compiled patterns — created once at import time.
_RE_CRLF = re.compile(r"\r\n?")
_RE_MULTI_NEWLINES = re.compile(r"\n{3,}")
_RE_HORIZONTAL_SPACE = re.compile(r"[^\S\n]{2,}")


def clean_text(text: str) -> str:
    """Clean and normalize *text* for chunking.

    Args:
        text: Raw extracted text from a document page or section.

    Returns:
        Cleaned text with normalized whitespace.  May return an empty
        string if the input contains only whitespace.
    """
    if not text:
        return ""

    # 1. Sanitize null bytes, byte-order mark, and zero-width spaces.
    result = text.replace("\x00", "").replace("\ufeff", "").replace("\u200b", "")

    # 2. Normalize non-breaking spaces to standard space.
    result = result.replace("\u00a0", " ")

    # 3. Normalize line endings.
    result = _RE_CRLF.sub("\n", result)

    # 4. Convert tabs to spaces (before collapsing).
    result = result.replace("\t", " ")

    # 5. Strip each line individually (removes leading/trailing
    #    spaces per line while preserving intentional blank lines).
    result = "\n".join(line.strip() for line in result.split("\n"))

    # 6. Collapse runs of horizontal whitespace within lines.
    result = _RE_HORIZONTAL_SPACE.sub(" ", result)

    # 7. Collapse 3+ consecutive newlines → double newline
    #    (preserves single paragraph breaks).
    result = _RE_MULTI_NEWLINES.sub("\n\n", result)

    # 8. Strip leading/trailing whitespace from full text.
    return result.strip()
