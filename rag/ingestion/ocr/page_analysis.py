"""Page-level analysis to determine whether a PDF page requires OCR.

Implements the conservative two-condition heuristic:
1. Native alphanumeric character count < min_native_chars.
2. Raster image coverage >= MIN_IMAGE_COVERAGE (30% of page area).

Legitimately sparse or blank text pages without large raster images are NOT OCR'd.
PDFs with an existing embedded/searchable text layer (e.g. pre-OCR'd) are preserved.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pymupdf

logger = logging.getLogger(__name__)

# Module constant for minimum page area covered by raster images to trigger OCR.
MIN_IMAGE_COVERAGE: float = 0.30


def count_alphanumeric_chars(text: str) -> int:
    """Count alphanumeric characters in text, ignoring punctuation and whitespace."""
    return sum(1 for c in text if c.isalnum())


def calculate_image_coverage(page: pymupdf.Page) -> float:
    """Calculate the fraction of page area covered by raster images.

    Clips image bounding rectangles to page boundary and computes
    total covered area over total page area.

    Args:
        page: A PyMuPDF Page object.

    Returns:
        float: Fraction of page area covered by raster images in [0.0, 1.0].
    """
    page_rect = page.rect
    page_area = page_rect.get_area()
    if page_area <= 0:
        return 0.0

    try:
        images = page.get_images(full=True)
    except Exception as exc:
        logger.warning(
            "Failed to inspect images on page %d: %s",
            page.number + 1,
            exc,
        )
        return 0.0

    if not images:
        return 0.0

    covered_area = 0.0
    seen_xrefs: set[int] = set()
    for img_info in images:
        xref = img_info[0]
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            continue

        for r in rects:
            clipped = r & page_rect
            if not clipped.is_empty:
                covered_area += clipped.get_area()

    # Cap coverage fraction at 1.0 (in case overlapping images sum > page_area)
    return min(1.0, covered_area / page_area)


def needs_ocr(
    page: pymupdf.Page,
    min_native_chars: int = 25,
) -> bool:
    """Determine whether a PDF page is a scanned image requiring OCR.

    A page needs OCR if and only if:
    1. Alphanumeric character count from native text < min_native_chars, AND
    2. Raster image coverage >= MIN_IMAGE_COVERAGE (30%).

    Args:
        page: PyMuPDF Page instance.
        min_native_chars: Minimum alphanumeric character threshold (default 25).

    Returns:
        bool: True if the page should be processed with OCR, False otherwise.
    """
    try:
        native_text = page.get_text("text")
    except Exception as exc:
        logger.warning(
            "Failed to extract native text for page analysis on page %d: %s",
            page.number + 1,
            exc,
        )
        native_text = ""

    native_chars = count_alphanumeric_chars(native_text)
    if native_chars >= min_native_chars:
        return False

    coverage = calculate_image_coverage(page)
    return coverage >= MIN_IMAGE_COVERAGE
