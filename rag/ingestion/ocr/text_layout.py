"""Pure functions for converting Tesseract TSV output into formatted text.

Handles:
- Header validation (raises OcrPageError if invalid).
- Skipping structural rows (conf == -1) and empty tokens.
- Sorting words by reading order: (block, par, line, word, left).
- Grouping words into lines and paragraphs (blank line on paragraph/block change).
- Resilient parsing (skips individual malformed rows without aborting).
- Calculating mean confidence across valid recognized words (0-100), or None.
"""

from __future__ import annotations

import csv
import io
import logging

from rag.ingestion.exceptions import OcrPageError

logger = logging.getLogger(__name__)

_REQUIRED_COLUMNS = (
    "level",
    "page_num",
    "block_num",
    "par_num",
    "line_num",
    "word_num",
    "left",
    "top",
    "width",
    "height",
    "conf",
    "text",
)


def parse_tsv_to_text(tsv_content: str) -> tuple[str, float | None]:
    """Parse Tesseract TSV output into plain text and mean word confidence.

    Args:
        tsv_content: Raw TSV output string from Tesseract.

    Returns:
        tuple[str, float | None]:
            - Formatted text with single newlines within paragraphs and
              double newlines between paragraphs/blocks.
            - Mean confidence as float in [0.0, 100.0], or None if no words recognized.

    Raises:
        OcrPageError: If TSV header is missing or does not contain required columns.
    """
    if not tsv_content or not tsv_content.strip():
        return "", None

    reader = csv.reader(
        io.StringIO(tsv_content), delimiter="\t", quoting=csv.QUOTE_NONE
    )

    try:
        header = next(reader)
    except StopIteration:
        return "", None

    # Validate header columns
    header_clean = [col.strip().lower() for col in header]
    for req in (
        "level",
        "block_num",
        "par_num",
        "line_num",
        "word_num",
        "conf",
        "text",
    ):
        if req not in header_clean:
            raise OcrPageError(
                f"Invalid TSV header: missing required column '{req}'. Header: {header}"
            )

    col_idx = {name: idx for idx, name in enumerate(header_clean)}
    idx_block = col_idx["block_num"]
    idx_par = col_idx["par_num"]
    idx_line = col_idx["line_num"]
    idx_word = col_idx["word_num"]
    idx_left = col_idx.get("left")
    idx_conf = col_idx["conf"]
    idx_text = col_idx["text"]

    # Structure: lines[(block, par, line)] = list of (word_num, left, text, conf)
    lines_map: dict[tuple[int, int, int], list[tuple[int, int, str, float]]] = {}
    valid_confidences: list[float] = []

    for row in reader:
        if len(row) < len(header_clean):
            continue  # Tolerate truncated row

        try:
            if "level" in col_idx:
                int(row[col_idx["level"]])
            if "page_num" in col_idx:
                int(row[col_idx["page_num"]])
            block = int(row[idx_block])
            par = int(row[idx_par])
            line = int(row[idx_line])
            word = int(row[idx_word])
            left = int(row[idx_left]) if idx_left is not None else 0
            conf_val = float(row[idx_conf])
            text = row[idx_text].strip()
        except (ValueError, IndexError):
            # Tolerate malformed field types
            continue

        # Skip rows with no text or conf == -1 (structural items)
        if not text or conf_val < 0:
            continue

        valid_confidences.append(conf_val)
        line_key = (block, par, line)
        if line_key not in lines_map:
            lines_map[line_key] = []
        lines_map[line_key].append((word, left, text, conf_val))

    if not lines_map:
        return "", None

    # Sort lines by (block, par, line)
    sorted_line_keys = sorted(lines_map.keys())

    paragraphs: list[list[str]] = []
    current_par_key: tuple[int, int] | None = None
    current_par_lines: list[str] = []

    for line_key in sorted_line_keys:
        par_key = (line_key[0], line_key[1])
        if current_par_key is not None and par_key != current_par_key:
            if current_par_lines:
                paragraphs.append(current_par_lines)
                current_par_lines = []
        current_par_key = par_key

        # Sort words within line by (word_num, left)
        words_in_line = sorted(lines_map[line_key], key=lambda item: (item[0], item[1]))
        line_str = " ".join(item[2] for item in words_in_line)
        if line_str:
            current_par_lines.append(line_str)

    if current_par_lines:
        paragraphs.append(current_par_lines)

    # Join lines with \n, paragraphs with \n\n
    par_strings = ["\n".join(lines) for lines in paragraphs if lines]
    full_text = "\n\n".join(par_strings).strip()

    mean_conf: float | None = (
        round(sum(valid_confidences) / len(valid_confidences), 2)
        if valid_confidences
        else None
    )

    return full_text, mean_conf
