"""File-format detection using magic bytes and file extension.

The detector is isolated from individual loaders so the routing
decision is made in one place.
"""

from __future__ import annotations

import enum
from pathlib import Path

from rag.ingestion.exceptions import UnsupportedFormatError

# Magic-byte signatures used to verify file content.
_PDF_MAGIC = b"%PDF"
_ZIP_MAGIC = b"PK\x03\x04"  # DOCX is a ZIP archive (OOXML)


class FileFormat(enum.Enum):
    """Supported document formats."""

    PDF = "pdf"
    DOCX = "docx"


# Map of lowercase extensions to formats.
_EXTENSION_MAP: dict[str, FileFormat] = {
    ".pdf": FileFormat.PDF,
    ".docx": FileFormat.DOCX,
}


def detect_format(file_path: Path) -> FileFormat:
    """Determine the file format by inspecting magic bytes and extension.

    Strategy:
        1. Read the first few bytes to check magic signatures.
        2. Cross-reference with the file extension.
        3. Magic bytes take priority — a ``.pdf`` file without the
           ``%PDF`` header is rejected.

    Raises:
        UnsupportedFormatError: If the format cannot be identified or
            the magic bytes contradict the extension.
    """
    magic = _read_magic_bytes(file_path)
    ext = file_path.suffix.lower()

    # Try magic bytes first.
    if magic.startswith(_PDF_MAGIC):
        return FileFormat.PDF
    if magic.startswith(_ZIP_MAGIC) and ext == ".docx":
        # ZIP could be many things — only accept as DOCX when the
        # extension confirms the intent.
        return FileFormat.DOCX

    # Fall through: extension didn't match any known magic pattern.
    if ext in _EXTENSION_MAP:
        raise UnsupportedFormatError(
            f"File '{file_path.name}' has extension '{ext}' but its content "
            "does not match the expected format signature."
        )

    raise UnsupportedFormatError(
        f"Unsupported file format: '{ext}'. "
        f"Supported formats: {', '.join(sorted(_EXTENSION_MAP))}."
    )


def _read_magic_bytes(file_path: Path, size: int = 8) -> bytes:
    """Read the first *size* bytes of a file for signature detection."""
    with open(file_path, "rb") as fh:
        return fh.read(size)
