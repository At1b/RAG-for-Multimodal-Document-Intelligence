"""File validation — runs before format detection and parsing.

Checks are ordered from cheapest to most expensive so we fail fast.
"""

from __future__ import annotations

from pathlib import Path

from rag.ingestion.exceptions import (
    EmptyFileError,
    FileNotFoundError,
    FileTooLargeError,
    UnsupportedFormatError,
)

# Extensions accepted by the ingestion pipeline.
_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".docx"})


def validate_file(file_path: Path, *, max_size_bytes: int | None = None) -> None:
    """Run all pre-processing validations on *file_path*.

    Args:
        file_path: Path to the candidate file.
        max_size_bytes: Optional upper bound on file size.
            Pass ``None`` to skip the size check.

    Raises:
        FileNotFoundError: File does not exist or is not a regular file.
        EmptyFileError: File is zero bytes.
        FileTooLargeError: File exceeds *max_size_bytes*.
        UnsupportedFormatError: Extension is not in the supported set.
    """
    _check_exists(file_path)
    _check_not_empty(file_path)
    if max_size_bytes is not None:
        _check_size(file_path, max_size_bytes)
    _check_extension(file_path)


# --- individual checks (kept small and testable) ---


def _check_exists(file_path: Path) -> None:
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"File not found: '{file_path.name}'.")


def _check_not_empty(file_path: Path) -> None:
    if file_path.stat().st_size == 0:
        raise EmptyFileError(f"File is empty: '{file_path.name}'.")


def _check_size(file_path: Path, max_bytes: int) -> None:
    size = file_path.stat().st_size
    if size > max_bytes:
        max_mb = max_bytes / (1024 * 1024)
        raise FileTooLargeError(
            f"File '{file_path.name}' is {size:,} bytes, "
            f"exceeding the {max_mb:.0f} MB limit."
        )


def _check_extension(file_path: Path) -> None:
    ext = file_path.suffix.lower()
    if ext not in _SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(
            f"Unsupported file extension: '{ext}'. "
            f"Supported: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}."
        )
