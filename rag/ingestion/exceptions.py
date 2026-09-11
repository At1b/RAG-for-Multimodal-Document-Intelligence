"""Ingestion-specific exceptions.

Each exception maps to a distinct failure mode in the ingestion pipeline.
The API layer translates these into appropriate HTTP responses.
"""

import builtins


class IngestionError(Exception):
    """Base exception for all ingestion-related errors."""


class FileNotFoundError(IngestionError, builtins.FileNotFoundError):
    """The requested file does not exist or is not accessible."""


class EmptyFileError(IngestionError):
    """The file exists but contains no data (zero bytes)."""


class UnsupportedFormatError(IngestionError):
    """The file format is not supported by any available loader."""


class InvalidDocumentError(IngestionError):
    """The file claims to be a supported format but is corrupted or unreadable."""


class FileTooLargeError(IngestionError):
    """The file exceeds the configured maximum upload size."""
