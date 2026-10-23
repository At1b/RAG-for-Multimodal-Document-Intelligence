"""OCR engine protocol and Tesseract 5 subprocess adapter.

Provides:
- OcrPageResult: normalized result dataclass (text, mean_confidence).
- OcrEngine: protocol for pluggable OCR backends.
- TesseractEngine: robust subprocess adapter for Tesseract 5 with hard timeouts,
  process isolation, and clear error mapping.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from rag.ingestion.exceptions import OcrPageError, OcrUnavailableError
from rag.ingestion.ocr.text_layout import parse_tsv_to_text

logger = logging.getLogger(__name__)

_DEFAULT_WINDOWS_TESSERACT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
_LANGUAGE_PATTERN = re.compile(r"^[a-z]{3}(_[A-Za-z]+)?(\+[a-z]{3}(_[A-Za-z]+)?)*$")


@dataclass(frozen=True)
class OcrPageResult:
    """Result of OCR extraction for a single page or image."""

    text: str
    mean_confidence: float | None = None


@runtime_checkable
class OcrEngine(Protocol):
    """Protocol defining the interface for OCR engines."""

    @property
    def version(self) -> str:
        """Return the engine identifier and version string."""
        ...

    def recognize(
        self,
        image_bytes: bytes,
        *,
        language: str = "eng",
        timeout: float = 60.0,
    ) -> OcrPageResult:
        """Run OCR on image bytes and return normalized page result."""
        ...


class TesseractEngine:
    """Tesseract 5 adapter invoking the CLI via isolated subprocess."""

    def __init__(self, tesseract_cmd: str | Path | None = None) -> None:
        self._custom_cmd = str(tesseract_cmd) if tesseract_cmd else None
        self._cached_version: str | None = None

    def _resolve_executable(self) -> str:
        """Locate the tesseract executable or raise OcrUnavailableError."""
        if self._custom_cmd:
            if shutil.which(self._custom_cmd) or os.path.isfile(self._custom_cmd):
                return self._custom_cmd
            raise OcrUnavailableError(
                f"Configured Tesseract executable not found: '{self._custom_cmd}'"
            )

        cmd = shutil.which("tesseract")
        if cmd:
            return cmd

        if _DEFAULT_WINDOWS_TESSERACT.is_file():
            return str(_DEFAULT_WINDOWS_TESSERACT)

        raise OcrUnavailableError(
            "Tesseract OCR executable not found. Please install Tesseract 5 "
            "and ensure it is on PATH or installed at "
            f"'{_DEFAULT_WINDOWS_TESSERACT}'."
        )

    @property
    def version(self) -> str:
        """Return Tesseract version string (e.g. 'tesseract 5.5.3')."""
        if self._cached_version is not None:
            return self._cached_version

        exe = self._resolve_executable()
        kwargs: dict = {}
        if hasattr(subprocess, "CREATE_NO_WINDOW"):
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        try:
            res = subprocess.run(
                [exe, "--version"],
                capture_output=True,
                text=True,
                timeout=10.0,
                check=False,
                **kwargs,
            )
            if res.returncode == 0 and res.stdout:
                first_line = res.stdout.strip().splitlines()[0]
                self._cached_version = first_line
                return self._cached_version
            raise OcrUnavailableError(
                f"Failed to query Tesseract version: return code {res.returncode}"
            )
        except (subprocess.SubprocessError, OSError) as exc:
            raise OcrUnavailableError(
                f"Failed to run Tesseract --version: {exc}"
            ) from exc

    def recognize(
        self,
        image_bytes: bytes,
        *,
        language: str = "eng",
        timeout: float = 60.0,
    ) -> OcrPageResult:
        """Execute Tesseract CLI via stdin piping and parse TSV output.

        Args:
            image_bytes: PNG-encoded image bytes.
            language: Tesseract language code (e.g. 'eng', 'eng+deu').
            timeout: Subprocess timeout in seconds.

        Returns:
            OcrPageResult containing plain text and mean confidence.

        Raises:
            OcrUnavailableError: If Tesseract or requested language pack is missing.
            OcrPageError: If OCR times out, encounters a non-zero exit code,
                or produces malformed TSV.
        """
        if not _LANGUAGE_PATTERN.match(language):
            raise OcrUnavailableError(
                f"Invalid OCR language specification: '{language}'"
            )

        exe = self._resolve_executable()

        args = [exe, "stdin", "stdout", "-l", language, "--psm", "3", "tsv"]
        kwargs: dict = {}
        if hasattr(subprocess, "CREATE_NO_WINDOW"):
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        try:
            proc = subprocess.run(
                args,
                input=image_bytes,
                capture_output=True,
                timeout=timeout,
                check=False,
                **kwargs,
            )
        except subprocess.TimeoutExpired as exc:
            raise OcrPageError(f"OCR process timed out after {timeout:.1f}s") from exc
        except FileNotFoundError as exc:
            raise OcrUnavailableError(
                "Tesseract executable not found during invocation."
            ) from exc
        except OSError as exc:
            raise OcrUnavailableError(
                f"Failed to spawn Tesseract process: {exc}"
            ) from exc

        if proc.returncode != 0:
            stderr_snippet = proc.stderr.decode("utf-8", errors="replace")[:500]
            logger.warning(
                "Tesseract exited with code %d. Stderr: %s",
                proc.returncode,
                stderr_snippet.strip(),
            )
            # Differentiate missing language data from generic OCR page failure
            if (
                "failed loading language" in stderr_snippet.lower()
                or "error opening data file" in stderr_snippet.lower()
                or "could not initialize tesseract" in stderr_snippet.lower()
            ):
                raise OcrUnavailableError(
                    f"Tesseract language data missing for '{language}'."
                )
            raise OcrPageError(
                f"Tesseract process returned non-zero exit code {proc.returncode}."
            )

        tsv_text = proc.stdout.decode("utf-8", errors="replace")
        text, mean_conf = parse_tsv_to_text(tsv_text)
        return OcrPageResult(text=text, mean_confidence=mean_conf)
