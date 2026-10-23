"""Unit tests for OCR engine protocol and TesseractEngine adapter."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from rag.ingestion.exceptions import OcrPageError, OcrUnavailableError
from rag.ingestion.ocr.engine import OcrEngine, OcrPageResult, TesseractEngine


def _mock_tsv_output() -> str:
    return (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        "5\t1\t1\t1\t1\t1\t10\t10\t20\t10\t95.0\tHello\n"
        "5\t1\t1\t1\t1\t2\t35\t10\t20\t10\t90.0\tWorld\n"
    )


class TestTesseractEngine:
    """Verify TesseractEngine implementation and error handling."""

    def test_protocol_conformance(self):
        engine = TesseractEngine()
        assert isinstance(engine, OcrEngine)

    def test_executable_discovery_custom_path(self, tmp_path):
        fake_exe = tmp_path / "custom_tesseract.exe"
        fake_exe.touch()
        engine = TesseractEngine(tesseract_cmd=fake_exe)
        assert engine._resolve_executable() == str(fake_exe)

    def test_executable_discovery_which(self):
        with patch("shutil.which", return_value="/usr/bin/tesseract"):
            engine = TesseractEngine()
            assert engine._resolve_executable() == "/usr/bin/tesseract"

    def test_executable_discovery_default_windows(self):
        with (
            patch("shutil.which", return_value=None),
            patch("pathlib.Path.is_file", return_value=True),
        ):
            engine = TesseractEngine()
            exe = engine._resolve_executable()
            assert "Tesseract-OCR" in exe

    def test_missing_executable_raises_ocr_unavailable(self):
        with (
            patch("shutil.which", return_value=None),
            patch("pathlib.Path.is_file", return_value=False),
        ):
            engine = TesseractEngine()
            with pytest.raises(OcrUnavailableError, match="not found"):
                engine._resolve_executable()

    def test_version_query_success(self):
        engine = TesseractEngine()
        mock_proc = MagicMock(
            returncode=0, stdout="tesseract v5.5.3\nleptonica-1.87.0\n"
        )
        with (
            patch.object(engine, "_resolve_executable", return_value="dummy_tesseract"),
            patch("subprocess.run", return_value=mock_proc),
        ):
            assert engine.version == "tesseract v5.5.3"
            # Second call uses cached version
            assert engine.version == "tesseract v5.5.3"

    def test_version_query_failure(self):
        engine = TesseractEngine()
        mock_proc = MagicMock(returncode=1, stdout="", stderr="Error")
        with (
            patch.object(engine, "_resolve_executable", return_value="dummy_tesseract"),
            patch("subprocess.run", return_value=mock_proc),
        ):
            with pytest.raises(OcrUnavailableError, match="Failed to query"):
                _ = engine.version

    def test_recognize_success(self):
        engine = TesseractEngine()
        mock_proc = MagicMock(
            returncode=0,
            stdout=_mock_tsv_output().encode("utf-8"),
            stderr=b"",
        )
        with (
            patch.object(engine, "_resolve_executable", return_value="dummy_tesseract"),
            patch("subprocess.run", return_value=mock_proc) as mock_run,
        ):
            result = engine.recognize(b"fake_png_data", language="eng", timeout=15.0)

            assert isinstance(result, OcrPageResult)
            assert result.text == "Hello World"
            assert result.mean_confidence == 92.5
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["input"] == b"fake_png_data"
            assert call_kwargs["timeout"] == 15.0

    def test_recognize_timeout_raises_ocr_page_error(self):
        engine = TesseractEngine()
        with (
            patch.object(engine, "_resolve_executable", return_value="dummy_tesseract"),
            patch(
                "subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="tesseract", timeout=5.0),
            ),
        ):
            with pytest.raises(OcrPageError, match="timed out"):
                engine.recognize(b"fake_png", timeout=5.0)

    def test_recognize_non_zero_exit_raises_ocr_page_error(self):
        engine = TesseractEngine()
        mock_proc = MagicMock(
            returncode=2,
            stdout=b"",
            stderr=b"Corrupted image format",
        )
        with (
            patch.object(engine, "_resolve_executable", return_value="dummy_tesseract"),
            patch("subprocess.run", return_value=mock_proc),
        ):
            with pytest.raises(OcrPageError, match="exit code 2"):
                engine.recognize(b"fake_png")

    def test_recognize_missing_language_raises_ocr_unavailable(self):
        engine = TesseractEngine()
        mock_proc = MagicMock(
            returncode=1,
            stdout=b"",
            stderr=(
                b"Failed loading language 'fra'\n"
                b"Tesseract couldn't load any languages!\n"
            ),
        )
        with (
            patch.object(engine, "_resolve_executable", return_value="dummy_tesseract"),
            patch("subprocess.run", return_value=mock_proc),
        ):
            with pytest.raises(
                OcrUnavailableError,
                match="Failed loading language|language data missing",
            ):
                engine.recognize(b"fake_png", language="fra")

    def test_invalid_language_string_rejected_before_spawning(self):
        engine = TesseractEngine(tesseract_cmd="dummy_tesseract")
        with patch("subprocess.run") as mock_run:
            with pytest.raises(OcrUnavailableError, match="Invalid OCR language"):
                engine.recognize(b"fake_png", language="eng;rm -rf /")
            mock_run.assert_not_called()

    def test_document_data_not_leaked_in_exceptions(self):
        engine = TesseractEngine()
        mock_proc = MagicMock(
            returncode=1,
            stdout=b"secret user text",
            stderr=b"Error occurred",
        )
        with (
            patch.object(engine, "_resolve_executable", return_value="dummy_tesseract"),
            patch("subprocess.run", return_value=mock_proc),
        ):
            with pytest.raises(OcrPageError) as exc_info:
                engine.recognize(b"secret payload")
            assert "secret user text" not in str(exc_info.value)
