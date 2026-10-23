"""Tests for Phase 10 configuration validation (backend/config.py)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.config import Settings


class TestPhase10ConfigDefaults:
    """Verify Phase 10 configuration defaults and customization."""

    def test_phase10_defaults(self):
        s = Settings()
        assert s.ocr_enabled is True
        assert s.ocr_language == "eng"
        assert s.ocr_dpi == 300
        assert s.ocr_min_native_chars == 25
        assert s.ocr_timeout_seconds == 60.0
        assert s.ocr_max_pages == 50
        assert s.ocr_max_image_pixels == 40_000_000

    def test_custom_phase10_settings(self):
        s = Settings(
            ocr_enabled=False,
            ocr_language="fra",
            ocr_dpi=200,
            ocr_min_native_chars=50,
            ocr_timeout_seconds=30.0,
            ocr_max_pages=10,
            ocr_max_image_pixels=20_000_000,
        )
        assert s.ocr_enabled is False
        assert s.ocr_language == "fra"
        assert s.ocr_dpi == 200
        assert s.ocr_min_native_chars == 50
        assert s.ocr_timeout_seconds == 30.0
        assert s.ocr_max_pages == 10
        assert s.ocr_max_image_pixels == 20_000_000

    def test_env_aliases(self, monkeypatch):
        monkeypatch.setenv("OCR_ENABLED", "false")
        monkeypatch.setenv("OCR_LANGUAGE", "eng+deu")
        monkeypatch.setenv("OCR_DPI", "200")
        monkeypatch.setenv("OCR_MIN_NATIVE_CHARS", "30")
        monkeypatch.setenv("OCR_TIMEOUT_SECONDS", "45.5")
        monkeypatch.setenv("OCR_MAX_PAGES", "25")
        monkeypatch.setenv("OCR_MAX_IMAGE_PIXELS", "10000000")

        s = Settings()
        assert s.ocr_enabled is False
        assert s.ocr_language == "eng+deu"
        assert s.ocr_dpi == 200
        assert s.ocr_min_native_chars == 30
        assert s.ocr_timeout_seconds == 45.5
        assert s.ocr_max_pages == 25
        assert s.ocr_max_image_pixels == 10_000_000


class TestOcrLanguageValidation:
    """Verify validation on ocr_language."""

    def test_valid_single_language(self):
        s = Settings(ocr_language="eng")
        assert s.ocr_language == "eng"

    def test_valid_compound_languages(self):
        s = Settings(ocr_language="eng+fra+spa")
        assert s.ocr_language == "eng+fra+spa"

    def test_valid_language_with_variant(self):
        s = Settings(ocr_language="chi_sim+chi_tra")
        assert s.ocr_language == "chi_sim+chi_tra"

    def test_empty_language_raises(self):
        with pytest.raises(ValidationError, match="ocr_language"):
            Settings(ocr_language="")

    def test_whitespace_language_raises(self):
        with pytest.raises(ValidationError, match="ocr_language"):
            Settings(ocr_language="   ")

    def test_injection_semicolon_raises(self):
        with pytest.raises(ValidationError, match="ocr_language"):
            Settings(ocr_language="eng;rm -rf /")

    def test_injection_flags_raises(self):
        with pytest.raises(ValidationError, match="ocr_language"):
            Settings(ocr_language="eng --psm 0")

    def test_invalid_length_raises(self):
        with pytest.raises(ValidationError, match="ocr_language"):
            Settings(ocr_language="en")


class TestOcrDpiValidation:
    """Verify bounds for ocr_dpi."""

    def test_min_dpi(self):
        s = Settings(ocr_dpi=72)
        assert s.ocr_dpi == 72

    def test_max_dpi(self):
        s = Settings(ocr_dpi=600)
        assert s.ocr_dpi == 600

    def test_dpi_below_min_raises(self):
        with pytest.raises(ValidationError, match="ocr_dpi"):
            Settings(ocr_dpi=71)

    def test_dpi_above_max_raises(self):
        with pytest.raises(ValidationError, match="ocr_dpi"):
            Settings(ocr_dpi=601)

    def test_dpi_bool_raises(self):
        with pytest.raises(ValidationError, match="ocr_dpi"):
            Settings(ocr_dpi=True)


class TestOcrMinNativeCharsValidation:
    """Verify bounds for ocr_min_native_chars."""

    def test_min_value(self):
        s = Settings(ocr_min_native_chars=1)
        assert s.ocr_min_native_chars == 1

    def test_max_value(self):
        s = Settings(ocr_min_native_chars=10_000)
        assert s.ocr_min_native_chars == 10_000

    def test_zero_raises(self):
        with pytest.raises(ValidationError, match="ocr_min_native_chars"):
            Settings(ocr_min_native_chars=0)

    def test_above_max_raises(self):
        with pytest.raises(ValidationError, match="ocr_min_native_chars"):
            Settings(ocr_min_native_chars=10_001)


class TestOcrTimeoutSecondsValidation:
    """Verify bounds for ocr_timeout_seconds."""

    def test_valid_float(self):
        s = Settings(ocr_timeout_seconds=0.5)
        assert s.ocr_timeout_seconds == 0.5

    def test_zero_or_negative_raises(self):
        with pytest.raises(ValidationError, match="ocr_timeout_seconds"):
            Settings(ocr_timeout_seconds=0.0)
        with pytest.raises(ValidationError, match="ocr_timeout_seconds"):
            Settings(ocr_timeout_seconds=-1.0)

    def test_above_max_raises(self):
        with pytest.raises(ValidationError, match="ocr_timeout_seconds"):
            Settings(ocr_timeout_seconds=600.1)


class TestOcrMaxPagesValidation:
    """Verify bounds for ocr_max_pages."""

    def test_min_value(self):
        s = Settings(ocr_max_pages=1)
        assert s.ocr_max_pages == 1

    def test_max_value(self):
        s = Settings(ocr_max_pages=1000)
        assert s.ocr_max_pages == 1000

    def test_zero_raises(self):
        with pytest.raises(ValidationError, match="ocr_max_pages"):
            Settings(ocr_max_pages=0)

    def test_above_max_raises(self):
        with pytest.raises(ValidationError, match="ocr_max_pages"):
            Settings(ocr_max_pages=1001)


class TestOcrMaxImagePixelsValidation:
    """Verify bounds for ocr_max_image_pixels."""

    def test_min_value(self):
        s = Settings(ocr_max_image_pixels=1_000_000)
        assert s.ocr_max_image_pixels == 1_000_000

    def test_max_value(self):
        s = Settings(ocr_max_image_pixels=200_000_000)
        assert s.ocr_max_image_pixels == 200_000_000

    def test_below_min_raises(self):
        with pytest.raises(ValidationError, match="ocr_max_image_pixels"):
            Settings(ocr_max_image_pixels=999_999)

    def test_above_max_raises(self):
        with pytest.raises(ValidationError, match="ocr_max_image_pixels"):
            Settings(ocr_max_image_pixels=200_000_001)
