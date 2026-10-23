"""Unit tests for TSV parsing and text layout formatting."""

from __future__ import annotations

import pytest

from rag.ingestion.exceptions import OcrPageError
from rag.ingestion.ocr.text_layout import parse_tsv_to_text


def _tsv_header() -> str:
    return (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\t"
        "left\ttop\twidth\theight\tconf\ttext\n"
    )


class TestTsvParsing:
    """Verify TSV parsing behaviors and edge cases."""

    def test_empty_tsv_returns_empty(self):
        text, conf = parse_tsv_to_text("")
        assert text == ""
        assert conf is None

        text, conf = parse_tsv_to_text("   \n\t\n")
        assert text == ""
        assert conf is None

    def test_header_only_returns_empty(self):
        text, conf = parse_tsv_to_text(_tsv_header())
        assert text == ""
        assert conf is None

    def test_single_line_text(self):
        tsv = (
            _tsv_header()
            + "1\t1\t0\t0\t0\t0\t0\t0\t100\t100\t-1\t\n"
            + "5\t1\t1\t1\t1\t1\t10\t10\t20\t10\t95.0\tHello\n"
            + "5\t1\t1\t1\t1\t2\t35\t10\t20\t10\t85.0\tWorld\n"
        )
        text, conf = parse_tsv_to_text(tsv)
        assert text == "Hello World"
        assert conf == 90.0

    def test_multi_line_same_paragraph(self):
        tsv = (
            _tsv_header()
            + "5\t1\t1\t1\t1\t1\t10\t10\t20\t10\t90.0\tLine\n"
            + "5\t1\t1\t1\t1\t2\t35\t10\t20\t10\t90.0\tOne\n"
            + "5\t1\t1\t1\t2\t1\t10\t25\t20\t10\t80.0\tLine\n"
            + "5\t1\t1\t1\t2\t2\t35\t25\t20\t10\t80.0\tTwo\n"
        )
        text, conf = parse_tsv_to_text(tsv)
        assert text == "Line One\nLine Two"
        assert conf == 85.0

    def test_multi_paragraph_and_multi_block(self):
        tsv = (
            _tsv_header()
            + "5\t1\t1\t1\t1\t1\t10\t10\t20\t10\t90.0\tParagraph\n"
            + "5\t1\t1\t1\t1\t2\t35\t10\t20\t10\t90.0\tOne\n"
            + "5\t1\t1\t2\t1\t1\t10\t30\t20\t10\t90.0\tParagraph\n"
            + "5\t1\t1\t2\t1\t2\t35\t30\t20\t10\t90.0\tTwo\n"
            + "5\t1\t2\t1\t1\t1\t10\t60\t20\t10\t90.0\tBlock\n"
            + "5\t1\t2\t1\t1\t2\t35\t60\t20\t10\t90.0\tTwo\n"
        )
        text, conf = parse_tsv_to_text(tsv)
        assert text == "Paragraph One\n\nParagraph Two\n\nBlock Two"
        assert conf == 90.0

    def test_skips_structural_and_negative_conf(self):
        tsv = (
            _tsv_header()
            + "1\t1\t0\t0\t0\t0\t0\t0\t500\t500\t-1\t\n"
            + "2\t1\t1\t0\t0\t0\t0\t0\t500\t500\t-1\t\n"
            + "5\t1\t1\t1\t1\t1\t10\t10\t20\t10\t-1\t\n"
            + "5\t1\t1\t1\t1\t2\t35\t10\t20\t10\t96.0\tValid\n"
        )
        text, conf = parse_tsv_to_text(tsv)
        assert text == "Valid"
        assert conf == 96.0

    def test_skips_whitespace_words(self):
        tsv = (
            _tsv_header()
            + "5\t1\t1\t1\t1\t1\t10\t10\t20\t10\t90.0\t   \n"
            + "5\t1\t1\t1\t1\t2\t35\t10\t20\t10\t80.0\tWord\n"
        )
        text, conf = parse_tsv_to_text(tsv)
        assert text == "Word"
        assert conf == 80.0

    def test_no_valid_words_gives_none_confidence(self):
        tsv = (
            _tsv_header()
            + "1\t1\t0\t0\t0\t0\t0\t0\t100\t100\t-1\t\n"
            + "5\t1\t1\t1\t1\t1\t10\t10\t20\t10\t-1\t\n"
        )
        text, conf = parse_tsv_to_text(tsv)
        assert text == ""
        assert conf is None

    def test_malformed_rows_are_tolerated(self):
        tsv = (
            _tsv_header()
            + "This is a corrupted non-tsv line\n"
            + "5\tnot_an_int\t1\t1\t1\t1\t10\t10\t20\t10\t90.0\tSkipped\n"
            + "5\t1\t1\t1\t1\t1\t10\t10\t20\t10\t88.0\tGood\n"
            + "short\trow\n"
            + "5\t1\t1\t1\t1\t2\t35\t10\t20\t10\t92.0\tWord\n"
        )
        text, conf = parse_tsv_to_text(tsv)
        assert text == "Good Word"
        assert conf == 90.0

    def test_wrong_header_raises_ocr_page_error(self):
        bad_tsv = "colA\tcolB\tcolC\n1\t2\t3\n"
        with pytest.raises(OcrPageError, match="Invalid TSV header"):
            parse_tsv_to_text(bad_tsv)
