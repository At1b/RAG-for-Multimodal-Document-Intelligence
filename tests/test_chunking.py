"""Comprehensive tests for Phase 2 — Normalization and Chunking.

Covers:
- Text cleaning / normalization
- Chunking with various text lengths
- Configurable chunk size and overlap
- Configuration validation
- Metadata preservation
- Page-number tracking
- Unique chunk IDs
- Unicode / non-ASCII text
- Edge cases and pathological inputs
"""

import pytest
from pydantic import ValidationError

from rag.chunking import Chunk, ChunkingConfig, chunk_document, clean_text
from rag.chunking.chunk_id import generate_chunk_id
from rag.ingestion.models import Document, PageContent

# ===================================================================
# Helpers
# ===================================================================


def _make_document(
    pages: list[tuple[int, str]] | None = None,
    *,
    document_id: str = "doc-001",
    document_name: str = "test.pdf",
    source_type: str = "pdf",
) -> Document:
    """Create a test Document from a list of (page_number, content) tuples."""
    if pages is None:
        pages = []
    return Document(
        document_id=document_id,
        document_name=document_name,
        source_type=source_type,
        pages=[PageContent(page_number=pn, content=text) for pn, text in pages],
        metadata={"total_pages": len(pages)},
    )


# ===================================================================
# Text Cleaning Tests
# ===================================================================


class TestCleanText:
    """Tests for the text_cleaning.clean_text function."""

    def test_empty_string(self):
        """Empty input returns empty output."""
        assert clean_text("") == ""

    def test_none_like_empty(self):
        """Falsy empty string returns empty."""
        assert clean_text("") == ""

    def test_plain_text_unchanged(self):
        """Clean text passes through without modification."""
        text = "Hello world. This is a test."
        assert clean_text(text) == text

    def test_leading_trailing_whitespace_stripped(self):
        """Leading and trailing whitespace on the full text is removed."""
        assert clean_text("  hello  ") == "hello"
        assert clean_text("\n\nhello\n\n") == "hello"

    def test_per_line_stripping(self):
        """Leading/trailing whitespace per line is stripped."""
        text = "  line one  \n  line two  "
        assert clean_text(text) == "line one\nline two"

    def test_crlf_normalized(self):
        """Windows line endings are normalized to LF."""
        text = "line1\r\nline2\r\nline3"
        result = clean_text(text)
        assert "\r" not in result
        assert result == "line1\nline2\nline3"

    def test_cr_normalized(self):
        """Old Mac line endings are normalized to LF."""
        text = "line1\rline2\rline3"
        result = clean_text(text)
        assert "\r" not in result
        assert result == "line1\nline2\nline3"

    def test_multiple_blank_lines_collapsed(self):
        """Three or more consecutive newlines collapse to double newline."""
        text = "para1\n\n\n\npara2"
        assert clean_text(text) == "para1\n\npara2"

    def test_double_newline_preserved(self):
        """A single paragraph break (double newline) is preserved."""
        text = "para1\n\npara2"
        assert clean_text(text) == "para1\n\npara2"

    def test_horizontal_space_collapsed(self):
        """Multiple spaces/tabs within a line collapse to one space."""
        text = "hello    world\ttab\t\there"
        result = clean_text(text)
        assert "  " not in result
        assert "\t" not in result
        assert "hello world tab here" == result

    def test_whitespace_only_returns_empty(self):
        """Input containing only whitespace returns empty string."""
        assert clean_text("   \n\n\t\t  \n  ") == ""

    def test_preserves_semantic_content(self):
        """Meaningful punctuation, casing, and structure are preserved."""
        text = "Dr. Smith's report (2024): Revenue grew 25%."
        assert clean_text(text) == text

    def test_unicode_preserved(self):
        """Non-ASCII characters survive cleaning."""
        text = "Ünïcödé — 日本語テスト 🚀"
        assert clean_text(text) == text


# ===================================================================
# ChunkingConfig Validation Tests
# ===================================================================


class TestChunkingConfig:
    """Tests for ChunkingConfig validation."""

    def test_defaults(self):
        """Default config has expected values."""
        cfg = ChunkingConfig()
        assert cfg.chunk_size == 1000
        assert cfg.chunk_overlap == 200

    def test_custom_values(self):
        """Custom config values are accepted."""
        cfg = ChunkingConfig(chunk_size=500, chunk_overlap=50)
        assert cfg.chunk_size == 500
        assert cfg.chunk_overlap == 50

    def test_zero_overlap_allowed(self):
        """Zero overlap is valid."""
        cfg = ChunkingConfig(chunk_size=100, chunk_overlap=0)
        assert cfg.chunk_overlap == 0

    def test_chunk_size_zero_rejected(self):
        """chunk_size=0 raises ValidationError."""
        with pytest.raises(ValidationError, match="chunk_size must be > 0"):
            ChunkingConfig(chunk_size=0)

    def test_chunk_size_negative_rejected(self):
        """Negative chunk_size raises ValidationError."""
        with pytest.raises(ValidationError, match="chunk_size must be > 0"):
            ChunkingConfig(chunk_size=-10)

    def test_overlap_negative_rejected(self):
        """Negative overlap raises ValidationError."""
        with pytest.raises(ValidationError, match="chunk_overlap must be >= 0"):
            ChunkingConfig(chunk_overlap=-1)

    def test_overlap_equals_size_rejected(self):
        """overlap == chunk_size raises ValueError."""
        with pytest.raises(ValidationError, match="chunk_overlap.*must be <"):
            ChunkingConfig(chunk_size=100, chunk_overlap=100)

    def test_overlap_exceeds_size_rejected(self):
        """overlap > chunk_size raises ValueError."""
        with pytest.raises(ValidationError, match="chunk_overlap.*must be <"):
            ChunkingConfig(chunk_size=100, chunk_overlap=200)


# ===================================================================
# Chunk ID Tests
# ===================================================================


class TestChunkId:
    """Tests for chunk ID generation."""

    def test_unique_ids(self):
        """Generated chunk IDs are unique."""
        ids = {generate_chunk_id() for _ in range(1000)}
        assert len(ids) == 1000

    def test_is_string(self):
        """Chunk ID is a string."""
        cid = generate_chunk_id()
        assert isinstance(cid, str)
        assert len(cid) > 0


# ===================================================================
# Chunk Model Tests
# ===================================================================


class TestChunkModel:
    """Tests for the Chunk Pydantic model."""

    def test_creation(self):
        """Chunk can be created with all fields."""
        chunk = Chunk(
            chunk_id="c-1",
            document_id="d-1",
            document_name="test.pdf",
            source_type="pdf",
            content="hello world",
            page_number=1,
            chunk_index=0,
            metadata={"char_count": 11},
        )
        assert chunk.chunk_id == "c-1"
        assert chunk.content == "hello world"
        assert chunk.page_number == 1
        assert chunk.chunk_index == 0

    def test_page_number_optional(self):
        """page_number defaults to None."""
        chunk = Chunk(
            chunk_id="c-1",
            document_id="d-1",
            document_name="test.pdf",
            source_type="pdf",
            content="text",
            chunk_index=0,
        )
        assert chunk.page_number is None

    def test_negative_chunk_index_rejected(self):
        """chunk_index < 0 is rejected."""
        with pytest.raises(ValidationError):
            Chunk(
                chunk_id="c-1",
                document_id="d-1",
                document_name="test.pdf",
                source_type="pdf",
                content="text",
                chunk_index=-1,
            )


# ===================================================================
# chunk_document — Normal Cases
# ===================================================================


class TestChunkDocumentNormal:
    """Tests for chunk_document with normal inputs."""

    def test_single_page_short_text(self):
        """Text shorter than chunk_size produces a single chunk."""
        doc = _make_document([(1, "Short text.")])
        chunks = chunk_document(doc, ChunkingConfig(chunk_size=100, chunk_overlap=0))
        assert len(chunks) == 1
        assert chunks[0].content == "Short text."

    def test_single_page_exact_size(self):
        """Text exactly chunk_size characters produces one chunk."""
        text = "a" * 100
        doc = _make_document([(1, text)])
        chunks = chunk_document(doc, ChunkingConfig(chunk_size=100, chunk_overlap=0))
        assert len(chunks) == 1
        assert chunks[0].content == text

    def test_multi_chunk_no_overlap(self):
        """Text longer than chunk_size splits into multiple chunks (no overlap)."""
        text = "a" * 250
        doc = _make_document([(1, text)])
        chunks = chunk_document(doc, ChunkingConfig(chunk_size=100, chunk_overlap=0))
        assert len(chunks) == 3
        assert chunks[0].content == "a" * 100
        assert chunks[1].content == "a" * 100
        assert chunks[2].content == "a" * 50

    def test_multi_chunk_with_overlap(self):
        """Overlap produces expected overlapping content."""
        text = "abcdefghij" * 10  # 100 chars
        doc = _make_document([(1, text)])
        config = ChunkingConfig(chunk_size=30, chunk_overlap=10)
        chunks = chunk_document(doc, config)

        # Verify overlap: end of chunk[i] should overlap with start of chunk[i+1]
        for i in range(len(chunks) - 1):
            overlap_from_prev = chunks[i].content[-10:]
            overlap_from_next = chunks[i + 1].content[:10]
            assert overlap_from_prev == overlap_from_next, (
                f"Chunk {i} and {i + 1} overlap mismatch"
            )

    def test_chunk_order_preserved(self):
        """Chunks are in text order (chunk_index is sequential)."""
        doc = _make_document([(1, "a" * 300)])
        chunks = chunk_document(doc, ChunkingConfig(chunk_size=100, chunk_overlap=0))
        indices = [c.chunk_index for c in chunks]
        assert indices == [0, 1, 2]

    def test_deterministic_content(self):
        """Same input produces same chunk content (IDs may differ)."""
        doc = _make_document([(1, "Deterministic test content.")])
        config = ChunkingConfig(chunk_size=100, chunk_overlap=0)
        chunks_a = chunk_document(doc, config)
        chunks_b = chunk_document(doc, config)
        assert [c.content for c in chunks_a] == [c.content for c in chunks_b]


# ===================================================================
# chunk_document — Edge Cases
# ===================================================================


class TestChunkDocumentEdgeCases:
    """Tests for chunk_document with edge-case inputs."""

    def test_empty_document_no_pages(self):
        """Document with no pages produces zero chunks."""
        doc = _make_document([])
        chunks = chunk_document(doc)
        assert chunks == []

    def test_empty_page_content(self):
        """Document with empty page content produces zero chunks."""
        doc = _make_document([(1, "")])
        chunks = chunk_document(doc)
        assert chunks == []

    def test_whitespace_only_document(self):
        """Document with whitespace-only content produces zero chunks."""
        doc = _make_document([(1, "   \n\n\t\t  \n  ")])
        chunks = chunk_document(doc)
        assert chunks == []

    def test_very_long_text(self):
        """Very long text is chunked without error or infinite loop."""
        # 100,000 characters
        text = "word " * 20000
        doc = _make_document([(1, text)])
        config = ChunkingConfig(chunk_size=1000, chunk_overlap=200)
        chunks = chunk_document(doc, config)
        assert len(chunks) > 1
        # All text is covered
        total_unique_chars = sum(len(c.content) for c in chunks)
        assert total_unique_chars >= len(text.strip())

    def test_text_just_over_chunk_size(self):
        """Text one character over chunk_size produces two chunks."""
        text = "a" * 101
        doc = _make_document([(1, text)])
        chunks = chunk_document(doc, ChunkingConfig(chunk_size=100, chunk_overlap=0))
        assert len(chunks) == 2
        assert chunks[0].content == "a" * 100
        assert chunks[1].content == "a"

    def test_minimal_chunk_size(self):
        """chunk_size=1 with overlap=0 works without infinite loop."""
        doc = _make_document([(1, "abc")])
        config = ChunkingConfig(chunk_size=1, chunk_overlap=0)
        chunks = chunk_document(doc, config)
        assert len(chunks) == 3
        assert [c.content for c in chunks] == ["a", "b", "c"]

    def test_large_overlap(self):
        """Large overlap (close to chunk_size) works correctly."""
        text = "a" * 100
        doc = _make_document([(1, text)])
        # step = 10 - 9 = 1
        config = ChunkingConfig(chunk_size=10, chunk_overlap=9)
        chunks = chunk_document(doc, config)
        assert len(chunks) > 1
        # Should not hang — step is 1, so we advance one char at a time
        # 91 full chunks of size 10, then trailing
        assert len(chunks) == 91


# ===================================================================
# chunk_document — Metadata Preservation
# ===================================================================


class TestChunkDocumentMetadata:
    """Tests for metadata preservation through chunking."""

    def test_document_id_preserved(self):
        """All chunks carry the source document_id."""
        doc = _make_document([(1, "hello world")], document_id="unique-doc-42")
        chunks = chunk_document(doc)
        for chunk in chunks:
            assert chunk.document_id == "unique-doc-42"

    def test_document_name_preserved(self):
        """All chunks carry the source document_name."""
        doc = _make_document(
            [(1, "content here")], document_name="Annual Report (2024).pdf"
        )
        chunks = chunk_document(doc)
        for chunk in chunks:
            assert chunk.document_name == "Annual Report (2024).pdf"

    def test_source_type_preserved(self):
        """All chunks carry the source_type."""
        doc = _make_document([(1, "content")], source_type="docx")
        chunks = chunk_document(doc)
        for chunk in chunks:
            assert chunk.source_type == "docx"

    def test_chunk_metadata_contains_config(self):
        """Each chunk's metadata records the chunking config used."""
        doc = _make_document([(1, "some text content")])
        config = ChunkingConfig(chunk_size=500, chunk_overlap=50)
        chunks = chunk_document(doc, config)
        for chunk in chunks:
            assert chunk.metadata["chunk_size"] == 500
            assert chunk.metadata["chunk_overlap"] == 50
            assert "char_count" in chunk.metadata

    def test_all_chunk_ids_unique(self):
        """Every chunk produced from a document has a unique ID."""
        text = "word " * 500
        doc = _make_document([(1, text)])
        config = ChunkingConfig(chunk_size=100, chunk_overlap=20)
        chunks = chunk_document(doc, config)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"

    def test_chunk_ids_unique_across_documents(self):
        """Chunk IDs are unique across different documents."""
        doc_a = _make_document([(1, "doc A content")], document_id="doc-a")
        doc_b = _make_document([(1, "doc B content")], document_id="doc-b")
        chunks_a = chunk_document(doc_a)
        chunks_b = chunk_document(doc_b)
        all_ids = [c.chunk_id for c in chunks_a + chunks_b]
        assert len(all_ids) == len(set(all_ids))


# ===================================================================
# chunk_document — Page Number Preservation
# ===================================================================


class TestChunkDocumentPageNumbers:
    """Tests for page-number assignment through chunking."""

    def test_single_page_number(self):
        """A single-page document assigns that page number to all chunks."""
        doc = _make_document([(5, "a" * 300)])
        chunks = chunk_document(doc, ChunkingConfig(chunk_size=100, chunk_overlap=0))
        for chunk in chunks:
            assert chunk.page_number == 5

    def test_multi_page_first_chunk_first_page(self):
        """First chunk gets the first page's number."""
        doc = _make_document([(1, "a" * 100), (2, "b" * 100)])
        chunks = chunk_document(doc, ChunkingConfig(chunk_size=50, chunk_overlap=0))
        assert chunks[0].page_number == 1

    def test_multi_page_later_chunks_later_pages(self):
        """Chunks from later pages get higher page numbers."""
        doc = _make_document(
            [
                (1, "a" * 200),
                (2, "b" * 200),
                (3, "c" * 200),
            ]
        )
        config = ChunkingConfig(chunk_size=100, chunk_overlap=0)
        chunks = chunk_document(doc, config)

        # Early chunks should come from page 1, later from page 2/3
        first_page = chunks[0].page_number
        last_page = chunks[-1].page_number
        assert first_page is not None
        assert last_page is not None
        assert first_page <= last_page

    def test_no_pages_gives_none(self):
        """Document with no pages produces no chunks (and no page errors)."""
        doc = _make_document([])
        chunks = chunk_document(doc)
        assert chunks == []


# ===================================================================
# chunk_document — Multiple Documents
# ===================================================================


class TestChunkDocumentMultiple:
    """Tests for chunking multiple documents independently."""

    def test_independent_documents(self):
        """Two documents chunked independently produce correct metadata."""
        doc_a = _make_document(
            [(1, "Doc A page one."), (2, "Doc A page two.")],
            document_id="id-a",
            document_name="report_a.pdf",
        )
        doc_b = _make_document(
            [(1, "Doc B content.")],
            document_id="id-b",
            document_name="report_b.docx",
            source_type="docx",
        )

        chunks_a = chunk_document(doc_a)
        chunks_b = chunk_document(doc_b)

        for c in chunks_a:
            assert c.document_id == "id-a"
            assert c.document_name == "report_a.pdf"
            assert c.source_type == "pdf"

        for c in chunks_b:
            assert c.document_id == "id-b"
            assert c.document_name == "report_b.docx"
            assert c.source_type == "docx"


# ===================================================================
# chunk_document — Unicode / Non-ASCII
# ===================================================================


class TestChunkDocumentUnicode:
    """Tests for Unicode and non-ASCII text handling."""

    def test_accented_characters(self):
        """Accented characters are preserved in chunks."""
        text = "Ünïcödé tëxt wïth àccénts" * 5
        doc = _make_document([(1, text)])
        chunks = chunk_document(doc, ChunkingConfig(chunk_size=50, chunk_overlap=0))
        reconstructed = "".join(c.content for c in chunks)
        assert reconstructed == text

    def test_cjk_characters(self):
        """CJK characters are handled correctly."""
        text = "日本語テスト文章です。" * 10
        doc = _make_document([(1, text)])
        chunks = chunk_document(doc, ChunkingConfig(chunk_size=30, chunk_overlap=0))
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.content  # Non-empty

    def test_emoji(self):
        """Emoji characters are preserved."""
        text = "Hello 🌍🚀 World 🎉" * 10
        doc = _make_document([(1, text)])
        chunks = chunk_document(doc, ChunkingConfig(chunk_size=40, chunk_overlap=0))
        all_content = "".join(c.content for c in chunks)
        assert "🌍" in all_content
        assert "🚀" in all_content

    def test_mixed_scripts(self):
        """Mixed Latin, Arabic, Cyrillic scripts are preserved."""
        text = "English العربية Русский 中文"
        doc = _make_document([(1, text)])
        chunks = chunk_document(doc, ChunkingConfig(chunk_size=1000, chunk_overlap=0))
        assert len(chunks) == 1
        assert "العربية" in chunks[0].content
        assert "Русский" in chunks[0].content


# ===================================================================
# chunk_document — Configurable Parameters
# ===================================================================


class TestChunkDocumentConfigurable:
    """Tests for configurable chunk_size and chunk_overlap."""

    def test_small_chunk_size(self):
        """Small chunk_size produces more chunks."""
        doc = _make_document([(1, "a" * 100)])
        small = chunk_document(doc, ChunkingConfig(chunk_size=10, chunk_overlap=0))
        large = chunk_document(doc, ChunkingConfig(chunk_size=50, chunk_overlap=0))
        assert len(small) > len(large)

    def test_larger_overlap_produces_more_chunks(self):
        """Larger overlap produces more chunks (smaller step)."""
        doc = _make_document([(1, "a" * 100)])
        no_overlap = chunk_document(doc, ChunkingConfig(chunk_size=20, chunk_overlap=0))
        with_overlap = chunk_document(
            doc, ChunkingConfig(chunk_size=20, chunk_overlap=10)
        )
        assert len(with_overlap) > len(no_overlap)

    def test_default_config_when_none(self):
        """Passing config=None uses default ChunkingConfig."""
        doc = _make_document([(1, "a" * 50)])
        chunks = chunk_document(doc, None)
        assert len(chunks) >= 1
        assert chunks[0].metadata["chunk_size"] == 1000
        assert chunks[0].metadata["chunk_overlap"] == 200


# ===================================================================
# Integration — Document from Phase 1 model
# ===================================================================


class TestIntegrationWithPhase1:
    """Tests that chunking integrates cleanly with Phase 1 Document model."""

    def test_phase1_document_accepted(self):
        """A Document constructed like Phase 1 loaders produce is accepted."""
        doc = Document(
            document_id="real-uuid-here",
            document_name="Annual_Report_2024.pdf",
            source_type="pdf",
            pages=[
                PageContent(
                    page_number=1,
                    content="Revenue grew by 25% in FY2024.",
                    metadata={"char_count": 30},
                ),
                PageContent(
                    page_number=2,
                    content="Operating costs decreased by 10%.",
                    metadata={"char_count": 33},
                ),
            ],
            metadata={"total_pages": 2, "file_size_bytes": 102400},
        )
        chunks = chunk_document(doc)
        assert len(chunks) >= 1
        assert chunks[0].document_id == "real-uuid-here"
        assert chunks[0].document_name == "Annual_Report_2024.pdf"
        assert chunks[0].source_type == "pdf"

    def test_docx_single_page_convention(self):
        """DOCX single-page convention (page_number=1) is preserved."""
        doc = Document(
            document_id="docx-id",
            document_name="proposal.docx",
            source_type="docx",
            pages=[
                PageContent(
                    page_number=1,
                    content="All DOCX content in one page. " * 20,
                ),
            ],
        )
        chunks = chunk_document(doc, ChunkingConfig(chunk_size=100, chunk_overlap=20))
        assert len(chunks) >= 1
        # All chunks should reference page 1 since DOCX has no page boundaries
        for chunk in chunks:
            assert chunk.page_number == 1
