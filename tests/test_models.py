"""Tests for the normalized Document and PageContent models."""

from rag.ingestion.models import Document, PageContent


def test_page_content_creation():
    """PageContent can be created with required fields."""
    page = PageContent(page_number=1, content="Hello world")
    assert page.page_number == 1
    assert page.content == "Hello world"
    assert page.metadata == {}


def test_page_content_empty_content():
    """PageContent accepts empty string content (e.g. blank PDF pages)."""
    page = PageContent(page_number=3, content="")
    assert page.content == ""
    assert page.page_number == 3


def test_page_content_with_metadata():
    """PageContent stores arbitrary per-page metadata."""
    page = PageContent(page_number=1, content="text", metadata={"char_count": 4})
    assert page.metadata["char_count"] == 4


def test_document_creation():
    """Document can be created with all required fields."""
    doc = Document(
        document_id="abc-123",
        document_name="report.pdf",
        source_type="pdf",
        pages=[PageContent(page_number=1, content="Page 1 text")],
        metadata={"total_pages": 1},
    )
    assert doc.document_id == "abc-123"
    assert doc.document_name == "report.pdf"
    assert doc.source_type == "pdf"
    assert len(doc.pages) == 1
    assert doc.metadata["total_pages"] == 1


def test_document_id_preserved():
    """Document ID is preserved as provided."""
    doc = Document(
        document_id="unique-id-42",
        document_name="file.docx",
        source_type="docx",
    )
    assert doc.document_id == "unique-id-42"


def test_document_name_preserved():
    """Document name is preserved exactly as provided."""
    doc = Document(
        document_id="id",
        document_name="My Report (Final).pdf",
        source_type="pdf",
    )
    assert doc.document_name == "My Report (Final).pdf"


def test_document_source_type_preserved():
    """Source type is preserved."""
    doc = Document(
        document_id="id",
        document_name="file.pdf",
        source_type="pdf",
    )
    assert doc.source_type == "pdf"


def test_document_page_numbers_preserved():
    """Page numbers on individual pages are preserved."""
    pages = [
        PageContent(page_number=1, content="first"),
        PageContent(page_number=2, content="second"),
        PageContent(page_number=3, content="third"),
    ]
    doc = Document(
        document_id="id",
        document_name="multi.pdf",
        source_type="pdf",
        pages=pages,
    )
    assert [p.page_number for p in doc.pages] == [1, 2, 3]


def test_document_defaults():
    """Document defaults to empty pages list and metadata dict."""
    doc = Document(
        document_id="id",
        document_name="file.pdf",
        source_type="pdf",
    )
    assert doc.pages == []
    assert doc.metadata == {}
