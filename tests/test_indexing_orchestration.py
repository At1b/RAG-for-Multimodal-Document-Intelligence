"""Unit tests for DocumentIndexingService.

Tests the orchestration logic using mocked dependencies —
no real file I/O, embedding, or vector store operations.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag.chunking.chunker import ChunkingConfig
from rag.chunking.models import Chunk
from rag.ingestion.models import Document, PageContent
from rag.orchestration.exceptions import DocumentIndexingError
from rag.orchestration.indexing_service import DocumentIndexingService, IndexingResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_document(
    *,
    document_id: str = "doc-001",
    document_name: str = "test.pdf",
    source_type: str = "pdf",
    pages: list[PageContent] | None = None,
) -> Document:
    """Create a test Document."""
    if pages is None:
        pages = [
            PageContent(page_number=1, content="Hello world. This is test content."),
        ]
    return Document(
        document_id=document_id,
        document_name=document_name,
        source_type=source_type,
        pages=pages,
    )


def _make_chunks(
    document: Document,
    count: int = 3,
) -> list[Chunk]:
    """Create a list of test Chunks."""
    return [
        Chunk(
            chunk_id=f"chunk-{i}",
            document_id=document.document_id,
            document_name=document.document_name,
            source_type=document.source_type,
            content=f"Chunk content {i}",
            page_number=1,
            chunk_index=i,
        )
        for i in range(count)
    ]


@pytest.fixture()
def mock_ingestion_service():
    """Create a mock IngestionService."""
    return MagicMock()


@pytest.fixture()
def mock_indexing_service():
    """Create a mock IndexingService."""
    return MagicMock()


@pytest.fixture()
def chunking_config():
    """Create a test ChunkingConfig."""
    return ChunkingConfig(chunk_size=100, chunk_overlap=20)


@pytest.fixture()
def service(mock_ingestion_service, chunking_config, mock_indexing_service):
    """Create a DocumentIndexingService with mocked dependencies."""
    return DocumentIndexingService(
        ingestion_service=mock_ingestion_service,
        chunking_config=chunking_config,
        indexing_service=mock_indexing_service,
    )


# ---------------------------------------------------------------------------
# Valid document flow
# ---------------------------------------------------------------------------


class TestValidDocumentFlow:
    """Tests for the happy-path indexing flow."""

    def test_index_document_returns_indexing_result(
        self, service, mock_ingestion_service, mock_indexing_service
    ):
        """index_document returns an IndexingResult on success."""
        document = _make_document()
        chunks = _make_chunks(document)

        mock_ingestion_service.ingest.return_value = document
        mock_indexing_service.index_chunks.return_value = len(chunks)

        with patch(
            "rag.orchestration.indexing_service.chunk_document",
            return_value=chunks,
        ):
            result = service.index_document(Path("test.pdf"))

        assert isinstance(result, IndexingResult)
        assert result.document_id == document.document_id
        assert result.document_name == document.document_name
        assert result.num_pages == len(document.pages)
        assert result.num_chunks == len(chunks)

    def test_correct_dependency_call_chain(
        self, service, mock_ingestion_service, mock_indexing_service
    ):
        """Dependencies are called in the correct order: ingest → chunk → index."""
        document = _make_document()
        chunks = _make_chunks(document)

        mock_ingestion_service.ingest.return_value = document
        mock_indexing_service.index_chunks.return_value = len(chunks)

        with patch(
            "rag.orchestration.indexing_service.chunk_document",
            return_value=chunks,
        ) as mock_chunk:
            service.index_document(Path("test.pdf"))

        # Ingest called once with the file path.
        mock_ingestion_service.ingest.assert_called_once()
        call_args = mock_ingestion_service.ingest.call_args
        assert call_args[0][0] == Path("test.pdf")

        # chunk_document called once with the document.
        mock_chunk.assert_called_once()

        # IndexingService.index_chunks called once with the chunks.
        mock_indexing_service.index_chunks.assert_called_once_with(chunks)

    def test_no_duplicated_processing(
        self, service, mock_ingestion_service, mock_indexing_service
    ):
        """Each dependency is called exactly once — no duplicated processing."""
        document = _make_document()
        chunks = _make_chunks(document)

        mock_ingestion_service.ingest.return_value = document
        mock_indexing_service.index_chunks.return_value = len(chunks)

        with patch(
            "rag.orchestration.indexing_service.chunk_document",
            return_value=chunks,
        ) as mock_chunk:
            service.index_document(Path("test.pdf"))

        assert mock_ingestion_service.ingest.call_count == 1
        assert mock_chunk.call_count == 1
        assert mock_indexing_service.index_chunks.call_count == 1


# ---------------------------------------------------------------------------
# Document name handling
# ---------------------------------------------------------------------------


class TestDocumentNameHandling:
    """Tests for document_name parameter forwarding."""

    def test_custom_document_name_forwarded(
        self, service, mock_ingestion_service, mock_indexing_service
    ):
        """A custom document_name is forwarded to IngestionService."""
        document = _make_document(document_name="custom_name.pdf")
        chunks = _make_chunks(document)

        mock_ingestion_service.ingest.return_value = document
        mock_indexing_service.index_chunks.return_value = len(chunks)

        with patch(
            "rag.orchestration.indexing_service.chunk_document",
            return_value=chunks,
        ):
            result = service.index_document(
                Path("test.pdf"), document_name="custom_name.pdf"
            )

        call_kwargs = mock_ingestion_service.ingest.call_args
        assert call_kwargs[1]["original_filename"] == "custom_name.pdf"
        assert result.document_name == "custom_name.pdf"

    def test_none_document_name_uses_filename(
        self, service, mock_ingestion_service, mock_indexing_service
    ):
        """When document_name is None, file_path.name is used."""
        document = _make_document(document_name="report.pdf")
        chunks = _make_chunks(document)

        mock_ingestion_service.ingest.return_value = document
        mock_indexing_service.index_chunks.return_value = len(chunks)

        with patch(
            "rag.orchestration.indexing_service.chunk_document",
            return_value=chunks,
        ):
            service.index_document(Path("report.pdf"))

        call_kwargs = mock_ingestion_service.ingest.call_args
        assert call_kwargs[1]["original_filename"] == "report.pdf"

    def test_string_file_path_converted_to_path(
        self, service, mock_ingestion_service, mock_indexing_service
    ):
        """A string file_path is converted to Path automatically."""
        document = _make_document()
        chunks = _make_chunks(document)

        mock_ingestion_service.ingest.return_value = document
        mock_indexing_service.index_chunks.return_value = len(chunks)

        with patch(
            "rag.orchestration.indexing_service.chunk_document",
            return_value=chunks,
        ):
            result = service.index_document("test.pdf")

        assert isinstance(result, IndexingResult)


# ---------------------------------------------------------------------------
# Ingestion failure
# ---------------------------------------------------------------------------


class TestIngestionFailure:
    """Tests for ingestion-stage failures."""

    def test_ingestion_error_raises_document_indexing_error(
        self, service, mock_ingestion_service
    ):
        """Ingestion failure raises DocumentIndexingError."""
        mock_ingestion_service.ingest.side_effect = RuntimeError("Parse failed")

        with pytest.raises(DocumentIndexingError, match="Ingestion failed"):
            service.index_document(Path("bad.pdf"))

    def test_ingestion_error_preserves_cause(self, service, mock_ingestion_service):
        """The original exception is preserved as __cause__."""
        cause = RuntimeError("Parse failed")
        mock_ingestion_service.ingest.side_effect = cause

        with pytest.raises(DocumentIndexingError) as exc_info:
            service.index_document(Path("bad.pdf"))

        assert exc_info.value.__cause__ is cause

    def test_ingestion_error_does_not_call_chunking_or_indexing(
        self, service, mock_ingestion_service, mock_indexing_service
    ):
        """Chunking and indexing are NOT called when ingestion fails."""
        mock_ingestion_service.ingest.side_effect = RuntimeError("fail")

        with pytest.raises(DocumentIndexingError):
            service.index_document(Path("bad.pdf"))

        mock_indexing_service.index_chunks.assert_not_called()


# ---------------------------------------------------------------------------
# Chunking failure
# ---------------------------------------------------------------------------


class TestChunkingFailure:
    """Tests for chunking-stage failures."""

    def test_chunking_exception_raises_document_indexing_error(
        self, service, mock_ingestion_service, mock_indexing_service
    ):
        """A chunking exception raises DocumentIndexingError."""
        mock_ingestion_service.ingest.return_value = _make_document()

        with patch(
            "rag.orchestration.indexing_service.chunk_document",
            side_effect=ValueError("Bad chunk config"),
        ):
            with pytest.raises(DocumentIndexingError, match="Chunking failed"):
                service.index_document(Path("test.pdf"))

        mock_indexing_service.index_chunks.assert_not_called()

    def test_zero_chunks_raises_document_indexing_error(
        self, service, mock_ingestion_service, mock_indexing_service
    ):
        """Zero chunks produced raises DocumentIndexingError."""
        mock_ingestion_service.ingest.return_value = _make_document()

        with patch(
            "rag.orchestration.indexing_service.chunk_document",
            return_value=[],
        ):
            with pytest.raises(DocumentIndexingError, match="zero chunks"):
                service.index_document(Path("test.pdf"))

        mock_indexing_service.index_chunks.assert_not_called()

    def test_chunking_error_preserves_cause(self, service, mock_ingestion_service):
        """The original chunking exception is preserved as __cause__."""
        cause = ValueError("Bad config")
        mock_ingestion_service.ingest.return_value = _make_document()

        with patch(
            "rag.orchestration.indexing_service.chunk_document",
            side_effect=cause,
        ):
            with pytest.raises(DocumentIndexingError) as exc_info:
                service.index_document(Path("test.pdf"))

        assert exc_info.value.__cause__ is cause


# ---------------------------------------------------------------------------
# Indexing failure
# ---------------------------------------------------------------------------


class TestIndexingFailure:
    """Tests for indexing-stage failures."""

    def test_indexing_error_raises_document_indexing_error(
        self, service, mock_ingestion_service, mock_indexing_service
    ):
        """IndexingService failure raises DocumentIndexingError."""
        document = _make_document()
        chunks = _make_chunks(document)

        mock_ingestion_service.ingest.return_value = document
        mock_indexing_service.index_chunks.side_effect = RuntimeError("Store failed")

        with patch(
            "rag.orchestration.indexing_service.chunk_document",
            return_value=chunks,
        ):
            with pytest.raises(DocumentIndexingError, match="Indexing failed"):
                service.index_document(Path("test.pdf"))

    def test_indexing_error_preserves_cause(
        self, service, mock_ingestion_service, mock_indexing_service
    ):
        """The original indexing exception is preserved as __cause__."""
        document = _make_document()
        chunks = _make_chunks(document)
        cause = RuntimeError("Store failed")

        mock_ingestion_service.ingest.return_value = document
        mock_indexing_service.index_chunks.side_effect = cause

        with patch(
            "rag.orchestration.indexing_service.chunk_document",
            return_value=chunks,
        ):
            with pytest.raises(DocumentIndexingError) as exc_info:
                service.index_document(Path("test.pdf"))

        assert exc_info.value.__cause__ is cause


# ---------------------------------------------------------------------------
# IndexingResult model
# ---------------------------------------------------------------------------


class TestIndexingResultModel:
    """Tests for the IndexingResult data model."""

    def test_indexing_result_fields(self):
        """IndexingResult has the expected fields."""
        result = IndexingResult(
            document_id="doc-1",
            document_name="test.pdf",
            num_pages=5,
            num_chunks=10,
        )
        assert result.document_id == "doc-1"
        assert result.document_name == "test.pdf"
        assert result.num_pages == 5
        assert result.num_chunks == 10

    def test_multi_page_document(
        self, service, mock_ingestion_service, mock_indexing_service
    ):
        """Multi-page documents report correct page count."""
        pages = [
            PageContent(page_number=i, content=f"Page {i} content") for i in range(1, 4)
        ]
        document = _make_document(pages=pages)
        chunks = _make_chunks(document, count=6)

        mock_ingestion_service.ingest.return_value = document
        mock_indexing_service.index_chunks.return_value = 6

        with patch(
            "rag.orchestration.indexing_service.chunk_document",
            return_value=chunks,
        ):
            result = service.index_document(Path("multi_page.pdf"))

        assert result.num_pages == 3
        assert result.num_chunks == 6


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    """Tests for the orchestration exception hierarchy."""

    def test_document_indexing_error_is_orchestration_error(self):
        """DocumentIndexingError is a subclass of OrchestrationError."""
        from rag.orchestration.exceptions import OrchestrationError

        assert issubclass(DocumentIndexingError, OrchestrationError)

    def test_document_indexing_error_is_exception(self):
        """DocumentIndexingError is a subclass of Exception."""
        assert issubclass(DocumentIndexingError, Exception)
