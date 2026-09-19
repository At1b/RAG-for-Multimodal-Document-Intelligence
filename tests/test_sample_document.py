"""Real sample document integration tests — Phase 6B.

Uses the sample PDF fixture (tests/fixtures/sample_ai_overview.pdf) to
verify the complete pipeline from ingestion through retrieval:

    1. Document is ingested
    2. Chunks are created
    3. Embeddings are generated
    4. Vectors are stored
    5. Query retrieves relevant content
    6. LLM generates an answer (skipped if Ollama unavailable)

The full LLM generation test is marked to skip when Ollama is unavailable,
while retrieval/indexing tests run without Ollama.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from rag.chunking.chunker import ChunkingConfig
from rag.embeddings.indexing import IndexingService
from rag.embeddings.sentence_transformer import SentenceTransformerEmbeddingService
from rag.ingestion.service import IngestionService
from rag.orchestration.indexing_service import DocumentIndexingService
from rag.retrieval.semantic import SemanticRetriever
from rag.vectorstore.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)

SAMPLE_PDF = Path(__file__).parent / "fixtures" / "sample_ai_overview.pdf"


def _ollama_available() -> bool:
    """Check if the Ollama server is reachable."""
    try:
        import ollama as ollama_sdk

        client = ollama_sdk.Client(host="http://localhost:11434", timeout=5.0)
        client.list()
        return True
    except Exception:
        return False


@pytest.fixture
def indexed_sample(tmp_path):
    """Index the sample PDF into a temporary vector store.

    Returns a tuple of (indexing_result, settings_dict) for use by tests.
    """
    persist_dir = str(tmp_path / "vectorstore")
    collection_name = "test_sample"

    ingestion_service = IngestionService(max_size_bytes=50 * 1024 * 1024)
    chunking_config = ChunkingConfig(chunk_size=500, chunk_overlap=100)
    embedding_service = SentenceTransformerEmbeddingService(
        model_name="all-MiniLM-L6-v2"
    )
    vector_store = ChromaVectorStore(
        persist_directory=persist_dir,
        collection_name=collection_name,
    )
    indexing_service = IndexingService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )
    doc_indexing_service = DocumentIndexingService(
        ingestion_service=ingestion_service,
        chunking_config=chunking_config,
        indexing_service=indexing_service,
    )

    result = doc_indexing_service.index_document(
        file_path=SAMPLE_PDF,
        document_name="sample_ai_overview.pdf",
    )

    return {
        "result": result,
        "vector_store": vector_store,
        "embedding_service": embedding_service,
        "persist_dir": persist_dir,
        "collection_name": collection_name,
    }


# ===========================================================================
# Step 1: Document ingestion
# ===========================================================================


class TestSampleIngestion:
    """Verify the sample PDF can be ingested."""

    def test_sample_pdf_exists(self):
        """The sample PDF fixture exists."""
        assert SAMPLE_PDF.exists(), f"Sample PDF not found: {SAMPLE_PDF}"

    def test_sample_ingested_successfully(self, indexed_sample):
        """The sample PDF is ingested without errors."""
        result = indexed_sample["result"]
        assert result.document_name == "sample_ai_overview.pdf"
        assert result.document_id  # non-empty

    def test_sample_has_pages(self, indexed_sample):
        """The ingested document has at least one page."""
        result = indexed_sample["result"]
        assert result.num_pages >= 1


# ===========================================================================
# Step 2: Chunks created
# ===========================================================================


class TestSampleChunking:
    """Verify chunks are created from the sample document."""

    def test_chunks_created(self, indexed_sample):
        """The sample document produces at least one chunk."""
        result = indexed_sample["result"]
        assert result.num_chunks >= 1

    def test_multiple_chunks_for_multipar_document(self, indexed_sample):
        """A multi-paragraph document should produce multiple chunks."""
        result = indexed_sample["result"]
        # The sample has 7 paragraphs; with chunk_size=500, expect multiple chunks.
        assert result.num_chunks >= 2


# ===========================================================================
# Step 3 & 4: Embeddings generated and vectors stored
# ===========================================================================


class TestSampleEmbeddingsAndStorage:
    """Verify embeddings are generated and stored in the vector store."""

    def test_vectors_stored(self, indexed_sample):
        """Vector store contains embeddings after indexing."""
        store = indexed_sample["vector_store"]
        assert store.count() >= 1

    def test_vector_count_matches_chunks(self, indexed_sample):
        """Number of vectors matches the number of chunks."""
        result = indexed_sample["result"]
        store = indexed_sample["vector_store"]
        assert store.count() == result.num_chunks


# ===========================================================================
# Step 5: Query retrieves relevant content
# ===========================================================================


class TestSampleRetrieval:
    """Verify retrieval returns relevant content from the sample."""

    def test_rag_query_retrieves_content(self, indexed_sample):
        """A relevant question retrieves content about RAG."""
        store = indexed_sample["vector_store"]
        emb_service = indexed_sample["embedding_service"]

        retriever = SemanticRetriever(
            embedding_service=emb_service,
            vector_store=store,
            default_top_k=5,
        )
        results = retriever.retrieve("What is Retrieval-Augmented Generation?")
        assert len(results) >= 1
        # The top result should contain RAG-related content.
        top_content = results[0].content.lower()
        assert any(
            keyword in top_content
            for keyword in ["retrieval", "generation", "rag", "knowledge"]
        )

    def test_transformer_query_retrieves_content(self, indexed_sample):
        """A question about Transformers retrieves relevant content."""
        store = indexed_sample["vector_store"]
        emb_service = indexed_sample["embedding_service"]

        retriever = SemanticRetriever(
            embedding_service=emb_service,
            vector_store=store,
            default_top_k=5,
        )
        results = retriever.retrieve("What is the Transformer architecture?")
        assert len(results) >= 1
        top_content = results[0].content.lower()
        assert any(
            keyword in top_content
            for keyword in ["transformer", "attention", "vaswani"]
        )

    def test_embedding_query_retrieves_content(self, indexed_sample):
        """A question about embeddings retrieves relevant content."""
        store = indexed_sample["vector_store"]
        emb_service = indexed_sample["embedding_service"]

        retriever = SemanticRetriever(
            embedding_service=emb_service,
            vector_store=store,
            default_top_k=5,
        )
        results = retriever.retrieve("What are vector embeddings?")
        assert len(results) >= 1
        top_content = results[0].content.lower()
        assert any(
            keyword in top_content
            for keyword in ["embedding", "vector", "representation", "numerical"]
        )

    def test_results_have_metadata(self, indexed_sample):
        """Retrieved results include expected metadata fields."""
        store = indexed_sample["vector_store"]
        emb_service = indexed_sample["embedding_service"]

        retriever = SemanticRetriever(
            embedding_service=emb_service,
            vector_store=store,
            default_top_k=3,
        )
        results = retriever.retrieve("What is deep learning?")
        assert len(results) >= 1
        for result in results:
            assert result.document_id
            assert result.document_name == "sample_ai_overview.pdf"
            assert result.content
            assert isinstance(result.score, float)


# ===========================================================================
# Step 6: LLM generates an answer (requires Ollama)
# ===========================================================================


class TestSampleLLMGeneration:
    """Full LLM generation tests using the real Ollama runtime.

    These tests are skipped when Ollama is not available (e.g. in CI).
    """

    @pytest.mark.skipif(
        not _ollama_available(),
        reason="Ollama server not available — skipping live LLM test",
    )
    def test_real_llm_generates_answer(self, indexed_sample):
        """The real LLM generates an answer from the sample document."""
        from rag.generation.ollama_generator import OllamaGenerator
        from rag.orchestration.query_service import RAGQueryService

        store = indexed_sample["vector_store"]
        emb_service = indexed_sample["embedding_service"]

        retriever = SemanticRetriever(
            embedding_service=emb_service,
            vector_store=store,
            default_top_k=3,
        )
        generator = OllamaGenerator(
            model="tinyllama",
            temperature=0.1,
            max_tokens=256,
            timeout=120.0,
        )
        query_service = RAGQueryService(
            retriever=retriever,
            generator=generator,
        )

        result = query_service.query("What is RAG and how does it work?")
        assert result.answer
        assert len(result.answer) > 10
        assert result.model_name == "tinyllama"
        assert result.num_chunks_retrieved >= 1
