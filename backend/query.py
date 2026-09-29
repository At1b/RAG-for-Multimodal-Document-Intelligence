"""Query API router.

Thin route handler: accepts a question, delegates to RAGQueryService,
maps exceptions to HTTP responses.

Phase 7: Returns structured source citations from retrieval metadata.
Phase 8: Supports configurable retrieval mode (semantic, keyword, hybrid).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.config import get_settings
from rag.embeddings.sentence_transformer import SentenceTransformerEmbeddingService
from rag.generation.exceptions import InvalidQuestionError
from rag.generation.ollama_generator import OllamaGenerator
from rag.orchestration.exceptions import EmptyRetrievalError, QueryError
from rag.orchestration.query_service import RAGQueryService
from rag.retrieval.base import Retriever
from rag.retrieval.bm25 import BM25Retriever
from rag.retrieval.exceptions import InvalidQueryError
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.index_builder import build_bm25_index_from_chroma
from rag.retrieval.semantic import SemanticRetriever
from rag.vectorstore.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])


class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""

    question: str = Field(..., min_length=1, description="The user's question.")
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=1000,
        description="Maximum number of context chunks to retrieve.",
    )


class SourceResponse(BaseModel):
    """A single source/citation in the query response.

    Phase 7: Structured source built from actual retrieval metadata.
    """

    document_id: str
    document_name: str
    page_number: int | None = None
    chunk_id: str
    score: float


class QueryResponse(BaseModel):
    """Response body for the /query endpoint.

    Phase 7: Includes structured sources/citations from retrieval.
    Phase 8: Includes retrieval_mode indicator.
    """

    answer: str
    model_name: str
    num_chunks_retrieved: int
    sources: list[SourceResponse] = []
    retrieval_mode: str = "semantic"


# Mapping from query/generation exception types to HTTP status codes.
_QUERY_ERROR_STATUS: dict[type, int] = {
    InvalidQueryError: 400,
    InvalidQuestionError: 400,
}


def _resolve_query_error(exc: QueryError) -> HTTPException:
    """Map a QueryError to an appropriate HTTPException.

    The RAGQueryService wraps pipeline failures in QueryError.
    This helper inspects the cause chain to find the underlying
    validation exception and maps it to the correct HTTP status code.
    """
    cause = exc.__cause__
    if cause is not None:
        for error_type, status_code in _QUERY_ERROR_STATUS.items():
            if isinstance(cause, error_type):
                return HTTPException(status_code=status_code, detail=str(cause))

    # Generic query pipeline failure (embedding, vector store, or LLM error).
    logger.error("Query pipeline failed: %s", exc, exc_info=True)
    return HTTPException(
        status_code=500,
        detail="An error occurred while processing the query.",
    )


def _build_retriever(settings) -> tuple[Retriever, str]:
    """Build the appropriate retriever based on settings.retrieval_mode.

    Returns:
        Tuple of (retriever_instance, retrieval_mode_name).
    """
    mode = settings.retrieval_mode

    embedding_service = SentenceTransformerEmbeddingService(
        model_name=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
    )
    vector_store = ChromaVectorStore(
        persist_directory=settings.vector_store_path,
        collection_name=settings.vector_store_collection,
    )

    if mode == "semantic":
        retriever = SemanticRetriever(
            embedding_service=embedding_service,
            vector_store=vector_store,
            default_top_k=settings.vector_search_top_k,
            min_score=settings.retrieval_min_score,
        )
        return retriever, "semantic"

    elif mode == "keyword":
        bm25_index = build_bm25_index_from_chroma(vector_store)
        retriever = BM25Retriever(
            bm25_index=bm25_index,
            default_top_k=settings.vector_search_top_k,
        )
        return retriever, "keyword"

    elif mode == "hybrid":
        semantic_retriever = SemanticRetriever(
            embedding_service=embedding_service,
            vector_store=vector_store,
            default_top_k=settings.hybrid_semantic_top_k,
            min_score=settings.retrieval_min_score,
        )
        bm25_index = build_bm25_index_from_chroma(vector_store)
        keyword_retriever = BM25Retriever(
            bm25_index=bm25_index,
            default_top_k=settings.hybrid_keyword_top_k,
        )
        retriever = HybridRetriever(
            semantic_retriever=semantic_retriever,
            keyword_retriever=keyword_retriever,
            semantic_weight=settings.hybrid_semantic_weight,
            keyword_weight=settings.hybrid_keyword_weight,
            rrf_k=settings.rrf_k,
            default_top_k=settings.vector_search_top_k,
            semantic_top_k=settings.hybrid_semantic_top_k,
            keyword_top_k=settings.hybrid_keyword_top_k,
        )
        return retriever, "hybrid"

    else:
        # Should not happen after config validation, but be safe.
        raise ValueError(f"Unknown retrieval_mode: {mode}")


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Ask a question and receive a document-grounded answer.

    The endpoint retrieves relevant document chunks from the
    vector store and generates an answer using the configured LLM.

    Phase 8: Retrieval mode (semantic/keyword/hybrid) is determined
    by the RETRIEVAL_MODE configuration setting.
    """
    settings = get_settings()

    # Build retriever based on configured mode.
    retriever, retrieval_mode = _build_retriever(settings)

    generator = OllamaGenerator(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        context_max_chars=settings.llm_context_max_chars,
        timeout=settings.llm_timeout,
    )

    # For semantic and hybrid modes, apply the relevance gate at the
    # orchestration level.  For keyword-only mode, the BM25 retriever
    # already returns only positive-scoring results.
    min_score = (
        settings.retrieval_min_score if retrieval_mode in ("semantic",) else None
    )

    query_service = RAGQueryService(
        retriever=retriever,
        generator=generator,
        min_score=min_score,
    )

    try:
        result = query_service.query(
            question=request.question,
            top_k=request.top_k,
        )
    except (InvalidQuestionError, InvalidQueryError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except EmptyRetrievalError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except QueryError as exc:
        raise _resolve_query_error(exc) from exc

    return QueryResponse(
        answer=result.answer,
        model_name=result.model_name,
        num_chunks_retrieved=result.num_chunks_retrieved,
        retrieval_mode=retrieval_mode,
        sources=[
            SourceResponse(
                document_id=s.document_id,
                document_name=s.document_name,
                page_number=s.page_number,
                chunk_id=s.chunk_id,
                score=s.score,
            )
            for s in result.sources
        ],
    )
