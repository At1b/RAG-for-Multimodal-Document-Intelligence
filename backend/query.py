"""Query API router.

Thin route handler: accepts a question, delegates to RAGQueryService,
maps exceptions to HTTP responses.
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
from rag.retrieval.exceptions import InvalidQueryError
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


class QueryResponse(BaseModel):
    """Response body for the /query endpoint."""

    answer: str
    model_name: str
    num_chunks_retrieved: int


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


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Ask a question and receive a document-grounded answer.

    The endpoint retrieves relevant document chunks from the
    vector store and generates an answer using the configured LLM.
    """
    settings = get_settings()

    # Construct service dependencies from settings.
    embedding_service = SentenceTransformerEmbeddingService(
        model_name=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
    )
    vector_store = ChromaVectorStore(
        persist_directory=settings.vector_store_path,
        collection_name=settings.vector_store_collection,
    )
    retriever = SemanticRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
        default_top_k=settings.vector_search_top_k,
    )
    generator = OllamaGenerator(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        context_max_chars=settings.llm_context_max_chars,
        timeout=settings.llm_timeout,
    )

    query_service = RAGQueryService(
        retriever=retriever,
        generator=generator,
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
    )
