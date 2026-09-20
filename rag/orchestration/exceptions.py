"""Orchestration-level exceptions.

These exceptions wrap module-level errors at the application boundary
to provide a clean API for callers of the orchestration services.

Each exception preserves the underlying cause via ``from exc`` chaining.
"""


class OrchestrationError(Exception):
    """Base exception for all orchestration-related errors."""


class DocumentIndexingError(OrchestrationError):
    """A failure occurred during the document indexing pipeline.

    Wraps ingestion, chunking, or indexing failures with the
    underlying cause preserved.
    """


class QueryError(OrchestrationError):
    """A failure occurred during the RAG query pipeline.

    Wraps retrieval or generation failures with the underlying
    cause preserved.
    """


class EmptyRetrievalError(QueryError):
    """Retrieval returned no usable context for the question.

    The LLM must NOT be called when this error is raised.
    No fallback answers should be fabricated.
    """


class InsufficientContextError(EmptyRetrievalError):
    """Retrieved context is insufficient or below the relevance threshold.

    Subclasses ``EmptyRetrievalError`` to maintain backward compatibility
    with callers catching ``EmptyRetrievalError``.
    The LLM must NOT be called when this error is raised.
    No fallback answers should be fabricated.
    """
