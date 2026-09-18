"""Generation-specific exceptions.

Each exception maps to a distinct failure mode in the generation pipeline.
Follows the same pattern as ``rag.ingestion.exceptions`` and
``rag.retrieval.exceptions``.
"""


class GenerationError(Exception):
    """Base exception for all generation-related errors."""


class InvalidQuestionError(GenerationError):
    """The question is invalid (empty, whitespace-only, wrong type)."""


class InvalidContextError(GenerationError):
    """The retrieval context is invalid or empty.

    Empty retrieval context must be treated as invalid generation context.
    The LLM must NOT be called when context is empty.
    """


class ModelInitializationError(GenerationError):
    """The LLM model or runtime failed to initialize."""


class ModelGenerationError(GenerationError):
    """The LLM failed during answer generation."""


class GenerationConfigError(GenerationError, ValueError):
    """The generation configuration is invalid."""
