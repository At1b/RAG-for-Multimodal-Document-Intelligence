"""Abstract generator interface.

All generator implementations must subclass ``Generator`` so the rest
of the application depends on the abstraction rather than a specific
LLM provider or runtime.

Replaceable: swap the concrete class — no downstream code changes required.
Future LLM backends (OpenAI, Anthropic, local transformers) can implement
this same interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag.generation.models import GenerationResult
from rag.vectorstore.models import VectorSearchResult


class Generator(ABC):
    """Base class for LLM generation backends.

    Subclasses must implement ``generate`` to produce an answer from
    a user question and retrieved context.

    The generator must NOT call the retriever.  It receives already-
    retrieved context and is responsible only for prompt construction
    and LLM invocation.
    """

    @abstractmethod
    def generate(
        self,
        question: str,
        context: list[VectorSearchResult],
    ) -> GenerationResult:
        """Generate an answer from a question and retrieved context.

        Args:
            question: The user's natural-language question.
            context: Retrieved document chunks from the retrieval layer.
                Must not be empty.

        Returns:
            A ``GenerationResult`` containing the generated answer and
            optional model metadata.

        Raises:
            InvalidQuestionError: If the question is invalid.
            InvalidContextError: If the context is empty or invalid.
            ModelInitializationError: If the LLM runtime fails to load.
            ModelGenerationError: If the LLM fails during generation.
        """
