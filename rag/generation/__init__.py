"""MM-RAG Generation Layer.

Provides the LLM generation interface and components for Phase 5.

Public API:
    - ``Generator`` — abstract base class for LLM backends.
    - ``OllamaGenerator`` — concrete Ollama adapter.
    - ``GenerationResult`` — result model for generated answers.
    - ``build_context`` — converts retrieval results to context string.
    - ``build_prompt`` — constructs chat messages for the LLM.
    - Exceptions: ``GenerationError``, ``InvalidQuestionError``,
      ``InvalidContextError``, ``ModelInitializationError``,
      ``ModelGenerationError``, ``GenerationConfigError``.
"""

from rag.generation.base import Generator
from rag.generation.context_builder import build_context
from rag.generation.exceptions import (
    GenerationConfigError,
    GenerationError,
    InvalidContextError,
    InvalidQuestionError,
    ModelGenerationError,
    ModelInitializationError,
)
from rag.generation.models import GenerationResult
from rag.generation.ollama_generator import OllamaGenerator
from rag.generation.prompt import build_prompt

__all__ = [
    "Generator",
    "OllamaGenerator",
    "GenerationResult",
    "build_context",
    "build_prompt",
    "GenerationError",
    "InvalidQuestionError",
    "InvalidContextError",
    "ModelInitializationError",
    "ModelGenerationError",
    "GenerationConfigError",
]
