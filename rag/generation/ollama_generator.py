"""Ollama LLM adapter — concrete ``Generator`` implementation.

Uses the ``ollama`` Python SDK to communicate with a locally running
Ollama server.  All Ollama-specific logic is contained in this module;
the rest of the application depends only on the ``Generator`` ABC.

Design decisions:
    - Lazy client initialization (no import-time cost).
    - Connection validated on first ``generate`` call.
    - Configurable model name, temperature, max tokens, base URL.
    - Never returns fabricated fallback answers on failure.
    - Treats retrieved context as untrusted reference data.
"""

from __future__ import annotations

import logging
from typing import Any

from rag.generation.base import Generator
from rag.generation.context_builder import build_context
from rag.generation.exceptions import (
    InvalidQuestionError,
    ModelGenerationError,
    ModelInitializationError,
)
from rag.generation.models import GenerationResult
from rag.generation.prompt import build_prompt
from rag.vectorstore.models import VectorSearchResult

logger = logging.getLogger(__name__)


class OllamaGenerator(Generator):
    """Generate answers using an Ollama-hosted LLM.

    Args:
        model: Ollama model name (e.g. ``"tinyllama"``).
        base_url: Ollama server URL (e.g. ``"http://localhost:11434"``).
        temperature: Sampling temperature (0.0–2.0).
        max_tokens: Maximum number of tokens to generate.
        context_max_chars: Maximum character length for the formatted
            context passed to the LLM.
    """

    def __init__(
        self,
        model: str = "tinyllama",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        max_tokens: int = 512,
        context_max_chars: int = 3000,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
            raise ValueError("temperature must be a number")
        if temperature < 0.0 or temperature > 2.0:
            raise ValueError(
                f"temperature must be between 0.0 and 2.0, got {temperature}"
            )
        if (
            not isinstance(max_tokens, int)
            or isinstance(max_tokens, bool)
            or max_tokens < 1
        ):
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        if (
            not isinstance(context_max_chars, int)
            or isinstance(context_max_chars, bool)
            or context_max_chars < 100
        ):
            raise ValueError(
                f"context_max_chars must be >= 100, got {context_max_chars}"
            )

        self._model = model.strip()
        self._base_url = base_url.strip()
        self._temperature = float(temperature)
        self._max_tokens = max_tokens
        self._context_max_chars = context_max_chars
        self._client: Any | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        question: str,
        context: list[VectorSearchResult],
    ) -> GenerationResult:
        """Generate an answer from a question and retrieved context.

        Steps:
            1. Validate the question.
            2. Build the formatted context (raises ``InvalidContextError``
               if context is empty).
            3. Build the chat prompt with grounding instructions.
            4. Call the Ollama LLM.
            5. Return a ``GenerationResult``.

        Args:
            question: The user's question.
            context: Retrieved chunks from the retrieval layer.

        Returns:
            A ``GenerationResult`` with the generated answer.

        Raises:
            InvalidQuestionError: If the question is invalid.
            InvalidContextError: If context is empty.
            ModelInitializationError: If Ollama is unreachable.
            ModelGenerationError: If the LLM fails.
        """
        self._validate_question(question)

        # Build context — raises InvalidContextError if empty.
        formatted_context = build_context(context, max_chars=self._context_max_chars)

        # Build prompt messages.
        messages = build_prompt(question.strip(), formatted_context)

        # Ensure client is ready.
        client = self._get_client()

        # Call the LLM.
        answer = self._call_llm(client, messages)

        return GenerationResult(
            answer=answer,
            model_name=self._model,
            metadata={
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
                "base_url": self._base_url,
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_question(question: str) -> None:
        """Validate the user question.

        Raises:
            InvalidQuestionError: If the question is invalid.
        """
        if not isinstance(question, str):
            raise InvalidQuestionError(
                f"question must be a string, got {type(question).__name__}"
            )
        cleaned = (
            question.replace("\x00", "")
            .replace("\ufeff", "")
            .replace("\u200b", "")
            .strip()
        )
        if not cleaned:
            raise InvalidQuestionError("question must be a non-empty string")

    def _get_client(self) -> Any:
        """Lazily initialize and return the Ollama client.

        Raises:
            ModelInitializationError: If the ``ollama`` package is not
                installed or the client cannot be created.
        """
        if self._client is not None:
            return self._client

        try:
            import ollama as ollama_sdk

            self._client = ollama_sdk.Client(host=self._base_url)
        except ImportError as exc:
            raise ModelInitializationError(
                "The 'ollama' Python package is required but not installed. "
                "Install it with: pip install ollama"
            ) from exc
        except Exception as exc:
            raise ModelInitializationError(
                f"Failed to initialize Ollama client: {exc}"
            ) from exc

        return self._client

    def _call_llm(self, client: Any, messages: list[dict[str, str]]) -> str:
        """Call the Ollama chat API and return the answer text.

        Args:
            client: Initialized Ollama client.
            messages: Chat messages (system + user).

        Returns:
            The generated answer string.

        Raises:
            ModelGenerationError: If the LLM call fails or returns
                an empty/invalid response.
        """
        try:
            response = client.chat(
                model=self._model,
                messages=messages,
                options={
                    "temperature": self._temperature,
                    "num_predict": self._max_tokens,
                },
            )
        except Exception as exc:
            raise ModelGenerationError(f"Ollama generation failed: {exc}") from exc

        # Extract the answer text from the response.
        try:
            answer = response["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise ModelGenerationError(
                f"Unexpected Ollama response format: {exc}"
            ) from exc

        if not isinstance(answer, str) or not answer.strip():
            raise ModelGenerationError("Ollama returned an empty response")

        return answer.strip()
