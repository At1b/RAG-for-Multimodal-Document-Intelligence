"""Tests for Phase 5 — LLM Generation.

Covers:
    - Generator ABC interface contract
    - Prompt builder
    - Context builder
    - OllamaGenerator adapter (mocked)
    - Integration (mock LLM)
    - Exception hierarchy
    - Configuration validation
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from backend.config import Settings
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
from rag.generation.prompt import SYSTEM_PROMPT, build_prompt
from rag.vectorstore.models import VectorSearchResult

# =====================================================================
# Helpers
# =====================================================================


def _make_result(
    *,
    chunk_id: str = "chunk-1",
    document_id: str = "doc-1",
    document_name: str = "report.pdf",
    content: str = "Revenue increased by 25%.",
    score: float = 0.95,
    page_number: int | None = 5,
    metadata: dict | None = None,
) -> VectorSearchResult:
    """Create a VectorSearchResult for testing."""
    meta = metadata or {}
    if page_number is not None:
        meta.setdefault("page_number", page_number)
    return VectorSearchResult(
        chunk_id=chunk_id,
        document_id=document_id,
        document_name=document_name,
        content=content,
        score=score,
        metadata=meta,
    )


class ConcreteGenerator(Generator):
    """Minimal concrete Generator for ABC tests."""

    def generate(
        self,
        question: str,
        context: list[VectorSearchResult],
    ) -> GenerationResult:
        return GenerationResult(answer="test answer", model_name="test")


# =====================================================================
# Generator Interface Tests
# =====================================================================


class TestGeneratorInterface:
    """Test the Generator ABC contract."""

    def test_cannot_instantiate_abc(self):
        """Generator ABC cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Generator()  # type: ignore[abstract]

    def test_concrete_subclass_conforms(self):
        """A concrete subclass can be instantiated."""
        gen = ConcreteGenerator()
        assert isinstance(gen, Generator)

    def test_generate_method_exists(self):
        """Concrete subclass has a generate method."""
        gen = ConcreteGenerator()
        result = gen.generate("question", [_make_result()])
        assert isinstance(result, GenerationResult)
        assert result.answer == "test answer"


# =====================================================================
# Prompt Builder Tests
# =====================================================================


class TestPromptBuilder:
    """Test the prompt builder."""

    def test_question_included_in_prompt(self):
        """The question appears in the user message."""
        messages = build_prompt("What is revenue?", "Some context here.")
        user_msg = messages[1]["content"]
        assert "What is revenue?" in user_msg

    def test_context_included_in_prompt(self):
        """The context appears in the user message."""
        messages = build_prompt("Question?", "Context about revenue.")
        user_msg = messages[1]["content"]
        assert "Context about revenue." in user_msg

    def test_instructions_included(self):
        """System message contains grounding instructions."""
        messages = build_prompt("Q?", "C.")
        system_msg = messages[0]["content"]
        assert "ONLY" in system_msg
        assert "context" in system_msg.lower()

    def test_system_role_present(self):
        """Messages include a system role."""
        messages = build_prompt("Q?", "C.")
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_two_messages_returned(self):
        """Prompt returns exactly two messages (system + user)."""
        messages = build_prompt("Q?", "C.")
        assert len(messages) == 2

    def test_no_hardcoded_answers(self):
        """The prompt template does not contain pre-baked answers."""
        messages = build_prompt("What is 2+2?", "Math context.")
        full_text = " ".join(m["content"] for m in messages)
        # Should not contain a direct answer
        assert "the answer is" not in full_text.lower()

    def test_instructions_prevent_override(self):
        """System prompt instructs the model to ignore embedded instructions."""
        lower_prompt = SYSTEM_PROMPT.lower()
        assert "ignore" in lower_prompt or "Ignore" in SYSTEM_PROMPT
        assert "override" in lower_prompt or "change your role" in lower_prompt

    def test_context_label_present(self):
        """User message labels the context section."""
        messages = build_prompt("Q?", "Some context.")
        user_msg = messages[1]["content"]
        assert "CONTEXT:" in user_msg

    def test_question_label_present(self):
        """User message labels the question section."""
        messages = build_prompt("Q?", "Some context.")
        user_msg = messages[1]["content"]
        assert "QUESTION:" in user_msg

    def test_insufficient_context_instruction(self):
        """System prompt instructs to state when context is insufficient."""
        lower_prompt = SYSTEM_PROMPT.lower()
        assert "insufficient" in lower_prompt or "not contain" in lower_prompt


# =====================================================================
# Context Builder Tests
# =====================================================================


class TestContextBuilder:
    """Test the context builder."""

    def test_content_included(self):
        """Chunk content appears in the built context."""
        result = _make_result(content="Revenue grew by 25%.")
        context = build_context([result])
        assert "Revenue grew by 25%." in context

    def test_document_name_preserved(self):
        """Document name appears in the context."""
        result = _make_result(document_name="annual_report.pdf")
        context = build_context([result])
        assert "annual_report.pdf" in context

    def test_page_number_present(self):
        """Page number appears when available."""
        result = _make_result(page_number=42)
        context = build_context([result])
        assert "42" in context

    def test_page_number_none(self):
        """Context works when page number is None."""
        result = _make_result(page_number=None, metadata={})
        context = build_context([result])
        assert "Page:" not in context

    def test_chunk_id_included(self):
        """Chunk ID appears in the context."""
        result = _make_result(chunk_id="chunk-abc-123")
        context = build_context([result])
        assert "chunk-abc-123" in context

    def test_multiple_documents(self):
        """Context includes chunks from multiple documents."""
        r1 = _make_result(document_name="doc_a.pdf", content="Content A.")
        r2 = _make_result(
            chunk_id="chunk-2",
            document_name="doc_b.pdf",
            content="Content B.",
        )
        context = build_context([r1, r2], max_chars=5000)
        assert "doc_a.pdf" in context
        assert "doc_b.pdf" in context
        assert "Content A." in context
        assert "Content B." in context

    def test_empty_results_raises(self):
        """Empty results raise InvalidContextError."""
        with pytest.raises(InvalidContextError, match="empty"):
            build_context([])

    def test_none_results_raises(self):
        """None results raise InvalidContextError."""
        with pytest.raises(InvalidContextError):
            build_context(None)  # type: ignore[arg-type]

    def test_ordering_preserved(self):
        """Chunks appear in the order they were provided."""
        r1 = _make_result(content="FIRST chunk content.")
        r2 = _make_result(chunk_id="c2", content="SECOND chunk content.")
        r3 = _make_result(chunk_id="c3", content="THIRD chunk content.")
        context = build_context([r1, r2, r3], max_chars=5000)
        pos1 = context.index("FIRST")
        pos2 = context.index("SECOND")
        pos3 = context.index("THIRD")
        assert pos1 < pos2 < pos3

    def test_context_limit_truncates_later_chunks(self):
        """When limit is reached, later chunks are omitted entirely."""
        short = _make_result(content="Short.")
        long = _make_result(
            chunk_id="c2",
            content="X" * 500,
        )
        # Set max_chars so only the first chunk fits.
        context = build_context([short, long], max_chars=200)
        assert "Short." in context
        assert "X" * 500 not in context

    def test_context_limit_keeps_first_chunk(self):
        """First chunk is always included even if it exceeds max_chars."""
        big = _make_result(content="Y" * 500)
        context = build_context([big], max_chars=100)
        assert "Y" * 500 in context

    def test_max_chars_minimum(self):
        """max_chars below 100 raises InvalidContextError."""
        with pytest.raises(InvalidContextError, match="max_chars"):
            build_context([_make_result()], max_chars=50)


# =====================================================================
# OllamaGenerator Tests (Mocked)
# =====================================================================


class TestOllamaGenerator:
    """Test the OllamaGenerator with mocked Ollama SDK."""

    def _make_generator(self, **kwargs) -> OllamaGenerator:
        """Create an OllamaGenerator with defaults."""
        defaults = {
            "model": "tinyllama",
            "base_url": "http://localhost:11434",
            "temperature": 0.1,
            "max_tokens": 512,
            "context_max_chars": 3000,
        }
        defaults.update(kwargs)
        return OllamaGenerator(**defaults)

    def test_successful_generation(self):
        """Mocked successful generation returns a GenerationResult."""
        gen = self._make_generator()

        mock_client = MagicMock()
        mock_client.chat.return_value = {
            "message": {"content": "Revenue increased by 25%."}
        }
        gen._client = mock_client

        result = gen.generate("What happened to revenue?", [_make_result()])
        assert isinstance(result, GenerationResult)
        assert result.answer == "Revenue increased by 25%."
        assert result.model_name == "tinyllama"

    def test_model_initialization_failure(self):
        """When ollama import fails, raises ModelInitializationError."""
        gen = self._make_generator()
        gen._client = None  # Force re-init

        with patch.dict("sys.modules", {"ollama": None}):
            with pytest.raises(ModelInitializationError, match="not installed"):
                gen.generate("Q?", [_make_result()])

    def test_generation_failure(self):
        """When Ollama chat raises, ModelGenerationError is raised."""
        gen = self._make_generator()

        mock_client = MagicMock()
        mock_client.chat.side_effect = ConnectionError("Server down")
        gen._client = mock_client

        with pytest.raises(ModelGenerationError, match="failed"):
            gen.generate("Q?", [_make_result()])

    def test_empty_response(self):
        """When Ollama returns empty content, ModelGenerationError is raised."""
        gen = self._make_generator()

        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": ""}}
        gen._client = mock_client

        with pytest.raises(ModelGenerationError, match="empty"):
            gen.generate("Q?", [_make_result()])

    def test_invalid_question_rejected(self):
        """Empty question raises InvalidQuestionError."""
        gen = self._make_generator()
        with pytest.raises(InvalidQuestionError):
            gen.generate("", [_make_result()])

    def test_whitespace_question_rejected(self):
        """Whitespace-only question raises InvalidQuestionError."""
        gen = self._make_generator()
        with pytest.raises(InvalidQuestionError):
            gen.generate("   \n\t  ", [_make_result()])

    def test_non_string_question_rejected(self):
        """Non-string question raises InvalidQuestionError."""
        gen = self._make_generator()
        with pytest.raises(InvalidQuestionError):
            gen.generate(123, [_make_result()])  # type: ignore[arg-type]

    def test_empty_context_raises_without_llm_call(self):
        """Empty context raises InvalidContextError; LLM is never called."""
        gen = self._make_generator()
        mock_client = MagicMock()
        gen._client = mock_client

        with pytest.raises(InvalidContextError):
            gen.generate("Q?", [])

        # LLM should NOT have been called.
        mock_client.chat.assert_not_called()

    def test_result_metadata_populated(self):
        """GenerationResult includes model metadata."""
        gen = self._make_generator()

        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": "Answer here."}}
        gen._client = mock_client

        result = gen.generate("Q?", [_make_result()])
        assert result.metadata["temperature"] == 0.1
        assert result.metadata["max_tokens"] == 512

    def test_invalid_model_name(self):
        """Empty model name raises ValueError."""
        with pytest.raises(ValueError, match="model"):
            OllamaGenerator(model="")

    def test_invalid_temperature(self):
        """Temperature out of range raises ValueError."""
        with pytest.raises(ValueError, match="temperature"):
            OllamaGenerator(temperature=3.0)

    def test_invalid_max_tokens(self):
        """max_tokens < 1 raises ValueError."""
        with pytest.raises(ValueError, match="max_tokens"):
            OllamaGenerator(max_tokens=0)

    def test_invalid_context_max_chars(self):
        """context_max_chars < 100 raises ValueError."""
        with pytest.raises(ValueError, match="context_max_chars"):
            OllamaGenerator(context_max_chars=50)

    def test_invalid_base_url(self):
        """Empty base_url raises ValueError."""
        with pytest.raises(ValueError, match="base_url"):
            OllamaGenerator(base_url="")


# =====================================================================
# Integration Tests (Mock LLM)
# =====================================================================


class TestGenerationIntegration:
    """Integration tests connecting context builder → prompt builder → generator."""

    def test_full_pipeline_mock(self):
        """Question + retrieval results → context → prompt → generator (mocked)."""
        # Simulate retrieval results from Phase 4.
        results = [
            _make_result(
                document_name="report_2023.pdf",
                content="Revenue was $100M in 2023.",
                page_number=10,
            ),
            _make_result(
                chunk_id="chunk-2",
                document_name="report_2024.pdf",
                content="Revenue was $125M in 2024.",
                page_number=12,
            ),
        ]

        gen = OllamaGenerator()
        mock_client = MagicMock()
        mock_client.chat.return_value = {
            "message": {"content": "Revenue increased from $100M to $125M."}
        }
        gen._client = mock_client

        result = gen.generate(
            "How did revenue change between 2023 and 2024?",
            results,
        )

        assert isinstance(result, GenerationResult)
        assert "125M" in result.answer

        # Verify the LLM was called with properly structured messages.
        call_args = mock_client.chat.call_args
        messages = call_args.kwargs.get("messages")
        assert messages is not None
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "report_2023.pdf" in messages[1]["content"]
        assert "report_2024.pdf" in messages[1]["content"]

    def test_empty_context_blocks_generation(self):
        """Empty retrieval context raises InvalidContextError before LLM call."""
        gen = OllamaGenerator()
        mock_client = MagicMock()
        gen._client = mock_client

        with pytest.raises(InvalidContextError):
            gen.generate("What is revenue?", [])

        mock_client.chat.assert_not_called()


# =====================================================================
# Exception Hierarchy Tests
# =====================================================================


class TestExceptionHierarchy:
    """Test exception subclass relationships."""

    def test_all_inherit_from_generation_error(self):
        """All generation exceptions inherit from GenerationError."""
        assert issubclass(InvalidQuestionError, GenerationError)
        assert issubclass(InvalidContextError, GenerationError)
        assert issubclass(ModelInitializationError, GenerationError)
        assert issubclass(ModelGenerationError, GenerationError)
        assert issubclass(GenerationConfigError, GenerationError)

    def test_generation_error_is_exception(self):
        """GenerationError inherits from Exception."""
        assert issubclass(GenerationError, Exception)

    def test_exception_chaining(self):
        """Exception chaining preserves the original cause."""
        original = ConnectionError("connection refused")
        wrapped = ModelGenerationError("LLM failed")
        wrapped.__cause__ = original
        assert wrapped.__cause__ is original


# =====================================================================
# Configuration Tests
# =====================================================================


class TestPhase5Configuration:
    """Test Phase 5 configuration validation."""

    def test_defaults(self):
        """Default Phase 5 settings are valid."""
        settings = Settings(
            _env_file=None,
            llm_model="tinyllama",
            llm_base_url="http://localhost:11434",
        )
        assert settings.llm_model == "tinyllama"
        assert settings.llm_base_url == "http://localhost:11434"
        assert settings.llm_temperature == 0.1
        assert settings.llm_max_tokens == 512
        assert settings.llm_context_max_chars == 3000

    def test_custom_values(self):
        """Custom Phase 5 settings are accepted."""
        settings = Settings(
            _env_file=None,
            llm_model="llama3",
            llm_base_url="https://my-server.com:8080",
            llm_temperature=0.7,
            llm_max_tokens=1024,
            llm_context_max_chars=5000,
        )
        assert settings.llm_model == "llama3"
        assert settings.llm_base_url == "https://my-server.com:8080"
        assert settings.llm_temperature == 0.7
        assert settings.llm_max_tokens == 1024
        assert settings.llm_context_max_chars == 5000

    def test_invalid_base_url_rejected(self):
        """Non-HTTP URL for llm_base_url is rejected."""
        with pytest.raises(ValidationError, match="HTTP"):
            Settings(
                _env_file=None,
                llm_base_url="ftp://server.com",
            )

    def test_empty_base_url_rejected(self):
        """Empty llm_base_url is rejected."""
        with pytest.raises(ValidationError, match="non-empty"):
            Settings(
                _env_file=None,
                llm_base_url="",
            )

    def test_invalid_temperature_rejected(self):
        """Temperature > 2.0 is rejected."""
        with pytest.raises(ValidationError, match="temperature"):
            Settings(
                _env_file=None,
                llm_temperature=3.0,
            )

    def test_negative_temperature_rejected(self):
        """Negative temperature is rejected."""
        with pytest.raises(ValidationError, match="temperature"):
            Settings(
                _env_file=None,
                llm_temperature=-0.1,
            )

    def test_invalid_max_tokens_rejected(self):
        """max_tokens < 1 is rejected."""
        with pytest.raises(ValidationError, match="max_tokens"):
            Settings(
                _env_file=None,
                llm_max_tokens=0,
            )

    def test_invalid_context_max_chars_rejected(self):
        """context_max_chars < 100 is rejected."""
        with pytest.raises(ValidationError, match="context_max_chars"):
            Settings(
                _env_file=None,
                llm_context_max_chars=50,
            )

    def test_empty_model_rejected(self):
        """Empty llm_model is rejected."""
        with pytest.raises(ValidationError, match="non-empty"):
            Settings(
                _env_file=None,
                llm_model="",
            )
