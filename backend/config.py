"""Application configuration loaded from environment variables."""

from typing import Self
from urllib.parse import urlparse

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings

from rag.generation.ollama_generator import DEFAULT_MODEL


class Settings(BaseSettings):
    """MM-RAG application settings.

    Values are loaded from environment variables or a .env file.
    """

    environment: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    max_upload_size_mb: int = 50
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Phase 3: Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 64

    # Phase 3: Vector Store
    vector_store_path: str = "data/vectorstore"
    vector_store_collection: str = "mmrag_chunks"
    vector_search_top_k: int = 10

    # Phase 5: LLM Generation
    llm_model: str = DEFAULT_MODEL
    llm_base_url: str = "http://localhost:11434"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 512
    llm_timeout: float = Field(
        default=120.0,
        validation_alias=AliasChoices("llm_timeout", "llm_timeout_seconds"),
    )
    llm_context_max_chars: int = Field(
        default=3000,
        validation_alias=AliasChoices(
            "llm_context_max_chars", "llm_context_char_limit"
        ),
    )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @field_validator("max_upload_size_mb")
    @classmethod
    def _max_upload_size_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"max_upload_size_mb must be > 0, got {v}")
        return v

    @field_validator("chunk_size")
    @classmethod
    def _chunk_size_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"chunk_size must be > 0, got {v}")
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"chunk_overlap must be >= 0, got {v}")
        return v

    @field_validator("embedding_model")
    @classmethod
    def _embedding_model_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("embedding_model must be a non-empty string")
        return v.strip()

    @field_validator("embedding_batch_size")
    @classmethod
    def _embedding_batch_size_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"embedding_batch_size must be >= 1, got {v}")
        return v

    @field_validator("vector_store_path")
    @classmethod
    def _vector_store_path_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("vector_store_path must be a non-empty string")
        return v.strip()

    @field_validator("vector_store_collection")
    @classmethod
    def _vector_store_collection_valid(cls, v: str) -> str:
        stripped = v.strip() if v else ""
        if not stripped:
            raise ValueError("vector_store_collection must be a non-empty string")
        if len(stripped) < 3 or len(stripped) > 512:
            raise ValueError(
                "vector_store_collection length must be between 3 and 512 characters"
            )
        if not (stripped[0].isalnum() and stripped[-1].isalnum()):
            raise ValueError(
                "vector_store_collection must start and end with an "
                "alphanumeric character"
            )
        return stripped

    @field_validator("vector_search_top_k")
    @classmethod
    def _vector_search_top_k_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"vector_search_top_k must be >= 1, got {v}")
        if v > 1000:
            raise ValueError(f"vector_search_top_k must be <= 1000, got {v}")
        return v

    @model_validator(mode="after")
    def _validate_chunk_overlap(self) -> Self:
        """Validate cross-field constraint: chunk_overlap < chunk_size."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be < "
                f"chunk_size ({self.chunk_size})"
            )
        return self

    # ------------------------------------------------------------------
    # Phase 5: LLM Generation validators
    # ------------------------------------------------------------------

    @field_validator("llm_model")
    @classmethod
    def _llm_model_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("llm_model must be a non-empty string")
        return v.strip()

    @field_validator("llm_base_url")
    @classmethod
    def _llm_base_url_valid(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("llm_base_url must be a non-empty string")
        stripped = v.strip()
        parsed = urlparse(stripped)
        if parsed.scheme.lower() not in ("http", "https") or not parsed.netloc:
            raise ValueError(
                "llm_base_url must be a valid HTTP or HTTPS URL with a host, "
                f"got '{stripped}'"
            )
        return stripped

    @field_validator("llm_temperature")
    @classmethod
    def _llm_temperature_range(cls, v: float) -> float:
        if v < 0.0 or v > 2.0:
            raise ValueError(f"llm_temperature must be between 0.0 and 2.0, got {v}")
        return v

    @field_validator("llm_max_tokens")
    @classmethod
    def _llm_max_tokens_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"llm_max_tokens must be >= 1, got {v}")
        if v > 32768:
            raise ValueError(f"llm_max_tokens must be <= 32768, got {v}")
        return v

    @field_validator("llm_timeout")
    @classmethod
    def _llm_timeout_range(cls, v: float) -> float:
        if v <= 0.0 or v > 600.0:
            raise ValueError(f"llm_timeout must be between 0 and 600 seconds, got {v}")
        return v

    @field_validator("llm_context_max_chars")
    @classmethod
    def _llm_context_max_chars_minimum(cls, v: int) -> int:
        if v < 100:
            raise ValueError(f"llm_context_max_chars must be >= 100, got {v}")
        if v > 500_000:
            raise ValueError(f"llm_context_max_chars must be <= 500000, got {v}")
        return v


def get_settings() -> Settings:
    """Return application settings instance."""
    return Settings()
