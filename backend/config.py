"""Application configuration loaded from environment variables."""

from typing import Self

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings


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

    @model_validator(mode="after")
    def _validate_chunk_overlap(self) -> Self:
        """Validate cross-field constraint: chunk_overlap < chunk_size."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be < "
                f"chunk_size ({self.chunk_size})"
            )
        return self


def get_settings() -> Settings:
    """Return application settings instance."""
    return Settings()
