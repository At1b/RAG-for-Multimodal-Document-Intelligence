"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """MM-RAG application settings.

    Values are loaded from environment variables or a .env file.
    """

    environment: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


def get_settings() -> Settings:
    """Return application settings instance."""
    return Settings()
