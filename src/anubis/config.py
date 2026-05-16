"""Application configuration via Pydantic Settings.

Reads from environment variables and/or a ``.env`` file.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global application settings loaded from environment variables."""

    app_name: str = "Anubis Assistant"
    debug: bool = False
    log_level: str = "INFO"

    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_default_model: str = "gpt-4o"
    llm_timeout: float = 120.0

    host: str = "0.0.0.0"  # noqa: S104
    port: int = 8000

    model_config = {"env_prefix": "ANUBIS_", "env_file": ".env", "extra": "ignore"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached instance of the application settings."""
    return Settings()
