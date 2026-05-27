"""
Central configuration — all settings via env vars / .env file.
Pydantic v2 BaseSettings; import `settings` everywhere.
"""
from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM providers ────────────────────────────────────────────────────────
    openai_api_key: str = Field(default="", description="OpenAI API key")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")

    # ── External services ────────────────────────────────────────────────────
    tavily_api_key: str = Field(default="", description="Tavily search API key")
    firecrawl_api_key: str = Field(default="", description="Firecrawl API key")
    cohere_api_key: str = Field(default="", description="Cohere API key for reranking")

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:password@localhost:5432/content_engine"
    )
    redis_url: str = Field(default="redis://localhost:6379/0")

    # ── Observability ─────────────────────────────────────────────────────────
    langsmith_tracing: bool = Field(default=False)
    langsmith_api_key: str = Field(default="")
    langsmith_project: str = Field(default="ai-content-engine")

    # ── Security ──────────────────────────────────────────────────────────────
    jwt_secret_key: str = Field(default="change-me")
    jwt_algorithm: str = Field(default="HS256")
    jwt_expire_minutes: int = Field(default=1440)

    # ── Engine behaviour ──────────────────────────────────────────────────────
    max_revision_iterations: int = Field(default=3)
    max_recursion_limit: int = Field(default=50)
    token_budget_per_run: int = Field(default=80_000)
    node_timeout_seconds: float = Field(default=60.0)
    global_run_timeout_seconds: float = Field(default=600.0)

    # ── Models ────────────────────────────────────────────────────────────────
    primary_model: str = Field(default="anthropic:claude-sonnet-4-6")
    review_model: str = Field(default="openai:gpt-4o")
    fast_model: str = Field(default="anthropic:claude-haiku-4-5-20251001")
    embedding_model: str = Field(default="text-embedding-3-small")

    # ── Publishing ────────────────────────────────────────────────────────────
    wordpress_url: str = Field(default="")
    wordpress_username: str = Field(default="")
    wordpress_app_password: str = Field(default="")

    @field_validator("primary_model", "review_model", "fast_model")
    @classmethod
    def validate_model_format(cls, v: str) -> str:
        if ":" not in v:
            raise ValueError(
                f"Model must be 'provider:model-name', got: {v!r}"
            )
        return v


settings = Settings()
