"""Single configuration entry point. Every environment variable is listed in `.env.example`."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["fake", "anthropic", "openai"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Provider
    llm_provider: Provider = "fake"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    # Models
    llm_model: str = "claude-sonnet-4-5"
    judge_model: str = "gpt-4.1"
    cheap_model: str = "claude-haiku-4-5"

    # Tracing
    trace_path: str = "traces/spans.jsonl"
    pricing_path: str | None = None
    trace_feature: str = "default"
    trace_user_id: str = "anon"
    otel_export: bool = False

    # Caching
    redis_url: str | None = None
    semantic_cache_threshold: float = 0.92
    embedding_provider: Literal["fake", "openai"] = "fake"
    embedding_model: str = "text-embedding-3-small"

    # Alerts
    slack_webhook_url: str | None = None
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password: str | None = None
    alert_email_from: str | None = None
    alert_email_to: str | None = None
    alerts_dir: str = "alerts"

    # Fake provider knobs
    fake_judge_thresholds: dict[str, float] = Field(default_factory=dict)

    @field_validator("fake_judge_thresholds", mode="before")
    @classmethod
    def _parse_thresholds(cls, v: object) -> object:
        if v is None or v == "":
            return {}
        if isinstance(v, str):
            return json.loads(v)
        return v

    @field_validator(
        "pricing_path",
        "redis_url",
        "anthropic_api_key",
        "openai_api_key",
        "slack_webhook_url",
        "smtp_host",
        "smtp_user",
        "smtp_password",
        "alert_email_from",
        "alert_email_to",
        mode="before",
    )
    @classmethod
    def _empty_to_none(cls, v: object) -> object:
        return None if v == "" else v

    def resolved_provider(self) -> Provider:
        """Provider actually used: falls back to `fake` when the chosen provider has no key."""
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            return "fake"
        if self.llm_provider == "openai" and not self.openai_api_key:
            return "fake"
        return self.llm_provider


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Testing helper: forget the cached Settings so env changes are picked up."""
    get_settings.cache_clear()
