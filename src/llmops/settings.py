"""Single configuration entry point. Every environment variable is listed in `.env.example`."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["fake", "openrouter", "anthropic", "openai"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Provider. None = auto: openrouter > anthropic > openai (whichever has a key) > fake.
    llm_provider: Provider | None = None
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_site_url: str | None = None
    openrouter_app_name: str | None = "llm-evalops"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    # Models (OpenRouter slugs; bare ids work with the direct backends).
    llm_model: str = "anthropic/claude-sonnet-4.5"
    judge_model: str = "openai/gpt-4.1-mini"
    cheap_model: str = "anthropic/claude-haiku-4.5"

    # Tracing
    trace_path: str = "traces/spans.jsonl"
    pricing_path: str | None = None
    trace_feature: str = "default"
    trace_user_id: str = "anon"
    otel_export: bool = False

    # Caching
    redis_url: str | None = None
    semantic_cache_threshold: float = 0.92
    embedding_provider: Literal["fake", "local", "openai"] = "fake"
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
        "llm_provider",
        "openrouter_api_key",
        "openrouter_site_url",
        "openrouter_app_name",
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
        """Provider actually used.

        Explicit `LLM_PROVIDER` wins when its key exists (falls back to fake otherwise). When unset,
        pick `openrouter` if `OPENROUTER_API_KEY` is set, else `anthropic`/`openai` by key, else
        `fake`.
        """
        keys = {
            "openrouter": self.openrouter_api_key,
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
        }
        if self.llm_provider == "fake":
            return "fake"
        if self.llm_provider is not None:
            return self.llm_provider if keys.get(self.llm_provider) else "fake"
        for name in ("openrouter", "anthropic", "openai"):
            if keys[name]:
                return name  # type: ignore[return-value]
        return "fake"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Testing helper: forget the cached Settings so env changes are picked up."""
    get_settings.cache_clear()
