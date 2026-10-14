from __future__ import annotations

import logging

from llmops.llm.fake import FakeClient, FakeConfig
from llmops.llm.types import LLMClient
from llmops.settings import Settings, get_settings

log = logging.getLogger("llmops.llm")


def get_client(settings: Settings | None = None, *, role: str = "generator") -> LLMClient:
    """Build the LLM client for `LLM_PROVIDER`.

    Resolution (see `Settings.resolved_provider`): an explicit `LLM_PROVIDER` wins when its key
    is present; when unset, `openrouter` is chosen if `OPENROUTER_API_KEY` exists, then the
    direct `anthropic` / `openai` backends, else `fake` with a one-line warning.

    `role` picks the default model: "generator" -> LLM_MODEL, "judge" -> JUDGE_MODEL,
    "cheap" -> CHEAP_MODEL.
    """
    s = settings or get_settings()
    models = {"generator": s.llm_model, "judge": s.judge_model, "cheap": s.cheap_model}
    model = models.get(role, s.llm_model)
    provider = s.resolved_provider()
    if provider == "fake" and s.llm_provider != "fake":
        wanted = s.llm_provider or "auto"
        log.warning("LLM_PROVIDER=%s has no API key available; using the fake provider", wanted)
    if provider == "openrouter":
        from llmops.llm.openrouter import OpenRouterClient

        return OpenRouterClient(
            api_key=s.openrouter_api_key or "",
            default_model=model,
            base_url=s.openrouter_base_url,
            site_url=s.openrouter_site_url,
            app_name=s.openrouter_app_name,
        )
    if provider == "anthropic":
        from llmops.llm.anthropic import AnthropicClient

        return AnthropicClient(api_key=s.anthropic_api_key or "", default_model=model)
    if provider == "openai":
        from llmops.llm.openai import OpenAIClient

        return OpenAIClient(api_key=s.openai_api_key or "", default_model=model)
    return FakeClient(FakeConfig(judge_thresholds=dict(s.fake_judge_thresholds), model=model))
