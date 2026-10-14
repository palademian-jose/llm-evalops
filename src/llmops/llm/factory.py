from __future__ import annotations

import logging

from llmops.llm.fake import FakeClient, FakeConfig
from llmops.llm.types import LLMClient
from llmops.settings import Settings, get_settings

log = logging.getLogger("llmops.llm")


def get_client(settings: Settings | None = None, *, role: str = "generator") -> LLMClient:
    """Build the LLM client selected by `LLM_PROVIDER`. Falls back to `fake` when no key is set.

    `role` picks the default model: "generator" -> LLM_MODEL, "judge" -> JUDGE_MODEL,
    "cheap" -> CHEAP_MODEL.
    """
    s = settings or get_settings()
    models = {"generator": s.llm_model, "judge": s.judge_model, "cheap": s.cheap_model}
    model = models.get(role, s.llm_model)
    provider = s.resolved_provider()
    if provider != s.llm_provider:
        log.warning("LLM_PROVIDER=%s has no API key; using the fake provider", s.llm_provider)
    if provider == "anthropic":
        from llmops.llm.anthropic import AnthropicClient

        return AnthropicClient(api_key=s.anthropic_api_key or "", default_model=model)
    if provider == "openai":
        from llmops.llm.openai import OpenAIClient

        return OpenAIClient(api_key=s.openai_api_key or "", default_model=model)
    if s.llm_provider == "fake":
        log.info("Using the fake LLM provider (LLM_PROVIDER=fake)")
    return FakeClient(FakeConfig(judge_thresholds=dict(s.fake_judge_thresholds), model=model))
