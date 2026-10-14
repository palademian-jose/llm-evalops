"""`CachedClient`: exact cache -> semantic cache -> traced provider call."""

from __future__ import annotations

import time

from llmops.cache.exact import ExactCache, request_key
from llmops.cache.semantic import SemanticCache
from llmops.llm.types import LLMClient, LLMResponse, Message, ToolSpec
from llmops.pricing import estimate_cost
from llmops.tracing.tracer import Tracer
from llmops.tracing.wrap import wrap_client


def prompt_text(messages: list[Message]) -> str:
    """Text used for semantic matching: system + last user message."""
    system = " ".join(m.text() for m in messages if m.role == "system")
    user = next((m.text() for m in reversed(messages) if m.role == "user"), "")
    return f"{system}\n{user}".strip()


class CachedClient:
    def __init__(
        self,
        client: LLMClient,
        *,
        exact: ExactCache | None = None,
        semantic: SemanticCache | None = None,
        tracer: Tracer | None = None,
        feature: str | None = None,
    ):
        self._client = wrap_client(client, tracer, feature=feature)
        self.exact = exact
        self.semantic = semantic

    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        key = request_key(
            messages, model=model, tools=tools, temperature=temperature, max_tokens=max_tokens
        )
        if self.exact is not None:
            hit = self.exact.lookup(key)
            if hit is not None:
                return hit
        text = prompt_text(messages)
        scope = f"{model}|{'tools' if tools else 'notools'}"
        if self.semantic is not None and not tools:
            hit = self.semantic.lookup(text, scope=scope)
            if hit is not None:
                return hit
        start = time.perf_counter()
        resp = self._client.complete(
            messages, tools=tools, model=model, temperature=temperature, max_tokens=max_tokens
        )
        latency = resp.latency_ms or (time.perf_counter() - start) * 1000
        cost = resp.cost_usd
        if cost is None:
            cost = (
                estimate_cost(resp.model, resp.usage.input_tokens, resp.usage.output_tokens) or 0.0
            )
        if self.exact is not None:
            self.exact.store(key, resp, cost_usd=cost, latency_ms=latency)
        if self.semantic is not None and not tools:
            self.semantic.store(text, resp, cost_usd=cost, latency_ms=latency, scope=scope)
        return resp
