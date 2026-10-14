"""Exact-match response cache keyed on the full request (messages, model, tools, params)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from llmops.cache.backends import Backend, MemoryLRU
from llmops.llm.types import LLMResponse, Message, ToolSpec
from llmops.tracing.tracer import Tracer, get_tracer


def request_key(
    messages: list[Message],
    *,
    model: str | None,
    tools: list[ToolSpec] | None,
    temperature: float,
    max_tokens: int,
) -> str:
    payload = {
        "messages": [m.model_dump(mode="json") for m in messages],
        "model": model,
        "tools": [t.model_dump(mode="json") for t in tools] if tools else None,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


class ExactCache:
    """Stores `{"response": LLMResponse, "cost_usd", "latency_ms"}` under a request hash.

    `lookup` emits one `kind="cache"` span per call with `cache_hit` and, on a hit,
    `meta.saved_cost_usd` / `meta.saved_ms` taken from the original call.
    """

    cache_type = "exact"

    def __init__(
        self,
        backend: Backend | None = None,
        *,
        ttl: int | None = None,
        tracer: Tracer | None = None,
    ):
        # `backend or MemoryLRU()` would discard an empty backend (it defines __len__).
        self.backend = backend if backend is not None else MemoryLRU()
        self.ttl = ttl
        self._tracer = tracer
        self.hits = 0
        self.misses = 0

    def lookup(self, key: str) -> LLMResponse | None:
        tracer = self._tracer or get_tracer()
        with tracer.span("cache.exact", "cache", meta={"cache_type": self.cache_type}) as span:
            entry = self.backend.get(key)
            if entry is None:
                self.misses += 1
                span.cache_hit = False
                return None
            self.hits += 1
            span.cache_hit = True
            span.meta["saved_cost_usd"] = entry.get("cost_usd", 0.0)
            span.meta["saved_ms"] = entry.get("latency_ms", 0.0)
            span.model = entry.get("model")
            return LLMResponse.model_validate(entry["response"])

    def store(self, key: str, response: LLMResponse, *, cost_usd: float, latency_ms: float) -> None:
        self.backend.set(
            key,
            {
                "response": response.model_dump(mode="json"),
                "cost_usd": cost_usd,
                "latency_ms": latency_ms,
                "model": response.model,
            },
            ttl=self.ttl,
        )

    def stats(self) -> dict[str, Any]:
        n = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / n if n else 0.0,
            "size": len(self.backend),
        }
