"""Semantic cache: cosine similarity over embeddings of previous prompts.

Correctness risks (documented in the README): a near-duplicate prompt can differ in exactly
the token that matters ("2024" vs "2025", "not"), so the threshold must be tuned per feature
against a labeled set and the cache must be scoped by model, tools and any user/tenant context.
Spans record the matched similarity so false hits can be audited.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from llmops.cache.embeddings import Embedder, HashedBoWEmbedder
from llmops.llm.types import LLMResponse
from llmops.pricing import estimate_cost
from llmops.tracing.tracer import Tracer, get_tracer


class SemanticCache:
    cache_type = "semantic"

    def __init__(
        self,
        embedder: Embedder | None = None,
        *,
        threshold: float = 0.92,
        max_entries: int = 2048,
        tracer: Tracer | None = None,
    ):
        self.embedder = embedder or HashedBoWEmbedder()
        self.threshold = threshold
        self.max_entries = max_entries
        self._tracer = tracer
        self._vectors: np.ndarray | None = None
        self._entries: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _embed(self, text: str) -> np.ndarray:
        tracer = self._tracer or get_tracer()
        with tracer.span("cache.embed", "other", model=self.embedder.model) as span:
            vec = self.embedder.embed([text])[0]
            tokens = max(1, len(text) // 4)
            span.input_tokens = tokens
            span.cost_usd = estimate_cost(self.embedder.model, tokens, 0) or 0.0
        return vec

    def lookup(self, text: str, *, scope: str = "") -> LLMResponse | None:
        tracer = self._tracer or get_tracer()
        with tracer.span("cache.semantic", "cache", meta={"cache_type": self.cache_type}) as span:
            vec = self._embed(text)
            best, best_sim = None, -1.0
            with self._lock:
                if self._vectors is not None and len(self._entries):
                    sims = self._vectors @ vec
                    for idx in np.argsort(-sims):
                        if self._entries[idx]["scope"] != scope:
                            continue
                        best, best_sim = self._entries[idx], float(sims[idx])
                        break
            span.meta["similarity"] = round(best_sim, 4) if best is not None else None
            span.meta["threshold"] = self.threshold
            if best is None or best_sim < self.threshold:
                self.misses += 1
                span.cache_hit = False
                return None
            self.hits += 1
            span.cache_hit = True
            span.model = best.get("model")
            span.meta["saved_cost_usd"] = best.get("cost_usd", 0.0)
            span.meta["saved_ms"] = best.get("latency_ms", 0.0)
            span.meta["matched_text"] = best["text"][:120]
            return LLMResponse.model_validate(best["response"])

    def store(
        self,
        text: str,
        response: LLMResponse,
        *,
        cost_usd: float,
        latency_ms: float,
        scope: str = "",
        vector: np.ndarray | None = None,
    ) -> None:
        vec = vector if vector is not None else self.embedder.embed([text])[0]
        entry = {
            "text": text,
            "scope": scope,
            "response": response.model_dump(mode="json"),
            "cost_usd": cost_usd,
            "latency_ms": latency_ms,
            "model": response.model,
        }
        with self._lock:
            self._entries.append(entry)
            self._vectors = (
                vec[None, :] if self._vectors is None else np.vstack([self._vectors, vec])
            )
            if len(self._entries) > self.max_entries:
                self._entries.pop(0)
                self._vectors = self._vectors[1:]

    def stats(self) -> dict[str, Any]:
        n = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / n if n else 0.0,
            "size": len(self._entries),
        }
