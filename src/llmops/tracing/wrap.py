"""`wrap_client(client)`: auto-instrument any `LLMClient.complete` call.

Each call emits one `kind="llm"` span carrying model, token counts, latency and cost. Cost is
the provider-reported figure when the response has one (OpenRouter `usage.cost`), otherwise
tokens x `pricing.json`. Both numbers are kept in `meta` so they can be reconciled later:
`meta.cost_source` is `reported` | `computed` | `unknown_model`.
"""

from __future__ import annotations

from typing import Any

from llmops.llm.types import LLMClient, LLMResponse, Message, ToolSpec
from llmops.pricing import estimate_cost
from llmops.tracing.tracer import Tracer, get_tracer


class TracedClient:
    def __init__(
        self,
        client: LLMClient,
        tracer: Tracer | None = None,
        *,
        name: str = "llm.complete",
        feature: str | None = None,
        system: str | None = None,
        meta: dict[str, Any] | None = None,
    ):
        self._client = client
        self._tracer = tracer
        self._name = name
        self._feature = feature
        self._system = system
        self._meta = dict(meta or {})

    @property
    def inner(self) -> LLMClient:
        return self._client

    def __getattr__(self, item: str) -> Any:  # delegate everything else
        return getattr(self._client, item)

    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        tracer = self._tracer or get_tracer()
        attrs: dict[str, Any] = {"model": model}
        if self._feature:
            attrs["feature"] = self._feature
        with tracer.span(self._name, "llm", **attrs) as span:
            resp = self._client.complete(
                messages, tools=tools, model=model, temperature=temperature, max_tokens=max_tokens
            )
            record_response(span, resp, system=self._system or _guess_system(self._client))
            span.meta.update(self._meta)
            return resp


def response_cost(resp: LLMResponse) -> float:
    """USD for one response by the rule `record_response` applies: the provider-reported cost
    when there is one, else tokens x `pricing.json`, else 0 (unknown model)."""
    if resp.cost_usd is not None:
        return float(resp.cost_usd)
    return estimate_cost(resp.model, resp.usage.input_tokens, resp.usage.output_tokens) or 0.0


def record_response(span: Any, resp: LLMResponse, *, system: str | None = None) -> None:
    """Fill an llm span from an LLMResponse (shared by wrap_client and the cache layer)."""
    span.model = resp.model
    span.input_tokens = resp.usage.input_tokens
    span.output_tokens = resp.usage.output_tokens
    if resp.latency_ms:
        span.latency_ms = round(resp.latency_ms, 3)
    computed = estimate_cost(resp.model, resp.usage.input_tokens, resp.usage.output_tokens)
    reported = resp.cost_usd
    if reported is not None:
        span.cost_usd = float(reported)
        source = "reported"
    elif computed is not None:
        span.cost_usd = computed
        source = "computed"
    else:
        span.cost_usd = 0.0
        source = "unknown_model"
    span.meta.update(
        {
            "cost_source": source,
            "cost_computed_usd": computed,
            "cost_reported_usd": reported,
            "stop_reason": resp.stop_reason,
            "tool_calls": len(resp.tool_calls),
            "output_chars": len(resp.text or ""),
        }
    )
    if system:
        span.meta.setdefault("gen_ai.system", system)


def _guess_system(client: Any) -> str:
    """`gen_ai.system`: the client's `provider` attribute when it has one, else its class name."""
    provider = getattr(client, "provider", None)
    if isinstance(provider, str) and provider:
        return provider
    name = type(client).__name__.lower()
    for key in ("openrouter", "anthropic", "openai", "fake"):
        if key in name:
            return key
    return "unknown"


def wrap_client(
    client: LLMClient,
    tracer: Tracer | None = None,
    *,
    name: str = "llm.complete",
    feature: str | None = None,
    system: str | None = None,
    meta: dict[str, Any] | None = None,
) -> TracedClient:
    """Return a drop-in client whose `complete` emits an llm span per call.

    `system` overrides the `gen_ai.system` meta value; by default it is the client's `provider`
    attribute (when present) or a guess from its class name. `meta` is merged into every span's
    `meta` (for example `{"run_id": ...}`); build one wrapper per call when the name or meta
    changes between calls.
    """
    if isinstance(client, TracedClient):
        return client
    return TracedClient(client, tracer, name=name, feature=feature, system=system, meta=meta)
