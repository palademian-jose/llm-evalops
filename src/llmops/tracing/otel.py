"""Optional OpenTelemetry exporter (install the `otel` extra).

Each finished llmops span becomes one OTel span whose attributes follow the GenAI semantic
conventions attribute names:
    gen_ai.system, gen_ai.operation.name, gen_ai.request.model, gen_ai.response.model,
    gen_ai.usage.input_tokens, gen_ai.usage.output_tokens, gen_ai.response.finish_reasons
plus `llmops.*` attributes for what the conventions do not cover (cost, feature, user, cache).
Parent/child links are preserved by replaying the llmops ids as OTel span context.
"""

from __future__ import annotations

from typing import Any

from llmops.tracing.span import Span, parse_ts

_OPERATION = {
    "llm": "chat",
    "retrieval": "retrieval",
    "rerank": "rerank",
    "tool": "execute_tool",
    "cache": "cache_lookup",
    "verify": "verify",
    "other": "other",
}


def genai_attributes(span: Span) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "gen_ai.operation.name": _OPERATION.get(span.kind, span.kind),
        "llmops.span.kind": span.kind,
        "llmops.feature": span.feature,
        "llmops.user_id": span.user_id,
        "llmops.cost_usd": span.cost_usd,
        "llmops.cache_hit": span.cache_hit,
        "llmops.latency_ms": span.latency_ms,
        "llmops.status": span.status,
    }
    if span.kind == "llm" or span.model:
        attrs["gen_ai.system"] = str(span.meta.get("gen_ai.system", _system_from_model(span.model)))
        attrs["gen_ai.request.model"] = span.model or ""
        attrs["gen_ai.response.model"] = span.model or ""
        attrs["gen_ai.usage.input_tokens"] = span.input_tokens
        attrs["gen_ai.usage.output_tokens"] = span.output_tokens
        if "stop_reason" in span.meta:
            attrs["gen_ai.response.finish_reasons"] = [str(span.meta["stop_reason"])]
    if span.session_id:
        attrs["llmops.session_id"] = span.session_id
    if span.error_kind:
        attrs["error.type"] = span.error_kind
    return attrs


def _system_from_model(model: str | None) -> str:
    from llmops.llm.types import model_family

    fam = model_family(model)
    return {"anthropic": "anthropic", "openai": "openai", "google": "gcp.gemini"}.get(fam, fam)


class OTelExporter:
    """Replays llmops spans into an OpenTelemetry TracerProvider."""

    def __init__(self, provider: Any | None = None, tracer_name: str = "llmops"):
        from opentelemetry import trace as ot

        self._ot = ot
        self._provider = provider or ot.get_tracer_provider()
        self._tracer = self._provider.get_tracer(tracer_name)

    def export(self, span: Span) -> None:
        from opentelemetry.trace import (
            NonRecordingSpan,
            SpanContext,
            TraceFlags,
            set_span_in_context,
        )

        trace_id = _uuid_to_int(span.trace_id) & ((1 << 128) - 1)
        span_id = _uuid_to_int(span.span_id) & ((1 << 64) - 1)
        parent_ctx = None
        if span.parent_span_id:
            parent_sc = SpanContext(
                trace_id=trace_id,
                span_id=_uuid_to_int(span.parent_span_id) & ((1 << 64) - 1),
                is_remote=False,
                trace_flags=TraceFlags(TraceFlags.SAMPLED),
            )
            parent_ctx = set_span_in_context(NonRecordingSpan(parent_sc))
        start_ns = int(parse_ts(span.start_ts).timestamp() * 1e9)
        end_ns = int(parse_ts(span.end_ts or span.start_ts).timestamp() * 1e9)
        ot_span = self._tracer.start_span(
            span.name, context=parent_ctx, attributes=genai_attributes(span), start_time=start_ns
        )
        if span.status == "error":
            from opentelemetry.trace import Status, StatusCode

            ot_span.set_status(Status(StatusCode.ERROR, span.error_kind or "error"))
        ot_span.set_attribute("llmops.span_id", span.span_id)
        ot_span.set_attribute("llmops.otel_span_id_hint", str(span_id))
        ot_span.end(end_time=max(end_ns, start_ns))


def _uuid_to_int(value: str) -> int:
    try:
        return int(value.replace("-", ""), 16)
    except ValueError:
        return abs(hash(value))
