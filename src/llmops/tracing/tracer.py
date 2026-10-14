"""Tracer with nested spans via contextvars.

    tracer = Tracer(JsonlSink("traces/spans.jsonl"))
    with tracer.trace(feature="rag.answer", user_id="u1"):
        with tracer.span("retrieve", kind="retrieval") as s:
            ...
            s.set(meta={"k": 20})
        with tracer.span("generate", kind="llm", model="anthropic/claude-sonnet-4.5") as s:
            ...

Spans opened inside another span get `parent_span_id` set automatically; a span opened outside
any trace starts a new trace. Each thread has its own context (new threads start clean).
"""

from __future__ import annotations

import contextvars
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol

from llmops.tracing.sink import JsonlSink, NullSink, Sink
from llmops.tracing.span import Span, format_ts, new_id, now_iso, parse_ts


class Exporter(Protocol):
    def export(self, span: Span) -> None: ...


@dataclass
class _TraceCtx:
    trace_id: str
    feature: str
    user_id: str
    session_id: str | None


_trace_ctx: contextvars.ContextVar[_TraceCtx | None] = contextvars.ContextVar(
    "llmops_trace", default=None
)
_span_stack: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "llmops_spans", default=()
)


class Tracer:
    def __init__(
        self,
        sink: Sink | None = None,
        *,
        feature: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        exporters: list[Exporter] | None = None,
    ):
        if sink is None:
            from llmops.settings import get_settings

            sink = JsonlSink(get_settings().trace_path)
        self.sink = sink
        self.exporters: list[Exporter] = list(exporters or [])
        self._feature = feature
        self._user_id = user_id
        self._session_id = session_id

    # -- defaults -----------------------------------------------------------------------------
    @property
    def feature(self) -> str:
        if self._feature is not None:
            return self._feature
        from llmops.settings import get_settings

        return get_settings().trace_feature

    @property
    def user_id(self) -> str:
        if self._user_id is not None:
            return self._user_id
        from llmops.settings import get_settings

        return get_settings().trace_user_id

    # -- context ------------------------------------------------------------------------------
    @contextmanager
    def trace(
        self,
        *,
        trace_id: str | None = None,
        feature: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> Iterator[str]:
        """Open a trace context: every span inside shares trace_id/feature/user/session."""
        ctx = _TraceCtx(
            trace_id=trace_id or new_id(),
            feature=feature or self.feature,
            user_id=user_id or self.user_id,
            session_id=session_id if session_id is not None else self._session_id,
        )
        tok_ctx = _trace_ctx.set(ctx)
        tok_stack = _span_stack.set(())
        try:
            yield ctx.trace_id
        finally:
            _trace_ctx.reset(tok_ctx)
            _span_stack.reset(tok_stack)

    @staticmethod
    def current_trace_id() -> str | None:
        ctx = _trace_ctx.get()
        return ctx.trace_id if ctx else None

    @staticmethod
    def current_span_id() -> str | None:
        stack = _span_stack.get()
        return stack[-1] if stack else None

    @contextmanager
    def span(self, name: str, kind: str = "other", **attrs: Any) -> Iterator[Span]:
        ctx = _trace_ctx.get()
        stack = _span_stack.get()
        own_trace = ctx is None
        if own_trace:
            ctx = _TraceCtx(
                trace_id=new_id(),
                feature=self.feature,
                user_id=self.user_id,
                session_id=self._session_id,
            )
            tok_ctx = _trace_ctx.set(ctx)
        span = Span(
            trace_id=ctx.trace_id,
            parent_span_id=stack[-1] if stack else None,
            name=name,
            kind=kind,  # type: ignore[arg-type]
            feature=ctx.feature,
            user_id=ctx.user_id,
            session_id=ctx.session_id,
        )
        span.set(**attrs)
        tok_stack = _span_stack.set(stack + (span.span_id,))
        start = time.perf_counter()
        try:
            yield span
        except BaseException as e:
            span.status = "error"
            span.error_kind = span.error_kind or type(e).__name__
            span.meta.setdefault("error", str(e)[:500])
            raise
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            if not span.latency_ms:
                span.latency_ms = round(elapsed, 3)
            if span.end_ts is None:
                span.end_ts = now_iso()
            _span_stack.reset(tok_stack)
            if own_trace:
                _trace_ctx.reset(tok_ctx)
            self.record(span)

    def record(self, span: Span) -> None:
        """Write a finished span to the sink and exporters."""
        if span.end_ts is None:
            span.end_ts = format_ts(parse_ts(span.start_ts))
        self.sink.write(span)
        for exp in self.exporters:
            exp.export(span)


# -- module-level default tracer -------------------------------------------------------------
_default: Tracer | None = None


def get_tracer() -> Tracer:
    global _default
    if _default is None:
        _default = Tracer()
    return _default


def set_tracer(tracer: Tracer | None) -> None:
    global _default
    _default = tracer


@contextmanager
def span(name: str, kind: str = "other", **attrs: Any) -> Iterator[Span]:
    with get_tracer().span(name, kind, **attrs) as s:
        yield s


@contextmanager
def trace(**kwargs: Any) -> Iterator[str]:
    with get_tracer().trace(**kwargs) as tid:
        yield tid


def null_tracer() -> Tracer:
    return Tracer(NullSink())
