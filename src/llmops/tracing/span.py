"""Span model (PLAN.md shared contract 2)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SpanKind = Literal["llm", "retrieval", "rerank", "tool", "cache", "verify", "other"]
SPAN_KINDS: tuple[str, ...] = ("llm", "retrieval", "rerank", "tool", "cache", "verify", "other")


def new_id() -> str:
    return str(uuid.uuid4())


def now_iso() -> str:
    return format_ts(datetime.now(UTC))


def parse_ts(ts: str) -> datetime:
    s = ts.replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def format_ts(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    dt = dt.astimezone(UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


class Span(BaseModel):
    """One JSON line in `TRACE_PATH`."""

    model_config = ConfigDict(extra="allow")

    trace_id: str = Field(default_factory=new_id)
    span_id: str = Field(default_factory=new_id)
    parent_span_id: str | None = None
    name: str = "span"
    kind: SpanKind = "other"
    start_ts: str = Field(default_factory=now_iso)
    end_ts: str | None = None
    latency_ms: float = 0.0
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cache_hit: bool = False
    feature: str = "default"
    user_id: str = "anon"
    session_id: str | None = None
    status: Literal["ok", "error"] = "ok"
    error_kind: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    def set(self, **attrs: Any) -> Span:
        """Set known fields; unknown keys land in `meta`."""
        for key, value in attrs.items():
            if key in type(self).model_fields and key != "meta":
                setattr(self, key, value)
            elif key == "meta" and isinstance(value, dict):
                self.meta.update(value)
            else:
                self.meta[key] = value
        return self

    @property
    def start(self) -> datetime:
        return parse_ts(self.start_ts)

    @property
    def end(self) -> datetime | None:
        return parse_ts(self.end_ts) if self.end_ts else None

    @property
    def is_root(self) -> bool:
        return self.parent_span_id is None

    def to_json(self) -> str:
        return self.model_dump_json(exclude_none=False)
