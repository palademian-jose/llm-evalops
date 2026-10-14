"""Cache statistics from spans: hit rate plus estimated dollars and milliseconds saved."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from llmops.tracing.span import Span


@dataclass
class CacheStatRow:
    cache_type: str
    lookups: int
    hits: int
    saved_cost_usd: float
    saved_ms: float

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0


def cache_stats(spans: list[Span]) -> list[CacheStatRow]:
    agg: dict[str, dict[str, float]] = defaultdict(
        lambda: {"lookups": 0, "hits": 0, "usd": 0.0, "ms": 0.0}
    )
    for s in spans:
        if s.kind != "cache":
            continue
        a = agg[str(s.meta.get("cache_type", "unknown"))]
        a["lookups"] += 1
        if s.cache_hit:
            a["hits"] += 1
            a["usd"] += float(s.meta.get("saved_cost_usd") or 0.0)
            a["ms"] += float(s.meta.get("saved_ms") or 0.0)
    rows = [
        CacheStatRow(
            cache_type=k,
            lookups=int(v["lookups"]),
            hits=int(v["hits"]),
            saved_cost_usd=v["usd"],
            saved_ms=v["ms"],
        )
        for k, v in agg.items()
    ]
    rows.sort(key=lambda r: r.cache_type)
    return rows


def overall_hit_rate(spans: list[Span]) -> float:
    """Share of requests (traces) served from any cache."""
    traces = {s.trace_id for s in spans}
    hit_traces = {s.trace_id for s in spans if s.kind == "cache" and s.cache_hit}
    return len(hit_traces) / len(traces) if traces else 0.0


def cache_markdown(rows: list[CacheStatRow], spans: list[Span] | None = None) -> str:
    lines = [
        "| cache | lookups | hits | hit rate | est. USD saved | est. ms saved |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.cache_type} | {r.lookups} | {r.hits} | {r.hit_rate:.1%} | {r.saved_cost_usd:.4f} | {r.saved_ms:,.0f} |"
        )
    if not rows:
        lines.append("| (no cache spans) | | | | | |")
    if spans is not None:
        lines.append("")
        lines.append(f"Requests served from any cache: {overall_hit_rate(spans):.1%}")
    return "\n".join(lines)
