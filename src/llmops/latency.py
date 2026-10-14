"""Latency percentiles and stage breakdown.

- `--by feature`: p50/p90/p95/p99 of root-span latency (one root span per request) per feature.
- `--by stage`: percentiles of span latency per span kind (retrieval/rerank/llm/tool/verify/...),
  plus each stage's share of total root time ("breakdown").
- TTFT: percentiles of `meta.ttft_ms` when present on llm spans.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import numpy as np

from llmops.tracing.span import Span

By = Literal["stage", "feature"]
PCTS = (50, 90, 95, 99)


@dataclass
class LatencyRow:
    key: str
    n: int
    p50: float
    p90: float
    p95: float
    p99: float
    mean: float
    share: float | None = None  # share of total root time (stage mode)
    ttft_p50: float | None = None
    ttft_p95: float | None = None


def _pcts(values: list[float]) -> tuple[float, float, float, float, float]:
    arr = np.asarray(values, dtype=float)
    p = np.percentile(arr, PCTS)
    return float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(arr.mean())


def root_spans(spans: list[Span]) -> list[Span]:
    return [s for s in spans if s.parent_span_id is None]


def latency_table(spans: list[Span], by: By = "feature") -> list[LatencyRow]:
    groups: dict[str, list[float]] = defaultdict(list)
    ttft: dict[str, list[float]] = defaultdict(list)
    if by == "feature":
        for s in root_spans(spans):
            groups[s.feature].append(s.latency_ms)
        for s in spans:
            if s.kind == "llm" and isinstance(s.meta.get("ttft_ms"), int | float):
                ttft[s.feature].append(float(s.meta["ttft_ms"]))
        total_root = None
    else:
        for s in spans:
            groups[s.kind].append(s.latency_ms)
            if s.kind == "llm" and isinstance(s.meta.get("ttft_ms"), int | float):
                ttft[s.kind].append(float(s.meta["ttft_ms"]))
        total_root = sum(s.latency_ms for s in root_spans(spans)) or None
    rows = []
    for key, vals in groups.items():
        p50, p90, p95, p99, mean = _pcts(vals)
        row = LatencyRow(key=key, n=len(vals), p50=p50, p90=p90, p95=p95, p99=p99, mean=mean)
        if total_root and by == "stage":
            # Child time only; root spans ("other" wrappers) would double count.
            child_time = sum(v for v in vals) if key != "other" else 0.0
            row.share = child_time / total_root
        if ttft.get(key):
            t = np.percentile(np.asarray(ttft[key]), [50, 95])
            row.ttft_p50, row.ttft_p95 = float(t[0]), float(t[1])
        rows.append(row)
    rows.sort(key=lambda r: (-r.p95, r.key))
    return rows


def latency_markdown(rows: list[LatencyRow], by: By) -> str:
    has_ttft = any(r.ttft_p50 is not None for r in rows)
    head = f"| {by} | n | p50 ms | p90 ms | p95 ms | p99 ms | mean ms |"
    sep = "|---|---|---|---|---|---|---|"
    if by == "stage":
        head += " share of request time |"
        sep += "---|"
    if has_ttft:
        head += " TTFT p50 | TTFT p95 |"
        sep += "---|---|"
    lines = [head, sep]
    for r in rows:
        line = f"| {r.key} | {r.n} | {r.p50:.1f} | {r.p90:.1f} | {r.p95:.1f} | {r.p99:.1f} | {r.mean:.1f} |"
        if by == "stage":
            line += f" {'—' if r.share is None else f'{r.share:.1%}'} |"
        if has_ttft:
            t50 = "—" if r.ttft_p50 is None else f"{r.ttft_p50:.1f}"
            t95 = "—" if r.ttft_p95 is None else f"{r.ttft_p95:.1f}"
            line += f" {t50} | {t95} |"
        lines.append(line)
    return "\n".join(lines)
