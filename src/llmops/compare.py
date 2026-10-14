"""Before/after comparison across trace directories (one directory per serving scenario).

The same code feeds `llmops before-after`, the dashboard's "Before / after" section and
`scripts/synth_traces.py --from-dirs`, so a synthetic set and a set written by a real app
(`docs-rag/traces/scenarios/*`) produce the same table.

A *source* is any of: a directory holding `*.jsonl` files (one scenario, named after the
directory), a directory whose sub-directories each hold `*.jsonl` files (one scenario per
sub-directory), a `.jsonl` file, or a glob of those. The scenario named `baseline` is the
reference for the delta columns (else the first one, or the one chosen with `baseline=`).
"""

from __future__ import annotations

import glob
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from llmops.cache.stats import overall_hit_rate
from llmops.ingest import load_spans
from llmops.tracing.span import Span

# Display order for the usual serving ladder; anything else follows alphabetically.
LADDER = ("baseline", "exact_cache", "semantic_cache", "cheaper_model")


@dataclass
class ScenarioStats:
    scenario: str
    requests: int
    usd_per_1k: float
    p50_ms: float
    p95_ms: float
    cache_hit_rate: float
    error_rate: float

    def as_dict(self) -> dict:
        return asdict(self)


def scenario_stats(name: str, spans: list[Span]) -> ScenarioStats:
    """One row: requests are distinct traces, cost sums every span, latency is the root span's."""
    traces = {s.trace_id for s in spans}
    roots = [s for s in spans if s.parent_span_id is None]
    n = len(traces)
    cost = sum(s.cost_usd for s in spans)
    lat = np.array([s.latency_ms for s in roots]) if roots else np.zeros(1)
    return ScenarioStats(
        scenario=name,
        requests=n,
        usd_per_1k=cost / n * 1000 if n else 0.0,
        p50_ms=float(np.percentile(lat, 50)),
        p95_ms=float(np.percentile(lat, 95)),
        cache_hit_rate=overall_hit_rate(spans),
        error_rate=(sum(1 for s in roots if s.status == "error") / len(roots)) if roots else 0.0,
    )


def _expand(source: str | Path) -> list[Path]:
    s = str(source)
    if any(ch in s for ch in "*?["):
        return sorted(Path(m) for m in glob.glob(s, recursive=True))
    return [Path(s)]


def resolve_scenarios(sources: Iterable[str | Path]) -> list[tuple[str, list[Path]]]:
    """Expand sources into `(scenario name, jsonl files)` pairs in display order."""
    found: list[tuple[str, Path, list[Path]]] = []  # name, owner dir, files
    for src in sources:
        for p in _expand(src):
            if p.is_file() and p.suffix == ".jsonl":
                found.append((p.parent.name if p.name == "spans.jsonl" else p.stem, p.parent, [p]))
            elif p.is_dir():
                direct = sorted(p.glob("*.jsonl"))
                if direct:
                    found.append((p.name, p.parent, direct))
                    continue
                for child in sorted(c for c in p.iterdir() if c.is_dir()):
                    files = sorted(child.rglob("*.jsonl"))
                    if files:
                        found.append((child.name, child.parent, files))
    unique: list[tuple[str, Path, list[Path]]] = []
    seen: set[Path] = set()
    for item in found:
        key = item[2][0].resolve()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    names = [n for n, _, _ in unique]
    out: list[tuple[str, list[Path]]] = []
    for name, owner, files in unique:
        label = f"{owner.name}/{name}" if names.count(name) > 1 else name
        out.append((label, files))

    def order(item: tuple[str, list[Path]]) -> tuple[int, str]:
        base = item[0].rsplit("/", 1)[-1]
        return (LADDER.index(base), item[0]) if base in LADDER else (len(LADDER), item[0])

    return sorted(out, key=order)


def compare_scenarios(sources: Iterable[str | Path]) -> list[ScenarioStats]:
    rows: list[ScenarioStats] = []
    for name, files in resolve_scenarios(sources):
        spans = load_spans(files)
        if spans:
            rows.append(scenario_stats(name, spans))
    return rows


def _pct(new: float, old: float) -> float:
    return (new - old) / old * 100 if old else 0.0


def deltas(
    rows: list[ScenarioStats], baseline: str | None = None
) -> list[tuple[ScenarioStats, float, float]]:
    """`(row, % change in $/1k req, % change in p95)` against the baseline row."""
    if not rows:
        return []
    ref = next((r for r in rows if r.scenario == (baseline or "baseline")), None)
    if ref is None:
        if baseline:
            raise ValueError(f"no scenario named {baseline!r} among {[r.scenario for r in rows]}")
        ref = rows[0]
    return [(r, _pct(r.usd_per_1k, ref.usd_per_1k), _pct(r.p95_ms, ref.p95_ms)) for r in rows]


def to_markdown(rows: list[ScenarioStats], baseline: str | None = None) -> str:
    lines = [
        "| Scenario | requests | $/1k req | Δ cost | p50 ms | p95 ms | Δ p95 "
        "| cache hit rate | error rate |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r, d_cost, d_p95 in deltas(rows, baseline):
        lines.append(
            f"| {r.scenario} | {r.requests} | {r.usd_per_1k:.2f} | {d_cost:+.0f}% | {r.p50_ms:.0f} "
            f"| {r.p95_ms:.0f} | {d_p95:+.0f}% | {r.cache_hit_rate:.0%} | {r.error_rate:.1%} |"
        )
    return "\n".join(lines)


def report(
    rows: list[ScenarioStats],
    *,
    title: str = "Before/after: serving strategies",
    note: str | None = None,
    baseline: str | None = None,
) -> str:
    parts = [f"# {title}", ""]
    if note:
        parts += [note, ""]
    parts.append(to_markdown(rows, baseline))
    return "\n".join(parts)
