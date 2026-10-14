"""Generate SYNTHETIC, seeded trace sets for four scenarios so the dashboard, the alert check
and the README before/after table work offline. Nothing here comes from a real model.

Scenarios (same request stream, different serving strategy):
- baseline        : every request hits anthropic/claude-sonnet-4.5; last hour carries an incident
                    (opus routing + slow provider) so `llmops check` has something to catch.
- exact_cache     : baseline + exact-match cache (~22 % of requests are repeats).
- semantic_cache  : exact cache + semantic cache (~13 % more near-duplicates served).
- cheaper_model   : both caches + a router that sends ~60 % of simple requests to haiku-4.5.

Writes `<out>/<scenario>/spans.jsonl`, `<out>/bill_baseline.csv` (a simulated provider bill with
a small rounding discrepancy) and, with --report, a Markdown before/after table.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from llmops.pricing import estimate_cost  # noqa: E402
from llmops.tracing.span import Span, format_ts  # noqa: E402


def _id(rng: np.random.Generator) -> str:
    """Deterministic UUID-shaped id drawn from the scenario rng."""
    h = rng.bytes(16).hex()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


SCENARIOS = ["baseline", "exact_cache", "semantic_cache", "cheaper_model"]
ANCHOR = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)
SONNET, HAIKU, OPUS = (
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-opus-4.1",
)
REFUSALS = [
    "I can't help with that request.",
    "I'm unable to verify that claim from the documents.",
]


def _llm_span(
    rng, *, trace_id, parent, start, feature, user, model, in_tok, out_tok, slow=1.0, error=False
):
    latency = (280 + out_tok * 9.0) * float(rng.lognormal(0, 0.15)) * slow
    cost = estimate_cost(model, in_tok, out_tok) or 0.0
    refusal = bool(rng.random() < 0.02)
    return (
        Span(
            trace_id=trace_id,
            span_id=_id(rng),
            parent_span_id=parent,
            name="generate",
            kind="llm",
            start_ts=format_ts(start),
            end_ts=format_ts(start + timedelta(milliseconds=latency)),
            latency_ms=round(latency, 2),
            model=model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=round(cost, 8),
            feature=feature,
            user_id=user,
            status="error" if error else "ok",
            error_kind="ProviderError" if error else None,
            meta={
                "ttft_ms": round(float(rng.normal(180, 30)) * slow, 1),
                "cost_source": "computed",
                "refusal": refusal,
                "output_text": REFUSALS[0] if refusal else "…",
                "synthetic": True,
            },
        ),
        latency,
        cost,
    )


def generate(
    scenario: str, n: int, seed: int, *, days: int = 7, incident: bool = True
) -> list[Span]:
    rng = np.random.default_rng(seed)
    exact = scenario in ("exact_cache", "semantic_cache", "cheaper_model")
    semantic = scenario in ("semantic_cache", "cheaper_model")
    cheaper = scenario == "cheaper_model"
    users = [f"u{i:03d}" for i in range(50)]
    user_w = 1 / np.arange(1, 51) ** 0.8
    user_w /= user_w.sum()
    spans: list[Span] = []
    offsets = np.sort(rng.uniform(0, days * 86400, size=n))
    for off in offsets:
        start = ANCHOR - timedelta(seconds=days * 86400) + timedelta(seconds=float(off))
        in_incident = incident and scenario == "baseline" and (ANCHOR - start) <= timedelta(hours=1)
        feature = "rag.answer" if rng.random() < 0.7 else "agent.run"
        user = str(rng.choice(users, p=user_w))
        tid = _id(rng)
        root_id = _id(rng)
        t = start
        children: list[Span] = []
        total_cost = 0.0
        err = rng.random() < (0.06 if in_incident else 0.015)
        exact_hit = exact and rng.random() < 0.22
        sem_hit = semantic and not exact_hit and rng.random() < 0.13 / 0.78
        in_tok = int(rng.lognormal(np.log(1500), 0.35))
        out_tok = int(rng.lognormal(np.log(240), 0.4))
        model = OPUS if in_incident and rng.random() < 0.5 else SONNET
        if cheaper and rng.random() < 0.6:
            model = HAIKU
        slow = 1.6 if in_incident else 1.0

        if feature == "rag.answer":
            lat = float(rng.lognormal(np.log(55), 0.3))
            children.append(
                Span(
                    trace_id=tid,
                    span_id=_id(rng),
                    parent_span_id=root_id,
                    name="retrieve",
                    kind="retrieval",
                    start_ts=format_ts(t),
                    end_ts=format_ts(t + timedelta(milliseconds=lat)),
                    latency_ms=round(lat, 2),
                    feature=feature,
                    user_id=user,
                    meta={"k": 20, "synthetic": True},
                )
            )
            t += timedelta(milliseconds=lat)
            lat = float(rng.lognormal(np.log(40), 0.3))
            rc = estimate_cost("rerank-v3.5", searches=1) or 0.0
            total_cost += rc
            children.append(
                Span(
                    trace_id=tid,
                    span_id=_id(rng),
                    parent_span_id=root_id,
                    name="rerank",
                    kind="rerank",
                    model="rerank-v3.5",
                    start_ts=format_ts(t),
                    end_ts=format_ts(t + timedelta(milliseconds=lat)),
                    latency_ms=round(lat, 2),
                    cost_usd=rc,
                    feature=feature,
                    user_id=user,
                    meta={"synthetic": True},
                )
            )
            t += timedelta(milliseconds=lat)

        would_be_llm, would_latency, would_cost = _llm_span(
            rng,
            trace_id=tid,
            parent=root_id,
            start=t,
            feature=feature,
            user=user,
            model=model,
            in_tok=in_tok,
            out_tok=out_tok,
            slow=slow,
            error=err,
        )
        if exact:
            children.append(
                Span(
                    trace_id=tid,
                    span_id=_id(rng),
                    parent_span_id=root_id,
                    name="cache.exact",
                    kind="cache",
                    start_ts=format_ts(t),
                    end_ts=format_ts(t + timedelta(milliseconds=1.2)),
                    latency_ms=1.2,
                    cache_hit=exact_hit,
                    feature=feature,
                    user_id=user,
                    meta={
                        "cache_type": "exact",
                        "saved_cost_usd": would_cost if exact_hit else None,
                        "saved_ms": round(would_latency, 1) if exact_hit else None,
                        "synthetic": True,
                    },
                )
            )
            t += timedelta(milliseconds=1.2)
        if semantic and not exact_hit:
            emb_cost = estimate_cost("text-embedding-3-small", in_tok // 4, 0) or 0.0
            total_cost += emb_cost
            children.append(
                Span(
                    trace_id=tid,
                    span_id=_id(rng),
                    parent_span_id=root_id,
                    name="cache.semantic",
                    kind="cache",
                    start_ts=format_ts(t),
                    end_ts=format_ts(t + timedelta(milliseconds=28)),
                    latency_ms=28.0,
                    cache_hit=sem_hit,
                    cost_usd=emb_cost,
                    feature=feature,
                    user_id=user,
                    meta={
                        "cache_type": "semantic",
                        "similarity": round(
                            float(rng.uniform(0.93, 0.99))
                            if sem_hit
                            else float(rng.uniform(0.4, 0.9)),
                            3,
                        ),
                        "saved_cost_usd": would_cost if sem_hit else None,
                        "saved_ms": round(would_latency, 1) if sem_hit else None,
                        "synthetic": True,
                    },
                )
            )
            t += timedelta(milliseconds=28)
        served_from_cache = exact_hit or sem_hit
        if not served_from_cache:
            if feature == "agent.run":
                for step in range(int(rng.integers(1, 3))):
                    s_span, lat, c = _llm_span(
                        rng,
                        trace_id=tid,
                        parent=root_id,
                        start=t,
                        feature=feature,
                        user=user,
                        model=model,
                        in_tok=in_tok // 2,
                        out_tok=60,
                        slow=slow,
                    )
                    children.append(s_span)
                    total_cost += c
                    t += timedelta(milliseconds=lat)
                    tl = float(rng.lognormal(np.log(120), 0.5))
                    children.append(
                        Span(
                            trace_id=tid,
                            span_id=_id(rng),
                            parent_span_id=root_id,
                            name="tool.search",
                            kind="tool",
                            start_ts=format_ts(t),
                            end_ts=format_ts(t + timedelta(milliseconds=tl)),
                            latency_ms=round(tl, 2),
                            feature=feature,
                            user_id=user,
                            meta={"step": step, "synthetic": True},
                        )
                    )
                    t += timedelta(milliseconds=tl)
            final, lat, c = _llm_span(
                rng,
                trace_id=tid,
                parent=root_id,
                start=t,
                feature=feature,
                user=user,
                model=model,
                in_tok=in_tok,
                out_tok=out_tok,
                slow=slow,
                error=err,
            )
            children.append(final)
            total_cost += c
            t += timedelta(milliseconds=lat)
            if feature == "agent.run":
                vl = float(rng.lognormal(np.log(15), 0.3))
                children.append(
                    Span(
                        trace_id=tid,
                        span_id=_id(rng),
                        parent_span_id=root_id,
                        name="verify",
                        kind="verify",
                        start_ts=format_ts(t),
                        end_ts=format_ts(t + timedelta(milliseconds=vl)),
                        latency_ms=round(vl, 2),
                        feature=feature,
                        user_id=user,
                        meta={"synthetic": True},
                    )
                )
                t += timedelta(milliseconds=vl)
        else:
            err = False
        root_latency = (t - start).total_seconds() * 1000 + float(rng.uniform(2, 6))
        spans.append(
            Span(
                trace_id=tid,
                span_id=root_id,
                parent_span_id=None,
                name=feature,
                kind="other",
                start_ts=format_ts(start),
                end_ts=format_ts(start + timedelta(milliseconds=root_latency)),
                latency_ms=round(root_latency, 2),
                feature=feature,
                user_id=user,
                session_id=f"s-{user}",
                status="error" if err else "ok",
                error_kind="ProviderError" if err else None,
                meta={"scenario": scenario, "synthetic": True},
            )
        )
        spans.extend(children)
    spans.sort(key=lambda s: s.start_ts)
    return spans


def summarize(spans: list[Span]) -> dict:
    roots = [s for s in spans if s.parent_span_id is None]
    n = len(roots)
    cost = sum(s.cost_usd for s in spans)
    lat = np.array([s.latency_ms for s in roots])
    hit_traces = {s.trace_id for s in spans if s.kind == "cache" and s.cache_hit}
    return {
        "requests": n,
        "cost_usd": round(cost, 4),
        "usd_per_1k": round(cost / n * 1000, 4) if n else 0.0,
        "p50_ms": round(float(np.percentile(lat, 50)), 1) if n else 0.0,
        "p95_ms": round(float(np.percentile(lat, 95)), 1) if n else 0.0,
        "cache_hit_rate": round(len(hit_traces) / n, 4) if n else 0.0,
        "error_rate": round(sum(1 for s in roots if s.status == "error") / n, 4) if n else 0.0,
    }


def write_bill(spans: list[Span], path: Path, drift: float = 0.018) -> None:
    agg: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0, 0, 0.0])
    for s in spans:
        if s.kind != "llm" or not s.model:
            continue
        key = (s.start_ts[:10], s.model)
        agg[key][0] += s.input_tokens
        agg[key][1] += s.output_tokens
        agg[key][2] += s.cost_usd
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "model", "input_tokens", "output_tokens", "cost_usd"])
        for (day, model), (i, o, c) in sorted(agg.items()):
            w.writerow([day, model, int(i), int(o), f"{c * (1 + drift):.4f}"])


def before_after_markdown(stats: dict[str, dict]) -> str:
    base = stats["baseline"]
    lines = [
        "Synthetic, seeded traffic (`scripts/synth_traces.py`); not real-model numbers.",
        "",
        "| Scenario | requests | $/1k req | Δ cost | p50 ms | p95 ms | Δ p95 | cache hit rate | error rate |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name in SCENARIOS:
        s = stats[name]
        d_cost = (
            (s["usd_per_1k"] - base["usd_per_1k"]) / base["usd_per_1k"] * 100
            if base["usd_per_1k"]
            else 0.0
        )
        d_p95 = (s["p95_ms"] - base["p95_ms"]) / base["p95_ms"] * 100 if base["p95_ms"] else 0.0
        lines.append(
            f"| {name} | {s['requests']} | {s['usd_per_1k']:.2f} | {d_cost:+.0f}% | {s['p50_ms']:.0f} | {s['p95_ms']:.0f} | "
            f"{d_p95:+.0f}% | {s['cache_hit_rate']:.0%} | {s['error_rate']:.1%} |"
        )
    return "\n".join(lines)


def write_scenarios(out: Path, n: int, seed: int, *, days: int = 7) -> dict[str, dict]:
    out.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict] = {}
    for i, scenario in enumerate(SCENARIOS):
        spans = generate(scenario, n, seed + i, days=days)
        d = out / scenario
        d.mkdir(parents=True, exist_ok=True)
        with (d / "spans.jsonl").open("w", encoding="utf-8") as fh:
            for s in spans:
                fh.write(s.to_json() + "\n")
        stats[scenario] = summarize(spans)
        if scenario == "baseline":
            write_bill(spans, out / "bill_baseline.csv")
    (out / "README.md").write_text(
        "# Synthetic traces\n\nGenerated by `scripts/synth_traces.py` with a fixed seed. "
        "These are NOT real model calls; they exist so the dashboard, `llmops check` and the "
        "README before/after table can run offline.\n",
        encoding="utf-8",
    )
    (out / "summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="traces/synthetic")
    ap.add_argument("--n", type=int, default=2000, help="requests per scenario")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--report", default=None, help="write the before/after Markdown table here")
    args = ap.parse_args()
    stats = write_scenarios(Path(args.out), args.n, args.seed, days=args.days)
    md = before_after_markdown(stats)
    print(md)
    if args.report:
        p = Path(args.report)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            "# Before/after: serving strategies (synthetic)\n\n" + md + "\n", encoding="utf-8"
        )
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
