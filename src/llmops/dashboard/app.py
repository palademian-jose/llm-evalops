"""Streamlit dashboard over llmops trace files.

Run with `llmops dashboard --traces traces/ [--compare traces/synthetic] [--rules alerts.yaml]`
or directly: `streamlit run src/llmops/dashboard/app.py -- --traces traces/`.
Environment fallbacks (used by tests): LLMOPS_DASH_TRACES, LLMOPS_DASH_COMPARE, LLMOPS_DASH_RULES.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from llmops.alerts.rules import evaluate_rules, load_alerts_config
from llmops.cache.stats import cache_stats, overall_hit_rate
from llmops.cost import cost_table
from llmops.ingest import load_spans
from llmops.latency import latency_table
from llmops.tracing.span import Span


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--traces", nargs="*", default=None)
    ap.add_argument("--compare", default=None)
    ap.add_argument("--rules", default=None)
    ns, _ = ap.parse_known_args(sys.argv[1:])
    if not ns.traces:
        env = os.environ.get("LLMOPS_DASH_TRACES")
        ns.traces = env.split(os.pathsep) if env else ["traces"]
    ns.compare = ns.compare or os.environ.get("LLMOPS_DASH_COMPARE")
    ns.rules = (
        ns.rules
        or os.environ.get("LLMOPS_DASH_RULES")
        or ("alerts.yaml" if Path("alerts.yaml").exists() else None)
    )
    return ns


@st.cache_data(show_spinner=False)
def _load(sources: tuple[str, ...]) -> list[Span]:
    return load_spans(list(sources))


def to_frame(spans: list[Span]) -> pd.DataFrame:
    rows = [
        {
            "trace_id": s.trace_id,
            "span_id": s.span_id,
            "root": s.parent_span_id is None,
            "kind": s.kind,
            "name": s.name,
            "start": pd.Timestamp(s.start_ts),
            "latency_ms": s.latency_ms,
            "model": s.model,
            "input_tokens": s.input_tokens,
            "output_tokens": s.output_tokens,
            "cost_usd": s.cost_usd,
            "cache_hit": s.cache_hit,
            "feature": s.feature,
            "user_id": s.user_id,
            "status": s.status,
        }
        for s in spans
    ]
    return pd.DataFrame(rows)


def _bucket(df: pd.DataFrame) -> str:
    span_days = (df["start"].max() - df["start"].min()).total_seconds() / 86400 if len(df) else 0
    return "D" if span_days > 3 else "h"


def kpi_row(spans: list[Span], df: pd.DataFrame) -> None:
    roots = df[df["root"]]
    n = max(1, len(roots))
    total = df["cost_usd"].sum()
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Requests", f"{len(roots):,}")
    c2.metric("Total cost", f"${total:,.2f}")
    c3.metric("$ / 1k requests", f"${total / n * 1000:,.2f}")
    c4.metric(
        "p50 / p95 latency",
        f"{roots['latency_ms'].quantile(0.5):,.0f} / {roots['latency_ms'].quantile(0.95):,.0f} ms",
    )
    c5.metric("Error rate", f"{(roots['status'] == 'error').mean():.1%}")
    c6.metric("Cache hit rate", f"{overall_hit_rate(spans):.1%}")


def section_cost(df: pd.DataFrame) -> None:
    st.subheader("Cost over time")
    freq = _bucket(df)
    series = df.set_index("start")["cost_usd"].resample(freq).sum()
    st.line_chart(series.rename("cost_usd"))
    st.subheader("Top-cost features")
    rows = cost_table([], "feature") if df.empty else None  # keep import used when df is empty
    feat = df.groupby("feature")["cost_usd"].sum().sort_values(ascending=False)
    st.bar_chart(feat)
    by_model = (
        df[df["model"].notna()]
        .groupby("model")[["cost_usd", "input_tokens", "output_tokens"]]
        .sum()
    )
    st.dataframe(by_model.sort_values("cost_usd", ascending=False), use_container_width=True)
    del rows


def section_latency(df: pd.DataFrame, spans: list[Span]) -> None:
    st.subheader("Latency over time (root spans)")
    roots = df[df["root"]].set_index("start")["latency_ms"]
    freq = _bucket(df)
    q = roots.resample(freq).agg([lambda s: s.quantile(0.5), lambda s: s.quantile(0.95)])
    q.columns = ["p50", "p95"]
    st.line_chart(q)
    st.subheader("Stage breakdown")
    stage = latency_table(spans, "stage")
    st.dataframe(
        pd.DataFrame(
            [
                {"stage": r.key, "n": r.n, "p50": r.p50, "p95": r.p95, "share": r.share}
                for r in stage
            ]
        ),
        use_container_width=True,
    )


def section_cache_errors(df: pd.DataFrame, spans: list[Span]) -> None:
    left, right = st.columns(2)
    with left:
        st.subheader("Cache hit rate over time")
        cache = df[df["kind"] == "cache"]
        if cache.empty:
            st.info("No cache spans in this trace set.")
        else:
            st.line_chart(
                cache.set_index("start")["cache_hit"]
                .astype(float)
                .resample(_bucket(df))
                .mean()
                .rename("hit_rate")
            )
            st.dataframe(
                pd.DataFrame([r.__dict__ | {"hit_rate": r.hit_rate} for r in cache_stats(spans)]),
                use_container_width=True,
            )
    with right:
        st.subheader("Error rate over time")
        roots = df[df["root"]]
        st.line_chart(
            (roots.set_index("start")["status"] == "error")
            .astype(float)
            .resample(_bucket(df))
            .mean()
            .rename("error_rate")
        )


def section_alerts(spans: list[Span], rules: str | None) -> None:
    st.subheader("Live alerts")
    if not rules or not Path(rules).exists():
        st.info("No alerts.yaml found; pass --rules to evaluate alert rules here.")
        return
    cfg = load_alerts_config(rules)
    window = st.text_input("Window", cfg.window)
    baseline = st.text_input("Baseline window", cfg.baseline_window)
    try:
        res = evaluate_rules(spans, cfg, window=window, baseline_window=baseline)
    except ValueError as e:
        st.error(str(e))
        return
    if res.ok:
        st.success(
            f"No violations in the last {window} (window ends {res.now:%Y-%m-%d %H:%M} UTC)."
        )
    else:
        st.error(f"{len(res.violations)} violation(s) in the last {window}")
    st.dataframe(pd.DataFrame([r.__dict__ for r in res.rows]), use_container_width=True)


def section_compare(compare: str | None) -> None:
    st.header("Before / after")
    dirs: list[Path] = []
    if compare and Path(compare).is_dir():
        dirs = sorted(p for p in Path(compare).iterdir() if p.is_dir() and list(p.glob("*.jsonl")))
    a = st.text_input("Trace directory A (before)", str(dirs[0]) if dirs else "")
    b = st.text_input("Trace directory B (after)", str(dirs[-1]) if dirs else "")
    rows = []
    for p in dirs or [Path(x) for x in (a, b) if x]:
        spans = _load((str(p),))
        if not spans:
            continue
        df = to_frame(spans)
        roots = df[df["root"]]
        n = max(1, len(roots))
        rows.append(
            {
                "scenario": p.name,
                "requests": len(roots),
                "$/1k req": round(df["cost_usd"].sum() / n * 1000, 2),
                "p50 ms": round(roots["latency_ms"].quantile(0.5), 0),
                "p95 ms": round(roots["latency_ms"].quantile(0.95), 0),
                "cache hit rate": round(overall_hit_rate(spans), 3),
                "error rate": round((roots["status"] == "error").mean(), 3),
            }
        )
    if not rows:
        st.info("Point --compare at a directory with one sub-directory of traces per scenario.")
        return
    table = pd.DataFrame(rows)
    base = table.iloc[0]
    table["Δ cost"] = ((table["$/1k req"] - base["$/1k req"]) / base["$/1k req"]).map(
        lambda x: f"{x:+.0%}"
    )
    table["Δ p95"] = ((table["p95 ms"] - base["p95 ms"]) / base["p95 ms"]).map(
        lambda x: f"{x:+.0%}"
    )
    st.dataframe(table, use_container_width=True, hide_index=True)


def main() -> None:
    ns = _args()
    st.set_page_config(page_title="llmops dashboard", layout="wide")
    st.title("llmops dashboard")
    sources = st.sidebar.text_input(
        "Trace sources (os.pathsep-separated)", os.pathsep.join(ns.traces)
    )
    spans = _load(tuple(s for s in sources.split(os.pathsep) if s))
    if not spans:
        st.warning(
            f"No spans found in {sources}. Run `make synth` or point --traces at a JSONL directory."
        )
        section_compare(ns.compare)
        return
    df = to_frame(spans)
    features = sorted(df["feature"].unique())
    chosen = st.sidebar.multiselect("Features", features, default=features)
    if chosen and len(chosen) < len(features):
        df = df[df["feature"].isin(chosen)]
        keep = set(df["trace_id"])
        spans = [s for s in spans if s.trace_id in keep]
    st.caption(
        f"{len(spans):,} spans, {df['start'].min():%Y-%m-%d %H:%M} to {df['start'].max():%Y-%m-%d %H:%M} UTC"
        + ("  (synthetic traces)" if any(s.meta.get("synthetic") for s in spans[:50]) else "")
    )
    kpi_row(spans, df)
    section_cost(df)
    section_latency(df, spans)
    section_cache_errors(df, spans)
    section_alerts(spans, ns.rules)
    section_compare(ns.compare)


main()
