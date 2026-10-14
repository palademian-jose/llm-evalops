"""Before/after comparison over scenario trace directories (llmops.compare, before-after CLI)."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llmops.cli import app
from llmops.compare import (
    ScenarioStats,
    compare_scenarios,
    deltas,
    resolve_scenarios,
    scenario_stats,
    to_markdown,
)
from llmops.ingest import load_spans
from llmops.tracing import JsonlSink, MemorySink, Tracer

ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


def _scenario(
    root: Path, name: str, *, requests: int, cost: float, hits: int = 0, latency: float = 100.0
) -> Path:
    """A scenario directory with `requests` traces: root span + llm child (+ cache hit spans)."""
    d = root / name
    tracer = Tracer(JsonlSink(d / "spans.jsonl", use_filelock=False))
    for i in range(requests):
        with tracer.trace(feature="rag.answer"):
            with tracer.span("rag.ask", "other") as r:
                r.latency_ms = latency + i
                if i < hits:
                    with tracer.span("cache.lookup", "cache") as c:
                        c.cache_hit = True
                else:
                    with tracer.span("generate", "llm", model="m") as g:
                        g.cost_usd = cost
    tracer.sink.close()
    return d


def test_scenario_stats_counts_traces_sums_cost_and_uses_root_latency():
    sink = MemorySink()
    t = Tracer(sink)
    for i in range(4):
        with t.trace(), t.span("root", "other") as r:
            r.latency_ms = 100.0 * (i + 1)
            if i == 3:
                r.status = "error"
            with t.span("generate", "llm") as g:
                g.cost_usd = 0.01
                g.latency_ms = 5.0
            if i < 2:
                with t.span("cache.lookup", "cache") as c:
                    c.cache_hit = True
    s = scenario_stats("x", sink.spans)
    assert s.requests == 4 and s.usd_per_1k == pytest.approx(10.0)
    assert s.p50_ms == pytest.approx(250.0) and s.p95_ms == pytest.approx(385.0)
    assert s.cache_hit_rate == 0.5 and s.error_rate == 0.25


def test_resolve_scenarios_accepts_dirs_parents_files_and_globs_in_ladder_order(tmp_path: Path):
    for name in ("cheaper_model", "baseline", "semantic_cache", "exact_cache", "zeta"):
        _scenario(tmp_path / "scen", name, requests=2, cost=0.01)
    names = [n for n, _ in resolve_scenarios([tmp_path / "scen"])]
    assert names == ["baseline", "exact_cache", "semantic_cache", "cheaper_model", "zeta"]
    globbed = [n for n, _ in resolve_scenarios([str(tmp_path / "scen" / "*")])]
    assert globbed == names
    one_dir = resolve_scenarios([tmp_path / "scen" / "exact_cache"])
    assert [n for n, _ in one_dir] == ["exact_cache"]
    one_file = resolve_scenarios([tmp_path / "scen" / "zeta" / "spans.jsonl"])
    assert [n for n, _ in one_file] == ["zeta"]
    # a repeated source is listed once; non-trace entries (README.md, summary.json) are skipped
    (tmp_path / "scen" / "summary.json").write_text("{}", encoding="utf-8")
    both = resolve_scenarios([str(tmp_path / "scen" / "*"), tmp_path / "scen" / "baseline"])
    assert [n for n, _ in both] == names


def test_same_scenario_name_in_two_places_is_disambiguated_by_parent(tmp_path: Path):
    _scenario(tmp_path / "a", "baseline", requests=1, cost=0.01)
    _scenario(tmp_path / "b", "baseline", requests=1, cost=0.02)
    names = [n for n, _ in resolve_scenarios([tmp_path / "a", tmp_path / "b"])]
    assert names == ["a/baseline", "b/baseline"]


def test_compare_scenarios_deltas_against_baseline(tmp_path: Path):
    _scenario(tmp_path, "baseline", requests=10, cost=0.01, latency=1000)
    _scenario(tmp_path, "exact_cache", requests=10, cost=0.01, hits=4, latency=500)
    rows = compare_scenarios([tmp_path])
    assert [r.scenario for r in rows] == ["baseline", "exact_cache"]
    (b, bc, bp), (e, ec, ep) = deltas(rows)
    assert b.usd_per_1k == pytest.approx(10.0) and bc == 0 and bp == 0
    assert e.usd_per_1k == pytest.approx(6.0) and ec == pytest.approx(-40.0)
    assert e.cache_hit_rate == pytest.approx(0.4) and ep < 0
    md = to_markdown(rows)
    assert "| exact_cache | 10 | 6.00 | -40% |" in md and "| 40% |" in md
    with pytest.raises(ValueError):
        deltas(rows, "nope")
    alt = deltas(rows, baseline="exact_cache")
    assert alt[0][1] == pytest.approx(66.67, abs=0.01)


def test_first_scenario_is_the_reference_when_none_is_called_baseline():
    rows = [
        ScenarioStats("b", 1, 10.0, 1, 1, 0, 0),
        ScenarioStats("a", 1, 5.0, 1, 1, 0, 0),
    ]
    assert deltas(rows)[1][1] == pytest.approx(-50.0)


def test_before_after_cli_dirs_positional_and_outputs(tmp_path: Path):
    scen = tmp_path / "scen"
    _scenario(scen, "baseline", requests=5, cost=0.02)
    _scenario(scen, "cheaper_model", requests=5, cost=0.005)
    md_path, json_path = tmp_path / "out" / "ba.md", tmp_path / "out" / "ba.json"
    res = runner.invoke(
        app,
        [
            "before-after",
            "--dirs",
            str(scen / "baseline"),
            str(scen / "cheaper_model"),  # what a shell glob after --dirs expands to
            "--title",
            "T",
            "--note",
            "fake provider, projected cost",
            "--md",
            str(md_path),
            "--json",
            str(json_path),
        ],
    )
    assert res.exit_code == 0, res.output
    text = md_path.read_text(encoding="utf-8")
    assert text.startswith("# T\n\nfake provider, projected cost\n\n| Scenario |")
    assert "| cheaper_model | 5 | 5.00 | -75% |" in text
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert [d["scenario"] for d in data] == ["baseline", "cheaper_model"]
    quoted = runner.invoke(app, ["before-after", "--dirs", str(scen / "*")])
    assert quoted.exit_code == 0 and "| baseline | 5 |" in quoted.output


def test_before_after_cli_errors(tmp_path: Path):
    assert runner.invoke(app, ["before-after"]).exit_code == 2
    assert runner.invoke(app, ["before-after", "--dirs", str(tmp_path / "missing")]).exit_code == 2
    _scenario(tmp_path, "baseline", requests=1, cost=0.01)
    res = runner.invoke(app, ["before-after", str(tmp_path), "--baseline", "nope"])
    assert res.exit_code == 2 and "nope" in res.output


def test_synth_traces_from_dirs_builds_the_same_table(tmp_path: Path):
    path = ROOT / "scripts" / "synth_traces.py"
    spec = importlib.util.spec_from_file_location("synth_traces", path)
    synth = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(synth)
    out = tmp_path / "synthetic"
    synth.write_scenarios(out, n=400, seed=5, days=1)
    report = tmp_path / "real.md"
    argv = ["synth_traces.py", "--from-dirs", str(out / "*"), "--report", str(report)]
    argv += ["--title", "docs-rag", "--note", "fake provider, projected cost"]
    old = sys.argv
    sys.argv = argv
    try:
        synth.main()
    finally:
        sys.argv = old
    text = report.read_text(encoding="utf-8")
    assert text.startswith("# docs-rag\n\nfake provider, projected cost\n")
    for name in synth.SCENARIOS:
        assert f"| {name} | 400 |" in text
    # the generator's own summary agrees with the shared code on cost and cache hit rate
    stats = {n: synth.summarize(load_spans(out / n)) for n in synth.SCENARIOS}
    shared = {r.scenario: r for r in compare_scenarios([out])}
    for n in synth.SCENARIOS:
        assert shared[n].usd_per_1k == pytest.approx(stats[n]["usd_per_1k"], abs=1e-3)
        assert shared[n].cache_hit_rate == pytest.approx(stats[n]["cache_hit_rate"], abs=1e-3)


def test_dashboard_before_after_accepts_scenario_directories_and_globs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from streamlit.testing.v1 import AppTest

    _scenario(tmp_path / "scen", "baseline", requests=6, cost=0.02)
    _scenario(tmp_path / "scen", "exact_cache", requests=6, cost=0.02, hits=3)
    monkeypatch.setenv("LLMOPS_DASH_TRACES", str(tmp_path / "scen" / "baseline"))
    sources = [str(tmp_path / "scen" / "baseline"), str(tmp_path / "scen" / "exact*")]
    monkeypatch.setenv("LLMOPS_DASH_COMPARE", os.pathsep.join(sources))
    monkeypatch.setenv("LLMOPS_DASH_RULES", "")
    monkeypatch.setattr(sys, "argv", ["app.py"])
    at = AppTest.from_file(
        str(ROOT / "src" / "llmops" / "dashboard" / "app.py"), default_timeout=120
    )
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    table = at.dataframe[-1].value
    assert list(table["scenario"]) == ["baseline", "exact_cache"]
    assert list(table["Δ cost"]) == ["+0%", "-50%"]
