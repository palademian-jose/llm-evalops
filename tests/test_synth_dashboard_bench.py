from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llmops.cli import app
from llmops.ingest import load_spans

ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory) -> Path:
    synth = _load_script("synth_traces")
    out = tmp_path_factory.mktemp("synthetic")
    stats = synth.write_scenarios(out, n=1500, seed=3, days=1)
    (out / "stats.json").write_text(json.dumps(stats), encoding="utf-8")
    return out


def test_synth_scenarios_are_seeded_and_ordered(synth_dir: Path):
    synth = _load_script("synth_traces")
    a = synth.generate("exact_cache", 50, 1, days=2)
    b = synth.generate("exact_cache", 50, 1, days=2)
    assert [s.model_dump(exclude={"span_id", "trace_id"}) for s in a] == [
        s.model_dump(exclude={"span_id", "trace_id"}) for s in b
    ]
    stats = json.loads((synth_dir / "stats.json").read_text(encoding="utf-8"))
    assert set(stats) == set(synth.SCENARIOS)
    assert stats["baseline"]["cache_hit_rate"] == 0.0
    assert stats["exact_cache"]["cache_hit_rate"] > 0.1
    assert stats["semantic_cache"]["cache_hit_rate"] > stats["exact_cache"]["cache_hit_rate"]
    assert (
        stats["cheaper_model"]["usd_per_1k"]
        < stats["semantic_cache"]["usd_per_1k"]
        < stats["baseline"]["usd_per_1k"]
    )
    md = synth.before_after_markdown(stats)
    assert "| cheaper_model |" in md and "Synthetic" in md
    spans = load_spans(synth_dir / "baseline")
    assert all(s.meta.get("synthetic") for s in spans)
    assert (synth_dir / "bill_baseline.csv").read_text(encoding="utf-8").startswith("date,model,")


def test_synthetic_baseline_incident_trips_alert_check(synth_dir: Path):
    res = runner.invoke(
        app,
        [
            "check",
            "--traces",
            str(synth_dir / "baseline"),
            "--rules",
            str(ROOT / "alerts.yaml"),
            "--window",
            "1h",
            "--baseline-window",
            "48h",
        ],
    )
    assert res.exit_code == 1, res.output
    assert "cost-per-request-spike" in res.output and "VIOLATION" in res.output
    res = runner.invoke(
        app,
        [
            "cost",
            "--traces",
            str(synth_dir / "baseline"),
            "--by",
            "day",
            "--validate-against",
            str(synth_dir / "bill_baseline.csv"),
        ],
    )
    assert res.exit_code == 0 and "discrepancy: -1.77%" in res.output


def test_dashboard_renders_without_exceptions(synth_dir: Path, monkeypatch: pytest.MonkeyPatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("LLMOPS_DASH_TRACES", str(synth_dir / "exact_cache"))
    monkeypatch.setenv("LLMOPS_DASH_COMPARE", str(synth_dir))
    monkeypatch.setenv("LLMOPS_DASH_RULES", str(ROOT / "alerts.yaml"))
    monkeypatch.setattr(sys, "argv", ["app.py"])
    at = AppTest.from_file(
        str(ROOT / "src" / "llmops" / "dashboard" / "app.py"), default_timeout=120
    )
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.title[0].value == "llmops dashboard"
    headers = [h.value for h in at.subheader]
    assert "Cost over time" in headers and "Live alerts" in headers
    assert any("Before / after" in h.value for h in at.header)
    assert len(at.metric) >= 6


def test_dashboard_cli_builds_streamlit_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    calls = {}

    def fake_run(cmd, **kw):
        calls["cmd"] = cmd

        class P:
            returncode = 0

        return P()

    import subprocess

    monkeypatch.setattr(subprocess, "run", fake_run)
    res = runner.invoke(
        app, ["dashboard", "--traces", str(tmp_path), "--port", "8600", "--compare", str(tmp_path)]
    )
    assert res.exit_code == 0, res.output
    cmd = calls["cmd"]
    assert "streamlit" in cmd and "run" in cmd and "--server.headless" in cmd and "8600" in cmd
    assert cmd[cmd.index("--") + 1 :][:2] == ["--traces", str(tmp_path)]


def test_bench_overhead_runs_quickly():
    bench = _load_script("bench_overhead")
    rows = bench.bench(300)
    assert [r[0] for r in rows][0].startswith("bare function")
    assert all(r[1] > 0 for r in rows)
    md = bench.to_markdown(rows, 300)
    assert "| span, MemorySink |" in md
