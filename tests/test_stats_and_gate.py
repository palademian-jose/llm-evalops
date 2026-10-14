from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from evalkit.cli import app
from evalkit.gate import GateReport, evaluate_gate, load_gate_config
from evalkit.io import write_eval_result
from evalkit.stats import bootstrap_ci, compare, mcnemar_exact, permutation_test
from evalkit.types import CaseResult, EvalResult, GateConfig

runner = CliRunner()


def _result(
    run_id: str, scores: dict[str, list], metrics: dict[str, float] | None = None
) -> EvalResult:
    n = len(next(iter(scores.values())))
    cases = [
        CaseResult(id=f"g-{i}", scores={m: vals[i] for m, vals in scores.items()}, passed=True)
        for i in range(n)
    ]
    agg = metrics or {m: float(np.mean([float(v) for v in vals])) for m, vals in scores.items()}
    return EvalResult(run_id=run_id, metrics=agg, cases=cases)


def test_bootstrap_is_deterministic_under_seed_and_brackets_true_effect():
    rng = np.random.default_rng(42)
    diffs = rng.normal(0.1, 0.2, size=200)
    ci1 = bootstrap_ci(diffs, n_boot=2000, seed=7)
    ci2 = bootstrap_ci(diffs, n_boot=2000, seed=7)
    ci3 = bootstrap_ci(diffs, n_boot=2000, seed=8)
    assert ci1 == ci2 and ci1 != ci3
    assert ci1[0] < 0.1 < ci1[1]
    assert permutation_test(diffs, n_perm=2000, seed=1) < 0.01
    assert permutation_test(np.zeros(10)) == 1.0


def test_mcnemar_exact_matches_binomial():
    assert mcnemar_exact(0, 0) == 1.0
    # b=1, c=9: two-sided exact binomial p = 2 * P(X <= 1 | n=10, 0.5) = 2 * 11/1024
    assert mcnemar_exact(1, 9) == pytest.approx(22 / 1024)
    assert mcnemar_exact(5, 5) == 1.0


def test_compare_pairs_by_id_and_flags_regression():
    a = _result("a", {"faithfulness": [True] * 18 + [False] * 2, "f1": [0.9] * 20})
    b = _result("b", {"faithfulness": [True] * 10 + [False] * 10, "f1": [0.9] * 20})
    comps = {c.metric: c for c in compare(a, b, n_boot=2000, seed=0)}
    f = comps["faithfulness"]
    assert f.n == 20 and f.delta == pytest.approx(-0.4) and f.is_bool
    assert f.mcnemar_b == 8 and f.mcnemar_c == 0
    assert f.p_mcnemar == pytest.approx(2 * 0.5**8)
    assert f.significant and f.ci_high < 0
    assert not comps["f1"].significant and comps["f1"].p_perm == 1.0


@pytest.mark.parametrize(
    ("baseline", "candidate", "thr", "expected_ok"),
    [
        (0.90, 0.90, {"min": 0.85, "max_drop": 0.05}, True),
        (0.90, 0.86, {"min": 0.85, "max_drop": 0.05}, True),
        (0.90, 0.84, {"min": 0.85, "max_drop": 0.05}, False),  # below min
        (0.95, 0.89, {"min": 0.85, "max_drop": 0.05}, False),  # drop too large
        (0.80, 0.95, {"min": 0.85}, True),
        (None, 0.80, {"min": 0.85}, False),
        (None, 0.90, {"max_drop": 0.05}, True),  # no baseline -> warn only
        (0.10, 0.10, {"max": 0.2, "max_increase": 0.05}, True),
        (0.10, 0.25, {"max": 0.2}, False),
        (0.10, 0.16, {"max_increase": 0.05}, False),
    ],
)
def test_gate_pass_fail_matrix(baseline, candidate, thr, expected_ok):
    cfg = GateConfig.model_validate({"metrics": {"m": thr}})
    base = EvalResult(run_id="base", metrics={"m": baseline}) if baseline is not None else None
    cand = EvalResult(run_id="cand", metrics={"m": candidate})
    rep = evaluate_gate(base, cand, cfg)
    assert rep.ok is expected_ok, rep.to_markdown()


def test_gate_missing_metric_and_min_cases():
    cfg = GateConfig.model_validate({"metrics": {"m": {"min": 0.5}}, "min_cases": 3})
    cand = EvalResult(run_id="c", metrics={}, cases=[CaseResult(id="1")])
    rep = evaluate_gate(None, cand, cfg)
    assert [r.status for r in rep.rows] == ["FAIL", "FAIL"]
    assert "metric missing" in rep.rows[1].reason


def test_gate_markdown_header_exact():
    rep = GateReport(rows=[])
    assert "| Metric | Baseline | Candidate | Δ | Threshold | Status |" in rep.to_markdown()


def _write_results(tmp_path: Path) -> tuple[Path, Path, Path]:
    a = _result("main", {"faithfulness": [True] * 19 + [False], "exact_match": [True] * 20})
    b = _result("pr", {"faithfulness": [True] * 14 + [False] * 6, "exact_match": [True] * 20})
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    write_eval_result(pa, a)
    write_eval_result(pb, b)
    cfg = tmp_path / "evals.yaml"
    cfg.write_text(
        "metrics:\n  faithfulness: {min: 0.85, max_drop: 0.05}\n  exact_match: {min: 0.9}\n",
        encoding="utf-8",
    )
    return pa, pb, cfg


def test_compare_cli_writes_markdown(tmp_path: Path):
    pa, pb, _ = _write_results(tmp_path)
    md = tmp_path / "compare.md"
    res = runner.invoke(
        app, ["compare", "--a", str(pa), "--b", str(pb), "--md", str(md), "--n-boot", "500"]
    )
    assert res.exit_code == 0, res.output
    text = md.read_text(encoding="utf-8")
    assert "faithfulness" in text and "McNemar" in text and "B worse" in text


def test_gate_cli_exit_codes_and_step_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pa, pb, cfg = _write_results(tmp_path)
    md = tmp_path / "gate.md"
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    res = runner.invoke(
        app,
        [
            "gate",
            "--baseline",
            str(pa),
            "--candidate",
            str(pb),
            "--config",
            str(cfg),
            "--md",
            str(md),
            "--annotate",
        ],
    )
    assert res.exit_code == 1, res.output
    assert "GATE FAILED" in res.output and "| faithfulness |" in res.output
    assert "GATE FAILED" in summary.read_text(encoding="utf-8")
    assert load_gate_config(cfg).metrics["faithfulness"].min == 0.85

    res = runner.invoke(
        app, ["gate", "--baseline", str(pa), "--candidate", str(pa), "--config", str(cfg)]
    )
    assert res.exit_code == 0, res.output

    res = runner.invoke(app, ["annotate", "--md", str(md)])
    assert res.exit_code == 0
    assert summary.read_text(encoding="utf-8").count("GATE FAILED") == 2

    monkeypatch.delenv("GITHUB_STEP_SUMMARY")
    res = runner.invoke(app, ["annotate", "--md", str(md)])
    assert res.exit_code == 0 and "GATE FAILED" in res.output
    assert os.environ.get("GITHUB_STEP_SUMMARY") is None


def test_gate_cli_via_subprocess(tmp_path: Path, env: dict[str, str]):
    import subprocess
    import sys

    pa, pb, cfg = _write_results(tmp_path)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "evalkit.cli",
            "gate",
            "--baseline",
            str(pa),
            "--candidate",
            str(pb),
            "--config",
            str(cfg),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 1
    assert "| Metric | Baseline | Candidate |" in proc.stdout
    assert json.loads(pa.read_text())["run_id"] == "main"
