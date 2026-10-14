from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from evalkit.cli import app
from evalkit.io import GoldenLoadError, load_golden, write_golden
from evalkit.types import EvalResult, GateConfig, GoldenCase, Verdict
from evalkit.validate import validate_cases, validate_file

runner = CliRunner()


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_golden_case_schema_roundtrip(tmp_path: Path):
    cases = [
        GoldenCase(
            id="g-1",
            input={"question": "q"},
            expected={"answer": "a"},
            failure_mode="hallucination",
            tags=["x"],
            source="error_analysis",
        )
    ]
    p = tmp_path / "g.jsonl"
    write_golden(p, cases)
    assert load_golden(p) == cases


def test_golden_rejects_unknown_fields_and_bad_source(tmp_path: Path):
    p = _write(tmp_path / "g.jsonl", [{"id": "g", "input": {}, "bogus": 1}])
    with pytest.raises(GoldenLoadError):
        load_golden(p)
    with pytest.raises(Exception):
        GoldenCase(id="g", input={}, source="made_up")


def test_eval_result_matches_plan_shape():
    r = EvalResult.model_validate(
        {
            "run_id": "r",
            "commit": "abc",
            "timestamp": "2026-10-15T00:00:00Z",
            "provider": "fake",
            "model": "m",
            "metrics": {"faithfulness": 0.9},
            "cases": [
                {
                    "id": "g-1",
                    "scores": {"faithfulness": 1},
                    "passed": True,
                    "latency_ms": 0,
                    "cost_usd": 0,
                }
            ],
        }
    )
    assert r.cases[0].scores["faithfulness"] == 1
    assert r.case_scores("faithfulness") == {"g-1": 1}


def test_gate_config_parses_plan_yaml():
    cfg = GateConfig.model_validate(
        {"metrics": {"faithfulness": {"min": 0.85, "max_drop": 0.05}}, "min_cases": 10}
    )
    assert cfg.metrics["faithfulness"].max_drop == 0.05
    with pytest.raises(Exception):
        GateConfig.model_validate({"metrics": {"x": {"minimum": 1}}})


def test_verdict_contract():
    v = Verdict.model_validate({"pass": False, "reason": "r"})
    assert v.as_dict() == {"pass": False, "reason": "r"}


def test_validate_flags_duplicates_near_duplicates_and_leaks():
    cases = [
        GoldenCase(
            id="a", input={"q": "what is the capital of france"}, expected={"answer": "Paris"}
        ),
        GoldenCase(
            id="b", input={"q": "what is the capital of france?"}, expected={"answer": "Paris"}
        ),
        GoldenCase(id="a", input={"q": "what is the capital of france"}, expected={}),
        GoldenCase(
            id="c",
            input={"q": "The treaty was signed in 1648 in Westphalia; when was it signed?"},
            expected={"answer": "signed in 1648 in Westphalia"},
        ),
        GoldenCase(
            id="d", input={"q": "how tall is the eiffel tower"}, expected={"answer": "330 m"}
        ),
    ]
    rep = validate_cases(cases)
    assert not rep.ok
    assert any("duplicate id `a`" in e for e in rep.errors)
    assert any("duplicate input" in e for e in rep.errors)
    assert any("near-duplicate" in w and "`b`" in w for w in rep.warnings)
    assert any("leaky case: `c`" in w for w in rep.warnings)
    assert not any("`d`" in w for w in rep.warnings)


def test_validate_file_reports_schema_errors(tmp_path: Path):
    p = _write(tmp_path / "g.jsonl", [{"id": "g", "input": {}}, {"id": "", "input": {}}])
    rep = validate_file(p)
    assert rep.n_cases == 1
    assert any("line 2" in e for e in rep.errors)


def test_validate_cli_exit_codes(tmp_path: Path):
    good = _write(
        tmp_path / "good.jsonl",
        [
            {"id": "1", "input": {"q": "alpha beta gamma"}, "expected": {"answer": "x"}},
            {"id": "2", "input": {"q": "delta epsilon zeta"}, "expected": {"answer": "y"}},
        ],
    )
    res = runner.invoke(app, ["validate", str(good)])
    assert res.exit_code == 0, res.output
    assert "Result: OK" in res.output

    bad = _write(tmp_path / "bad.jsonl", [{"id": "1", "input": {}}, {"id": "1", "input": {}}])
    res = runner.invoke(app, ["validate", str(bad)])
    assert res.exit_code == 1
    assert "duplicate id" in res.output

    leaky = _write(
        tmp_path / "leaky.jsonl",
        [
            {
                "id": "1",
                "input": {"q": "the answer is forty-two obviously"},
                "expected": {"answer": "forty-two obviously"},
            }
        ],
    )
    assert runner.invoke(app, ["validate", str(leaky)]).exit_code == 0
    assert runner.invoke(app, ["validate", str(leaky), "--strict"]).exit_code == 1
