from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from evalkit.cli import app
from evalkit.io import load_eval_result, write_golden
from evalkit.runner import RunConfig, load_callable, run_eval
from evalkit.types import GoldenCase

runner = CliRunner()


def _golden(tmp_path: Path, n: int = 6) -> Path:
    cases = [
        GoldenCase(
            id=f"g-{i}",
            input={"question": f"question number {i}"},
            expected={"answer": f"answer {i}"},
        )
        for i in range(n)
    ]
    p = tmp_path / "g.jsonl"
    write_golden(p, cases)
    return p


def echo_target(input: dict) -> dict:
    from llmops.llm import FakeClient, Message
    from llmops.tracing import wrap_client

    client = wrap_client(FakeClient(), feature="test")
    client.complete([Message(role="user", content=input["question"])], model="openai/gpt-4o-mini")
    n = input["question"].rsplit(" ", 1)[-1]
    flaky = input.get("_run_index", 0) == 2 and n == "1"
    return {
        "answer": f"answer {n}" if not flaky else "wrong",
        "contexts": [f"answer {n} is the answer"],
    }


def failing_target(input: dict) -> dict:
    if input["question"].endswith("3"):
        raise RuntimeError("boom")
    return {"answer": "answer " + input["question"].rsplit(" ", 1)[-1]}


def test_run_eval_captures_scores_latency_cost_and_traces(tmp_path: Path):
    from evalkit.io import load_golden

    cases = load_golden(_golden(tmp_path))
    trace = tmp_path / "t.jsonl"
    res = run_eval(
        cases,
        echo_target,
        RunConfig(metrics=["exact_match", "faithfulness"], trace_path=str(trace), run_id="r1"),
    )
    assert (
        res.run_id == "r1"
        and res.provider == "fake"
        and res.meta["provider_note"] == "fake-provider"
    )
    assert res.metrics["exact_match"] == 1.0 and res.metrics["faithfulness"] == 1.0
    assert all(c.passed and c.cost_usd > 0 and c.latency_ms > 0 for c in res.cases)
    assert "latency_p95_ms" in res.metrics and res.metrics["cost_usd_per_case"] > 0
    lines = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert {s["kind"] for s in lines} == {"llm", "other"}
    assert all(s["feature"] in ("eval.run", "test") and s["session_id"] == "r1" for s in lines)
    assert len({s["trace_id"] for s in lines}) == 6


def test_run_eval_k_repeats_and_pass_pow_k(tmp_path: Path):
    from evalkit.io import load_golden

    cases = load_golden(_golden(tmp_path, n=3))
    res = run_eval(
        cases, echo_target, RunConfig(metrics=["task_success", "pass_pow_k"], k=3, trace_path="")
    )
    by_id = {c.id: c for c in res.cases}
    assert by_id["g-1"].scores["pass_pow_k"] is False
    assert by_id["g-1"].scores["task_success"] == 2 / 3
    assert by_id["g-0"].scores["pass_pow_k"] is True and by_id["g-0"].runs == 3
    assert res.metrics["pass_pow_k"] == pytest.approx(2 / 3, abs=1e-5)


def test_run_eval_survives_target_errors_and_concurrency(tmp_path: Path):
    from evalkit.io import load_golden

    cases = load_golden(_golden(tmp_path))
    res = run_eval(
        cases, failing_target, RunConfig(metrics=["exact_match"], concurrency=4, trace_path="")
    )
    errored = [c for c in res.cases if c.error]
    assert len(errored) == 1 and errored[0].id == "g-3" and not errored[0].passed
    assert (
        res.metrics["error_rate"] == pytest.approx(1 / 6, abs=1e-5)
        and res.metrics["exact_match"] == 1.0
    )
    assert [c.id for c in res.cases] == [f"g-{i}" for i in range(6)]


def test_load_callable_and_cli_run(tmp_path: Path, monkeypatch):
    golden = _golden(tmp_path)
    assert load_callable("tests.test_runner:echo_target").__name__ == "echo_target"
    out = tmp_path / "reports"
    res = runner.invoke(
        app,
        [
            "run",
            "--golden",
            str(golden),
            "--target",
            "tests.test_runner:echo_target",
            "--metrics",
            "exact_match,f1_token",
            "--out",
            str(out),
            "--run-id",
            "cli",
            "--no-trace",
            "--seed",
            "3",
            "--limit",
            "4",
        ],
    )
    assert res.exit_code == 0, res.output
    result = load_eval_result(out / "eval-cli.json")
    assert (
        result.meta["n_cases"] == 4
        and result.meta["seed"] == 3
        and result.metrics["f1_token"] == 1.0
    )
    bad = runner.invoke(
        app,
        [
            "run",
            "--golden",
            str(golden),
            "--target",
            "tests.test_runner:echo_target",
            "--metrics",
            "nope",
            "--out",
            str(out),
        ],
    )
    assert bad.exit_code != 0


def test_demo_targets_run_offline(env: dict[str, str], repo_root: Path, tmp_path: Path):
    code = (
        "from examples.targets import rag_baseline, rag_regressed, agent_baseline, agent_regressed\n"
        "o = rag_baseline({'question': 'How high is Mount Kilimanjaro?'})\n"
        "assert o['answer'] == 'Mount Kilimanjaro rises to 5895 metres above sea level.', o\n"
        "assert o['retrieved'][0] == 'geo-01'\n"
        "assert rag_baseline({'question': 'What is the population of Reykjavik?'})['answer'] == \"I don't know.\"\n"
        "a = agent_baseline({'task': 'Use the calculator to compute 17 * 23.'})\n"
        "assert a['answer'] == '391' and a['steps'] == 1, a\n"
        "b = agent_regressed({'task': 'Look up how high Mount Kilimanjaro is.', '_run_index': 0})\n"
        "assert isinstance(b['completed'], bool)\n"
        "print('ok')\n"
    )
    env = dict(env, TRACE_PATH=str(tmp_path / "t.jsonl"))
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=repo_root, env=env, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
