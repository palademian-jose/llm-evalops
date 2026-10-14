from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from evalkit.cli import app
from evalkit.judge import (
    Judge,
    JudgeFamilyError,
    JudgeParseError,
    check_family,
    extract_json,
    judge,
)
from evalkit.judge.calibrate import calibrate, cohens_kappa, pearson
from llmops.llm import FakeClient, FakeConfig

runner = CliRunner()


def test_judge_returns_binary_justified_verdict():
    j = Judge("faithfulness", FakeClient(), model="openai/gpt-4.1-mini")
    v = j(
        {"question": "capital of france?"},
        {"answer": "paris is the capital"},
        {"contexts": ["Paris is the capital of France"]},
    )
    assert v.pass_ is True and v.reason
    assert judge(
        "faithfulness", {"question": "q"}, {"answer": "zebra giraffe"}, {"contexts": ["Paris"]}
    ) == {
        "pass": False,
        "reason": judge(
            "faithfulness", {"question": "q"}, {"answer": "zebra giraffe"}, {"contexts": ["Paris"]}
        )["reason"],
    }


def test_judge_repairs_one_malformed_answer_then_fails():
    j = Judge("faithfulness", FakeClient(FakeConfig(malformed_judge_answers=1)), model="m")
    v = j({"question": "q"}, {"answer": "paris"}, {"contexts": ["paris"]})
    assert v.pass_ is True and j.calls == 2

    j2 = Judge("faithfulness", FakeClient(FakeConfig(malformed_judge_answers=2)), model="m")
    with pytest.raises(JudgeParseError):
        j2({"question": "q"}, {"answer": "paris"}, {"contexts": ["paris"]})
    assert j2.calls == 2


def test_extract_json_tolerates_fences_and_prose():
    assert extract_json('```json\n{"pass": true, "reason": "ok"}\n```') == {
        "pass": True,
        "reason": "ok",
    }
    assert extract_json('Verdict: {"pass": false, "reason": "x"} thanks') == {
        "pass": False,
        "reason": "x",
    }
    with pytest.raises(JudgeParseError):
        extract_json("nothing here")


def test_family_check_parses_openrouter_slugs():
    assert check_family("anthropic/claude-sonnet-4.5", "openai/gpt-4.1-mini", strict=True) is None
    assert check_family("anthropic/claude-sonnet-4.5", "claude-haiku-4-5", strict=False)
    with pytest.raises(JudgeFamilyError):
        check_family("openai/gpt-4.1", "gpt-4o", strict=True)


def test_cohens_kappa_against_reference_table():
    # Classic worked example: 20 both-yes, 5 A-yes/B-no, 10 A-no/B-yes, 15 both-no -> kappa 0.4
    assert cohens_kappa(tp=20, fn=5, fp=10, tn=15) == pytest.approx(0.4)
    # Perfect agreement
    assert cohens_kappa(10, 0, 0, 10) == pytest.approx(1.0)
    # Agreement no better than chance: po = 0.5, pe = 0.5
    assert cohens_kappa(5, 5, 5, 5) == pytest.approx(0.0)
    # 45/15/25/15: po = 0.60, pe = (60*70 + 40*30) / 100^2 = 0.54 -> kappa = 0.06 / 0.46
    assert cohens_kappa(tp=45, fn=15, fp=25, tn=15) == pytest.approx(0.06 / 0.46)
    assert pearson([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)


def _labels() -> list[dict]:
    rows = []
    ctx = "Paris is the capital of France and sits on the Seine river"
    for i in range(12):
        supported = i % 3 != 0
        answer = (
            "Paris is the capital of France"
            if supported
            else "Lyon is famous for gastronomy and silk"
        )
        # one deliberately noisy human label to keep kappa < 1
        label = supported if i != 4 else (not supported)
        rows.append(
            {
                "id": f"h-{i}",
                "criterion": "faithfulness",
                "input": {"question": "capital of france?"},
                "output": {"answer": answer + (" indeed" * (i % 4))},
                "reference": {"contexts": [ctx]},
                "label": label,
            }
        )
    rows.append(
        {
            "id": "p-1",
            "criterion": "faithfulness",
            "input": {"question": "capital of france?"},
            "output_a": {"answer": "Paris is the capital of France"},
            "output_b": {"answer": "Lyon has silk"},
            "reference": {"contexts": [ctx]},
            "label": "a",
        }
    )
    return rows


def test_calibrate_reports_kappa_confusion_and_robustness(tmp_path: Path):
    rep = calibrate(
        _labels(), "faithfulness", Judge("faithfulness", FakeClient(), model="openai/gpt-4.1-mini")
    )
    assert rep.confusion.n == 12 and rep.pairwise_n == 1
    assert (rep.confusion.tp, rep.confusion.fp, rep.confusion.fn, rep.confusion.tn) == (7, 1, 0, 4)
    assert 0 < rep.kappa < 1
    assert rep.position_swap_consistency == 1.0
    md = rep.to_markdown()
    assert "Cohen's kappa" in md and "Position-swap consistency" in md and "Length bias" in md


def test_calibrate_cli(tmp_path: Path):
    labels = tmp_path / "labels.jsonl"
    labels.write_text("\n".join(json.dumps(r) for r in _labels()) + "\n", encoding="utf-8")
    md = tmp_path / "cal.md"
    res = runner.invoke(
        app, ["calibrate", "--labels", str(labels), "--criterion", "faithfulness", "--md", str(md)]
    )
    assert res.exit_code == 0, res.output
    assert "Confusion matrix" in md.read_text(encoding="utf-8")
    res = runner.invoke(
        app,
        [
            "calibrate",
            "--labels",
            str(labels),
            "--criterion",
            "faithfulness",
            "--min-kappa",
            "0.99",
        ],
    )
    assert res.exit_code == 1
    res = runner.invoke(
        app,
        [
            "calibrate",
            "--labels",
            str(labels),
            "--criterion",
            "faithfulness",
            "--judge-model",
            "anthropic/claude-haiku-4.5",
            "--strict-family",
        ],
    )
    assert res.exit_code == 1 and "same family" in res.output
