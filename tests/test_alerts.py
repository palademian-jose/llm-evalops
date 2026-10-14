from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from llmops.alerts import (
    AlertsConfig,
    RuleConfig,
    SlackNotifier,
    evaluate_rules,
    load_outcomes,
    parse_duration,
    precision_table,
    record_outcome,
    result_to_dict,
    save_last_check,
)
from llmops.alerts.notify import EmailNotifier, format_text
from llmops.cli import app
from llmops.settings import Settings
from llmops.tracing import JsonlSink, Span, format_ts

runner = CliRunner()
NOW = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)


def _traffic(
    *,
    hours_back: float,
    n: int,
    rng: np.random.Generator,
    cost: float = 0.01,
    latency: float = 800,
    error_rate: float = 0.01,
    out_tokens_mean: float = 200,
    refusal_rate: float = 0.02,
) -> list[Span]:
    spans = []
    for i in range(n):
        t = NOW - timedelta(hours=hours_back) + timedelta(seconds=float(rng.uniform(0, 3600)))
        tid = f"{hours_back}-{i}"
        lat = float(rng.lognormal(np.log(latency), 0.2))
        err = rng.random() < error_rate
        spans.append(
            Span(
                trace_id=tid,
                span_id=tid + "r",
                name="rag",
                kind="other",
                feature="rag.answer",
                start_ts=format_ts(t),
                end_ts=format_ts(t + timedelta(milliseconds=lat)),
                latency_ms=lat,
                status="error" if err else "ok",
                error_kind="Timeout" if err else None,
            )
        )
        out_tokens = int(rng.normal(out_tokens_mean, 30))
        spans.append(
            Span(
                trace_id=tid,
                span_id=tid + "l",
                parent_span_id=tid + "r",
                name="generate",
                kind="llm",
                feature="rag.answer",
                start_ts=format_ts(t),
                end_ts=format_ts(t + timedelta(milliseconds=lat)),
                latency_ms=lat * 0.8,
                model="anthropic/claude-sonnet-4.5",
                input_tokens=1000,
                output_tokens=max(1, out_tokens),
                cost_usd=cost,
                meta={"refusal": bool(rng.random() < refusal_rate)},
            )
        )
    return spans


def _config() -> AlertsConfig:
    return AlertsConfig(
        window="1h",
        baseline_window="24h",
        rules=[
            RuleConfig(name="cost", type="cost_spike", max_increase_pct=50),
            RuleConfig(name="p95", type="p95_regression", max_increase_pct=30),
            RuleConfig(name="errors", type="error_rate_jump", max_increase_abs=0.02),
            RuleConfig(name="length", type="output_length_drift", ks_pvalue_lt=0.01),
            RuleConfig(name="refusals", type="refusal_rate_drift", max_increase_abs=0.05),
        ],
    )


def _baseline_spans(rng: np.random.Generator) -> list[Span]:
    spans = []
    for h in range(2, 26):
        spans += _traffic(hours_back=h, n=40, rng=rng)
    return spans


def test_parse_duration():
    assert parse_duration("90m") == timedelta(minutes=90)
    assert parse_duration("1.5h") == timedelta(hours=1.5)
    with pytest.raises(ValueError):
        parse_duration("soon")


def test_rules_quiet_when_window_matches_baseline():
    rng = np.random.default_rng(0)
    spans = _baseline_spans(rng) + _traffic(hours_back=1, n=60, rng=rng)
    res = evaluate_rules(spans, _config(), now=NOW)
    assert res.ok, res.to_markdown()
    assert {r.status for r in res.rows} == {"OK"}


def test_rules_fire_on_cost_latency_error_and_refusal_jumps():
    rng = np.random.default_rng(1)
    spans = _baseline_spans(rng) + _traffic(
        hours_back=1, n=60, rng=rng, cost=0.03, latency=1400, error_rate=0.10, refusal_rate=0.25
    )
    res = evaluate_rules(spans, _config(), now=NOW)
    fired = {r.rule for r in res.violations}
    assert {"cost", "p95", "errors", "refusals"} <= fired
    assert "length" not in fired
    md = res.to_markdown()
    assert "| cost | cost_spike |" in md and "VIOLATION" in md
    payload = result_to_dict(res)
    assert len(payload["violations"]) >= 4 and all(v["alert_id"] for v in payload["violations"])


def test_output_length_drift_detected_by_ks_test():
    rng = np.random.default_rng(2)
    spans = _baseline_spans(rng) + _traffic(hours_back=1, n=80, rng=rng, out_tokens_mean=320)
    res = evaluate_rules(spans, _config(), now=NOW)
    length = next(r for r in res.rows if r.rule == "length")
    assert length.status == "VIOLATION" and "KS stat" in length.detail
    # Significant but tiny shift does not fire thanks to min_shift_pct.
    cfg = AlertsConfig(
        rules=[
            RuleConfig(
                name="length", type="output_length_drift", ks_pvalue_lt=0.5, min_shift_pct=80
            )
        ]
    )
    res2 = evaluate_rules(spans, cfg, now=NOW)
    assert res2.rows[0].status == "OK"


def test_rules_skip_with_too_few_samples_and_default_now():
    rng = np.random.default_rng(3)
    spans = _baseline_spans(rng) + _traffic(hours_back=1, n=5, rng=rng)
    res = evaluate_rules(spans, _config(), now=NOW)
    assert all(r.status == "SKIPPED" for r in res.rows)
    assert res.now <= NOW


def test_feature_filter_and_pct_abs_thresholds():
    rng = np.random.default_rng(4)
    spans = _baseline_spans(rng) + _traffic(hours_back=1, n=60, rng=rng, error_rate=0.05)
    cfg = AlertsConfig(
        rules=[
            RuleConfig(
                name="e-abs", type="error_rate_jump", max_increase_abs=0.02, feature="rag.answer"
            ),
            RuleConfig(
                name="e-other", type="error_rate_jump", max_increase_abs=0.02, feature="agent.run"
            ),
        ]
    )
    res = evaluate_rules(spans, cfg, now=NOW)
    by = {r.rule: r for r in res.rows}
    assert by["e-abs"].status == "VIOLATION" and by["e-other"].status == "SKIPPED"


def test_notifiers_stdout_slack_email(capsys, monkeypatch):
    rng = np.random.default_rng(1)
    spans = _baseline_spans(rng) + _traffic(hours_back=1, n=60, rng=rng, cost=0.05)
    res = evaluate_rules(spans, _config(), now=NOW)
    posted = {}

    class Resp:
        def raise_for_status(self):
            return None

    SlackNotifier(
        "https://hooks.example/abc",
        post=lambda url, json: posted.update(url=url, body=json) or Resp(),
    ).send(res)
    assert posted["url"].startswith("https://hooks") and "cost" in posted["body"]["text"]
    assert format_text(res).startswith("llmops alert check: ")

    sent = {}

    class FakeSMTP:
        def __init__(self, host, port, timeout=0):
            sent["host"] = host

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def starttls(self):
            pass

        def login(self, u, p):
            sent["login"] = u

        def send_message(self, msg):
            sent["to"] = msg["To"]
            sent["subject"] = msg["Subject"]

    s = Settings(
        smtp_host="smtp.example", smtp_user="u", smtp_password="p", alert_email_to="a@x,b@x"
    )
    EmailNotifier(s).send(res, smtp_factory=FakeSMTP)
    assert sent == {
        "host": "smtp.example",
        "login": "u",
        "to": "a@x, b@x",
        "subject": "[llmops] alert rule violations",
    }
    with pytest.raises(RuntimeError):
        EmailNotifier(Settings()).send(res, smtp_factory=FakeSMTP)


def test_outcome_log_and_precision(tmp_path: Path):
    rng = np.random.default_rng(1)
    spans = _baseline_spans(rng) + _traffic(hours_back=1, n=60, rng=rng, cost=0.05, latency=2000)
    res = evaluate_rules(spans, _config(), now=NOW)
    save_last_check(result_to_dict(res), tmp_path)
    assert record_outcome(True, directory=tmp_path) == len(res.violations)
    cost_id = next(v.alert_id for v in res.violations if v.rule == "cost")
    assert record_outcome(False, alert_id=cost_id, directory=tmp_path) == 1
    rows = load_outcomes(tmp_path)
    assert len(rows) == len(res.violations) + 1
    table = precision_table(rows)
    assert "| cost | 2 | 1 | 1 | 50% | 50% |" in table
    assert record_outcome(True, directory=tmp_path / "empty") == 0


def test_check_cli_exit_codes_and_record_outcome(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    rng = np.random.default_rng(5)
    path = tmp_path / "spans.jsonl"
    sink = JsonlSink(path)
    for s in _baseline_spans(rng) + _traffic(hours_back=1, n=60, rng=rng, cost=0.05):
        sink.write(s)
    rules = tmp_path / "alerts.yaml"
    rules.write_text(
        "window: 1h\nbaseline_window: 24h\nrules:\n"
        "  - {name: cost, type: cost_spike, max_increase_pct: 50}\n"
        "  - {name: p95, type: p95_regression, max_increase_pct: 30}\n"
        "notifiers:\n  - type: stdout\n",
        encoding="utf-8",
    )
    alerts_dir = tmp_path / "alerts"
    monkeypatch.setenv("ALERTS_DIR", str(alerts_dir))
    from llmops.settings import reset_settings_cache

    reset_settings_cache()
    md = tmp_path / "alerts.md"
    res = runner.invoke(
        app,
        [
            "check",
            "--traces",
            str(path),
            "--rules",
            str(rules),
            "--now",
            format_ts(NOW),
            "--md",
            str(md),
            "--notify",
        ],
    )
    assert res.exit_code == 1, res.output
    assert "VIOLATION" in res.output and "llmops alert check: 1 violation" in res.output
    assert (alerts_dir / "last_check.json").exists() and "cost_spike" in md.read_text(
        encoding="utf-8"
    )

    res = runner.invoke(app, ["check", "--record-outcome", "false", "--note", "deploy rollout"])
    assert res.exit_code == 0, res.output
    res = runner.invoke(app, ["alert-stats"])
    assert res.exit_code == 0 and "| cost | 1 | 0 | 1 | 0% | 100% |" in res.output

    res = runner.invoke(
        app,
        [
            "check",
            "--traces",
            str(path),
            "--rules",
            str(rules),
            "--now",
            format_ts(NOW),
            "--window",
            "30h",
            "--baseline-window",
            "1h",
        ],
    )
    assert res.exit_code == 0, res.output  # everything in one window -> skipped baseline
    assert (
        json.loads((alerts_dir / "last_check.json").read_text(encoding="utf-8"))["violations"] == []
    )
