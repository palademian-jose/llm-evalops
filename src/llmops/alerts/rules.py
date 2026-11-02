"""Alert rules evaluated over a recent window against a baseline window.

Windows: `now` defaults to the latest `end_ts` in the loaded spans (so offline trace files can
be checked without a wall clock); the window is `[now - window, now]` and the baseline is
`[now - window - baseline_window, now - window)`.

Rule types (all accept `feature` to restrict to one feature and `min_samples`):
- cost_spike:          mean cost per request (USD/trace), `max_increase_pct`.
- p95_regression:      p95 of root-span latency, `max_increase_pct`.
- error_rate_jump:     share of root spans with status=error, `max_increase_abs` (percentage points
                       as a fraction, e.g. 0.02) and/or `max_increase_pct`.
- output_length_drift: two-sample Kolmogorov-Smirnov test on llm `output_tokens`, fires when
                       p < `ks_pvalue_lt` (default 0.01) and the median moved by more than
                       `min_shift_pct` (default 10) so a trivially significant large-n test
                       with no practical shift does not fire.
- refusal_rate_drift:  share of llm spans flagged as refusals (`meta.refusal` boolean, else any
                       `refusal_patterns` regex over `meta.output_text`), `max_increase_abs`.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats

from llmops.tracing.span import Span, format_ts, parse_ts

RuleType = Literal[
    "cost_spike", "p95_regression", "error_rate_jump", "output_length_drift", "refusal_rate_drift"
]

DEFAULT_REFUSAL_PATTERNS = [
    r"\bI can(?:'|no)t\b",
    r"\bI(?:'m| am) unable\b",
    r"\bI cannot help\b",
    r"\bI(?:'m| am) not able to\b",
    r"\bas an AI\b",
]


class RuleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: RuleType
    feature: str | None = None
    min_samples: int = 20
    max_increase_pct: float | None = None
    max_increase_abs: float | None = None
    ks_pvalue_lt: float | None = None
    min_shift_pct: float = 10.0
    refusal_patterns: list[str] | None = None


class NotifierConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["stdout", "slack", "email"]
    webhook_url: str | None = None
    webhook_url_env: str = "SLACK_WEBHOOK_URL"
    to: list[str] | None = None
    subject: str = "[llmops] alert rule violations"


class AlertsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window: str = "1h"
    baseline_window: str = "24h"
    rules: list[RuleConfig] = Field(default_factory=list)
    notifiers: list[NotifierConfig] = Field(default_factory=lambda: [NotifierConfig(type="stdout")])


def load_alerts_config(path: str | Path) -> AlertsConfig:
    return AlertsConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})


_DUR = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhdw])\s*$")


def parse_duration(text: str) -> timedelta:
    m = _DUR.match(text)
    if not m:
        raise ValueError(f"bad duration {text!r}; use e.g. 30m, 1h, 24h, 7d")
    value, unit = float(m.group(1)), m.group(2)
    mult = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}[unit]
    return timedelta(seconds=value * mult)


@dataclass
class CheckRow:
    rule: str
    type: str
    window_value: float | None
    baseline_value: float | None
    delta: float | None
    threshold: str
    status: str  # OK | VIOLATION | SKIPPED
    detail: str = ""
    window_n: int = 0
    baseline_n: int = 0

    @property
    def alert_id(self) -> str:
        return hashlib.sha1(f"{self.rule}:{self.type}:{self.detail}".encode()).hexdigest()[:10]


@dataclass
class CheckResult:
    now: datetime
    window: timedelta
    baseline_window: timedelta
    rows: list[CheckRow] = field(default_factory=list)

    @property
    def violations(self) -> list[CheckRow]:
        return [r for r in self.rows if r.status == "VIOLATION"]

    @property
    def ok(self) -> bool:
        return not self.violations

    def to_markdown(self) -> str:
        lines = [
            f"Window: {_fmt_td(self.window)} ending {format_ts(self.now)}; "
            f"baseline: previous {_fmt_td(self.baseline_window)}.",
            "",
            "| Rule | Type | Baseline | Window | Δ | Threshold | Status |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in self.rows:
            lines.append(
                f"| {r.rule} | {r.type} | {_fmt(r.baseline_value)} ({r.baseline_n}) | "
                f"{_fmt(r.window_value)} ({r.window_n}) | {_fmt(r.delta, signed=True)} | "
                f"{r.threshold} | {r.status}{' — ' + r.detail if r.detail else ''} |"
            )
        verdict = "no violations" if self.ok else f"{len(self.violations)} violation(s)"
        lines += ["", f"Result: {verdict}"]
        return "\n".join(lines)


def _fmt(x: float | None, signed: bool = False) -> str:
    if x is None:
        return "—"
    return f"{x:+.4g}" if signed else f"{x:.4g}"


def _fmt_td(td: timedelta) -> str:
    s = int(td.total_seconds())
    if s % 86400 == 0:
        return f"{s // 86400}d"
    if s % 3600 == 0:
        return f"{s // 3600}h"
    if s % 60 == 0:
        return f"{s // 60}m"
    return f"{s}s"


def latest_timestamp(spans: list[Span]) -> datetime | None:
    if not spans:
        return None
    return max(parse_ts(s.end_ts or s.start_ts) for s in spans)


def split_windows(
    spans: list[Span], now: datetime, window: timedelta, baseline_window: timedelta
) -> tuple[list[Span], list[Span]]:
    w_start = now - window
    b_start = w_start - baseline_window
    cur, base = [], []
    for s in spans:
        t = parse_ts(s.start_ts)
        if w_start <= t <= now:
            cur.append(s)
        elif b_start <= t < w_start:
            base.append(s)
    return cur, base


# ---- metric extractors -----------------------------------------------------------------------


def _roots(spans: list[Span]) -> list[Span]:
    return [s for s in spans if s.parent_span_id is None]


def cost_per_request(spans: list[Span]) -> tuple[float | None, int]:
    traces = {s.trace_id for s in spans}
    if not traces:
        return None, 0
    return sum(s.cost_usd for s in spans) / len(traces), len(traces)


def p95_latency(spans: list[Span]) -> tuple[float | None, int]:
    roots = _roots(spans)
    if not roots:
        return None, 0
    return float(np.percentile([s.latency_ms for s in roots], 95)), len(roots)


def error_rate(spans: list[Span]) -> tuple[float | None, int]:
    roots = _roots(spans)
    if not roots:
        return None, 0
    return sum(1 for s in roots if s.status == "error") / len(roots), len(roots)


def output_lengths(spans: list[Span]) -> list[float]:
    return [float(s.output_tokens) for s in spans if s.kind == "llm" and not s.cache_hit]


def is_refusal(span: Span, patterns: list[re.Pattern[str]]) -> bool | None:
    flag = span.meta.get("refusal")
    if isinstance(flag, bool):
        return flag
    text = span.meta.get("output_text")
    if not isinstance(text, str):
        return None
    return any(p.search(text) for p in patterns)


def refusal_rate(spans: list[Span], patterns: list[re.Pattern[str]]) -> tuple[float | None, int]:
    flags = [f for s in spans if s.kind == "llm" and (f := is_refusal(s, patterns)) is not None]
    if not flags:
        return None, 0
    return sum(flags) / len(flags), len(flags)


# ---- evaluation ------------------------------------------------------------------------------


def _pct_increase(base: float, cur: float) -> float | None:
    if base == 0:
        return None if cur == 0 else float("inf")
    return (cur - base) / abs(base) * 100


def evaluate_rule(rule: RuleConfig, window: list[Span], baseline: list[Span]) -> CheckRow:
    if rule.feature:
        window = [s for s in window if s.feature == rule.feature]
        baseline = [s for s in baseline if s.feature == rule.feature]
    row = CheckRow(rule=rule.name, type=rule.type, window_value=None, baseline_value=None,
                   delta=None, threshold="", status="OK")

    if rule.type == "output_length_drift":
        cur, base = output_lengths(window), output_lengths(baseline)
        p_lt = rule.ks_pvalue_lt if rule.ks_pvalue_lt is not None else 0.01
        row.threshold = f"KS p < {p_lt:g} and median shift > {rule.min_shift_pct:g}%"
        row.window_n, row.baseline_n = len(cur), len(base)
        if len(cur) < rule.min_samples or len(base) < rule.min_samples:
            row.status = "SKIPPED"
            row.detail = f"need {rule.min_samples} samples on both sides"
            return row
        ks = stats.ks_2samp(cur, base)
        med_b, med_c = float(np.median(base)), float(np.median(cur))
        shift = _pct_increase(med_b, med_c)
        row.baseline_value, row.window_value = med_b, med_c
        row.delta = med_c - med_b
        row.detail = f"KS stat {ks.statistic:.3f}, p={ks.pvalue:.3g}, median {med_b:.0f}->{med_c:.0f}"
        if ks.pvalue < p_lt and shift is not None and abs(shift) > rule.min_shift_pct:
            row.status = "VIOLATION"
        return row

    if rule.type == "refusal_rate_drift":
        pats = [re.compile(p, re.I) for p in (rule.refusal_patterns or DEFAULT_REFUSAL_PATTERNS)]
        cur_v, cur_n = refusal_rate(window, pats)
        base_v, base_n = refusal_rate(baseline, pats)
    elif rule.type == "cost_spike":
        cur_v, cur_n = cost_per_request(window)
        base_v, base_n = cost_per_request(baseline)
    elif rule.type == "p95_regression":
        cur_v, cur_n = p95_latency(window)
        base_v, base_n = p95_latency(baseline)
    elif rule.type == "error_rate_jump":
        cur_v, cur_n = error_rate(window)
        base_v, base_n = error_rate(baseline)
    else:  # pragma: no cover - pydantic restricts the literal
        raise ValueError(rule.type)

    row.window_n, row.baseline_n = cur_n, base_n
    row.window_value, row.baseline_value = cur_v, base_v
    parts = []
    if rule.max_increase_pct is not None:
        parts.append(f"+{rule.max_increase_pct:g}%")
    if rule.max_increase_abs is not None:
        parts.append(f"+{rule.max_increase_abs:g} abs")
    row.threshold = " or ".join(parts) or "—"
    if cur_v is None or base_v is None or cur_n < rule.min_samples or base_n < rule.min_samples:
        row.status = "SKIPPED"
        row.detail = f"need {rule.min_samples} samples on both sides"
        return row
    row.delta = cur_v - base_v
    pct = _pct_increase(base_v, cur_v)
    reasons = []
    if rule.max_increase_pct is not None and pct is not None and pct > rule.max_increase_pct:
        reasons.append(f"+{pct:.1f}% > {rule.max_increase_pct:g}%")
    if rule.max_increase_abs is not None and row.delta > rule.max_increase_abs:
        reasons.append(f"+{row.delta:.4g} > {rule.max_increase_abs:g}")
    if reasons:
        row.status = "VIOLATION"
        row.detail = "; ".join(reasons)
    elif pct is not None and pct != float("inf"):
        row.detail = f"{pct:+.1f}%"
    return row


def evaluate_rules(
    spans: list[Span],
    config: AlertsConfig,
    *,
    now: datetime | None = None,
    window: str | None = None,
    baseline_window: str | None = None,
) -> CheckResult:
    w = parse_duration(window or config.window)
    b = parse_duration(baseline_window or config.baseline_window)
    ts = now or latest_timestamp(spans)
    if ts is None:
        raise ValueError("no spans to check")
    cur, base = split_windows(spans, ts, w, b)
    result = CheckResult(now=ts, window=w, baseline_window=b)
    for rule in config.rules:
        result.rows.append(evaluate_rule(rule, cur, base))
    return result


def result_to_dict(result: CheckResult) -> dict[str, Any]:
    return {
        "now": format_ts(result.now),
        "window": _fmt_td(result.window),
        "baseline_window": _fmt_td(result.baseline_window),
        "violations": [
            {
                "alert_id": r.alert_id,
                "rule": r.rule,
                "type": r.type,
                "baseline": r.baseline_value,
                "window": r.window_value,
                "delta": r.delta,
                "threshold": r.threshold,
                "detail": r.detail,
            }
            for r in result.violations
        ],
        "checks": [r.__dict__ | {"alert_id": r.alert_id} for r in result.rows],
    }
