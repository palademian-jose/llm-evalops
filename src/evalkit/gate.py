"""CI gate: compare a candidate EvalResult with a baseline under `evals.yaml` thresholds.

Per metric:
- `min`: candidate value must be >= min.
- `max_drop`: candidate must not be below baseline by more than max_drop (needs a baseline).
- `max`: candidate value must be <= max (lower-is-better metrics such as latency or cost).
- `max_increase`: candidate must not exceed baseline by more than max_increase.
- A metric listed in the config but missing from the candidate is a violation.
`min_cases`: the candidate must contain at least this many cases.
Exit code 1 on any violation. Output is a Markdown table:
| Metric | Baseline | Candidate | Δ | Threshold | Status |
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from evalkit.types import EvalResult, GateConfig, MetricThreshold

HEADER = "| Metric | Baseline | Candidate | Δ | Threshold | Status |"


@dataclass
class GateRow:
    metric: str
    baseline: float | None
    candidate: float | None
    threshold: str
    status: str  # PASS | FAIL | WARN
    reason: str = ""

    @property
    def delta(self) -> float | None:
        if self.baseline is None or self.candidate is None:
            return None
        return self.candidate - self.baseline


@dataclass
class GateReport:
    rows: list[GateRow] = field(default_factory=list)
    baseline_id: str | None = None
    candidate_id: str | None = None

    @property
    def violations(self) -> list[GateRow]:
        return [r for r in self.rows if r.status == "FAIL"]

    @property
    def ok(self) -> bool:
        return not self.violations

    def to_markdown(self) -> str:
        lines = [HEADER, "|---|---|---|---|---|---|"]
        for r in self.rows:
            note = f" — {r.reason}" if r.reason else ""
            lines.append(
                f"| {r.metric} | {_fmt(r.baseline)} | {_fmt(r.candidate)} | "
                f"{_fmt_delta(r.delta)} | {r.threshold} | {r.status}{note} |"
            )
        verdict = "GATE PASSED" if self.ok else f"GATE FAILED ({len(self.violations)} violation(s))"
        head = f"baseline `{self.baseline_id or '-'}` -> candidate `{self.candidate_id or '-'}`"
        return "\n".join([f"**{verdict}** — {head}", ""] + lines)


def _fmt(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x:.4g}" if abs(x) >= 100 else f"{x:.3f}"


def _fmt_delta(x: float | None) -> str:
    return "—" if x is None else f"{x:+.3f}"


def _describe(t: MetricThreshold) -> str:
    parts = []
    if t.min is not None:
        parts.append(f"min {t.min:g}")
    if t.max_drop is not None:
        parts.append(f"max_drop {t.max_drop:g}")
    if t.max is not None:
        parts.append(f"max {t.max:g}")
    if t.max_increase is not None:
        parts.append(f"max_increase {t.max_increase:g}")
    return ", ".join(parts) or "—"


def load_gate_config(path: str | Path) -> GateConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return GateConfig.model_validate(data)


def evaluate_gate(
    baseline: EvalResult | None, candidate: EvalResult, config: GateConfig
) -> GateReport:
    rep = GateReport(
        baseline_id=baseline.run_id if baseline else None, candidate_id=candidate.run_id
    )
    if config.min_cases is not None:
        n = len(candidate.cases)
        ok = n >= config.min_cases
        rep.rows.append(
            GateRow(
                metric="cases",
                baseline=float(len(baseline.cases)) if baseline else None,
                candidate=float(n),
                threshold=f"min_cases {config.min_cases}",
                status="PASS" if ok else "FAIL",
                reason="" if ok else f"only {n} cases",
            )
        )
    eps = 1e-9
    for name, t in config.metrics.items():
        base = baseline.metrics.get(name) if baseline else None
        cand = candidate.metrics.get(name)
        row = GateRow(
            metric=name, baseline=base, candidate=cand, threshold=_describe(t), status="PASS"
        )
        reasons: list[str] = []
        if cand is None:
            row.status = "FAIL"
            reasons.append("metric missing in candidate")
        else:
            if t.min is not None and cand + eps < t.min:
                reasons.append(f"{cand:.3f} < min {t.min:g}")
            if t.max is not None and cand - eps > t.max:
                reasons.append(f"{cand:.3f} > max {t.max:g}")
            if t.max_drop is not None:
                if base is None:
                    row.status = "WARN"
                    reasons.append("no baseline for max_drop")
                elif (base - cand) - eps > t.max_drop:
                    reasons.append(f"dropped {base - cand:.3f} > max_drop {t.max_drop:g}")
            if t.max_increase is not None:
                if base is None:
                    row.status = "WARN"
                    reasons.append("no baseline for max_increase")
                elif (cand - base) - eps > t.max_increase:
                    reasons.append(f"rose {cand - base:.3f} > max_increase {t.max_increase:g}")
            if any(r for r in reasons if not r.startswith("no baseline")):
                row.status = "FAIL"
        row.reason = "; ".join(reasons)
        rep.rows.append(row)
    return rep


def write_step_summary(markdown: str) -> Path | None:
    """Append Markdown to $GITHUB_STEP_SUMMARY when running in GitHub Actions."""
    target = os.environ.get("GITHUB_STEP_SUMMARY")
    if not target:
        return None
    p = Path(target)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(markdown.rstrip() + "\n\n")
    return p
