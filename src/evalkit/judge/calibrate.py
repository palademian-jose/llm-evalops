"""Judge calibration against human labels.

Label file (`evals/human_labels.jsonl`), one JSON object per line:
- binary criteria: `{"id", "criterion", "input", "output", "reference"?, "label": true|false}`
- pairwise items:  `{"id", "criterion", "input", "output_a", "output_b", "reference"?,
                     "label": "a"|"b"}`

Reported:
- Cohen's kappa, raw agreement %, confusion matrix (human label as truth).
- Position-swap consistency: for pairwise items the judge sees (a, b) and (b, a); the verdict is
  consistent when it picks the same underlying output both times.
- Length bias: Pearson correlation between output length (characters) and the judge's pass
  verdict, compared with the same correlation for the human labels. A judge that is much more
  length-correlated than the humans is rewarding verbosity.
- Family check: warns (or fails with --strict-family) when the judge model shares a family with
  the generator model.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evalkit.io import iter_jsonl
from evalkit.judge.judge import Judge, JudgeParseError, check_family, render_value
from llmops.settings import get_settings


@dataclass
class Confusion:
    tp: int = 0  # human pass, judge pass
    fp: int = 0  # human fail, judge pass
    fn: int = 0  # human pass, judge fail
    tn: int = 0  # human fail, judge fail

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    def add(self, human: bool, judge: bool) -> None:
        if human and judge:
            self.tp += 1
        elif human and not judge:
            self.fn += 1
        elif not human and judge:
            self.fp += 1
        else:
            self.tn += 1


def cohens_kappa(tp: int, fp: int, fn: int, tn: int) -> float:
    """Cohen's kappa for two raters on a binary label.

    po = observed agreement, pe = chance agreement from the marginals:
    pe = P(both say yes) + P(both say no) = (r1_yes * r2_yes + r1_no * r2_no) / n^2.
    kappa = (po - pe) / (1 - pe); returns 1.0 for perfect agreement with pe == 1.
    """
    n = tp + fp + fn + tn
    if n == 0:
        return float("nan")
    po = (tp + tn) / n
    human_yes, judge_yes = tp + fn, tp + fp
    human_no, judge_no = fp + tn, fn + tn
    pe = (human_yes * judge_yes + human_no * judge_no) / (n * n)
    if math.isclose(pe, 1.0):
        return 1.0 if math.isclose(po, 1.0) else 0.0
    return (po - pe) / (1 - pe)


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return float("nan")
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    return sxy / math.sqrt(sxx * syy)


@dataclass
class CalibrationReport:
    criterion: str
    judge_model: str
    generator_model: str
    confusion: Confusion = field(default_factory=Confusion)
    kappa: float = float("nan")
    agreement: float = float("nan")
    parse_failures: int = 0
    pairwise_n: int = 0
    pairwise_consistent: int = 0
    pairwise_agreement: int = 0
    length_corr_judge: float = float("nan")
    length_corr_human: float = float("nan")
    length_buckets: list[tuple[str, int, float, float]] = field(default_factory=list)
    family_warning: str | None = None
    disagreements: list[dict[str, Any]] = field(default_factory=list)

    @property
    def position_swap_consistency(self) -> float:
        return self.pairwise_consistent / self.pairwise_n if self.pairwise_n else float("nan")

    def to_markdown(self) -> str:
        c = self.confusion
        fmt = _fmt
        pair_agree = self.pairwise_agreement / self.pairwise_n if self.pairwise_n else float("nan")
        lines = [
            f"# Judge calibration: `{self.criterion}`",
            "",
            f"- judge model: `{self.judge_model}` (family {_family(self.judge_model)})",
            f"- generator model: `{self.generator_model}` (family {_family(self.generator_model)})",
            f"- labeled items: {c.n} binary, {self.pairwise_n} pairwise",
            f"- parse failures (after one repair): {self.parse_failures}",
        ]
        if self.family_warning:
            lines.append(f"- WARNING: {self.family_warning}")
        lines += [
            "",
            "## Agreement with human labels",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Cohen's kappa | {fmt(self.kappa)} |",
            f"| Agreement | {fmt(self.agreement, pct=True)} |",
            f"| Judge pass rate | {fmt((c.tp + c.fp) / c.n if c.n else float('nan'), pct=True)} |",
            f"| Human pass rate | {fmt((c.tp + c.fn) / c.n if c.n else float('nan'), pct=True)} |",
            "",
            "## Confusion matrix (rows: human, columns: judge)",
            "",
            "| | judge PASS | judge FAIL |",
            "|---|---|---|",
            f"| human PASS | {c.tp} | {c.fn} |",
            f"| human FAIL | {c.fp} | {c.tn} |",
            "",
            "## Robustness checks",
            "",
            "| Check | Value | Note |",
            "|---|---|---|",
            f"| Position-swap consistency | {fmt(self.position_swap_consistency, pct=True)} | "
            f"pairwise items judged in both orders ({self.pairwise_n}) |",
            f"| Pairwise agreement with humans | {fmt(pair_agree, pct=True)} | "
            "winner matches the human label (original order) |",
            f"| Length bias r (judge) | {fmt(self.length_corr_judge)} | "
            "Pearson r between output length and judge PASS |",
            f"| Length bias r (human) | {fmt(self.length_corr_human)} | "
            "same correlation for human labels; a gap > 0.2 suggests verbosity bias |",
        ]
        if self.length_buckets:
            lines += [
                "",
                "## Pass rate by output length",
                "",
                "| Length bucket | n | judge pass | human pass |",
                "|---|---|---|---|",
            ]
            for name, n, jp, hp in self.length_buckets:
                lines.append(f"| {name} | {n} | {fmt(jp, pct=True)} | {fmt(hp, pct=True)} |")
        if self.disagreements:
            lines += ["", "## Disagreements (first 10)", ""]
            for d in self.disagreements[:10]:
                lines.append(
                    f"- `{d['id']}`: human={d['human']} judge={d['judge']} — {d['reason'][:140]}"
                )
        lines.append("")
        lines.append(_interpret_kappa(self.kappa))
        return "\n".join(lines)


def _family(model: str) -> str:
    from llmops.llm import model_family

    return model_family(model)


def _fmt(x: float, pct: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x * 100:.1f}%" if pct else f"{x:.3f}"


def _interpret_kappa(k: float) -> str:
    if math.isnan(k):
        return "Interpretation: not enough binary labels for kappa."
    if k < 0.2:
        band = "slight"
    elif k < 0.4:
        band = "fair"
    elif k < 0.6:
        band = "moderate"
    elif k < 0.8:
        band = "substantial"
    else:
        band = "almost perfect"
    return (
        f"Interpretation: kappa {k:.2f} = {band} agreement (Landis & Koch bands). "
        "Gate on judge-backed metrics only when kappa >= 0.6."
    )


def load_labels(path: str | Path, criterion: str | None = None) -> list[dict[str, Any]]:
    rows = [obj for _, obj in iter_jsonl(path)]
    if criterion:
        rows = [r for r in rows if r.get("criterion", criterion) == criterion]
    return rows


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "pass", "yes", "1")
    return bool(v)


def calibrate(
    labels: list[dict[str, Any]],
    criterion: str,
    judge: Judge | None = None,
    *,
    strict_family: bool = False,
) -> CalibrationReport:
    settings = get_settings()
    j = judge or Judge(criterion)
    rep = CalibrationReport(
        criterion=criterion, judge_model=j.model, generator_model=settings.llm_model
    )
    try:
        rep.family_warning = check_family(settings.llm_model, j.model, strict=strict_family)
    except Exception:
        raise

    lengths_j: list[tuple[float, float, float]] = []  # (len, judge, human)
    for row in labels:
        if "output_a" in row and "output_b" in row:
            _calibrate_pairwise(row, j, rep)
            continue
        human = _as_bool(row["label"])
        try:
            verdict = j(row["input"], row["output"], row.get("reference"))
        except JudgeParseError:
            rep.parse_failures += 1
            continue
        rep.confusion.add(human, verdict.pass_)
        length = float(len(render_value(row["output"])))
        lengths_j.append((length, 1.0 if verdict.pass_ else 0.0, 1.0 if human else 0.0))
        if human != verdict.pass_:
            rep.disagreements.append(
                {
                    "id": row.get("id", "?"),
                    "human": human,
                    "judge": verdict.pass_,
                    "reason": verdict.reason,
                }
            )

    c = rep.confusion
    rep.kappa = cohens_kappa(c.tp, c.fp, c.fn, c.tn)
    rep.agreement = (c.tp + c.tn) / c.n if c.n else float("nan")
    if lengths_j:
        ls = [x[0] for x in lengths_j]
        rep.length_corr_judge = pearson(ls, [x[1] for x in lengths_j])
        rep.length_corr_human = pearson(ls, [x[2] for x in lengths_j])
        rep.length_buckets = _length_buckets(lengths_j)
    return rep


def _calibrate_pairwise(row: dict[str, Any], j: Judge, rep: CalibrationReport) -> None:
    human = str(row["label"]).strip().lower()
    try:
        first = j.pairwise(row["input"], row["output_a"], row["output_b"], row.get("reference"))
        swapped = j.pairwise(row["input"], row["output_b"], row["output_a"], row.get("reference"))
    except JudgeParseError:
        rep.parse_failures += 1
        return
    rep.pairwise_n += 1
    # In the swapped order, "a" refers to the original b.
    swapped_original = "b" if swapped["winner"] == "a" else "a"
    if first["winner"] == swapped_original:
        rep.pairwise_consistent += 1
    if first["winner"] == human:
        rep.pairwise_agreement += 1


def _length_buckets(rows: list[tuple[float, float, float]]) -> list[tuple[str, int, float, float]]:
    ordered = sorted(rows, key=lambda r: r[0])
    n = len(ordered)
    if n < 3:
        return []
    thirds = [ordered[: n // 3], ordered[n // 3 : 2 * n // 3], ordered[2 * n // 3 :]]
    names = ["short", "medium", "long"]
    out = []
    for name, chunk in zip(names, thirds, strict=True):
        if not chunk:
            continue
        lo, hi = int(chunk[0][0]), int(chunk[-1][0])
        jp = sum(r[1] for r in chunk) / len(chunk)
        hp = sum(r[2] for r in chunk) / len(chunk)
        out.append((f"{name} ({lo}-{hi} chars)", len(chunk), jp, hp))
    return out


def report_json(rep: CalibrationReport) -> str:
    c = rep.confusion
    return json.dumps(
        {
            "criterion": rep.criterion,
            "judge_model": rep.judge_model,
            "generator_model": rep.generator_model,
            "n": c.n,
            "kappa": None if math.isnan(rep.kappa) else rep.kappa,
            "agreement": None if math.isnan(rep.agreement) else rep.agreement,
            "confusion": {"tp": c.tp, "fp": c.fp, "fn": c.fn, "tn": c.tn},
            "position_swap_consistency": None
            if math.isnan(rep.position_swap_consistency)
            else rep.position_swap_consistency,
            "length_corr_judge": None
            if math.isnan(rep.length_corr_judge)
            else rep.length_corr_judge,
            "length_corr_human": None
            if math.isnan(rep.length_corr_human)
            else rep.length_corr_human,
            "parse_failures": rep.parse_failures,
            "family_warning": rep.family_warning,
        },
        indent=2,
    )
