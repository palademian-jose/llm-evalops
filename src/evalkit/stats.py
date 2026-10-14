"""Paired statistics for A/B comparison of two eval results.

For every metric present in both results the cases are paired by id and:
- delta = mean(B) - mean(A) over the paired, non-None scores;
- 95% CI from a paired bootstrap of the per-case differences (default 10 000 resamples, seeded);
- a paired permutation (sign-flip) test p-value: the fraction of random sign assignments of
  the per-case differences whose |mean| is >= the observed |mean| (two-sided, +1 smoothing);
- for boolean metrics, McNemar's exact test on the discordant pairs (A pass/B fail = b,
  A fail/B pass = c): two-sided exact binomial p-value of b given n = b + c, p = 0.5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from evalkit.types import EvalResult


@dataclass
class MetricComparison:
    metric: str
    n: int
    mean_a: float
    mean_b: float
    delta: float
    ci_low: float
    ci_high: float
    p_perm: float
    is_bool: bool
    mcnemar_b: int | None = None
    mcnemar_c: int | None = None
    p_mcnemar: float | None = None

    @property
    def significant(self) -> bool:
        return not (self.ci_low <= 0.0 <= self.ci_high)

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__ | {"significant": self.significant}


def paired_scores(a: EvalResult, b: EvalResult, metric: str) -> tuple[np.ndarray, np.ndarray, bool]:
    """Scores for cases present in both runs with non-None values for `metric`."""
    sa, sb = a.case_scores(metric), b.case_scores(metric)
    xs, ys = [], []
    is_bool = True
    for cid in sa:
        va, vb = sa.get(cid), sb.get(cid)
        if va is None or vb is None or cid not in sb:
            continue
        if not isinstance(va, bool) or not isinstance(vb, bool):
            is_bool = False
        xs.append(float(va))
        ys.append(float(vb))
    return np.asarray(xs), np.asarray(ys), is_bool and bool(xs)


def bootstrap_ci(
    diffs: np.ndarray, n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of paired differences (vectorized, seeded)."""
    n = len(diffs)
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diffs[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def permutation_test(diffs: np.ndarray, n_perm: int = 10_000, seed: int = 0) -> float:
    """Two-sided paired sign-flip permutation p-value with +1 smoothing."""
    n = len(diffs)
    if n == 0:
        return float("nan")
    observed = abs(diffs.mean())
    if observed == 0.0:
        return 1.0
    rng = np.random.default_rng(seed + 1)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    perm_means = np.abs((signs * diffs).mean(axis=1))
    return float((np.sum(perm_means >= observed - 1e-12) + 1) / (n_perm + 1))


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    return float(stats.binomtest(b, n=n, p=0.5, alternative="two-sided").pvalue)


def compare_metric(
    a: EvalResult, b: EvalResult, metric: str, *, n_boot: int = 10_000, seed: int = 0
) -> MetricComparison | None:
    xs, ys, is_bool = paired_scores(a, b, metric)
    if len(xs) == 0:
        return None
    diffs = ys - xs
    lo, hi = bootstrap_ci(diffs, n_boot=n_boot, seed=seed)
    comp = MetricComparison(
        metric=metric,
        n=int(len(xs)),
        mean_a=float(xs.mean()),
        mean_b=float(ys.mean()),
        delta=float(diffs.mean()),
        ci_low=lo,
        ci_high=hi,
        p_perm=permutation_test(diffs, n_perm=n_boot, seed=seed),
        is_bool=is_bool,
    )
    if is_bool:
        comp.mcnemar_b = int(np.sum((xs == 1.0) & (ys == 0.0)))
        comp.mcnemar_c = int(np.sum((xs == 0.0) & (ys == 1.0)))
        comp.p_mcnemar = mcnemar_exact(comp.mcnemar_b, comp.mcnemar_c)
    return comp


def compare(
    a: EvalResult,
    b: EvalResult,
    metrics: list[str] | None = None,
    *,
    n_boot: int = 10_000,
    seed: int = 0,
) -> list[MetricComparison]:
    names = metrics or [m for m in a.metrics if m in b.metrics]
    out = []
    for m in names:
        comp = compare_metric(a, b, m, n_boot=n_boot, seed=seed)
        if comp is not None:
            out.append(comp)
    return out


def _p(x: float | None) -> str:
    if x is None or x != x:
        return "n/a"
    return "<0.001" if x < 0.001 else f"{x:.3f}"


def compare_markdown(
    comps: list[MetricComparison], *, label_a: str = "A", label_b: str = "B", n_boot: int = 10_000
) -> str:
    lines = [
        f"# Compare: `{label_a}` vs `{label_b}`",
        "",
        f"Paired by case id. 95% CI from {n_boot:,} paired bootstrap resamples; p from a paired "
        "sign-flip permutation test; McNemar exact p for boolean metrics.",
        "",
        "| Metric | n | A | B | Δ (B−A) | 95% CI | p (perm) | p (McNemar) | Verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in comps:
        if c.significant:
            verdict = "B better" if c.delta > 0 else "B worse"
        else:
            verdict = "no significant difference"
        lines.append(
            f"| {c.metric} | {c.n} | {c.mean_a:.3f} | {c.mean_b:.3f} | {c.delta:+.3f} | "
            f"[{c.ci_low:+.3f}, {c.ci_high:+.3f}] | {_p(c.p_perm)} | {_p(c.p_mcnemar)} | "
            f"{verdict} |"
        )
    if not comps:
        lines.append("| (no shared metrics) | | | | | | | | |")
    return "\n".join(lines)
