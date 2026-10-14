"""Pluggable metric registry.

A metric is a per-case function `fn(case, output, ctx) -> float | bool | None` registered with
`@metric("name")`. `None` means "not applicable for this case" and is excluded from aggregation.
Aggregation defaults to the mean over non-None scores (booleans count as 0/1); a metric can
supply its own aggregate.

Names of the form `<base>_at_<k>` (e.g. `recall_at_20`) resolve to the registered `<base>_at_k`
metric with `k` overridden, so gate configs can reference `recall_at_20` directly.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

from evalkit.types import GoldenCase, Score

if TYPE_CHECKING:
    from evalkit.judge.judge import Judge

Kind = Literal["score", "bool"]
Verifier = Callable[[GoldenCase, dict[str, Any]], bool]
JudgeFactory = Callable[[str], "Judge"]


@dataclass
class MetricContext:
    """Everything a metric may need beyond the case and the output."""

    k: int = 10
    judge_factory: JudgeFactory | None = None
    verifier: Verifier | None = None
    # All outputs produced for this case when the runner repeats it (pass^k); None otherwise.
    runs: list[dict[str, Any]] | None = None
    pass_threshold: float = 0.5
    _judges: dict[str, Judge] = field(default_factory=dict, repr=False)

    def judge(self, criterion: str) -> Judge:
        if self.judge_factory is None:
            from evalkit.judge.judge import Judge

            self.judge_factory = lambda c: Judge(c)
        if criterion not in self._judges:
            self._judges[criterion] = self.judge_factory(criterion)
        return self._judges[criterion]


MetricFn = Callable[[GoldenCase, dict[str, Any], MetricContext], Score]
Aggregate = Callable[[list[Score]], float | None]


def mean_aggregate(scores: list[Score]) -> float | None:
    vals = [float(s) for s in scores if s is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


@dataclass(frozen=True)
class Metric:
    name: str
    fn: MetricFn
    kind: Kind = "score"
    needs_judge: bool = False
    description: str = ""
    aggregate: Aggregate = mean_aggregate
    k: int | None = None  # bound k for `<base>_at_<k>` aliases

    def __call__(self, case: GoldenCase, output: dict[str, Any], ctx: MetricContext) -> Score:
        if self.k is not None and ctx.k != self.k:
            ctx = replace(ctx, k=self.k, _judges=ctx._judges)
        return self.fn(case, output, ctx)


REGISTRY: dict[str, Metric] = {}
_AT_K = re.compile(r"^(?P<base>[a-z0-9_]+)_at_(?P<k>\d+)$")


def metric(
    name: str,
    *,
    kind: Kind = "score",
    needs_judge: bool = False,
    description: str = "",
    aggregate: Aggregate = mean_aggregate,
) -> Callable[[MetricFn], MetricFn]:
    def deco(fn: MetricFn) -> MetricFn:
        REGISTRY[name] = Metric(
            name=name,
            fn=fn,
            kind=kind,
            needs_judge=needs_judge,
            description=description or (fn.__doc__ or "").strip(),
            aggregate=aggregate,
        )
        return fn

    return deco


def get_metric(name: str) -> Metric:
    if name in REGISTRY:
        return REGISTRY[name]
    m = _AT_K.match(name)
    if m and f"{m.group('base')}_at_k" in REGISTRY:
        base = REGISTRY[f"{m.group('base')}_at_k"]
        return replace(base, name=name, k=int(m.group("k")))
    raise KeyError(f"unknown metric `{name}`; known: {', '.join(sorted(REGISTRY))}")


def list_metrics() -> list[Metric]:
    return [REGISTRY[k] for k in sorted(REGISTRY)]


def compute_case_scores(
    case: GoldenCase,
    output: dict[str, Any],
    metric_names: list[str],
    ctx: MetricContext,
) -> dict[str, Score]:
    return {name: get_metric(name)(case, output, ctx) for name in metric_names}


def aggregate_scores(metric_names: list[str], per_case: list[dict[str, Score]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in metric_names:
        agg = get_metric(name).aggregate([scores.get(name) for scores in per_case])
        if agg is not None:
            out[name] = round(float(agg), 6)
    return out


def case_passed(scores: dict[str, Score], threshold: float = 0.5) -> bool:
    """A case passes when every applicable score is truthy (bool) or >= threshold (float)."""
    applicable = [s for s in scores.values() if s is not None]
    if not applicable:
        return False
    for s in applicable:
        if isinstance(s, bool):
            if not s:
                return False
        elif float(s) < threshold:
            return False
    return True
