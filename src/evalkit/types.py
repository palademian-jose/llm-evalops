"""Core evalkit types (PLAN.md shared contract 3)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Score = float | bool | None


class GoldenCase(BaseModel):
    """One golden-set case. `evals/golden/*.jsonl`, one case per line."""

    model_config = ConfigDict(extra="forbid")

    id: str
    input: dict[str, Any]
    expected: dict[str, Any] = Field(default_factory=dict)
    failure_mode: str | None = None
    tags: list[str] = Field(default_factory=list)
    source: Literal["error_analysis", "synthetic", "human"] = "synthetic"

    @field_validator("id")
    @classmethod
    def _non_empty_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("id must be non-empty")
        return v


class CaseResult(BaseModel):
    """Per-case entry inside an EvalResult."""

    model_config = ConfigDict(extra="allow")

    id: str
    scores: dict[str, Score] = Field(default_factory=dict)
    passed: bool = False
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    output: dict[str, Any] | None = None
    error: str | None = None
    runs: int = 1


class EvalResult(BaseModel):
    """`reports/eval-<run_id>.json`."""

    model_config = ConfigDict(extra="allow")

    run_id: str
    commit: str | None = None
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    provider: str = "fake"
    model: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    cases: list[CaseResult] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metrics", mode="before")
    @classmethod
    def _drop_unmeasured(cls, v: Any) -> Any:
        """A metric reported as null was not measured in this run (for example a recovery rate
        in a run without fault injection); treat it as absent so older reports still load."""
        if isinstance(v, dict):
            return {k: x for k, x in v.items() if x is not None}
        return v

    def case_scores(self, metric: str) -> dict[str, Score]:
        return {c.id: c.scores.get(metric) for c in self.cases}


class MetricThreshold(BaseModel):
    """Gate thresholds for one metric.

    Higher-is-better metrics use `min` (absolute floor on the candidate) and `max_drop`
    (candidate may not fall more than this below the baseline).
    Lower-is-better metrics (latency, cost) use `max` and `max_increase`.
    `optional: true` skips the metric (status SKIP) when the candidate did not measure it,
    instead of failing the gate on a missing metric.
    """

    model_config = ConfigDict(extra="forbid")

    min: float | None = None
    max_drop: float | None = None
    max: float | None = None
    max_increase: float | None = None
    optional: bool = False


class GateConfig(BaseModel):
    """`evals.yaml`: metric thresholds plus an optional minimum case count."""

    model_config = ConfigDict(extra="forbid")

    metrics: dict[str, MetricThreshold] = Field(default_factory=dict)
    min_cases: int | None = None


class Verdict(BaseModel):
    """Judge output (PLAN.md shared contract 4)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    pass_: bool = Field(alias="pass")
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"pass": self.pass_, "reason": self.reason}
