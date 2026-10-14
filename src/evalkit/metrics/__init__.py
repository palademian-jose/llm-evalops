"""Metric registry and built-in metrics. Importing this package registers the built-ins."""

from evalkit.metrics import builtin as _builtin  # noqa: F401  (registers metrics)
from evalkit.metrics.registry import (
    REGISTRY,
    Metric,
    MetricContext,
    aggregate_scores,
    case_passed,
    compute_case_scores,
    get_metric,
    list_metrics,
    metric,
)

__all__ = [
    "REGISTRY",
    "Metric",
    "MetricContext",
    "aggregate_scores",
    "case_passed",
    "compute_case_scores",
    "get_metric",
    "list_metrics",
    "metric",
]
