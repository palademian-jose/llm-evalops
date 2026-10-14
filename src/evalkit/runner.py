"""Runner: execute a target callable over a golden set and write an EvalResult.

The target is `module:callable` with signature `(input: dict) -> dict`. The runner injects
`_run_index` and `_seed` into a copy of the input so repeated runs (`--k`) can vary their seed.
Every case runs inside an llmops trace (`feature="eval.run"`), so a target that calls an LLM
through `llmops.tracing.wrap_client` has its latency and cost captured automatically:
`latency_ms` is the wall time of the target call, `cost_usd` is the sum of `cost_usd` over
all spans emitted during that call.

With `--k > 1`, numeric metrics are averaged across the k runs; boolean metrics become the
fraction of passing runs; `pass_pow_k` is true only when ALL k runs succeed.
"""

from __future__ import annotations

import importlib
import logging
import secrets
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from evalkit.metrics import (
    MetricContext,
    aggregate_scores,
    case_passed,
    compute_case_scores,
    get_metric,
)
from evalkit.types import CaseResult, EvalResult, GoldenCase, Score
from llmops.settings import get_settings
from llmops.tracing import (
    JsonlSink,
    MemorySink,
    MultiSink,
    NullSink,
    Tracer,
    get_tracer,
    set_tracer,
)

log = logging.getLogger("evalkit.runner")

Target = Callable[[dict[str, Any]], dict[str, Any]]


def load_callable(spec: str) -> Callable[..., Any]:
    """Import `module:attr` (or `module.attr`), with the current directory on sys.path."""
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    if ":" in spec:
        mod_name, attr = spec.split(":", 1)
    else:
        mod_name, _, attr = spec.rpartition(".")
    if not mod_name:
        raise ValueError(f"target must look like module:callable, got {spec!r}")
    mod = importlib.import_module(mod_name)
    obj: Any = mod
    for part in attr.split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise TypeError(f"{spec} is not callable")
    return obj


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


@dataclass
class RunConfig:
    metrics: list[str]
    k: int = 1
    seed: int = 0
    concurrency: int = 1
    run_id: str | None = None
    verifier: Callable[[GoldenCase, dict[str, Any]], bool] | None = None
    judge_model: str | None = None
    trace_path: str | None = None  # None -> settings.trace_path; "" -> no file
    limit: int | None = None
    pass_threshold: float = 0.5
    extra_meta: dict[str, Any] = field(default_factory=dict)


def _default_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + secrets.token_hex(2)


def _merge_runs(per_run: list[dict[str, Score]]) -> dict[str, Score]:
    if len(per_run) == 1:
        return per_run[0]
    merged: dict[str, Score] = {}
    for name in per_run[0]:
        vals = [r.get(name) for r in per_run]
        known = [v for v in vals if v is not None]
        if not known:
            merged[name] = None
        elif name == "pass_pow_k":
            merged[name] = all(bool(v) for v in known)
        else:
            merged[name] = float(np.mean([float(v) for v in known]))
    return merged


def run_eval(cases: list[GoldenCase], target: Target, cfg: RunConfig) -> EvalResult:
    settings = get_settings()
    for name in cfg.metrics:
        get_metric(name)  # fail fast on unknown metrics
    if cfg.limit:
        cases = cases[: cfg.limit]
    run_id = cfg.run_id or _default_run_id()

    memory = MemorySink()
    trace_path = settings.trace_path if cfg.trace_path is None else cfg.trace_path
    file_sink = JsonlSink(trace_path) if trace_path else NullSink()
    tracer = Tracer(MultiSink(memory, file_sink), feature="eval.run", session_id=run_id)
    previous = get_tracer()
    set_tracer(tracer)

    judge_factory = None
    if cfg.judge_model:
        from evalkit.judge import Judge

        judge_factory = lambda c: Judge(c, model=cfg.judge_model)  # noqa: E731

    def run_case(case: GoldenCase) -> CaseResult:
        ctx = MetricContext(
            judge_factory=judge_factory, verifier=cfg.verifier, pass_threshold=cfg.pass_threshold
        )
        outputs: list[dict[str, Any]] = []
        latencies: list[float] = []
        cost = 0.0
        error: str | None = None
        for i in range(cfg.k):
            payload = dict(case.input)
            payload["_run_index"] = i
            payload["_seed"] = cfg.seed + i
            with tracer.trace(user_id=case.id) as trace_id:
                start = time.perf_counter()
                try:
                    with tracer.span("target", "other", meta={"case_id": case.id, "run": i}):
                        out = target(payload)
                        if not isinstance(out, dict):
                            raise TypeError(f"target returned {type(out).__name__}, expected dict")
                except Exception as e:  # noqa: BLE001 - the runner must survive target failures
                    error = f"{type(e).__name__}: {e}"
                    log.warning("case %s run %d failed: %s", case.id, i, error)
                    out = {}
                latencies.append((time.perf_counter() - start) * 1000)
            cost += sum(
                s.cost_usd for s in memory.spans if s.trace_id == trace_id and s.kind != "other"
            )
            outputs.append(out)
        per_run: list[dict[str, Score]] = []
        for out in outputs:
            if error and not out:
                per_run.append({m: None for m in cfg.metrics})
                continue
            ctx.runs = outputs
            per_run.append(compute_case_scores(case, out, cfg.metrics, ctx))
        scores = _merge_runs(per_run)
        return CaseResult(
            id=case.id,
            scores=scores,
            passed=case_passed(scores, cfg.pass_threshold) and error is None,
            latency_ms=round(float(np.mean(latencies)), 3),
            cost_usd=round(cost, 8),
            output=outputs[0] if outputs else None,
            error=error,
            runs=cfg.k,
        )

    try:
        if cfg.concurrency > 1:
            with ThreadPoolExecutor(max_workers=cfg.concurrency) as ex:
                results = list(ex.map(run_case, cases))
        else:
            results = [run_case(c) for c in cases]
    finally:
        set_tracer(previous)

    metrics = aggregate_scores(cfg.metrics, [r.scores for r in results])
    lat = np.array([r.latency_ms for r in results]) if results else np.array([0.0])
    total_cost = float(sum(r.cost_usd for r in results))
    n_runs = max(1, len(results) * cfg.k)
    metrics["latency_ms_mean"] = round(float(lat.mean()), 3)
    metrics["latency_p95_ms"] = round(float(np.percentile(lat, 95)), 3)
    metrics["cost_usd_per_case"] = round(total_cost / n_runs, 8)
    metrics["error_rate"] = round(sum(1 for r in results if r.error) / max(1, len(results)), 6)

    return EvalResult(
        run_id=run_id,
        commit=git_commit(),
        provider=settings.resolved_provider(),
        model=settings.llm_model,
        metrics=metrics,
        cases=results,
        meta={
            "metrics_requested": cfg.metrics,
            "k": cfg.k,
            "seed": cfg.seed,
            "concurrency": cfg.concurrency,
            "n_cases": len(results),
            "n_errors": sum(1 for r in results if r.error),
            "cost_usd_total": round(total_cost, 8),
            "judge_model": cfg.judge_model or settings.judge_model,
            "trace_path": trace_path or None,
            "provider_note": "fake-provider" if settings.resolved_provider() == "fake" else None,
            **cfg.extra_meta,
        },
    )
