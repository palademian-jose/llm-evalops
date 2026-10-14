"""`evalkit` command line interface."""

from __future__ import annotations

import sys
from pathlib import Path

import typer


def _utf8_stdio() -> None:
    """Markdown tables use `Δ`; make Windows consoles and pipes handle it."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


_utf8_stdio()

app = typer.Typer(
    name="evalkit",
    help="Golden sets, calibrated LLM judge, A/B statistics and a CI gate.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def _root() -> None:
    """evalkit: golden sets, calibrated judge, A/B statistics and a CI gate."""


def _write_md(md: str, out: Path | None) -> None:
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md + "\n", encoding="utf-8")
        typer.echo(f"wrote {out}", err=True)


@app.command()
def validate(
    golden: Path = typer.Argument(..., help="Golden set JSONL file."),
    near_dup: float = typer.Option(0.9, help="Token-overlap threshold flagged as near-dup."),
    min_leak_chars: int = typer.Option(12, help="Min expected-string length checked for leaks."),
    strict: bool = typer.Option(False, help="Treat warnings (near-dups, leaks) as failures."),
) -> None:
    """Validate a golden set: schema, duplicate/near-duplicate inputs, leaky cases."""
    from evalkit.validate import validate_file

    rep = validate_file(golden, near_dup_threshold=near_dup, min_leak_chars=min_leak_chars)
    typer.echo(rep.to_markdown())
    if not rep.ok or (strict and rep.warnings):
        raise typer.Exit(code=1)


@app.command()
def metrics() -> None:
    """List registered metrics with their definitions."""
    from evalkit.metrics import list_metrics

    typer.echo("| Metric | Kind | Judge | Definition |")
    typer.echo("|---|---|---|---|")
    for m in list_metrics():
        first = m.description.splitlines()[0] if m.description else ""
        typer.echo(f"| `{m.name}` | {m.kind} | {'yes' if m.needs_judge else 'no'} | {first} |")


@app.command()
def calibrate(
    labels: Path = typer.Option(..., "--labels", help="Human labels JSONL."),
    criterion: str = typer.Option(..., "--criterion", help="Judge criterion to calibrate."),
    md: Path | None = typer.Option(None, "--md", help="Write the Markdown report here."),
    json_out: Path | None = typer.Option(None, "--json", help="Write a JSON summary here."),
    judge_model: str | None = typer.Option(None, help="Override JUDGE_MODEL."),
    strict_family: bool = typer.Option(
        False, help="Exit 1 when judge and generator share a model family."
    ),
    min_kappa: float | None = typer.Option(
        None, help="Exit 1 when Cohen's kappa is below this value."
    ),
) -> None:
    """Compare judge verdicts with human labels: kappa, agreement, confusion, bias checks."""
    import math

    from evalkit.judge.calibrate import calibrate as run_calibration
    from evalkit.judge.calibrate import load_labels, report_json
    from evalkit.judge.judge import Judge, JudgeFamilyError

    rows = load_labels(labels, criterion)
    if not rows:
        typer.echo(f"no labels for criterion `{criterion}` in {labels}", err=True)
        raise typer.Exit(code=2)
    try:
        rep = run_calibration(
            rows, criterion, Judge(criterion, model=judge_model), strict_family=strict_family
        )
    except JudgeFamilyError as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(code=1) from e
    out_md = rep.to_markdown()
    typer.echo(out_md)
    _write_md(out_md, md)
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(report_json(rep), encoding="utf-8")
    if min_kappa is not None and (math.isnan(rep.kappa) or rep.kappa < min_kappa):
        typer.echo(f"kappa {rep.kappa:.3f} below required {min_kappa}", err=True)
        raise typer.Exit(code=1)


@app.command()
def compare(
    a: Path = typer.Option(..., "--a", help="Baseline EvalResult JSON."),
    b: Path = typer.Option(..., "--b", help="Candidate EvalResult JSON."),
    metrics: str | None = typer.Option(None, help="Comma-separated metrics (default: shared)."),
    n_boot: int = typer.Option(10_000, help="Bootstrap / permutation resamples."),
    seed: int = typer.Option(0, help="Random seed for resampling."),
    md: Path | None = typer.Option(None, "--md", help="Write the Markdown report here."),
    json_out: Path | None = typer.Option(None, "--json", help="Write per-metric stats as JSON."),
) -> None:
    """Per-metric deltas with paired bootstrap CI, permutation and McNemar tests."""
    import json

    from evalkit.io import load_eval_result
    from evalkit.stats import compare as run_compare
    from evalkit.stats import compare_markdown

    ra, rb = load_eval_result(a), load_eval_result(b)
    names = [m.strip() for m in metrics.split(",")] if metrics else None
    comps = run_compare(ra, rb, names, n_boot=n_boot, seed=seed)
    out_md = compare_markdown(comps, label_a=ra.run_id, label_b=rb.run_id, n_boot=n_boot)
    typer.echo(out_md)
    _write_md(out_md, md)
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps([c.as_dict() for c in comps], indent=2), encoding="utf-8")


@app.command()
def gate(
    candidate: Path = typer.Option(..., "--candidate", help="Candidate EvalResult JSON."),
    baseline: Path | None = typer.Option(None, "--baseline", help="Baseline EvalResult JSON."),
    config: Path = typer.Option(Path("evals.yaml"), "--config", help="Gate config YAML."),
    md: Path | None = typer.Option(None, "--md", help="Write the Markdown table here."),
    annotate: bool = typer.Option(False, help="Append the table to $GITHUB_STEP_SUMMARY."),
) -> None:
    """Fail (exit 1) when the candidate violates min / max_drop thresholds."""
    from evalkit.gate import evaluate_gate, load_gate_config, write_step_summary
    from evalkit.io import load_eval_result

    cfg = load_gate_config(config)
    base = load_eval_result(baseline) if baseline else None
    rep = evaluate_gate(base, load_eval_result(candidate), cfg)
    out_md = rep.to_markdown()
    typer.echo(out_md)
    _write_md(out_md, md)
    if annotate:
        write_step_summary(out_md)
    if not rep.ok:
        raise typer.Exit(code=1)


@app.command()
def annotate(
    md: Path = typer.Option(..., "--md", help="Markdown file to publish."),
) -> None:
    """Append a Markdown file to $GITHUB_STEP_SUMMARY (prints it when not on Actions)."""
    from evalkit.gate import write_step_summary

    text = md.read_text(encoding="utf-8")
    target = write_step_summary(text)
    if target is None:
        typer.echo(text)
    else:
        typer.echo(f"appended to {target}", err=True)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
