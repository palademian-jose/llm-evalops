"""`evalkit` command line interface."""

from __future__ import annotations

from pathlib import Path

import typer

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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
