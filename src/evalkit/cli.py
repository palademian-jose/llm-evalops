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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
