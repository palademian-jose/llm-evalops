"""`llmops` command line interface."""

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
    name="llmops",
    help="Tracing, cost attribution, latency, caching, alerts and a dashboard.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def _root() -> None:
    """llmops: tracing, cost, latency, caching, alerts and a dashboard."""


def _load(traces: list[str]):
    from llmops.ingest import load_spans

    spans = load_spans(traces)
    if not spans:
        typer.echo(f"no spans found in {traces}", err=True)
        raise typer.Exit(code=2)
    return spans


def _write_md(md: str, out: Path | None) -> None:
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md + "\n", encoding="utf-8")
        typer.echo(f"wrote {out}", err=True)


@app.command()
def cost(
    traces: list[str] = typer.Option(..., "--traces", help="JSONL files, dirs or globs."),
    by: str = typer.Option("feature", "--by", help="request | feature | user | model | day"),
    top: int | None = typer.Option(None, help="Only show the top N rows."),
    validate_against: Path | None = typer.Option(
        None, "--validate-against", help="Provider bill CSV to reconcile against."
    ),
    md: Path | None = typer.Option(None, "--md", help="Write Markdown here."),
) -> None:
    """Cost attribution tables and optional reconciliation against a provider bill."""
    from llmops.cost import cost_markdown, cost_table, reconcile

    if by not in ("request", "feature", "user", "model", "day"):
        typer.echo("--by must be one of request|feature|user|model|day", err=True)
        raise typer.Exit(code=2)
    spans = _load(traces)
    rows = cost_table(spans, by, top=top)  # type: ignore[arg-type]
    out = [f"# Cost by {by}", "", cost_markdown(rows, by)]  # type: ignore[arg-type]
    if validate_against:
        out += ["", "# Reconciliation", "", reconcile(spans, validate_against).to_markdown()]
    text = "\n".join(out)
    typer.echo(text)
    _write_md(text, md)


@app.command()
def latency(
    traces: list[str] = typer.Option(..., "--traces", help="JSONL files, dirs or globs."),
    by: str = typer.Option("feature", "--by", help="stage | feature"),
    md: Path | None = typer.Option(None, "--md", help="Write Markdown here."),
) -> None:
    """p50/p90/p95/p99 latency per feature or per stage, TTFT when recorded."""
    from llmops.latency import latency_markdown, latency_table

    if by not in ("stage", "feature"):
        typer.echo("--by must be stage or feature", err=True)
        raise typer.Exit(code=2)
    spans = _load(traces)
    rows = latency_table(spans, by)  # type: ignore[arg-type]
    text = f"# Latency by {by}\n\n" + latency_markdown(rows, by)  # type: ignore[arg-type]
    typer.echo(text)
    _write_md(text, md)


@app.command("cache-stats")
def cache_stats_cmd(
    traces: list[str] = typer.Option(..., "--traces", help="JSONL files, dirs or globs."),
    md: Path | None = typer.Option(None, "--md", help="Write Markdown here."),
) -> None:
    """Cache hit rate and estimated dollars / milliseconds saved."""
    from llmops.cache.stats import cache_markdown, cache_stats

    spans = _load(traces)
    text = "# Cache statistics\n\n" + cache_markdown(cache_stats(spans), spans)
    typer.echo(text)
    _write_md(text, md)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
