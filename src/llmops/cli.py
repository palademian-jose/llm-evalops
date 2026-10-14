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


@app.command()
def check(
    traces: list[str] | None = typer.Option(None, "--traces", help="JSONL files, dirs or globs."),
    rules: Path = typer.Option(Path("alerts.yaml"), "--rules", help="Alert rules YAML."),
    window: str | None = typer.Option(None, help="Recent window, e.g. 1h (default from rules)."),
    baseline_window: str | None = typer.Option(None, help="Baseline window, e.g. 24h."),
    now: str | None = typer.Option(
        None, help="ISO timestamp for the window end (default: latest span)."
    ),
    notify_: bool = typer.Option(
        False, "--notify", help="Send violations through the configured notifiers."
    ),
    md: Path | None = typer.Option(None, "--md", help="Write the Markdown report here."),
    record_outcome: str | None = typer.Option(
        None,
        "--record-outcome",
        help="true|false: label the last check's violations (alert precision log).",
    ),
    alert_id: str | None = typer.Option(
        None, help="Only label this alert id when recording an outcome."
    ),
    note: str = typer.Option("", help="Free-text note stored with the outcome."),
) -> None:
    """Evaluate alert rules (cost spike, p95, error rate, length drift, refusal drift); exit 1 on violations."""
    from llmops.alerts import outcomes as oc

    if record_outcome is not None:
        flag = record_outcome.strip().lower() in ("true", "1", "yes", "tp")
        n = oc.record_outcome(flag, alert_id=alert_id, note=note)
        typer.echo(f"recorded {n} outcome(s) as {'true' if flag else 'false'} positive")
        if n:
            typer.echo(oc.precision_table(oc.load_outcomes()))
        return

    from llmops.alerts import evaluate_rules, load_alerts_config, result_to_dict
    from llmops.alerts.notify import notify
    from llmops.tracing.span import parse_ts

    if not traces:
        typer.echo("--traces is required (unless --record-outcome)", err=True)
        raise typer.Exit(code=2)
    cfg = load_alerts_config(rules)
    spans = _load(traces)
    result = evaluate_rules(
        spans,
        cfg,
        now=parse_ts(now) if now else None,
        window=window,
        baseline_window=baseline_window,
    )
    text = "# Alert check\n\n" + result.to_markdown()
    outcomes = oc.load_outcomes()
    if outcomes:
        text += "\n\n## Alert precision so far\n\n" + oc.precision_table(outcomes)
    typer.echo(text)
    _write_md(text, md)
    oc.save_last_check(result_to_dict(result))
    if notify_ and result.violations:
        notify(result, cfg.notifiers)
    if not result.ok:
        raise typer.Exit(code=1)


@app.command()
def dashboard(
    traces: list[str] = typer.Option(["traces"], "--traces", help="Trace files, dirs or globs."),
    compare: str | None = typer.Option(
        None, help="Directory with one sub-directory of traces per scenario."
    ),
    rules: str | None = typer.Option(None, help="alerts.yaml for the live alerts panel."),
    port: int = typer.Option(8501, help="Port for the Streamlit server."),
    headless: bool = typer.Option(True, help="Run Streamlit headless (no browser auto-open)."),
) -> None:
    """Launch the Streamlit dashboard (requires the `dashboard` extra)."""
    import subprocess

    from llmops.dashboard import APP_PATH

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        "--server.headless",
        "true" if headless else "false",
        "--server.port",
        str(port),
        "--browser.gatherUsageStats",
        "false",
        "--",
        "--traces",
        *traces,
    ]
    if compare:
        cmd += ["--compare", compare]
    if rules:
        cmd += ["--rules", rules]
    typer.echo(f"streamlit on http://localhost:{port}", err=True)
    proc = subprocess.run(cmd)
    raise typer.Exit(code=proc.returncode)


@app.command("alert-stats")
def alert_stats() -> None:
    """Precision / false-alarm rate of recorded alert outcomes."""
    from llmops.alerts import outcomes as oc

    typer.echo("# Alert precision\n\n" + oc.precision_table(oc.load_outcomes()))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
