"""`llmops` command line interface."""

from __future__ import annotations

import typer

app = typer.Typer(
    name="llmops",
    help="Tracing, cost attribution, latency, caching, alerts and a dashboard.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def _root() -> None:
    """llmops: tracing, cost, latency, caching, alerts and a dashboard."""


def main() -> None:
    app()


if __name__ == "__main__":
    main()
