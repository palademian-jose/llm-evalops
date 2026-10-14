"""`llmops` command line interface."""

from __future__ import annotations

import sys

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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
