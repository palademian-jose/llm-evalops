"""Measure the latency `llmops.tracing` adds per span and per traced LLM call.

Reports microseconds per operation for: a bare function call (floor), a span with a NullSink,
a span with a MemorySink, a span with the JSONL sink (file + lock), and a `wrap_client` call
around the fake provider (span + cost lookup). Writes a Markdown table to --report.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from llmops.llm import FakeClient, Message  # noqa: E402
from llmops.tracing import JsonlSink, MemorySink, NullSink, Tracer, wrap_client  # noqa: E402


def _time(fn, n: int, repeats: int = 5) -> tuple[float, float]:
    """Median and min microseconds per call over `repeats` batches of `n` calls."""
    per_call = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(n):
            fn()
        per_call.append((time.perf_counter() - start) / n * 1e6)
    return statistics.median(per_call), min(per_call)


def bench(n: int) -> list[tuple[str, float, float]]:
    rows = []

    def noop() -> None:
        return None

    rows.append(("bare function call (floor)",) + _time(noop, n))

    null = Tracer(NullSink())

    def span_null() -> None:
        with null.span("op", "tool", meta={"k": 1}):
            pass

    rows.append(("span, NullSink",) + _time(span_null, n))

    mem = Tracer(MemorySink())

    def span_mem() -> None:
        with mem.span("op", "tool", meta={"k": 1}):
            pass

    rows.append(("span, MemorySink",) + _time(span_mem, n))

    with tempfile.TemporaryDirectory() as tmp:
        jsonl = Tracer(JsonlSink(Path(tmp) / "spans.jsonl"))

        def span_file() -> None:
            with jsonl.span("op", "tool", meta={"k": 1}):
                pass

        rows.append(("span, JsonlSink (file + lock)",) + _time(span_file, max(200, n // 10)))

        nolock = Tracer(JsonlSink(Path(tmp) / "spans2.jsonl", use_filelock=False))

        def span_file_nolock() -> None:
            with nolock.span("op", "tool", meta={"k": 1}):
                pass

        rows.append(
            ("span, JsonlSink (file, no cross-process lock)",)
            + _time(span_file_nolock, max(200, n // 10))
        )
        jsonl.sink.close()  # type: ignore[attr-defined]
        nolock.sink.close()  # type: ignore[attr-defined]

    raw = FakeClient()
    msgs = [Message(role="user", content="benchmark prompt " * 20)]

    def call_raw() -> None:
        raw.complete(msgs, model="anthropic/claude-sonnet-4.5")

    rows.append(("fake client call, untraced (floor)",) + _time(call_raw, n // 2))
    traced = wrap_client(FakeClient(), Tracer(MemorySink()))

    def call_traced() -> None:
        traced.complete(msgs, model="anthropic/claude-sonnet-4.5")

    rows.append(("fake client call, wrap_client + MemorySink",) + _time(call_traced, n // 2))
    return rows


def to_markdown(rows: list[tuple[str, float, float]], n: int) -> str:
    floor_span = rows[0][1]
    floor_call = next(r for r in rows if r[0].startswith("fake client call, untraced"))[1]
    lines = [
        f"Python {sys.version.split()[0]}, {n:,} iterations x 5 batches, median of batch means.",
        "",
        "| Operation | median µs/op | min µs/op | added vs floor |",
        "|---|---|---|---|",
    ]
    for name, med, mn in rows:
        floor = floor_call if name.startswith("fake client") else floor_span
        lines.append(f"| {name} | {med:.1f} | {mn:.1f} | {med - floor:+.1f} µs |")
    lines += [
        "",
        "Per-span overhead with an in-memory sink is the relevant number for request latency; the "
        "JSONL sink cost is dominated by the file append and lock and is paid off the hot path when "
        "spans are written by a background thread or a sidecar.",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    rows = bench(args.n)
    md = to_markdown(rows, args.n)
    print(md)
    if args.report:
        p = Path(args.report)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# Tracing overhead benchmark\n\n" + md + "\n", encoding="utf-8")
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
