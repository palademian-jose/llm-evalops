"""Span sinks: append-only JSONL (thread- and process-safe), in-memory, and fan-out."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import IO, Protocol

from filelock import FileLock

from llmops.tracing.span import Span


class Sink(Protocol):
    def write(self, span: Span) -> None: ...


class MemorySink:
    def __init__(self) -> None:
        self.spans: list[Span] = []
        self._lock = threading.Lock()

    def write(self, span: Span) -> None:
        with self._lock:
            self.spans.append(span)

    def clear(self) -> None:
        with self._lock:
            self.spans.clear()


class JsonlSink:
    """Append one JSON line per span.

    The file is opened once in append mode and kept open. Each write is a single `write()` of a
    complete line followed by `flush()` under a process-local lock, so threads never interleave
    partial lines. With `use_filelock=True` (default) a cross-process `filelock` on
    `<path>.lock` additionally serialises writers from different processes; disabling it relies
    on O_APPEND atomicity for lines shorter than the OS pipe buffer and is several times faster.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        use_filelock: bool = True,
        per_process: bool = False,
    ):
        """`per_process=True` writes to `<stem>-<pid>.jsonl` and skips the cross-process lock;
        ingest the directory (or a glob) to read every process's file."""
        p = Path(path)
        if per_process:
            p = p.with_name(f"{p.stem}-{os.getpid()}{p.suffix}")
            use_filelock = False
        self.path = p
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._flock = FileLock(str(self.path) + ".lock") if use_filelock else None
        self._fh: IO[str] | None = None

    def _handle(self) -> IO[str]:
        if self._fh is None or self._fh.closed:
            self._fh = self.path.open("a", encoding="utf-8")
        return self._fh

    def write(self, span: Span) -> None:
        line = span.to_json() + "\n"
        with self._lock:
            if self._flock is not None:
                with self._flock:
                    self._append(line)
            else:
                self._append(line)

    def _append(self, line: str) -> None:
        fh = self._handle()
        fh.write(line)
        fh.flush()

    def close(self) -> None:
        with self._lock:
            if self._fh is not None and not self._fh.closed:
                self._fh.close()

    def __del__(self) -> None:  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass


class MultiSink:
    def __init__(self, *sinks: Sink):
        self.sinks = list(sinks)

    def write(self, span: Span) -> None:
        for s in self.sinks:
            s.write(span)


class NullSink:
    def write(self, span: Span) -> None:
        return None
