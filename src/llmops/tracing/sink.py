"""Span sinks: append-only JSONL (thread- and process-safe), in-memory, and fan-out."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Protocol

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

    Each write is a single `write()` of a complete line followed by `flush()`, under a
    process-local lock and a cross-process `filelock` on `<path>.lock`, so concurrent writers
    never interleave partial lines.
    """

    def __init__(self, path: str | os.PathLike[str], *, use_filelock: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._flock = FileLock(str(self.path) + ".lock") if use_filelock else None

    def write(self, span: Span) -> None:
        line = span.to_json() + "\n"
        with self._lock:
            if self._flock is not None:
                with self._flock:
                    self._append(line)
            else:
                self._append(line)

    def _append(self, line: str) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()


class MultiSink:
    def __init__(self, *sinks: Sink):
        self.sinks = list(sinks)

    def write(self, span: Span) -> None:
        for s in self.sinks:
            s.write(span)


class NullSink:
    def write(self, span: Span) -> None:
        return None
