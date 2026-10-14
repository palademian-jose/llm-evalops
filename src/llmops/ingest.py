"""Load spans from JSONL files, directories or glob patterns (tolerating bad lines)."""

from __future__ import annotations

import glob
import json
import logging
from collections.abc import Iterable
from pathlib import Path

from pydantic import ValidationError

from llmops.tracing.span import Span

log = logging.getLogger("llmops.ingest")


def resolve_paths(sources: Iterable[str | Path]) -> list[Path]:
    out: list[Path] = []
    for src in sources:
        s = str(src)
        p = Path(s)
        if p.is_dir():
            out.extend(sorted(p.rglob("*.jsonl")))
        elif any(ch in s for ch in "*?["):
            out.extend(sorted(Path(x) for x in glob.glob(s, recursive=True)))
        elif p.is_file():
            out.append(p)
        else:
            log.warning("trace source not found: %s", s)
    seen: set[Path] = set()
    unique = []
    for p in out:
        rp = p.resolve()
        if rp not in seen and rp.suffix == ".jsonl":
            seen.add(rp)
            unique.append(p)
    return unique


def load_spans(sources: Iterable[str | Path] | str | Path) -> list[Span]:
    if isinstance(sources, str | Path):
        sources = [sources]
    spans: list[Span] = []
    bad = 0
    for path in resolve_paths(sources):
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    spans.append(Span.model_validate(json.loads(line)))
                except (json.JSONDecodeError, ValidationError):
                    bad += 1
    if bad:
        log.warning("skipped %d malformed span lines", bad)
    spans.sort(key=lambda s: s.start_ts)
    return spans
