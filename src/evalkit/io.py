"""JSONL and JSON loaders/writers for golden sets and eval results."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from evalkit.types import EvalResult, GoldenCase


class GoldenLoadError(ValueError):
    def __init__(self, path: Path, line_no: int, message: str):
        super().__init__(f"{path}:{line_no}: {message}")
        self.path = path
        self.line_no = line_no
        self.message = message


def iter_jsonl(path: str | Path) -> Iterator[tuple[int, dict[str, Any]]]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise GoldenLoadError(p, i, f"invalid JSON: {e.msg}") from e
            if not isinstance(obj, dict):
                raise GoldenLoadError(p, i, "each line must be a JSON object")
            yield i, obj


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_golden(path: str | Path) -> list[GoldenCase]:
    """Load a golden set, raising GoldenLoadError on the first invalid line."""
    cases: list[GoldenCase] = []
    p = Path(path)
    for line_no, obj in iter_jsonl(p):
        try:
            cases.append(GoldenCase.model_validate(obj))
        except ValidationError as e:
            msgs = "; ".join(
                f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()
            )
            raise GoldenLoadError(p, line_no, msgs) from e
    return cases


def write_golden(path: str | Path, cases: Iterable[GoldenCase]) -> int:
    return write_jsonl(path, (c.model_dump(mode="json") for c in cases))


def load_eval_result(path: str | Path) -> EvalResult:
    return EvalResult.model_validate_json(Path(path).read_text(encoding="utf-8"))


def write_eval_result(path: str | Path, result: EvalResult) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return p
