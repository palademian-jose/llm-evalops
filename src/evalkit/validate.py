"""Golden-set validation: schema, duplicate ids/inputs, near-duplicates, and leaky cases.

- duplicate input: two cases whose normalized input JSON is identical.
- near-duplicate input: normalized token overlap (Jaccard over lowercase alphanumeric tokens
  of the flattened input text) greater than `near_dup_threshold` (default 0.9).
- leaky case: a string value from `expected` (at least `min_leak_chars` long) appears verbatim,
  case-insensitively, inside the flattened input text. Such a case rewards copying the input.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from evalkit.io import GoldenLoadError, iter_jsonl
from evalkit.types import GoldenCase

_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass
class ValidationReport:
    path: str
    n_cases: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    failure_modes: dict[str, int] = field(default_factory=dict)
    sources: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_markdown(self) -> str:
        lines = [f"# Golden set validation: `{self.path}`", "", f"- cases: {self.n_cases}"]
        if self.failure_modes:
            fm = ", ".join(f"{k}={v}" for k, v in sorted(self.failure_modes.items()))
            lines.append(f"- failure modes: {fm}")
        if self.sources:
            src = ", ".join(f"{k}={v}" for k, v in sorted(self.sources.items()))
            lines.append(f"- sources: {src}")
        lines.append(f"- errors: {len(self.errors)}")
        lines.append(f"- warnings: {len(self.warnings)}")
        if self.errors:
            lines += ["", "## Errors", ""] + [f"- {e}" for e in self.errors]
        if self.warnings:
            lines += ["", "## Warnings", ""] + [f"- {w}" for w in self.warnings]
        lines.append("")
        lines.append("Result: " + ("OK" if self.ok else "FAILED"))
        return "\n".join(lines)


def flatten_text(value: Any) -> str:
    """Concatenate every string leaf of a nested structure, in key order."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(flatten_text(value[k]) for k in sorted(value))
    if isinstance(value, list | tuple):
        return " ".join(flatten_text(v) for v in value)
    if value is None:
        return ""
    return str(value)


def normalize_tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def string_leaves(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in string_leaves(v)]
    if isinstance(value, list | tuple):
        return [s for v in value for s in string_leaves(v)]
    return []


def validate_cases(
    cases: list[GoldenCase],
    *,
    near_dup_threshold: float = 0.9,
    min_leak_chars: int = 12,
    report: ValidationReport | None = None,
) -> ValidationReport:
    rep = report or ValidationReport(path="<memory>")
    rep.n_cases = len(cases)

    seen_ids: dict[str, int] = {}
    seen_inputs: dict[str, str] = {}
    token_sets: list[tuple[str, set[str]]] = []

    for c in cases:
        rep.failure_modes[c.failure_mode or "unspecified"] = (
            rep.failure_modes.get(c.failure_mode or "unspecified", 0) + 1
        )
        rep.sources[c.source] = rep.sources.get(c.source, 0) + 1

        if c.id in seen_ids:
            rep.errors.append(f"duplicate id `{c.id}`")
        seen_ids[c.id] = seen_ids.get(c.id, 0) + 1

        key = json.dumps(c.input, sort_keys=True, ensure_ascii=False)
        exact_dup = key in seen_inputs
        if exact_dup:
            rep.errors.append(f"duplicate input: `{c.id}` is identical to `{seen_inputs[key]}`")
        else:
            seen_inputs[key] = c.id

        input_text = flatten_text(c.input)
        toks = normalize_tokens(input_text)
        if not exact_dup:
            for other_id, other_toks in token_sets:
                if other_toks and toks:
                    sim = jaccard(toks, other_toks)
                    if sim > near_dup_threshold:
                        rep.warnings.append(
                            f"near-duplicate input: `{c.id}` ~ `{other_id}` (overlap {sim:.2f})"
                        )
        token_sets.append((c.id, toks))

        low_input = input_text.lower()
        for leaf in string_leaves(c.expected):
            s = leaf.strip()
            if len(s) >= min_leak_chars and s.lower() in low_input:
                rep.warnings.append(
                    f"leaky case: `{c.id}` expected text appears verbatim in input: `{s[:60]}`"
                )
                break
    return rep


def validate_file(
    path: str | Path,
    *,
    near_dup_threshold: float = 0.9,
    min_leak_chars: int = 12,
) -> ValidationReport:
    p = Path(path)
    rep = ValidationReport(path=str(p))
    cases: list[GoldenCase] = []
    try:
        for line_no, obj in iter_jsonl(p):
            try:
                cases.append(GoldenCase.model_validate(obj))
            except ValidationError as e:
                for err in e.errors():
                    loc = ".".join(str(x) for x in err["loc"]) or "<root>"
                    rep.errors.append(f"line {line_no}: {loc}: {err['msg']}")
    except GoldenLoadError as e:
        rep.errors.append(f"line {e.line_no}: {e.message}")
    except FileNotFoundError:
        rep.errors.append(f"file not found: {p}")
        return rep
    return validate_cases(
        cases, near_dup_threshold=near_dup_threshold, min_leak_chars=min_leak_chars, report=rep
    )
