"""Cost attribution tables and bill reconciliation.

Attribution sums `cost_usd` over spans grouped by request (trace_id), feature, user, model or
day. Requests are counted as distinct trace ids. Only spans that carry cost (llm, rerank,
cache embeds) contribute dollars; every span contributes to the request count.

Reconciliation compares computed cost per (day, model) with a provider export. Two CSV formats
are accepted and auto-detected from the header:

1. Simple daily format (shipped as `examples/bill_sample.csv`):
       date,model,input_tokens,output_tokens,cost_usd
   where `date` is YYYY-MM-DD and `model` is a pricing key or OpenRouter slug.
2. OpenRouter activity export. Assumed columns (extra columns are ignored):
       created_at   ISO timestamp of the generation (UTC)
       model        OpenRouter slug, e.g. anthropic/claude-sonnet-4.5
       tokens_prompt, tokens_completion   token counts
       cost_total   USD charged (falls back to `cost` or `usage` when present)
   Detected when the header contains `tokens_prompt`.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from llmops.pricing import canonical_model, load_pricing
from llmops.tracing.span import Span, parse_ts

GroupBy = Literal["request", "feature", "user", "model", "day"]


def _group_key(span: Span, by: GroupBy) -> str:
    if by == "request":
        return span.trace_id
    if by == "feature":
        return span.feature
    if by == "user":
        return span.user_id
    if by == "model":
        return span.model or "(none)"
    if by == "day":
        return parse_ts(span.start_ts).strftime("%Y-%m-%d")
    raise ValueError(f"unknown grouping {by}")


@dataclass
class CostRow:
    key: str
    requests: int
    spans: int
    input_tokens: int
    output_tokens: int
    cost_usd: float

    @property
    def cost_per_1k_requests(self) -> float:
        return self.cost_usd / self.requests * 1000 if self.requests else 0.0


def cost_table(
    spans: list[Span], by: GroupBy = "feature", *, top: int | None = None
) -> list[CostRow]:
    agg: dict[str, dict] = defaultdict(
        lambda: {"traces": set(), "spans": 0, "in": 0, "out": 0, "cost": 0.0}
    )
    for s in spans:
        if by == "model" and not s.model:
            continue
        a = agg[_group_key(s, by)]
        a["traces"].add(s.trace_id)
        a["spans"] += 1
        a["in"] += s.input_tokens
        a["out"] += s.output_tokens
        a["cost"] += s.cost_usd
    rows = [
        CostRow(
            key=k,
            requests=len(v["traces"]),
            spans=v["spans"],
            input_tokens=v["in"],
            output_tokens=v["out"],
            cost_usd=v["cost"],
        )
        for k, v in agg.items()
    ]
    if by == "day":
        rows.sort(key=lambda r: r.key)
    else:
        rows.sort(key=lambda r: (-r.cost_usd, r.key))
    return rows[:top] if top else rows


def cost_markdown(rows: list[CostRow], by: GroupBy) -> str:
    total = sum(r.cost_usd for r in rows)
    lines = [
        f"| {by} | requests | spans | input tokens | output tokens | cost USD | $/1k req | share |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        share = r.cost_usd / total if total else 0.0
        lines.append(
            f"| {r.key} | {r.requests} | {r.spans} | {r.input_tokens:,} | {r.output_tokens:,} | "
            f"{r.cost_usd:.4f} | {r.cost_per_1k_requests:.2f} | {share:.1%} |"
        )
    lines.append(
        f"| **total** | {sum(r.requests for r in rows)} | {sum(r.spans for r in rows)} | "
        f"{sum(r.input_tokens for r in rows):,} | {sum(r.output_tokens for r in rows):,} | "
        f"**{total:.4f}** | | |"
    )
    return "\n".join(lines)


# ---- reconciliation ------------------------------------------------------------------------


@dataclass
class BillLine:
    day: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass
class ReconcileRow:
    day: str
    model: str
    computed_usd: float
    billed_usd: float
    computed_tokens: int
    billed_tokens: int

    @property
    def diff_usd(self) -> float:
        return self.computed_usd - self.billed_usd

    @property
    def diff_pct(self) -> float | None:
        return (self.diff_usd / self.billed_usd * 100) if self.billed_usd else None


@dataclass
class ReconcileReport:
    rows: list[ReconcileRow] = field(default_factory=list)
    bill_format: str = "simple"

    @property
    def computed_total(self) -> float:
        return sum(r.computed_usd for r in self.rows)

    @property
    def billed_total(self) -> float:
        return sum(r.billed_usd for r in self.rows)

    @property
    def discrepancy_pct(self) -> float | None:
        if not self.billed_total:
            return None
        return (self.computed_total - self.billed_total) / self.billed_total * 100

    def to_markdown(self) -> str:
        lines = [
            f"Bill format: {self.bill_format}",
            "",
            "| day | model | computed USD | billed USD | diff USD | diff % | computed tokens | billed tokens |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in self.rows:
            pct = "n/a" if r.diff_pct is None else f"{r.diff_pct:+.2f}%"
            lines.append(
                f"| {r.day} | {r.model} | {r.computed_usd:.4f} | {r.billed_usd:.4f} | {r.diff_usd:+.4f} | "
                f"{pct} | {r.computed_tokens:,} | {r.billed_tokens:,} |"
            )
        d = self.discrepancy_pct
        lines += [
            "",
            f"Computed total: {self.computed_total:.4f} USD; billed total: {self.billed_total:.4f} USD; "
            f"discrepancy: {'n/a' if d is None else f'{d:+.2f}%'}",
        ]
        return "\n".join(lines)


def _bill_model_key(model: str) -> str:
    return canonical_model(model, load_pricing())


def read_bill(path: str | Path) -> tuple[list[BillLine], str]:
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        headers = [h.strip() for h in reader.fieldnames or []]
        rows = list(reader)
    if "tokens_prompt" in headers:
        fmt = "openrouter"
        lines = []
        for r in rows:
            ts = (r.get("created_at") or "").strip()
            day = ts[:10]
            cost_raw = r.get("cost_total") or r.get("cost") or r.get("usage") or "0"
            lines.append(
                BillLine(
                    day=day,
                    model=_bill_model_key((r.get("model") or "").strip()),
                    input_tokens=int(float(r.get("tokens_prompt") or 0)),
                    output_tokens=int(float(r.get("tokens_completion") or 0)),
                    cost_usd=float(cost_raw or 0),
                )
            )
        return lines, fmt
    required = {"date", "model", "cost_usd"}
    if not required <= set(headers):
        raise ValueError(
            f"unrecognized bill CSV header {headers}; expected {sorted(required)} or an OpenRouter export"
        )
    lines = [
        BillLine(
            day=(r["date"] or "").strip()[:10],
            model=_bill_model_key((r["model"] or "").strip()),
            input_tokens=int(float(r.get("input_tokens") or 0)),
            output_tokens=int(float(r.get("output_tokens") or 0)),
            cost_usd=float(r["cost_usd"] or 0),
        )
        for r in rows
    ]
    return lines, "simple"


def reconcile(
    spans: list[Span], bill_path: str | Path, *, kinds: tuple[str, ...] = ("llm",)
) -> ReconcileReport:
    """Compare computed cost per (day, model) with a bill. Only `kinds` spans count (a provider
    bill covers its own LLM calls; rerank or embedding vendors are reconciled separately)."""
    bill, fmt = read_bill(bill_path)
    billed: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0])
    for b in bill:
        acc = billed[(b.day, b.model)]
        acc[0] += b.cost_usd
        acc[1] += b.input_tokens + b.output_tokens
    computed: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0])
    pricing = load_pricing()
    for s in spans:
        if s.kind not in kinds or not s.model or s.cache_hit:
            continue
        key = (parse_ts(s.start_ts).strftime("%Y-%m-%d"), canonical_model(s.model, pricing))
        acc = computed[key]
        acc[0] += s.cost_usd
        acc[1] += s.input_tokens + s.output_tokens
    rep = ReconcileReport(bill_format=fmt)
    for key in sorted(set(billed) | set(computed)):
        c, b = computed.get(key, [0.0, 0]), billed.get(key, [0.0, 0])
        rep.rows.append(
            ReconcileRow(
                day=key[0],
                model=key[1],
                computed_usd=c[0],
                billed_usd=b[0],
                computed_tokens=int(c[1]),
                billed_tokens=int(b[1]),
            )
        )
    return rep
