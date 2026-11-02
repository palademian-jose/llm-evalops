"""Alert-precision log: record whether fired alerts were real, report false-alarm rates.

`llmops check` writes the latest violations to `<ALERTS_DIR>/last_check.json`.
`llmops check --record-outcome true|false [--alert-id ID]` appends one line per violation (or
the one matching alert) to `<ALERTS_DIR>/outcomes.jsonl`. `llmops alert-stats` reports
precision = true / (true + false) per rule, which is the number to tune thresholds against.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def alerts_dir(path: str | Path | None = None) -> Path:
    if path is None:
        from llmops.settings import get_settings

        path = get_settings().alerts_dir
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_last_check(payload: dict[str, Any], directory: str | Path | None = None) -> Path:
    p = alerts_dir(directory) / "last_check.json"
    p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return p


def load_last_check(directory: str | Path | None = None) -> dict[str, Any] | None:
    p = alerts_dir(directory) / "last_check.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def record_outcome(
    outcome: bool, *, alert_id: str | None = None, note: str = "", directory: str | Path | None = None
) -> int:
    last = load_last_check(directory)
    if not last or not last.get("violations"):
        return 0
    targets = [v for v in last["violations"] if alert_id is None or v["alert_id"] == alert_id]
    p = alerts_dir(directory) / "outcomes.jsonl"
    with p.open("a", encoding="utf-8") as fh:
        for v in targets:
            fh.write(
                json.dumps(
                    {
                        "recorded_at": datetime.now(UTC).isoformat(),
                        "checked_at": last.get("now"),
                        "alert_id": v["alert_id"],
                        "rule": v["rule"],
                        "type": v["type"],
                        "true_positive": bool(outcome),
                        "note": note,
                    }
                )
                + "\n"
            )
    return len(targets)


def load_outcomes(directory: str | Path | None = None) -> list[dict[str, Any]]:
    p = alerts_dir(directory) / "outcomes.jsonl"
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def precision_table(rows: list[dict[str, Any]]) -> str:
    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        agg[r["rule"]][0 if r["true_positive"] else 1] += 1
    lines = ["| Rule | fired | true | false | precision | false-alarm rate |", "|---|---|---|---|---|---|"]
    tot_t = tot_f = 0
    for rule, (t, f) in sorted(agg.items()):
        n = t + f
        tot_t += t
        tot_f += f
        lines.append(f"| {rule} | {n} | {t} | {f} | {t / n:.0%} | {f / n:.0%} |")
    n = tot_t + tot_f
    if n:
        lines.append(f"| **all** | {n} | {tot_t} | {tot_f} | **{tot_t / n:.0%}** | {tot_f / n:.0%} |")
    else:
        lines.append("| (no outcomes recorded) | | | | | |")
    return "\n".join(lines)
