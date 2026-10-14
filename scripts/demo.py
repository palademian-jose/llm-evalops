"""End-to-end offline demo (`make demo`).

1. Build the golden sets and sample human labels.
2. Validate the golden set.
3. Run the baseline RAG target and a deliberately regressed candidate.
4. Run the baseline and regressed agent targets with k=3 (pass^k).
5. Calibrate the faithfulness judge against the human labels.
6. Compare baseline vs candidate with paired statistics.
7. Gate the candidate against evals.yaml: it MUST fail. Gate baseline vs itself: it must pass.

Exits 0 only when every expectation holds (so CI can assert that the gate catches the
regression). Writes everything to reports/ and a combined reports/demo_summary.md.
All numbers come from the fake provider and are labeled as such.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
RAG_METRICS = "exact_match,f1_token,recall_at_5,mrr,ndcg_at_5,faithfulness,answer_relevance"
AGENT_METRICS = "task_success,pass_pow_k"


def run(args: list[str], *, expect: int = 0, capture: bool = True) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", *args]
    print(f"\n$ {' '.join(a for a in cmd[2:])}", flush=True)
    env = dict(os.environ, LLM_PROVIDER="fake", PYTHONIOENCODING="utf-8")
    env.setdefault("TRACE_PATH", str(ROOT / "traces" / "demo" / "spans.jsonl"))
    proc = subprocess.run(
        cmd, cwd=ROOT, env=env, capture_output=capture, text=True, encoding="utf-8"
    )
    if capture:
        print(proc.stdout)
        if proc.stderr.strip():
            print(proc.stderr, file=sys.stderr)
    if proc.returncode != expect:
        print(f"expected exit {expect}, got {proc.returncode}", file=sys.stderr)
        sys.exit(1)
    return proc


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    trace_file = ROOT / "traces" / "demo" / "spans.jsonl"
    if trace_file.exists():
        trace_file.unlink()

    run(["examples.build_golden"])
    run(["evalkit.cli", "validate", "evals/golden/rag.jsonl"])
    run(["evalkit.cli", "validate", "evals/golden/agent.jsonl"])

    common = [
        "--golden",
        "evals/golden/rag.jsonl",
        "--metrics",
        RAG_METRICS,
        "--out",
        "reports",
        "--seed",
        "0",
    ]
    run(
        [
            "evalkit.cli",
            "run",
            *common,
            "--target",
            "examples.targets:rag_baseline",
            "--run-id",
            "rag-baseline",
        ]
    )
    run(
        [
            "evalkit.cli",
            "run",
            *common,
            "--target",
            "examples.targets:rag_regressed",
            "--run-id",
            "rag-candidate",
        ]
    )

    agent = [
        "--golden",
        "evals/golden/agent.jsonl",
        "--metrics",
        AGENT_METRICS,
        "--out",
        "reports",
        "--k",
        "3",
    ]
    run(
        [
            "evalkit.cli",
            "run",
            *agent,
            "--target",
            "examples.targets:agent_baseline",
            "--run-id",
            "agent-baseline",
        ]
    )
    run(
        [
            "evalkit.cli",
            "run",
            *agent,
            "--target",
            "examples.targets:agent_regressed",
            "--run-id",
            "agent-candidate",
        ]
    )

    run(
        [
            "evalkit.cli",
            "calibrate",
            "--labels",
            "evals/human_labels.jsonl",
            "--criterion",
            "faithfulness",
            "--md",
            "reports/calibration.md",
            "--json",
            "reports/calibration.json",
        ]
    )
    run(
        [
            "evalkit.cli",
            "compare",
            "--a",
            "reports/eval-rag-baseline.json",
            "--b",
            "reports/eval-rag-candidate.json",
            "--md",
            "reports/compare-rag.md",
            "--json",
            "reports/compare-rag.json",
        ]
    )
    run(
        [
            "evalkit.cli",
            "compare",
            "--a",
            "reports/eval-agent-baseline.json",
            "--b",
            "reports/eval-agent-candidate.json",
            "--md",
            "reports/compare-agent.md",
        ]
    )

    print("\n=== Gate: baseline vs regressed candidate (must FAIL) ===")
    gate = run(
        [
            "evalkit.cli",
            "gate",
            "--baseline",
            "reports/eval-rag-baseline.json",
            "--candidate",
            "reports/eval-rag-candidate.json",
            "--config",
            "evals.yaml",
            "--md",
            "reports/gate-rag.md",
        ],
        expect=1,
    )
    print("\n=== Gate: agent baseline vs regressed candidate (must FAIL) ===")
    run(
        [
            "evalkit.cli",
            "gate",
            "--baseline",
            "reports/eval-agent-baseline.json",
            "--candidate",
            "reports/eval-agent-candidate.json",
            "--config",
            "examples/evals-agent.yaml",
            "--md",
            "reports/gate-agent.md",
        ],
        expect=1,
    )
    print("\n=== Gate: baseline vs itself (must PASS) ===")
    run(
        [
            "evalkit.cli",
            "gate",
            "--baseline",
            "reports/eval-rag-baseline.json",
            "--candidate",
            "reports/eval-rag-baseline.json",
            "--config",
            "evals.yaml",
            "--md",
            "reports/gate-self.md",
        ],
        expect=0,
    )

    summary = [
        "# Demo summary (fake-provider numbers)",
        "",
        "All numbers below come from `LLM_PROVIDER=fake`; they demonstrate the pipeline, not model quality.",
        "",
        "## Gate: RAG baseline vs regressed candidate",
        "",
        (REPORTS / "gate-rag.md").read_text(encoding="utf-8").strip(),
        "",
        "## Gate: agent baseline vs regressed candidate (pass^3)",
        "",
        (REPORTS / "gate-agent.md").read_text(encoding="utf-8").strip(),
        "",
        "## Paired statistics (RAG)",
        "",
        (REPORTS / "compare-rag.md").read_text(encoding="utf-8").strip(),
        "",
        "## Paired statistics (agent)",
        "",
        (REPORTS / "compare-agent.md").read_text(encoding="utf-8").strip(),
        "",
        "## Judge calibration",
        "",
        (REPORTS / "calibration.md").read_text(encoding="utf-8").strip(),
        "",
    ]
    (REPORTS / "demo_summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(f"\nwrote {REPORTS / 'demo_summary.md'}")
    assert "GATE FAILED" in gate.stdout
    print("\nDemo OK: the gate failed the regressed candidate and passed the baseline.")


if __name__ == "__main__":
    main()
