"""Build the demo golden sets and sample human labels from the mini corpus.

Writes:
- evals/golden/rag.jsonl      (answerable + unanswerable RAG cases)
- evals/golden/agent.jsonl    (tool-use tasks)
- evals/human_labels.jsonl    (faithfulness labels incl. pairwise items; see --labels)

Everything is deterministic. Golden cases are tagged `source=error_analysis` for the cases that
mirror a failure mode seen in error analysis and `synthetic` for the rest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from evalkit.io import write_golden
from evalkit.types import GoldenCase
from examples.corpus import AGENT_TASKS, DOCS, FACTS, UNANSWERABLE


def rag_cases() -> list[GoldenCase]:
    cases = []
    for i, (doc_id, question, answer, failure_mode) in enumerate(FACTS, start=1):
        cases.append(
            GoldenCase(
                id=f"rag-{i:04d}",
                input={"question": question},
                expected={"answer": answer, "relevant": [doc_id]},
                failure_mode=failure_mode,
                tags=[doc_id.split("-")[0], "answerable"],
                source="error_analysis" if failure_mode == "hallucination" else "synthetic",
            )
        )
    for j, question in enumerate(UNANSWERABLE, start=1):
        cases.append(
            GoldenCase(
                id=f"rag-u{j:03d}",
                input={"question": question},
                expected={"answer": "I don't know."},
                failure_mode="refusal_miss",
                tags=["unanswerable"],
                source="error_analysis",
            )
        )
    return cases


def agent_cases() -> list[GoldenCase]:
    return [
        GoldenCase(
            id=f"agent-{i:04d}",
            input={"task": task},
            expected={"answer": answer, "max_steps": 2},
            failure_mode=failure_mode,
            tags=["calculator" if "calculator" in task else "search"],
            source="error_analysis",
        )
        for i, (task, answer, failure_mode) in enumerate(AGENT_TASKS, start=1)
    ]


def human_labels() -> list[dict]:
    """Faithfulness labels: supported answers, fabricated answers and a few noisy human labels."""
    rows = []
    fabricated = [
        "The figure was revised in 2031 by the international survey commission.",
        "Local guides report that the number doubled after the last census.",
        "Historians now attribute this to a different inventor entirely.",
    ]
    for i, (doc_id, question, answer, _) in enumerate(FACTS):
        h = int(hashlib.sha256(question.encode()).hexdigest()[:8], 16)
        supported = h % 3 != 0
        padding = " Sources agree on this point." * (h % 3)
        output = answer + padding if supported else fabricated[h % len(fabricated)] + padding
        label = supported
        if i in (5, 17):  # deliberate annotator noise so kappa is realistic
            label = not label
        rows.append(
            {
                "id": f"lab-{i + 1:04d}",
                "criterion": "faithfulness",
                "input": {"question": question},
                "output": {"answer": output},
                "reference": {"contexts": [DOCS[doc_id]]},
                "label": label,
            }
        )
    for i, (doc_id, question, answer, _) in enumerate(FACTS[:6]):
        rows.append(
            {
                "id": f"pair-{i + 1:03d}",
                "criterion": "faithfulness",
                "input": {"question": question},
                "output_a": {"answer": answer},
                "output_b": {"answer": fabricated[i % len(fabricated)]},
                "reference": {"contexts": [DOCS[doc_id]]},
                "label": "a",
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="evals/golden", help="Golden set directory.")
    ap.add_argument("--labels", default="evals/human_labels.jsonl", help="Human labels path.")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_rag = write_golden(out / "rag.jsonl", rag_cases())
    n_agent = write_golden(out / "agent.jsonl", agent_cases())
    labels = Path(args.labels)
    labels.parent.mkdir(parents=True, exist_ok=True)
    with labels.open("w", encoding="utf-8") as fh:
        rows = human_labels()
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {n_rag} RAG cases, {n_agent} agent cases, {len(rows)} human labels")


if __name__ == "__main__":
    main()
