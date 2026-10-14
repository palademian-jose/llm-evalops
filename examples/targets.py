"""Toy targets for `evalkit run`, fully offline with the fake provider.

- `rag_baseline`   : BM25-lite retrieval over the mini corpus + extractive generation.
- `rag_regressed`  : the same pipeline after a deliberately bad change: shallower retrieval
                     (k cut from 5 to 2 and only the first three query terms are scored) and a
                     prompt change that invites unsupported elaboration (simulated by appending
                     an invented sentence to about a third of the answers).
- `agent_baseline` : a tool-using loop (search, calculator) that stops after one tool result.
- `agent_regressed`: the same loop that intermittently forgets to stop and burns its step
                     budget (a tool loop), so `pass_pow_k` collapses while `task_success` on a
                     single run only dips.

Each target takes `(input: dict) -> dict` and uses `llmops.tracing.wrap_client` so that the
runner captures latency and cost from spans.
"""

from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache
from typing import Any

from examples.corpus import DOCS
from llmops.llm import LLMClient, Message, ToolSpec, get_client
from llmops.tracing import span, wrap_client

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {
    "the",
    "a",
    "an",
    "of",
    "is",
    "in",
    "to",
    "and",
    "on",
    "at",
    "how",
    "what",
    "who",
    "where",
    "when",
    "which",
    "why",
    "does",
    "do",
    "did",
    "it",
    "its",
    "for",
    "was",
    "by",
}


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@lru_cache(maxsize=1)
def _idf() -> dict[str, float]:
    n = len(DOCS)
    df: dict[str, int] = {}
    for text in DOCS.values():
        for t in set(_tokens(text)):
            df[t] = df.get(t, 0) + 1
    return {t: math.log(1 + n / c) for t, c in df.items()}


def retrieve(query: str, k: int = 5, max_terms: int | None = None) -> list[str]:
    """Rank docs by summed idf of matching query terms (ties broken by doc id)."""
    terms = _tokens(query)
    if max_terms is not None:
        terms = terms[:max_terms]
    idf = _idf()
    scored = []
    for doc_id, text in DOCS.items():
        doc_terms = set(_tokens(text))
        score = sum(idf.get(t, 0.0) for t in set(terms) if t in doc_terms)
        scored.append((-score, doc_id))
    scored.sort()
    return [d for s, d in scored[:k] if s < 0]


@lru_cache(maxsize=1)
def _client() -> LLMClient:
    return wrap_client(get_client(), feature="examples.rag")


def _stable_fraction(key: str) -> float:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


# ---- RAG ---------------------------------------------------------------------------------------


def _rag(question: str, *, k: int, max_terms: int | None, drift_rate: float, run_key: str) -> dict:
    with span("retrieve", "retrieval", meta={"k": k}) as s:
        ranked = retrieve(question, k=k, max_terms=max_terms)
        s.meta["hits"] = len(ranked)
    contexts = [DOCS[d] for d in ranked[:3]]
    prompt = (
        "Answer the question using only the context. If the context does not contain the "
        'answer, say "I don\'t know."\n<contexts>\n' + "\n".join(contexts) + "\n</contexts>\n"
        f"<question>{question}</question>"
    )
    resp = _client().complete([Message(role="user", content=prompt)], max_tokens=200)
    answer = resp.text.strip()
    if drift_rate and answer != "I don't know." and _stable_fraction(run_key) < drift_rate:
        answer += (
            " This figure was later revised upward by the international survey commission of "
            "2031 according to several widely cited reports."
        )
    return {"answer": answer, "retrieved": ranked, "contexts": contexts}


def rag_baseline(input: dict[str, Any]) -> dict[str, Any]:
    return _rag(str(input["question"]), k=5, max_terms=None, drift_rate=0.0, run_key="")


def rag_regressed(input: dict[str, Any]) -> dict[str, Any]:
    q = str(input["question"])
    return _rag(q, k=2, max_terms=3, drift_rate=0.35, run_key=q)


# ---- agent -------------------------------------------------------------------------------------

TOOLS = [
    ToolSpec(
        name="search",
        description="Search the knowledge base and return the best matching sentence.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    ),
    ToolSpec(
        name="calculator",
        description="Evaluate an arithmetic expression.",
        input_schema={
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
            "additionalProperties": False,
        },
    ),
]


def tool_search(query: str) -> str:
    ranked = retrieve(query, k=1)
    if not ranked:
        return "No results."
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", DOCS[ranked[0]]) if s.strip()]
    q = set(_tokens(query))
    return max(sentences, key=lambda s: (len(q & set(_tokens(s))), -len(s)))


def tool_calculator(expression: str) -> str:
    runs = [m.group(0) for m in re.finditer(r"[-+*/().\d\s]+", expression.replace("x", "*"))]
    runs = [r.strip(" .") for r in runs if re.search(r"\d", r)]
    if not runs:
        return "error: no expression"
    expr = max(runs, key=len)
    try:
        value = eval(expr, {"__builtins__": {}}, {})  # noqa: S307 - digits and operators only
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"
    return str(int(value)) if float(value).is_integer() else str(value)


def _agent(task: str, *, loop_rate: float, run_key: str, max_steps: int = 4) -> dict[str, Any]:
    client = _client()
    messages = [Message(role="user", content=task)]
    trajectory: list[dict[str, Any]] = []
    forgets_to_stop = loop_rate and _stable_fraction(run_key) < loop_rate
    answer = ""
    for step in range(max_steps):
        resp = client.complete(messages, tools=TOOLS, max_tokens=200)
        if resp.tool_calls:
            call = resp.tool_calls[0]
            with span(f"tool.{call.name}", "tool", meta={"step": step}):
                if call.name == "calculator":
                    result = tool_calculator(str(call.arguments.get("expression", task)))
                else:
                    result = tool_search(str(call.arguments.get("query", task)))
            trajectory.append(
                {"step": step, "tool": call.name, "args": call.arguments, "result": result}
            )
            messages.append(Message(role="assistant", content=f"[tool call {call.name}]"))
            messages.append(Message(role="tool", content=result, tool_call_id=call.id))
            if forgets_to_stop:
                # A buggy agent that drops the tool result and re-plans from scratch.
                messages = [Message(role="user", content=task)]
            continue
        answer = resp.text.strip()
        break
    else:
        answer = "I could not complete the task within the step budget."
    with span("verify", "verify", meta={"steps": len(trajectory)}):
        completed = bool(answer) and not answer.startswith("I could not")
    # No `success` key: the runner's verifier decides from `expected.answer`.
    return {
        "answer": answer,
        "trajectory": trajectory,
        "steps": len(trajectory),
        "completed": completed,
    }


def agent_baseline(input: dict[str, Any]) -> dict[str, Any]:
    return _agent(str(input["task"]), loop_rate=0.0, run_key="")


def agent_regressed(input: dict[str, Any]) -> dict[str, Any]:
    key = f"{input['task']}|{input.get('_run_index', 0)}"
    return _agent(str(input["task"]), loop_rate=0.45, run_key=key)
