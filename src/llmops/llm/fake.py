"""Deterministic fake provider.

Behaviour, in priority order:
1. Judge prompts (detected by the `<criterion>` tag emitted by evalkit's templates) are answered
   by a configurable lexical heuristic and return strict `{"pass": bool, "reason": str}` JSON.
2. Fixtures: `FakeClient(fixtures={"substring": "response"})` returns the first fixture whose key
   appears in the last user message.
3. Tool use: when tools are offered and no tool result is in the conversation yet, the fake calls
   the tool whose name is mentioned in the last user message (else the first tool) with arguments
   filled from the schema's string/number properties.
4. RAG-style prompts containing `<contexts>` and `<question>` get an extractive answer: the
   context sentence with the highest token overlap with the question.
5. Otherwise an echo of the last user message prefixed with `FAKE:`.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from llmops.llm.types import LLMResponse, Message, ToolCall, ToolSpec, Usage, approx_tokens

_TAG = re.compile(
    r"<(criterion|input|output|reference|contexts|question|output_a|output_b)>(.*?)</\1>", re.S
)
_TOKEN = re.compile(r"[a-z0-9]+")

DEFAULT_JUDGE_THRESHOLDS: dict[str, float] = {
    "faithfulness": 0.5,
    "answer_relevance": 0.2,
    "context_precision": 0.5,
    "default": 0.5,
}


def tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def overlap(a: str, b: str) -> float:
    """Fraction of `a`'s tokens that also occur in `b`. 0.0 when `a` is empty."""
    ta, tb = tokens(a), tokens(b)
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


def parse_tags(text: str) -> dict[str, str]:
    return {m.group(1): m.group(2).strip() for m in _TAG.finditer(text)}


@dataclass
class FakeConfig:
    judge_thresholds: dict[str, float] = field(default_factory=dict)
    # When > 0 the first N judge answers are deliberately malformed, to exercise the repair path.
    malformed_judge_answers: int = 0
    # Latency reported on every response (no real sleep), so traces carry non-zero numbers.
    simulated_latency_ms: float = 0.0
    model: str = "fake-model"


class FakeClient:
    def __init__(self, config: FakeConfig | None = None, fixtures: dict[str, str] | None = None):
        self.config = config or FakeConfig()
        self.fixtures = fixtures or {}
        self.calls: list[list[Message]] = []
        self._malformed_left = self.config.malformed_judge_answers

    # -- public API --------------------------------------------------------------------------
    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        start = time.perf_counter()
        self.calls.append(list(messages))
        prompt_text = "\n".join(m.text() for m in messages)
        last_user = next((m.text() for m in reversed(messages) if m.role == "user"), "")
        tool_calls: list[ToolCall] = []
        stop_reason = "end_turn"

        tags = parse_tags(prompt_text)
        fixture = self._fixture(last_user)
        if "criterion" in tags:
            text = self._judge(tags)
        elif fixture is not None:
            text = fixture
        elif tools and not any(m.role == "tool" for m in messages):
            tool_calls = [self._pick_tool(tools, last_user)]
            text = ""
            stop_reason = "tool_use"
        elif "contexts" in tags and "question" in tags:
            text = self._extractive_answer(tags["contexts"], tags["question"])
        elif any(m.role == "tool" for m in messages):
            tool_text = next(m.text() for m in reversed(messages) if m.role == "tool")
            text = tool_text.strip()[: max_tokens * 4]
        else:
            text = f"FAKE: {last_user.strip()[:200]}"

        usage = Usage(
            input_tokens=approx_tokens(prompt_text),
            output_tokens=approx_tokens(text) if text else 4,
        )
        latency = (time.perf_counter() - start) * 1000 + self.config.simulated_latency_ms
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            usage=usage,
            model=model or self.config.model,
            latency_ms=latency,
            stop_reason=stop_reason,
        )

    # -- helpers -----------------------------------------------------------------------------
    def _fixture(self, last_user: str) -> str | None:
        for key, value in self.fixtures.items():
            if key in last_user:
                return value
        return None

    def _judge(self, tags: dict[str, str]) -> str:
        if self._malformed_left > 0:
            self._malformed_left -= 1
            return "Sure! Here is my verdict: PASS because it looks fine."
        criterion = tags.get("criterion", "default")
        thresholds = {**DEFAULT_JUDGE_THRESHOLDS, **self.config.judge_thresholds}
        thr = thresholds.get(criterion, thresholds["default"])
        output = tags.get("output", "")
        reference = tags.get("reference", "")
        inp = tags.get("input", "")

        if criterion == "pairwise":
            a, b = tags.get("output_a", ""), tags.get("output_b", "")
            basis = reference or inp
            sa, sb = overlap(basis, a), overlap(basis, b)
            winner = "a" if sa >= sb else "b"
            return json.dumps({"winner": winner, "reason": f"overlap a={sa:.2f} b={sb:.2f}"})
        if criterion == "faithfulness":
            score = overlap(output, reference) if reference else 0.0
            reason = f"{score:.2f} of the answer tokens are supported by the context"
        elif criterion == "answer_relevance":
            score = overlap(inp, output)
            reason = f"{score:.2f} of the question tokens are addressed by the answer"
        elif criterion == "context_precision":
            ctxs = [c for c in reference.splitlines() if c.strip()]
            hits = [c for c in ctxs if overlap(inp, c) >= 0.3]
            score = len(hits) / len(ctxs) if ctxs else 0.0
            reason = f"{len(hits)}/{len(ctxs)} contexts are relevant to the question"
        else:
            score = overlap(reference, output) if reference else overlap(inp, output)
            reason = f"{score:.2f} lexical overlap with the reference"
        return json.dumps({"pass": bool(score >= thr), "reason": reason})

    @staticmethod
    def _pick_tool(tools: list[ToolSpec], last_user: str) -> ToolCall:
        low = last_user.lower()
        chosen = next((t for t in tools if t.name.lower() in low), tools[0])
        args: dict = {}
        props = chosen.input_schema.get("properties", {})
        nums = re.findall(r"-?\d+(?:\.\d+)?", last_user)
        for name, spec in props.items():
            typ = spec.get("type")
            if typ == "string":
                args[name] = last_user.strip()[:200]
            elif typ == "number":
                args[name] = float(nums[0]) if nums else 0.0
            elif typ == "integer":
                args[name] = int(float(nums[0])) if nums else 0
            elif typ == "boolean":
                args[name] = True
        call_id = f"call_{abs(hash(chosen.name + low)) % 10_000_000}"
        return ToolCall(id=call_id, name=chosen.name, arguments=args)

    @staticmethod
    def _extractive_answer(contexts: str, question: str) -> str:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", contexts) if s.strip()]
        if not sentences:
            return "I don't know."
        best = max(sentences, key=lambda s: (overlap(question, s), -len(s)))
        if overlap(question, best) == 0.0:
            return "I don't know."
        return best
