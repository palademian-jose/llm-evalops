"""Binary, criterion-specific, justified LLM judge (PLAN.md shared contract 4).

`Judge(criterion, client)` renders a prompt template from `prompts/<criterion>.txt` (falling back
to `prompts/default.txt`), calls the judge model and strictly parses `{"pass": bool, "reason":
str}`. On a parse failure it sends one repair turn (`prompts/repair.txt`) and parses again; a
second failure raises `JudgeParseError`.
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from evalkit.types import Verdict
from llmops.llm import LLMClient, Message, get_client, model_family
from llmops.settings import get_settings

log = logging.getLogger("evalkit.judge")

PROMPTS_DIR = Path(__file__).parent / "prompts"
_JSON_OBJ = re.compile(r"\{.*\}", re.S)


class JudgeParseError(ValueError):
    pass


class JudgeFamilyError(RuntimeError):
    pass


def render_value(value: Any) -> str:
    """Render dict/list/str content for a prompt. Lists go one item per line."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(render_value(v) for v in value)
    if isinstance(value, dict):
        if len(value) == 1:
            return render_value(next(iter(value.values())))
        return "\n".join(f"{k}: {render_value(v)}" for k, v in value.items())
    return json.dumps(value, ensure_ascii=False)


@lru_cache(maxsize=32)
def load_template(name: str, templates_dir: str | None = None) -> str:
    base = Path(templates_dir) if templates_dir else PROMPTS_DIR
    path = base / f"{name}.txt"
    if not path.exists():
        path = base / "default.txt"
    return path.read_text(encoding="utf-8")


def extract_json(text: str) -> dict[str, Any]:
    """Return the first JSON object in `text`; tolerate code fences and surrounding prose."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.S)
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    m = _JSON_OBJ.search(cleaned)
    if not m:
        raise JudgeParseError(f"no JSON object in judge output: {text[:120]!r}")
    # Shrink from the right until it parses (handles trailing prose).
    candidate = m.group(0)
    while candidate:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
            break
        except json.JSONDecodeError:
            end = candidate.rfind("}", 0, len(candidate) - 1)
            if end == -1:
                break
            candidate = candidate[: end + 1]
    raise JudgeParseError(f"unparseable JSON in judge output: {text[:120]!r}")


def parse_verdict(text: str) -> Verdict:
    obj = extract_json(text)
    try:
        return Verdict.model_validate(obj)
    except ValidationError as e:
        raise JudgeParseError(f"judge JSON does not match {{pass, reason}}: {obj}") from e


def check_family(
    generator_model: str | None, judge_model: str | None, *, strict: bool
) -> str | None:
    """Warn (or raise when strict) when judge and generator share a model family."""
    g, j = model_family(generator_model), model_family(judge_model)
    if g == j and g not in ("fake", "unknown"):
        msg = (
            f"judge model `{judge_model}` and generator model `{generator_model}` are the same "
            f"family ({g}); self-preference bias is likely. Use a cross-family judge."
        )
        if strict:
            raise JudgeFamilyError(msg)
        log.warning(msg)
        return msg
    return None


class Judge:
    def __init__(
        self,
        criterion: str,
        client: LLMClient | None = None,
        *,
        model: str | None = None,
        templates_dir: str | None = None,
        max_repairs: int = 1,
        temperature: float = 0.0,
    ):
        self.criterion = criterion
        self.client = client or get_client(role="judge")
        self.model = model or get_settings().judge_model
        self.templates_dir = templates_dir
        self.max_repairs = max_repairs
        self.temperature = temperature
        self.calls = 0

    # -- single-output criteria ---------------------------------------------------------------
    def __call__(
        self, input: dict[str, Any], output: dict[str, Any], reference: dict[str, Any] | None = None
    ) -> Verdict:
        prompt = load_template(self.criterion, self.templates_dir).format_map(
            _SafeDict(
                criterion=self.criterion,
                input=render_value(input),
                output=render_value(output),
                reference=render_value(reference) if reference else "(none)",
            )
        )
        text = self._complete([Message(role="user", content=prompt)])
        try:
            return parse_verdict(text)
        except JudgeParseError as first:
            if self.max_repairs < 1:
                raise
            repair = load_template("repair", self.templates_dir)
            text2 = self._complete(
                [
                    Message(role="user", content=prompt),
                    Message(role="assistant", content=text),
                    Message(role="user", content=repair),
                ]
            )
            try:
                return parse_verdict(text2)
            except JudgeParseError as second:
                raise JudgeParseError(f"{first}; after repair: {second}") from second

    def judge(
        self, input: dict[str, Any], output: dict[str, Any], reference: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Contract form: `{"pass": bool, "reason": str}`."""
        return self(input, output, reference).as_dict()

    # -- pairwise -----------------------------------------------------------------------------
    def pairwise(
        self,
        input: dict[str, Any],
        output_a: dict[str, Any],
        output_b: dict[str, Any],
        reference: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return `{"winner": "a"|"b", "reason": str}` using `prompts/pairwise.txt`."""
        prompt = load_template("pairwise", self.templates_dir).format_map(
            _SafeDict(
                criterion=self.criterion,
                input=render_value(input),
                output_a=render_value(output_a),
                output_b=render_value(output_b),
                reference=render_value(reference) if reference else "(none)",
            )
        )
        text = self._complete([Message(role="user", content=prompt)])
        obj = extract_json(text)
        winner = str(obj.get("winner", "")).strip().lower()
        if winner not in ("a", "b"):
            raise JudgeParseError(f"pairwise winner must be 'a' or 'b', got {obj}")
        return {"winner": winner, "reason": str(obj.get("reason", ""))}

    def _complete(self, messages: list[Message]) -> str:
        self.calls += 1
        resp = self.client.complete(
            messages, model=self.model, temperature=self.temperature, max_tokens=400
        )
        return resp.text


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


_DEFAULT_JUDGES: dict[str, Judge] = {}


def judge(
    criterion: str,
    input: dict[str, Any],
    output: dict[str, Any],
    reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Module-level judge matching PLAN.md shared contract 4."""
    if criterion not in _DEFAULT_JUDGES:
        _DEFAULT_JUDGES[criterion] = Judge(criterion)
    return _DEFAULT_JUDGES[criterion].judge(input, output, reference)
