"""OpenRouter backend: OpenAI-compatible chat completions through the `openai` SDK.

- `base_url=https://openrouter.ai/api/v1`, `OPENROUTER_API_KEY`, model ids are OpenRouter slugs
  (`anthropic/claude-sonnet-4.5`, `openai/gpt-4.1-mini`, `google/gemini-2.5-flash`).
- Every request asks for `usage: {"include": true}` so the response carries OpenRouter's own
  `usage.cost`; it is surfaced as `LLMResponse.cost_usd` and preferred by `llmops.tracing`.
- Optional `HTTP-Referer` / `X-Title` headers identify the app on the OpenRouter leaderboard.
Only exercised with a key.
"""

from __future__ import annotations

import json
import time
from typing import Any

from llmops.llm.types import LLMResponse, Message, ToolCall, ToolSpec, Usage

BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        default_model: str,
        *,
        base_url: str = BASE_URL,
        site_url: str | None = None,
        app_name: str | None = None,
        timeout: float = 60.0,
        sdk_client: Any | None = None,
    ):
        self._model = default_model
        if sdk_client is not None:
            self._sdk = sdk_client
        else:
            from openai import OpenAI

            headers: dict[str, str] = {}
            if site_url:
                headers["HTTP-Referer"] = site_url
            if app_name:
                headers["X-Title"] = app_name
            self._sdk = OpenAI(
                api_key=api_key, base_url=base_url, default_headers=headers or None, timeout=timeout
            )

    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": model or self._model,
            "messages": [to_openai_message(m) for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_body": {"usage": {"include": True}},
        }
        if tools:
            kwargs["tools"] = [to_openai_tool(t) for t in tools]
        start = time.perf_counter()
        resp = self._sdk.chat.completions.create(**kwargs)
        latency = (time.perf_counter() - start) * 1000
        return parse_openai_response(resp, requested_model=kwargs["model"], latency_ms=latency)


def to_openai_message(m: Message) -> dict[str, Any]:
    if m.role == "tool":
        return {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.text()}
    content = m.content if isinstance(m.content, str) else m.text()
    return {"role": m.role, "content": content}


def to_openai_tool(t: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": t.name, "description": t.description, "parameters": t.input_schema},
    }


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    value = getattr(obj, name, None)
    if value is None:
        extra = getattr(obj, "model_extra", None) or {}
        value = extra.get(name, default)
    return default if value is None else value


def parse_openai_response(resp: Any, *, requested_model: str, latency_ms: float) -> LLMResponse:
    """Parse an SDK `ChatCompletion` (or an equivalent dict) into the shared LLMResponse."""
    choice = _get(resp, "choices", [])[0]
    msg = _get(choice, "message")
    calls: list[ToolCall] = []
    for tc in _get(msg, "tool_calls", None) or []:
        fn = _get(tc, "function")
        raw_args = _get(fn, "arguments", "{}") or "{}"
        args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
        calls.append(ToolCall(id=_get(tc, "id"), name=_get(fn, "name"), arguments=args))
    usage = _get(resp, "usage")
    reported_cost = _get(usage, "cost", None) if usage is not None else None
    return LLMResponse(
        text=_get(msg, "content", "") or "",
        tool_calls=calls,
        usage=Usage(
            input_tokens=int(_get(usage, "prompt_tokens", 0) or 0) if usage is not None else 0,
            output_tokens=int(_get(usage, "completion_tokens", 0) or 0) if usage is not None else 0,
        ),
        model=_get(resp, "model", requested_model) or requested_model,
        latency_ms=latency_ms,
        stop_reason=_get(choice, "finish_reason", "stop") or "stop",
        cost_usd=float(reported_cost) if reported_cost is not None else None,
    )
