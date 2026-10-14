"""Anthropic Messages API backend (native tool use) over httpx. Only exercised with a key."""

from __future__ import annotations

import time

import httpx

from llmops.llm.types import LLMResponse, Message, ToolCall, ToolSpec, Usage

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"


class AnthropicClient:
    def __init__(
        self, api_key: str, default_model: str, timeout: float = 60.0, base_url: str = API_URL
    ):
        self._key = api_key
        self._model = default_model
        self._http = httpx.Client(timeout=timeout)
        self._url = base_url

    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        system = "\n".join(m.text() for m in messages if m.role == "system")
        body: dict = {
            "model": model or self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [self._to_api(m) for m in messages if m.role != "system"],
        }
        if system:
            body["system"] = system
        if tools:
            body["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in tools
            ]
        headers = {
            "x-api-key": self._key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }
        start = time.perf_counter()
        resp = self._http.post(self._url, headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()
        latency = (time.perf_counter() - start) * 1000

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                text_parts.append(block["text"])
            elif block.get("type") == "tool_use":
                calls.append(
                    ToolCall(id=block["id"], name=block["name"], arguments=block.get("input") or {})
                )
        usage = data.get("usage", {})
        return LLMResponse(
            text="".join(text_parts),
            tool_calls=calls,
            usage=Usage(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
            ),
            model=data.get("model", body["model"]),
            latency_ms=latency,
            stop_reason=data.get("stop_reason") or "end_turn",
        )

    @staticmethod
    def _to_api(m: Message) -> dict:
        if m.role == "tool":
            return {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.text()}
                ],
            }
        return {"role": m.role, "content": m.content}


def tool_use_block(call: ToolCall) -> dict:
    """Assistant content block for replaying a tool call in Anthropic format."""
    return {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
