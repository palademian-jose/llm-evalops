"""OpenAI chat completions backend (tools) over httpx. Only exercised with a key."""

from __future__ import annotations

import json
import time

import httpx

from llmops.llm.types import LLMResponse, Message, ToolCall, ToolSpec, Usage

API_URL = "https://api.openai.com/v1/chat/completions"


class OpenAIClient:
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
        body: dict = {
            "model": model or self._model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [self._to_api(m) for m in messages],
        }
        if tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in tools
            ]
        headers = {"Authorization": f"Bearer {self._key}", "content-type": "application/json"}
        start = time.perf_counter()
        resp = self._http.post(self._url, headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()
        latency = (time.perf_counter() - start) * 1000

        choice = data["choices"][0]
        msg = choice["message"]
        calls = [
            ToolCall(
                id=tc["id"],
                name=tc["function"]["name"],
                arguments=json.loads(tc["function"].get("arguments") or "{}"),
            )
            for tc in msg.get("tool_calls") or []
        ]
        usage = data.get("usage", {})
        return LLMResponse(
            text=msg.get("content") or "",
            tool_calls=calls,
            usage=Usage(
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            ),
            model=data.get("model", body["model"]),
            latency_ms=latency,
            stop_reason=choice.get("finish_reason") or "stop",
        )

    @staticmethod
    def _to_api(m: Message) -> dict:
        if m.role == "tool":
            return {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.text()}
        content = m.content if isinstance(m.content, str) else m.text()
        return {"role": m.role, "content": content}
