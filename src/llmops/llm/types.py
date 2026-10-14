"""Shared LLM provider contract (PLAN.md, shared contract 1)."""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict]
    tool_call_id: str | None = None

    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        parts: list[str] = []
        for block in self.content:
            if isinstance(block, dict):
                if "text" in block:
                    parts.append(str(block["text"]))
                elif "content" in block:
                    parts.append(str(block["content"]))
        return "\n".join(parts)


class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict  # JSON Schema, additionalProperties false


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict = Field(default_factory=dict)


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class LLMResponse(BaseModel):
    text: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    model: str
    latency_ms: float = 0.0
    stop_reason: str = "end_turn"
    # Provider-reported cost (OpenRouter `usage.cost`); None when the provider does not report it.
    cost_usd: float | None = None


@runtime_checkable
class LLMClient(Protocol):
    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse: ...


_VENDOR_ALIASES = {
    "anthropic": "anthropic",
    "openai": "openai",
    "google": "google",
    "meta-llama": "meta",
    "mistralai": "mistral",
    "deepseek": "deepseek",
    "qwen": "qwen",
    "cohere": "cohere",
    "x-ai": "xai",
}


def model_family(model: str | None) -> str:
    """Coarse model family used for the cross-family judge check.

    OpenRouter slugs carry the vendor as a prefix (`anthropic/claude-sonnet-4.5`), which is used
    directly. Bare ids fall back to name heuristics (`claude-*` -> anthropic, `gpt-*` -> openai).
    """
    if not model:
        return "unknown"
    m = model.lower().strip()
    if "/" in m:
        vendor = m.split("/", 1)[0]
        return _VENDOR_ALIASES.get(vendor, vendor)
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith(("gpt", "o1", "o3", "o4", "text-embedding", "chatgpt")):
        return "openai"
    if m.startswith("gemini"):
        return "google"
    if m.startswith("fake"):
        return "fake"
    if m.startswith("llama"):
        return "meta"
    return m.split("-")[0]


def approx_tokens(text: str) -> int:
    """Cheap, deterministic token estimate (~4 chars per token) for the fake provider."""
    return max(1, (len(text) + 3) // 4)
