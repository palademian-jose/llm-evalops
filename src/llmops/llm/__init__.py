"""LLM provider abstraction shared by evalkit and llmops (PLAN.md shared contract 1)."""

from llmops.llm.factory import get_client
from llmops.llm.fake import FakeClient, FakeConfig
from llmops.llm.types import (
    LLMClient,
    LLMResponse,
    Message,
    ToolCall,
    ToolSpec,
    Usage,
    approx_tokens,
    model_family,
)

__all__ = [
    "FakeClient",
    "FakeConfig",
    "LLMClient",
    "LLMResponse",
    "Message",
    "ToolCall",
    "ToolSpec",
    "Usage",
    "approx_tokens",
    "get_client",
    "model_family",
]
