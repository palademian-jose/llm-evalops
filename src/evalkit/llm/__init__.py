"""Re-export of the shared provider abstraction so `evalkit.llm` matches the PLAN.md layout."""

from llmops.llm import (
    FakeClient,
    FakeConfig,
    LLMClient,
    LLMResponse,
    Message,
    ToolCall,
    ToolSpec,
    Usage,
    get_client,
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
    "get_client",
    "model_family",
]
