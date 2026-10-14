"""llm-evalops is the source of truth for the settings the sibling repos share
(docs-rag and briefing-agent test against these names and defaults)."""

from __future__ import annotations

import json
from pathlib import Path

from llmops.pricing import bundled_pricing_path
from llmops.settings import Settings

ROOT = Path(__file__).resolve().parents[1]
SHARED = [
    "LLM_PROVIDER",
    "OPENROUTER_API_KEY",
    "LLM_MODEL",
    "JUDGE_MODEL",
    "CHEAP_MODEL",
    "TRACE_PATH",
]


def _env_names() -> set[str]:
    names = set()
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            names.add(line.split("=", 1)[0].strip())
    return names


def test_env_example_lists_the_shared_variables_and_every_variable_is_a_setting():
    names = _env_names()
    assert set(SHARED) <= names
    assert {n.lower() for n in names} <= set(Settings.model_fields) | {"pricing_path"}


def test_env_example_model_defaults_equal_the_settings_defaults():
    values = {}
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            values[k.strip()] = v.strip()
    defaults = Settings.model_fields
    for var in ("LLM_MODEL", "JUDGE_MODEL", "CHEAP_MODEL", "TRACE_PATH"):
        assert values[var] == defaults[var.lower()].default, var


def test_pricing_json_has_the_models_every_repo_prices():
    table = json.loads(bundled_pricing_path().read_text(encoding="utf-8"))
    for model in (
        "anthropic/claude-sonnet-4.5",
        "anthropic/claude-haiku-4.5",
        "openai/gpt-4.1-mini",
        "text-embedding-3-large",
        "rerank-v3.5",
        "fake-extractive-v1",
    ):
        assert model in table, model
