from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _fake_provider(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Every test runs with the fake provider and an isolated trace path."""
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("TRACE_PATH", str(tmp_path / "traces" / "spans.jsonl"))
    monkeypatch.setenv("ALERTS_DIR", str(tmp_path / "alerts"))
    monkeypatch.delenv("PRICING_PATH", raising=False)
    from llmops.settings import reset_settings_cache

    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture
def env() -> dict[str, str]:
    """Environment for subprocess-based CLI tests."""
    e = dict(os.environ)
    e["LLM_PROVIDER"] = "fake"
    e.pop("OPENROUTER_API_KEY", None)
    return e


@pytest.fixture
def repo_root() -> Path:
    return ROOT
