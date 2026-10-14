"""Streamlit dashboard (`llmops dashboard`). Requires the `dashboard` extra."""

from pathlib import Path

APP_PATH = Path(__file__).with_name("app.py")

__all__ = ["APP_PATH"]
