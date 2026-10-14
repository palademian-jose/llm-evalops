"""Pricing table loader and cost math. `llmops` is the source of truth for `pricing.json`."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from importlib import resources
from pathlib import Path

from pydantic import BaseModel


class Price(BaseModel):
    input_per_m: float = 0.0
    output_per_m: float = 0.0
    per_search: float = 0.0


def bundled_pricing_path() -> Path:
    return Path(str(resources.files("llmops").joinpath("pricing.json")))


def pricing_path() -> Path:
    override = os.environ.get("PRICING_PATH")
    if override:
        return Path(override)
    return bundled_pricing_path()


@lru_cache(maxsize=8)
def _load(path: str) -> dict[str, Price]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k.lower(): Price(**v) for k, v in raw.items() if not k.startswith("_")}


def load_pricing(path: str | os.PathLike[str] | None = None) -> dict[str, Price]:
    """Load the pricing table. Order: explicit `path` > `PRICING_PATH` env > bundled file."""
    p = Path(path) if path else pricing_path()
    return _load(str(p))


def reset_pricing_cache() -> None:
    _load.cache_clear()


def _variants(model: str) -> list[str]:
    """Candidate keys for a model id, most specific first."""
    m = model.lower().strip()
    out = [m]
    bare = m.split("/", 1)[1] if "/" in m else m
    if bare != m:
        out.append(bare)
    for base in list(out):
        # OpenRouter uses `claude-sonnet-4.5`; vendors use `claude-sonnet-4-5`. Accept both.
        dashed = base.replace(".", "-")
        dotted_head = base
        if dashed not in out:
            out.append(dashed)
        if dotted_head not in out:
            out.append(dotted_head)
    return out


def normalize_model(model: str, table: dict[str, Price] | None = None) -> str:
    """Map OpenRouter slugs, dated ids and aliases onto a pricing key.

    Order: exact key > vendor prefix stripped > dots/dashes normalized > longest key that is a
    prefix of the id (dated ids such as `claude-sonnet-4-5-20250929`). Returns the input when
    nothing matches.
    """
    table = table or load_pricing()
    variants = _variants(model)
    for v in variants:
        if v in table:
            return v
    for v in variants:
        candidates = [k for k in table if v.startswith(k) and not k.startswith("fake")]
        if candidates:
            return max(candidates, key=len)
    return model.lower().strip()


def canonical_model(model: str, table: dict[str, Price] | None = None) -> str:
    """One display key per model: the OpenRouter slug when the table has one, else the bare key.

    `claude-sonnet-4-5`, `claude-sonnet-4.5` and `anthropic/claude-sonnet-4.5` all map to
    `anthropic/claude-sonnet-4.5`, so bills and spans reconcile regardless of spelling.
    """
    table = table or load_pricing()
    key = normalize_model(model, table)
    if "/" in key:
        return key
    dashed = key.replace(".", "-")
    for k in table:
        if "/" in k and k.split("/", 1)[1].replace(".", "-") == dashed:
            return k
    return key


def estimate_cost(
    model: str | None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    *,
    searches: int = 0,
    pricing: dict[str, Price] | None = None,
) -> float | None:
    """USD cost for one call, or None when the model is not in the pricing table."""
    if not model:
        return None
    table = pricing or load_pricing()
    price = table.get(normalize_model(model, table))
    if price is None:
        return None
    cost = (input_tokens / 1_000_000) * price.input_per_m
    cost += (output_tokens / 1_000_000) * price.output_per_m
    cost += searches * price.per_search
    return round(cost, 10)
