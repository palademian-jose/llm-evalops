"""Embedding backends for the semantic cache.

- `HashedBoWEmbedder` (`fake` / `local`): deterministic hashed bag-of-words with sublinear tf,
  L2-normalized. Good enough to make the semantic cache testable offline; not a real embedding.
- `OpenAIEmbedder` (`openai`): `text-embedding-3-*` over httpx; needs OPENAI_API_KEY.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import httpx
import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    model: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashedBoWEmbedder:
    model = "hashed-bow"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            counts: dict[int, int] = {}
            tokens = _TOKEN.findall(text.lower())
            for a, b in zip(tokens, tokens[1:] + [""], strict=True):
                for gram in (a, f"{a}_{b}" if b else None):
                    if gram is None:
                        continue
                    h = int(hashlib.blake2b(gram.encode(), digest_size=8).hexdigest(), 16)
                    counts[h % self.dim] = counts.get(h % self.dim, 0) + 1
            for idx, c in counts.items():
                out[i, idx] += 1.0 + math.log(c)
            norm = np.linalg.norm(out[i])
            if norm:
                out[i] /= norm
        return out


class OpenAIEmbedder:
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        *,
        base_url: str = "https://api.openai.com/v1/embeddings",
        timeout: float = 30.0,
    ):
        self.model = model
        self.dim = 3072 if "large" in model else 1536
        self._key = api_key
        self._url = base_url
        self._http = httpx.Client(timeout=timeout)

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self._http.post(
            self._url,
            headers={"Authorization": f"Bearer {self._key}"},
            json={"model": self.model, "input": texts},
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        arr = np.asarray(
            [d["embedding"] for d in sorted(data, key=lambda d: d["index"])], dtype=np.float32
        )
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


def get_embedder() -> Embedder:
    from llmops.settings import get_settings

    s = get_settings()
    if s.embedding_provider == "openai" and s.openai_api_key:
        return OpenAIEmbedder(s.openai_api_key, s.embedding_model)
    return HashedBoWEmbedder()
