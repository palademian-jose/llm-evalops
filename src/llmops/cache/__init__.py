"""Exact and semantic response caches that emit cache spans."""

from llmops.cache.backends import MemoryLRU, RedisBackend, backend_from_settings
from llmops.cache.client import CachedClient
from llmops.cache.embeddings import HashedBoWEmbedder, OpenAIEmbedder, get_embedder
from llmops.cache.exact import ExactCache, request_key
from llmops.cache.semantic import SemanticCache
from llmops.cache.stats import cache_markdown, cache_stats, overall_hit_rate

__all__ = [
    "CachedClient",
    "ExactCache",
    "HashedBoWEmbedder",
    "MemoryLRU",
    "OpenAIEmbedder",
    "RedisBackend",
    "SemanticCache",
    "backend_from_settings",
    "cache_markdown",
    "cache_stats",
    "get_embedder",
    "overall_hit_rate",
    "request_key",
]
