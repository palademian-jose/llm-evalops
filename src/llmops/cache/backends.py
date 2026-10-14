"""Key/value backends for the exact cache: in-memory LRU and optional Redis."""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from typing import Any, Protocol


class Backend(Protocol):
    def get(self, key: str) -> dict[str, Any] | None: ...
    def set(self, key: str, value: dict[str, Any], ttl: int | None = None) -> None: ...
    def __len__(self) -> int: ...


class MemoryLRU:
    def __init__(self, maxsize: int = 1024):
        self.maxsize = maxsize
        self._data: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def set(self, key: str, value: dict[str, Any], ttl: int | None = None) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


class RedisBackend:
    """Redis-backed store (requires the `redis` extra and `REDIS_URL`)."""

    def __init__(self, url: str, *, prefix: str = "llmops:cache:", client: Any | None = None):
        if client is None:
            import redis

            client = redis.Redis.from_url(url)
        self._r = client
        self._prefix = prefix

    def get(self, key: str) -> dict[str, Any] | None:
        raw = self._r.get(self._prefix + key)
        if raw is None:
            return None
        return json.loads(raw)

    def set(self, key: str, value: dict[str, Any], ttl: int | None = None) -> None:
        payload = json.dumps(value)
        if ttl:
            self._r.set(self._prefix + key, payload, ex=ttl)
        else:
            self._r.set(self._prefix + key, payload)

    def __len__(self) -> int:
        return len(list(self._r.scan_iter(self._prefix + "*")))


def backend_from_settings(maxsize: int = 1024) -> Backend:
    from llmops.settings import get_settings

    url = get_settings().redis_url
    if url:
        return RedisBackend(url)
    return MemoryLRU(maxsize)
