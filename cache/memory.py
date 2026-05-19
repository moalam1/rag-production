"""
cache/memory.py — In-process LRU cache with TTL.

Best for:  Single-instance deployments, local dev, HF Spaces.
Limit:     Data lost on restart. Not shared between pods.
"""
import time
import hashlib
import json
from collections import OrderedDict
from typing import Any

from cache.base import BaseCache
from config import settings


class MemoryCache(BaseCache):

    def __init__(self, max_size: int = settings.CACHE_MAX_SIZE,
                 default_ttl: int = settings.CACHE_TTL_SECONDS):
        self._store: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self.max_size    = max_size
        self.default_ttl = default_ttl

    # ── Public API ────────────────────────────────────────────────

    def get(self, key: str) -> Any | None:
        if key not in self._store:
            return None
        value, expires_at = self._store[key]
        if time.time() > expires_at:
            del self._store[key]
            return None
        # LRU: move to end
        self._store.move_to_end(key)
        return value

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        ttl = ttl or self.default_ttl
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = (value, time.time() + ttl)
        # Evict oldest if over capacity
        if len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    def stats(self) -> dict:
        now = time.time()
        alive = sum(1 for _, (_, exp) in self._store.items() if exp > now)
        return {"total_keys": len(self._store), "alive_keys": alive, "max_size": self.max_size}

    # ── Key helpers ───────────────────────────────────────────────

    @staticmethod
    def make_key(namespace: str, data: Any) -> str:
        """Deterministic cache key from any serialisable data."""
        raw = json.dumps(data, sort_keys=True, default=str)
        digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return f"rag:{namespace}:{digest}"
