"""
cache/redis_cache.py — Redis-backed cache.

Best for:  Multi-pod production deployments, shared cache across instances.
Requires:  pip install redis
Setup:     Set CACHE_BACKEND=redis and REDIS_URL=redis://host:6379
"""
import json
import hashlib
from typing import Any

from cache.base import BaseCache
from config import settings


class RedisCache(BaseCache):

    def __init__(self, url: str = settings.REDIS_URL,
                 default_ttl: int = settings.CACHE_TTL_SECONDS):
        try:
            import redis
            self._client = redis.from_url(url, decode_responses=True)
            self._client.ping()
        except Exception as e:
            raise RuntimeError(f"Redis connection failed: {e}")
        self.default_ttl = default_ttl

    # ── Public API ────────────────────────────────────────────────

    def get(self, key: str) -> Any | None:
        raw = self._client.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        ttl = ttl or self.default_ttl
        self._client.setex(key, ttl, json.dumps(value, default=str))

    def delete(self, key: str) -> None:
        self._client.delete(key)

    def clear(self) -> None:
        self._client.flushdb()

    def exists(self, key: str) -> bool:
        return bool(self._client.exists(key))

    def stats(self) -> dict:
        info = self._client.info("memory")
        return {
            "used_memory_human": info.get("used_memory_human"),
            "connected_clients": self._client.info("clients").get("connected_clients"),
            "total_keys":        self._client.dbsize(),
        }

    # ── Key helpers ───────────────────────────────────────────────

    @staticmethod
    def make_key(namespace: str, data: Any) -> str:
        raw    = json.dumps(data, sort_keys=True, default=str)
        digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return f"rag:{namespace}:{digest}"
