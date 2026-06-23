"""
cache/factory.py — Returns the configured cache backend.
Switch backends by changing CACHE_BACKEND env var — no code changes needed.

  CACHE_BACKEND=memory    → MemoryCache  (default, good for single instance)
  CACHE_BACKEND=redis     → RedisCache   (recommended for production)
"""
from cache.base import BaseCache
from config import settings


def get_cache() -> BaseCache:
    backend = settings.CACHE_BACKEND.lower()

    if backend == "redis":
        from cache.redis_cache import RedisCache
        return RedisCache()

    if backend == "dynamodb":
        from cache.dynamodb_cache import DynamoDBCache
        return DynamoDBCache()

    # Default: in-memory
    from cache.memory import MemoryCache
    return MemoryCache()


# Singleton — one cache instance for the whole process
_cache: BaseCache | None = None


def cache() -> BaseCache:
    global _cache
    if _cache is None:
        _cache = get_cache()
    return _cache
