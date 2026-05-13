"""
pipeline/analytics.py — Query analytics using Redis sorted sets.

Tracks:
  - Top queries (sorted by frequency)
  - Query volume per hour
  - Namespace usage
  - Cache hit rate
"""
import logging
from datetime import datetime
import redis
import os

log = logging.getLogger(__name__)

def _r():
    return redis.from_url(os.getenv("REDIS_URL", ""))

def log_query(query: str, namespace: str, cached: bool, lang: str) -> None:
    """Log a query for analytics."""
    try:
        r   = _r()
        now = datetime.utcnow()
        hour_key = f"analytics:hourly:{now.strftime('%Y%m%d%H')}"

        # Top queries — sorted set, score = frequency
        r.zincrby("analytics:top_queries", 1, query.strip().lower())

        # Hourly volume
        r.incr(hour_key)
        r.expire(hour_key, 7 * 24 * 3600)  # keep 7 days

        # Namespace usage
        r.zincrby("analytics:namespaces", 1, namespace or "all")

        # Language usage
        r.zincrby("analytics:languages", 1, lang or "en")

        # Cache hit rate
        r.incr("analytics:total_queries")
        if cached:
            r.incr("analytics:cached_queries")

    except Exception as e:
        log.warning("Analytics log error: %s", e)


def get_top_queries(limit: int = 20) -> list[dict]:
    """Return top N most searched queries."""
    try:
        r       = _r()
        results = r.zrevrange("analytics:top_queries", 0, limit - 1, withscores=True)
        return [{"query": q.decode(), "count": int(score)} for q, score in results]
    except Exception as e:
        log.warning("Analytics get error: %s", e)
        return []


def get_stats() -> dict:
    """Return overall analytics stats."""
    try:
        r      = _r()
        total  = int(r.get("analytics:total_queries") or 0)
        cached = int(r.get("analytics:cached_queries") or 0)

        namespaces = {
            ns.decode(): int(score)
            for ns, score in r.zrevrange("analytics:namespaces", 0, -1, withscores=True)
        }
        languages = {
            lang.decode(): int(score)
            for lang, score in r.zrevrange("analytics:languages", 0, 4, withscores=True)
        }

        # Hourly volume — last 24 hours
        now   = datetime.utcnow()
        hours = []
        for i in range(24):
            from datetime import timedelta
            h   = now - timedelta(hours=i)
            key = f"analytics:hourly:{h.strftime('%Y%m%d%H')}"
            vol = int(r.get(key) or 0)
            hours.append({"hour": h.strftime('%Y-%m-%d %H:00'), "count": vol})

        return {
            "total_queries":    total,
            "cached_queries":   cached,
            "cache_hit_rate":   round(cached / total * 100, 1) if total > 0 else 0,
            "namespaces":       namespaces,
            "top_languages":    languages,
            "hourly_volume":    list(reversed(hours)),
        }
    except Exception as e:
        log.warning("Analytics stats error: %s", e)
        return {}
"""
pipeline/analytics.py — Query analytics using Redis sorted sets.

Tracks:
  - Top queries (sorted by frequency)
  - Query volume per hour
  - Namespace usage
  - Cache hit rate
"""
import logging
from datetime import datetime
import redis
import os

log = logging.getLogger(__name__)

def _r():
    return redis.from_url(os.getenv("REDIS_URL", ""))

def log_query(query: str, namespace: str, cached: bool, lang: str) -> None:
    """Log a query for analytics."""
    try:
        r   = _r()
        now = datetime.utcnow()
        hour_key = f"analytics:hourly:{now.strftime('%Y%m%d%H')}"

        # Top queries — sorted set, score = frequency
        r.zincrby("analytics:top_queries", 1, query.strip().lower())

        # Hourly volume
        r.incr(hour_key)
        r.expire(hour_key, 7 * 24 * 3600)  # keep 7 days

        # Namespace usage
        r.zincrby("analytics:namespaces", 1, namespace or "all")

        # Language usage
        r.zincrby("analytics:languages", 1, lang or "en")

        # Cache hit rate
        r.incr("analytics:total_queries")
        if cached:
            r.incr("analytics:cached_queries")

    except Exception as e:
        log.warning("Analytics log error: %s", e)


def get_top_queries(limit: int = 20) -> list[dict]:
    """Return top N most searched queries."""
    try:
        r       = _r()
        results = r.zrevrange("analytics:top_queries", 0, limit - 1, withscores=True)
        return [{"query": q.decode(), "count": int(score)} for q, score in results]
    except Exception as e:
        log.warning("Analytics get error: %s", e)
        return []


def get_stats() -> dict:
    """Return overall analytics stats."""
    try:
        r      = _r()
        total  = int(r.get("analytics:total_queries") or 0)
        cached = int(r.get("analytics:cached_queries") or 0)

        namespaces = {
            ns.decode(): int(score)
            for ns, score in r.zrevrange("analytics:namespaces", 0, -1, withscores=True)
        }
        languages = {
            lang.decode(): int(score)
            for lang, score in r.zrevrange("analytics:languages", 0, 4, withscores=True)
        }

        # Hourly volume — last 24 hours
        now   = datetime.utcnow()
        hours = []
        for i in range(24):
            from datetime import timedelta
            h   = now - timedelta(hours=i)
            key = f"analytics:hourly:{h.strftime('%Y%m%d%H')}"
            vol = int(r.get(key) or 0)
            hours.append({"hour": h.strftime('%Y-%m-%d %H:00'), "count": vol})

        return {
            "total_queries":    total,
            "cached_queries":   cached,
            "cache_hit_rate":   round(cached / total * 100, 1) if total > 0 else 0,
            "namespaces":       namespaces,
            "top_languages":    languages,
            "hourly_volume":    list(reversed(hours)),
        }
    except Exception as e:
        log.warning("Analytics stats error: %s", e)
        return {}
