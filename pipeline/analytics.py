"""
pipeline/analytics.py — Query analytics using Redis sorted sets.

Tracks:
  - Top queries (sorted by frequency)
  - Query volume per hour
  - Namespace usage
  - Cache hit rate
  - Intent distribution        (new)
  - Product demand signals     (new)
  - Use case demand            (new)
  - Filter effectiveness       (new)
  - Query rewrite rate         (new)
"""
import logging
from datetime import datetime, timedelta
import redis
import os

log = logging.getLogger(__name__)


def _r():
    return redis.from_url(os.getenv("REDIS_URL", ""))


def log_query(
    query:             str,
    namespace:         str,
    cached:            bool,
    lang:              str,
    intent:            str   = "unknown",
    confidence:        float = 0.0,
    rewritten_query:   str   = "",
    detected_products: list  = None,
    detected_use_case: str   = "",
    filter_applied:    bool  = False,
    filter_fallback:   bool  = False,
    top_score:         float = 0.0,
    chunk_count:       int   = 0,
) -> None:
    """
    Log a query for analytics.
    All new fields are optional with safe defaults — existing callers unaffected.
    """
    try:
        r   = _r()
        now = datetime.utcnow()
        hour_key = f"analytics:hourly:{now.strftime('%Y%m%d%H')}"

        # ── Existing tracking ─────────────────────────────────────────────────
        r.zincrby("analytics:top_queries", 1, query.strip().lower())
        r.incr(hour_key)
        r.expire(hour_key, 7 * 24 * 3600)   # keep 7 days
        r.zincrby("analytics:namespaces", 1, namespace or "all")
        r.zincrby("analytics:languages", 1, lang or "en")
        r.incr("analytics:total_queries")
        if cached:
            r.incr("analytics:cached_queries")

        # ── Intent distribution ───────────────────────────────────────────────
        if intent and intent != "unknown":
            r.zincrby("analytics:intents", 1, intent)
            # Hourly intent breakdown for trend analysis
            intent_hour = f"analytics:intent_hourly:{intent}:{now.strftime('%Y%m%d%H')}"
            r.incr(intent_hour)
            r.expire(intent_hour, 7 * 24 * 3600)

        # ── Product demand signals ────────────────────────────────────────────
        for product in (detected_products or []):
            r.zincrby("analytics:products", 1, product)

        # ── Use case demand ───────────────────────────────────────────────────
        if detected_use_case:
            r.zincrby("analytics:use_cases", 1, detected_use_case)

        # ── Filter effectiveness ──────────────────────────────────────────────
        if filter_applied:
            r.incr("analytics:total_filtered_queries")
        else:
            r.incr("analytics:total_unfiltered_queries")
        if filter_fallback:
            r.incr("analytics:filter_fallbacks")

        # ── Query rewrite rate ────────────────────────────────────────────────
        if rewritten_query and rewritten_query.strip() != query.strip():
            r.incr("analytics:rewritten_queries")

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
    """Return overall analytics stats including intent and product signals."""
    try:
        r      = _r()
        total  = int(r.get("analytics:total_queries") or 0)
        cached = int(r.get("analytics:cached_queries") or 0)

        # Namespace usage
        namespaces = {
            ns.decode(): int(score)
            for ns, score in r.zrevrange("analytics:namespaces", 0, -1, withscores=True)
        }

        # Language distribution
        languages = {
            lang.decode(): int(score)
            for lang, score in r.zrevrange("analytics:languages", 0, 4, withscores=True)
        }

        # Hourly volume — last 24 hours
        now   = datetime.utcnow()
        hours = []
        for i in range(24):
            h   = now - timedelta(hours=i)
            key = f"analytics:hourly:{h.strftime('%Y%m%d%H')}"
            vol = int(r.get(key) or 0)
            hours.append({"hour": h.strftime('%Y-%m-%d %H:00'), "count": vol})

        # ── New: intent distribution ──────────────────────────────────────────
        intents = {
            i.decode(): int(score)
            for i, score in r.zrevrange("analytics:intents", 0, -1, withscores=True)
        }

        # ── New: top queried products ─────────────────────────────────────────
        top_products = [
            {"product": p.decode(), "count": int(score)}
            for p, score in r.zrevrange("analytics:products", 0, 9, withscores=True)
        ]

        # ── New: top queried use cases ────────────────────────────────────────
        top_use_cases = [
            {"use_case": u.decode(), "count": int(score)}
            for u, score in r.zrevrange("analytics:use_cases", 0, 9, withscores=True)
        ]

        # ── New: filter effectiveness ─────────────────────────────────────────
        total_filtered   = int(r.get("analytics:total_filtered_queries") or 0)
        total_unfiltered = int(r.get("analytics:total_unfiltered_queries") or 0)
        filter_fallbacks = int(r.get("analytics:filter_fallbacks") or 0)
        rewritten        = int(r.get("analytics:rewritten_queries") or 0)

        filter_hit_rate = (
            round(total_filtered / (total_filtered + total_unfiltered) * 100, 1)
            if (total_filtered + total_unfiltered) > 0 else 0
        )
        fallback_rate = (
            round(filter_fallbacks / total_filtered * 100, 1)
            if total_filtered > 0 else 0
        )

        return {
            # Existing
            "total_queries":    total,
            "cached_queries":   cached,
            "cache_hit_rate":   round(cached / total * 100, 1) if total > 0 else 0,
            "namespaces":       namespaces,
            "top_languages":    languages,
            "hourly_volume":    list(reversed(hours)),
            # New
            "intent_distribution": intents,
            "top_products":        top_products,
            "top_use_cases":       top_use_cases,
            "filter_hit_rate":     filter_hit_rate,
            "filter_fallback_rate": fallback_rate,
            "rewrite_rate":        round(rewritten / total * 100, 1) if total > 0 else 0,
            "total_filtered":      total_filtered,
            "filter_fallbacks":    filter_fallbacks,
        }

    except Exception as e:
        log.warning("Analytics stats error: %s", e)
        return {}
