"""
api/deps.py — Shared dependencies for all API routers.

Extracted from api/search.py (Tier 1b refactor) so route modules
(search, analytics, admin, visitor, system) share auth + dynamic config
without importing from each other (avoids circular imports).
"""
import logging
import threading as _threading
import time as _time

from fastapi import Depends, HTTPException
from fastapi.security.api_key import APIKeyHeader

from config import settings

log = logging.getLogger("api.deps")

# ── API key auth ──────────────────────────────────────────────────
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(key: str = Depends(api_key_header)):
    if settings.API_KEY and key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return key


# ── Dynamic config from DynamoDB rag-config ─────────────────────────
_config_cache      = {}
_config_lock       = _threading.Lock()
_config_loaded_at  = 0
_CONFIG_TTL        = 300  # refresh every 5 minutes


def _load_config() -> dict:
    """Load config categories from the rag-config DynamoDB table (cached)."""
    global _config_cache, _config_loaded_at
    now = _time.time()
    if _config_cache and (now - _config_loaded_at) < _CONFIG_TTL:
        return _config_cache
    with _config_lock:
        if _config_cache and (now - _config_loaded_at) < _CONFIG_TTL:
            return _config_cache
        try:
            import boto3 as _b3
            _ddb   = _b3.resource("dynamodb", region_name="us-east-1")
            _table = _ddb.Table("rag-config")
            keys   = ["workload_signals", "product_signals",
                      "commercial_keywords", "workload_badge_styles",
                      "sections"]
            loaded = {}
            for k in keys:
                resp = _table.get_item(Key={"config_key": k})
                if "Item" in resp:
                    loaded[k] = resp["Item"].get("data", {})
            if loaded:
                _config_cache     = loaded
                _config_loaded_at = now
                log.info("rag-config loaded from DynamoDB: %s", list(loaded.keys()))
        except Exception as _ce:
            log.warning("rag-config load failed — using hardcoded fallback: %s", _ce)
    return _config_cache


def get_config(key: str, fallback):
    """Get a config value, falling back to hardcoded if DynamoDB unavailable."""
    cfg = _load_config()
    return cfg.get(key, fallback)


def invalidate_config():
    """Bust the in-process config cache so it reloads on the next get_config."""
    global _config_loaded_at
    _config_loaded_at = 0


# ── Section → namespace resolution (Option B; grandfather model) ──────────────
# resources is grandfathered to its 3 legacy type-namespaces; new sections get
# their own single namespace. A "section" maps to a LIST of namespaces.
_SECTIONS_FALLBACK = {
    "resources": {"namespaces": ["technical", "business", "media"], "source_type": "sitemap"},
}


def resolve_section_namespaces(section: str = None) -> list:
    """Resolve a section name -> its Pinecone namespace list.
    None / "" / "all" -> union of every section's namespaces.
    A known section name -> that section's namespaces.
    Anything else -> [section] (back-compat: a literal namespace passed directly).
    """
    sections = get_config("sections", _SECTIONS_FALLBACK)
    if not section or section.strip().lower() in ("", "all"):
        out, seen = [], set()
        for s in sections.values():
            for ns in s.get("namespaces", []):
                if ns not in seen:
                    out.append(ns); seen.add(ns)
        return out
    if section in sections:
        return sections[section].get("namespaces", [])
    return [section.strip()]
