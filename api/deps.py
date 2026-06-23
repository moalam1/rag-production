"""
api/deps.py — Shared web dependencies for all API routers (auth).

The dynamic-config + section-resolution logic was extracted to config_dynamic.py
(web-free, so the RAG pipeline can import it without dragging in FastAPI). It is
re-exported here for backward compatibility — existing
`from api.deps import get_config` / `resolve_section_namespaces` / etc. keep working.
"""
import logging
from fastapi import Depends, HTTPException
from fastapi.security.api_key import APIKeyHeader

from config import settings

# Re-export the web-free config/section helpers (back-compat for existing callers)
from config_dynamic import (
    get_config,
    invalidate_config,
    resolve_section_namespaces,
    list_released_sections,
    validate_search_namespace,
    resolve_write_namespace,
    _load_config,
    _SECTIONS_FALLBACK,
)

log = logging.getLogger("api.deps")

# ── API key auth (the only genuinely web-coupled dependency) ──────────
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(key: str = Depends(api_key_header)):
    if settings.API_KEY and key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return key
