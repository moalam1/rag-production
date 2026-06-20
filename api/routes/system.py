"""
api/routes/system.py — System / operational endpoints.

Extracted from api/search.py (Tier 1e). Small utility endpoints: health check
(unauthenticated), cache stats + clear, document registry listing, and the
badge-styles config read used by the UI. Auth + config from api.deps. Endpoint
bodies are verbatim moves.

Note: /health is intentionally UNAUTHENTICATED (no verify_api_key) — it's the
uptime probe used by the gateway/load balancer.
"""
from fastapi import APIRouter, Depends

from api.deps import verify_api_key, get_config
from config import settings

router = APIRouter(prefix="/api/v1", tags=["system"])


@router.get("/config/badge-styles")
async def badge_styles(_: str = Depends(verify_api_key)):
    """Return workload badge styles from rag-config — used by HF Space UI."""
    styles = get_config("workload_badge_styles", {
        "Distributed AI":        {"icon":"🤖","bg":"#0c2340","color":"#93c5fd"},
        "AI & Machine Learning": {"icon":"🤖","bg":"#0c2340","color":"#93c5fd"},
        "SD-WAN":                {"icon":"🔀","bg":"#1a1200","color":"#fcd34d"},
        "Hybrid Multicloud":     {"icon":"☁️", "bg":"#0a1628","color":"#60a5fa"},
        "Financial Services":    {"icon":"🏦","bg":"#0a1a0a","color":"#86efac"},
        "Network Modernization": {"icon":"🔧","bg":"#1a0a1a","color":"#d8b4fe"},
        "Colocation":            {"icon":"🏢","bg":"#1a1a0a","color":"#fde68a"},
        "Interconnection":       {"icon":"🔗","bg":"#0f1a1a","color":"#5eead4"},
    })
    return {"badge_styles": styles}


@router.get("/health")
async def health():
    return {"status": "ok", "environment": settings.ENVIRONMENT}


@router.get("/cache/stats")
async def cache_stats(_: str = Depends(verify_api_key)):
    from cache.factory import cache
    c = cache()
    if hasattr(c, "stats"):
        return c.stats()
    return {"message": "Stats not available for this backend"}


@router.delete("/cache")
async def clear_cache(_: str = Depends(verify_api_key)):
    from cache.factory import cache
    cache().clear()
    return {"message": "Cache cleared"}


@router.get("/registry")
async def list_registry(_: str = Depends(verify_api_key)):
    """List all documents in the DynamoDB registry."""
    from pipeline.registry import list_documents
    docs = list_documents()
    return {
        "total": len(docs),
        "documents": docs,
    }
    
