"""
pipeline/semantic_cache.py — Pinecone semantic cache (Layer 1).

Sits between Redis exact cache and the full RAG pipeline.
Stores past query embeddings + answers in a dedicated Pinecone namespace.
Returns cached answer if a similar past query is found (cosine >= 0.90, TTL 7 days).

Cache hierarchy:
  Layer 1: Redis exact     hash(query+lang) < 100ms  — identical queries
  Layer 2: Pinecone sem.   cosine >= 0.92   ~300ms   — similar queries  <- THIS FILE
  Layer 3: Full pipeline   embed+retrieve   4-8s     — new queries
"""
import json
import logging
import time
import hashlib
from typing import Optional

from pinecone import Pinecone
from config import settings

log = logging.getLogger(__name__)

CACHE_NAMESPACE      = "semantic-cache"
SIMILARITY_THRESHOLD = 0.90
MAX_METADATA_CHARS   = 5000
TTL_DAYS             = 7      # cached answers expire after 7 days
CACHE_VERSION        = 6       # bump when stored format changes
from pipeline.prompt_registry import get_prompt_version as _gpv
_PROMPT_VERSION_FALLBACK = 2
def _pv(): return _gpv("generation", _PROMPT_VERSION_FALLBACK)
                               # must match pv= in generator.py cache key
                               # mismatch = cache miss → fresh generation       # v2/v3 missing _intent field       # v2 had intent=general bug       # bump when stored format changes — auto-invalidates old entries
                               # prevents stale answers after content updates
                               # weekly nightly crawl + rebuild_bm25 keeps cache fresh

try:
    _pc        = Pinecone(api_key=settings.PINECONE_API_KEY)
    _index     = _pc.Index(settings.PINECONE_INDEX)
    _AVAILABLE = True
    log.info("Semantic cache initialised — namespace: %s threshold: %.2f",
             CACHE_NAMESPACE, SIMILARITY_THRESHOLD)
except Exception as e:
    _index     = None
    _AVAILABLE = False
    log.warning("Semantic cache unavailable: %s", e)


def _make_id(query: str, lang: str) -> str:
    raw = f"{lang}:{query.strip().lower()}"
    return "sc:" + hashlib.sha256(raw.encode()).hexdigest()[:24]


def get(query_embedding: list, lang: str = "en") -> Optional[dict]:
    if not _AVAILABLE or not _index:
        return None
    try:
        results = _index.query(
            vector=query_embedding,
            top_k=1,
            include_metadata=True,
            namespace=CACHE_NAMESPACE,
            filter={"lang": {"$eq": lang}},
        )
        if not results.matches:
            return None
        top        = results.matches[0]
        similarity = top.score
        if similarity < SIMILARITY_THRESHOLD:
            log.debug("Semantic cache NEAR-MISS: %.4f < %.2f", similarity, SIMILARITY_THRESHOLD)
            return None
        meta = top.metadata
        log.info("Semantic cache HIT: %.4f | '%s...'", similarity, meta.get("query", "")[:50])
        print(f"[CACHE DEBUG] HIT id={top.id} score={similarity:.4f} stored_query={meta.get('query','')[:80]}", flush=True)
        try:
            stored = json.loads(meta.get("answer_json", "{}"))

            if int(stored.get("prompt_version", 0)) != _pv():
                log.info("Semantic cache PV-MISS (path %s): stored=%s current=%s",
                             1, stored.get("prompt_version"), _pv())
                return None
        except json.JSONDecodeError:
            return None

        # Version check — treat old format entries as cache miss
        if stored.get("cache_version", 1) < CACHE_VERSION:
            log.info("Semantic cache VERSION-MISS: entry v%d < current v%d",
                     stored.get("cache_version", 1), CACHE_VERSION)
            return None

        # Version check — treat old format entries as cache miss
        if stored.get("cache_version", 1) < CACHE_VERSION:
            log.info("Semantic cache VERSION-MISS: entry v%d < current v%d",
                     stored.get("cache_version", 1), CACHE_VERSION)
            return None

        # TTL check — treat expired vectors as cache miss
        expires_at = stored.get("expires_at")
        if expires_at:
            from datetime import datetime, timezone
            try:
                exp = datetime.fromisoformat(expires_at)
                if exp < datetime.now(timezone.utc):
                    log.info("Semantic cache EXPIRED: %.4f | cached entry past %dd TTL",
                             similarity, TTL_DAYS)
                    return None  # expired — full pipeline runs, re-populates cache
            except Exception:
                pass  # malformed date — treat as valid
        # Prompt-version check — entries from older prompts are stale by definition
        if int(stored.get("prompt_version", 0)) != _pv():
            log.info("Semantic cache PV-MISS: stored=%s current=%s",
                     stored.get("prompt_version"), _pv())
            return None

        # Intent sanity check — reject general intent if query has product keywords
        # Prevents a cached general answer from polluting specific product queries
        cached_intent = stored.get("_intent", stored.get("intent", "general"))
        if cached_intent == "general":
            _PRODUCT_SIGNALS = ["fabric", "metal", "network edge", "equinix", "xscale",
                                "interconnect", "colocation", "ibx", "fcr"]
            q_lower = meta.get("query", "").lower()
            if any(p in q_lower for p in _PRODUCT_SIGNALS):
                log.info("Semantic cache INTENT-MISS: general intent cached for product query '%s'",
                         q_lower[:50])
                return None

        _increment_hit_count(top.id, meta)
        return {
            "answer":            stored.get("answer", ""),
            "followups":         stored.get("followups", []),
            "sources":           stored.get("sources", []),
            "cache_hit":         True,
            "semantic_hit":      True,
            "similarity":        round(similarity, 4),
            # Intent fields — stored at cache time, returned on cache hit
            "_intent":            stored.get("intent", "general"),
            "_detected_products": stored.get("detected_products", []),
            "_detected_use_case": stored.get("detected_use_case", ""),
            "_rewritten_query":   stored.get("rewritten_query", ""),
            "_confidence":        stored.get("confidence", 0.0),
        }
    except Exception as e:
        log.warning("Semantic cache get error (skipping): %s", e)
        return None


def set(query: str, query_embedding: list, result: dict, lang: str = "en") -> None:
    if not _AVAILABLE or not _index:
        return
    try:
        answer_json = json.dumps({
            "answer":            result.get("answer", ""),
            "followups":         result.get("followups", []),
            "sources":           result.get("sources", []),
            "intent":            result.get("_intent", "general"),
            "detected_products": result.get("_detected_products", []),
            "detected_use_case": result.get("_detected_use_case", ""),
            "rewritten_query":   result.get("_rewritten_query", ""),
            "confidence":        result.get("_confidence", 0.0),
        }, ensure_ascii=False)
        if len(answer_json) > MAX_METADATA_CHARS:
            # Too large — drop sources first, then truncate answer
            answer_json = json.dumps({
                "answer":    result.get("answer", ""),
                "followups": result.get("followups", []),
                "sources":   [],
            }, ensure_ascii=False)
        if len(answer_json) > MAX_METADATA_CHARS:
            answer_json = json.dumps({
                "answer":    result.get("answer", "")[:MAX_METADATA_CHARS - 100],
                "followups": [],
                "sources":   [],
            }, ensure_ascii=False)
        # Embed TTL inside answer_json — Pinecone has no native per-vector TTL
        from datetime import datetime, timedelta, timezone
        expires_at = (datetime.now(timezone.utc) + timedelta(days=TTL_DAYS)).isoformat()
        try:
            stored = json.loads(answer_json)

            if int(stored.get("prompt_version", 0)) != _pv():
                log.info("Semantic cache PV-MISS (path %s): stored=%s current=%s",
                             2, stored.get("prompt_version"), _pv())
                return None
            stored["expires_at"]     = expires_at
            stored["cache_version"]  = CACHE_VERSION
            stored["prompt_version"] = _pv()
            answer_json = json.dumps(stored, ensure_ascii=False)
        except Exception:
            pass  # if re-encode fails, store without TTL

        _index.upsert(
            vectors=[{
                "id":     _make_id(query, lang),
                "values": query_embedding,
                "metadata": {
                    "query":       query[:200],
                    "lang":        lang,
                    "answer_json": answer_json,
                    "cached_at":   int(time.time()),
                    "hit_count":   0,
                },
            }],
            namespace=CACHE_NAMESPACE,
        )
        log.debug("Semantic cache SET: lang=%s ttl=%dd query='%s...'",
                  lang, TTL_DAYS, query[:40])
    except Exception as e:
        log.warning("Semantic cache set error (non-blocking): %s", e)


def _increment_hit_count(vector_id: str, current_meta: dict) -> None:
    try:
        _index.update(
            id=vector_id,
            namespace=CACHE_NAMESPACE,
            set_metadata={**current_meta, "hit_count": current_meta.get("hit_count", 0) + 1,
                          "last_hit": int(time.time())},
        )
    except Exception:
        pass


def stats() -> dict:
    if not _AVAILABLE or not _index:
        return {"available": False}
    try:
        ns_stats = _index.describe_index_stats().get("namespaces", {}).get(CACHE_NAMESPACE, {})
        return {
            "available":    True,
            "namespace":    CACHE_NAMESPACE,
            "vector_count": ns_stats.get("vector_count", 0),
            "threshold":    SIMILARITY_THRESHOLD,
            "ttl_days":     TTL_DAYS,
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


def clear() -> None:
    """Delete all entries from semantic cache namespace by explicit IDs."""
    if not _AVAILABLE or not _index:
        return
    try:
        dummy = [0.0] * 1024
        results = _index.query(
            vector=dummy,
            top_k=100,
            include_metadata=False,
            namespace=CACHE_NAMESPACE,
        )
        ids = [m.id for m in results.matches]
        if ids:
            _index.delete(ids=ids, namespace=CACHE_NAMESPACE)
            log.info("Semantic cache cleared — deleted %d vectors by ID", len(ids))
        else:
            log.info("Semantic cache already empty")
    except Exception as e:
        log.warning("Semantic cache clear error: %s", e)

