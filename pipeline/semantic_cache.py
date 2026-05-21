"""
pipeline/semantic_cache.py — Pinecone semantic cache (Layer 1).

Sits between Redis exact cache and the full RAG pipeline.
Stores past query embeddings + answers in a dedicated Pinecone namespace.
Returns cached answer if a similar past query is found (cosine >= 0.92).

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
SIMILARITY_THRESHOLD = 0.92
MAX_METADATA_CHARS   = 3800

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
        try:
            stored = json.loads(meta.get("answer_json", "{}"))
        except json.JSONDecodeError:
            return None
        _increment_hit_count(top.id, meta)
        return {
            "answer":       stored.get("answer", ""),
            "followups":    stored.get("followups", []),
            "sources":      stored.get("sources", []),
            "cache_hit":    True,
            "semantic_hit": True,
            "similarity":   round(similarity, 4),
        }
    except Exception as e:
        log.warning("Semantic cache get error (skipping): %s", e)
        return None


def set(query: str, query_embedding: list, result: dict, lang: str = "en") -> None:
    if not _AVAILABLE or not _index:
        return
    try:
        answer_json = json.dumps({
            "answer":    result.get("answer", ""),
            "followups": result.get("followups", []),
            "sources":   result.get("sources", []),
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
        log.debug("Semantic cache SET: lang=%s query='%s...'", lang, query[:40])
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
