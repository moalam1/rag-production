
"""
pipeline/reranker.py — Cohere reranking with freshness decay.

Fixes applied:
  1. Cohere returns TOP_K_RERANK * 2 candidates (more headroom for decay re-sort)
  2. Freshness decay applied to every candidate's rerank_score
  3. Candidates re-sorted by adjusted score AFTER decay — not before
  4. Final top TOP_K_RERANK selected from decay-adjusted ranking
  5. published_date pulled from both chunk top-level and metadata (defensive)
"""
import logging
from datetime import datetime

import cohere

from config import settings
from langsmith import traceable
from cache.factory import cache
from cache.memory import MemoryCache

log = logging.getLogger(__name__)
_co = cohere.Client(settings.COHERE_API_KEY)

# Fetch 2× candidates from Cohere so decay re-sorting has room to work
_COHERE_CANDIDATES = min(settings.TOP_K_RERANK * 2, 20)


def _decay_multiplier(published_date: str) -> float:
    """
    Return freshness multiplier based on document age.
    Unknown/unparseable dates treated as 1 year old (neutral penalty).
    """
    try:
        days_old = (datetime.now() - datetime.fromisoformat(published_date)).days
    except (ValueError, TypeError, AttributeError):
        days_old = 365

    if days_old < 30:
        return 1.05
    elif days_old < 180:
        return 1.00
    elif days_old < 365:
        return 0.90
    elif days_old < 730:
        return 0.80
    else:
        return 0.70


@traceable(name="rerank-chunks", run_type="chain")
def rerank_chunks(query: str, chunks: list[dict]) -> list[dict]:
    """
    Rerank chunks by relevance using Cohere, then apply freshness decay
    and re-sort by the adjusted score before returning top N.

    Flow:
      1. Cohere reranks → returns _COHERE_CANDIDATES (2× top_k)
      2. Freshness decay multiplier applied to each candidate's score
      3. Re-sort by adjusted score descending
      4. Return top TOP_K_RERANK
    """
    if not chunks:
        return []

    c   = cache()
    key = MemoryCache.make_key(
        "rerank", {"q": query, "ids": [ch["id"] for ch in chunks]}
    )
    cached = c.get(key)
    if cached is not None:
        log.debug("rerank cache HIT")
        return cached

    # ── 1. Cohere rerank — fetch more candidates than final top_k ────────────
    docs     = [ch["text"] for ch in chunks]
    response = _co.rerank(
        model=settings.RERANK_MODEL,
        query=query,
        documents=docs,
        top_n=_COHERE_CANDIDATES,   # FIX 2: more candidates before decay
    )

    candidates = []
    for result in response.results:
        chunk = chunks[result.index].copy()
        chunk["rerank_score"] = result.relevance_score
        candidates.append(chunk)

    # ── 2. Apply freshness decay to every candidate ───────────────────────────
    for chunk in candidates:
        pub = (
            chunk.get("published_date")
            or chunk.get("metadata", {}).get("published_date", "")
        )
        multiplier              = _decay_multiplier(pub)
        chunk["rerank_score"]   = round(chunk["rerank_score"] * multiplier, 4)
        chunk["freshness_multiplier"] = multiplier
        chunk["days_old"]       = _days_old(pub)

    # ── 3. Re-sort by adjusted score AFTER decay ──────────────────────────────
    # FIX 1: this was missing — Cohere order was preserved even after decay
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    # ── 4. Take final top N ───────────────────────────────────────────────────
    top = candidates[:settings.TOP_K_RERANK]

    log.info(
        "Reranked %d → %d candidates → top %d after decay",
        len(chunks), len(candidates), len(top),
    )
    if top:
        log.debug(
            "Top chunk: score=%.4f multiplier=%.2f days_old=%s file=%s",
            top[0]["rerank_score"],
            top[0].get("freshness_multiplier", 1.0),
            top[0].get("days_old", "?"),
            top[0].get("filename", "?"),
        )

    c.set(key, top)
    return top


def _days_old(published_date: str) -> int | str:
    """Return age in days or '?' if unparseable."""
    try:
        return (datetime.now() - datetime.fromisoformat(published_date)).days
    except (ValueError, TypeError, AttributeError):
        return "?"


def build_context(chunks: list[dict]) -> str:
    """Format top chunks into a numbered context string for the LLM."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Chunk {i} | Source: {chunk['clean_name']} | "
            f"Page: {chunk['page']} | Score: {chunk['rerank_score']}]\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)
    
