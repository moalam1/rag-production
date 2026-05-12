"""
pipeline/reranker.py — Cohere reranking with cache.
"""
import logging
import cohere

from config import settings
from cache.factory import cache
from cache.memory import MemoryCache

log = logging.getLogger(__name__)

_co = cohere.Client(settings.COHERE_API_KEY)


def rerank_chunks(query: str, chunks: list[dict]) -> list[dict]:
    """
    Rerank chunks by relevance to query using Cohere.
    Cache key: hash of (query, chunk ids) so same query+chunks = cached.
    """
    if not chunks:
        return []

    c   = cache()
    key = MemoryCache.make_key("rerank", {"q": query, "ids": [ch["id"] for ch in chunks]})

    cached = c.get(key)
    if cached is not None:
        log.debug("rerank cache HIT")
        return cached

    docs = [ch["text"] for ch in chunks]
    response = _co.rerank(
        model=settings.RERANK_MODEL,
        query=query,
        documents=docs,
        top_n=settings.TOP_K_RERANK,
    )

    reranked = []
    for result in response.results:
        chunk = chunks[result.index].copy()
        chunk["rerank_score"] = result.relevance_score
        reranked.append(chunk)

    # ── Freshness decay ───────────────────────────────────────────
    from datetime import datetime
    now = datetime.now()
    for chunk in reranked:
        pub = chunk.get("published_date", "")
        try:
            days_old = (now - datetime.fromisoformat(pub)).days
        except (ValueError, TypeError):
            days_old = 365  # unknown age — treat as 1 year old

        if days_old < 30:
            multiplier = 1.05
        elif days_old < 180:
            multiplier = 1.00
        elif days_old < 365:
            multiplier = 0.90
        elif days_old < 730:
            multiplier = 0.80
        else:
            multiplier = 0.70

        chunk["rerank_score"] = round(chunk["rerank_score"] * multiplier, 4)
        chunk["freshness_multiplier"] = multiplier

    c.set(key, reranked)
    return reranked


def build_context(chunks: list[dict]) -> str:
    """Format chunks into a numbered context string for the LLM."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Chunk {i} | Source: {chunk['filename']} | Page: {chunk['page']}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)
