
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

from config import settings
from langsmith import traceable
from cache.factory import cache
from cache.memory import MemoryCache

log = logging.getLogger(__name__)

# Rerank backend: "cohere" (direct API, needs COHERE_API_KEY) or "bedrock"
# (hosted Cohere Rerank 3.5, auth via IAM — no key). Toggle via RERANK_BACKEND.
_RERANK_BACKEND = getattr(settings, "RERANK_BACKEND", "cohere").lower()

if _RERANK_BACKEND == "cohere":
    import cohere
    _co = cohere.Client(settings.COHERE_API_KEY)
else:
    import boto3, json as _json
    _bedrock = boto3.client("bedrock-runtime",
                            region_name=getattr(settings, "AWS_REGION", "us-east-1"))


def _rerank_call(query: str, docs: list[str], top_n: int) -> list[tuple[int, float]]:
    """Backend-agnostic rerank. Returns [(index, relevance_score), ...].
    Both backends use Cohere Rerank; bedrock hosts it (IAM auth, no key)."""
    if _RERANK_BACKEND == "cohere":
        resp = _co.rerank(
            model=settings.RERANK_MODEL, query=query, documents=docs, top_n=top_n,
        )
        return [(r.index, r.relevance_score) for r in resp.results]
    # bedrock invoke_model — Cohere Rerank 3.5
    body = _json.dumps({
        "query": query, "documents": docs, "top_n": top_n, "api_version": 2,
    })
    resp = _bedrock.invoke_model(
        modelId=getattr(settings, "BEDROCK_RERANK_MODEL_ID", "cohere.rerank-v3-5:0"),
        body=body,
    )
    results = _json.loads(resp["body"].read())["results"]
    return [(r["index"], r["relevance_score"]) for r in results]

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
    docs = [ch["text"] for ch in chunks]
    try:
        ranked = _rerank_call(query, docs, _COHERE_CANDIDATES)
        candidates = []
        for idx, score in ranked:
            chunk = chunks[idx].copy()
            chunk["rerank_score"] = score
            candidates.append(chunk)
    except Exception as e:
        # Graceful degradation (item 21): a rerank failure (Cohere 429 quota,
        # 5xx, timeout, network) must NOT crash /search. Fall back to the
        # retriever's existing order (chunks arrive pre-sorted by RRF), using
        # a descending proxy rerank_score that stays above the caller's 0.10
        # secondary-fallback threshold so we don't thrash an extra rerank call.
        # The downstream decay/sort/top-N path runs unchanged. Search stays up,
        # ranking is slightly less optimal. rerank_fallback flag = observable.
        log.warning("Rerank failed (%s) — falling back to un-reranked retriever order", e)
        candidates = []
        n = min(len(chunks), _COHERE_CANDIDATES)
        for i, ch in enumerate(chunks[:n]):
            chunk = ch.copy()
            # preserve retriever order: descending 1.0 .. (>0.10), above threshold
            chunk["rerank_score"]    = round(1.0 - (i / max(n, 1)) * 0.85, 4)
            chunk["rerank_fallback"] = True
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
            f"[Chunk {i} | Source: {chunk.get('clean_name', chunk.get('filename', 'Unknown'))} | "
            f"Page: {chunk.get('page', '?')} | Score: {chunk.get('rerank_score', 0.0)}]\n"
            f"{chunk.get('text', '')}"
        )
    return "\n\n---\n\n".join(parts)
    
