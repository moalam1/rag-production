"""
pipeline/retriever.py — Pinecone retrieval + BM25 hybrid search (Option B).

Hybrid retrieval using Reciprocal Rank Fusion (RRF):
  - Pinecone ANN   : semantic similarity (dense vectors)
  - BM25 in-memory : lexical / keyword matching (rank_bm25)
  - RRF merge      : combines both ranked lists by position, not raw score
  - Cohere rerank  : already downstream, works on merged top-20

Why hybrid matters:
  Semantic alone misses: product codes (IBX SV5), measurements (10Gbps),
  partner names (Megaport), acronyms (AMER, xScale). BM25 finds these
  by exact string match. Hybrid handles both.

BM25 index:
  - Built in-memory at module load from a Pinecone warm-up fetch
  - Rebuilds on EC2 restart — no persistence needed
  - Zero external dependency — rank_bm25 runs on CPU in microseconds
  - Zero vendor lock-in — same RRF logic works with any backend

Fixes retained:
  1. is_latest filter boolean True (not string "true")
  2. No fallback unfiltered query — deprecated chunks never surface
  3. Filter applied at Pinecone query time, not post-filter
"""
import json
import logging
import threading
import time
from typing import Optional

from pinecone import Pinecone
from rank_bm25 import BM25Okapi
import pickle
import zlib

from config import settings
from langsmith import traceable
from cache.factory import cache
from cache.memory import MemoryCache

log = logging.getLogger(__name__)

_pc    = Pinecone(api_key=settings.PINECONE_API_KEY)
_index = _pc.Index(settings.PINECONE_INDEX)

RETRIEVAL_CACHE_TTL = min(settings.CACHE_TTL_SECONDS, 3600)  # 1hr max — retrieval cache
ALL_NAMESPACES      = ["technical", "business", "media"]
RRF_K               = 60    # standard RRF constant
ALPHA               = 0.7   # semantic weight; 1-ALPHA = lexical weight

_LATEST_FILTER = {"is_latest": {"$eq": True}}

# BM25 in-memory state
_bm25_lock   = threading.Lock()
_bm25_index  = None
_bm25_chunks = []
_bm25_built  = False

# Redis key for persisted BM25 index
BM25_REDIS_KEY = "rag:bm25:index"
BM25_REDIS_TTL = 86400 * 7   # 7 days TTL


def _tokenise(text):
    return text.lower().split()


def _get_redis_binary():
    """Redis connection in binary mode for pickle storage."""
    import redis as redis_lib
    from config import settings
    return redis_lib.from_url(settings.REDIS_URL, decode_responses=False)


def _serialize_bm25(index, chunks) -> bytes:
    """Serialize BM25 index + chunks to compressed bytes."""
    import pickle, zlib
    return zlib.compress(pickle.dumps({"index": index, "chunks": chunks}, protocol=4), level=6)


def _deserialize_bm25(data: bytes):
    """Deserialize BM25 index + chunks from bytes."""
    import pickle, zlib
    payload = pickle.loads(zlib.decompress(data))
    return payload["index"], payload["chunks"]


def _build_bm25_index(chunks):
    """Build BM25 index from chunks, persist to Redis, update in-memory state."""
    global _bm25_index, _bm25_chunks, _bm25_built
    if not chunks:
        log.warning("BM25: no chunks provided — index not built")
        return
    light_chunks = [{"id": c["id"], "text": c.get("text", "")} for c in chunks]
    tokenised    = [_tokenise(c["text"]) for c in light_chunks]
    new_index    = BM25Okapi(tokenised)

    # Persist to Redis
    try:
        r       = _get_redis_binary()
        payload = _serialize_bm25(new_index, light_chunks)
        r.setex(BM25_REDIS_KEY, BM25_REDIS_TTL, payload)
        log.info("BM25 persisted to Redis — %d chunks, %.1f KB compressed",
                 len(light_chunks), len(payload)/1024)
    except Exception as e:
        log.warning("BM25 Redis persist failed (still in memory): %s", e)

    with _bm25_lock:
        _bm25_chunks = light_chunks
        _bm25_index  = new_index
        _bm25_built  = True
    log.info("BM25 index ready — %d chunks", len(light_chunks))


def _load_bm25_from_redis() -> bool:
    """Load BM25 from Redis. Returns True if successful. Completes in <1s."""
    global _bm25_index, _bm25_chunks, _bm25_built
    try:
        r    = _get_redis_binary()
        data = r.get(BM25_REDIS_KEY)
        if not data:
            log.info("BM25: no cached index in Redis — will rebuild from Pinecone")
            return False
        index, chunks = _deserialize_bm25(data)
        with _bm25_lock:
            _bm25_index  = index
            _bm25_chunks = chunks
            _bm25_built  = True
        log.info("BM25 loaded from Redis — %d chunks, %.1f KB (instant startup)",
                 len(chunks), len(data)/1024)
        return True
    except Exception as e:
        log.warning("BM25 Redis load failed — will rebuild from Pinecone: %s", e)
        return False


def _warm_bm25():
    """
    BM25 warmup:
      1. Try Redis cache first  (<1s — fast path)
      2. Fall back to Pinecone rebuild if cache miss (~30s)
      3. Persist rebuilt index to Redis for next startup
    """
    try:
        if _load_bm25_from_redis():
            return   # fast path — done in <1s

        # Slow path — rebuild from Pinecone
        log.info("BM25: rebuilding from Pinecone...")
        t0         = time.time()
        all_chunks = []
        dummy      = [0.0] * 1024
        for ns in ALL_NAMESPACES:
            try:
                for _ in range(50):
                    results = _index.query(
                        vector=dummy, top_k=200,
                        include_metadata=True,
                        namespace=ns,
                        filter=_LATEST_FILTER,
                    )
                    for match in results.matches:
                        chunk = _parse_match(match, ns)
                        if chunk:
                            all_chunks.append(chunk)
                    if len(results.matches) < 200:
                        break
            except Exception as e:
                log.warning("BM25 Pinecone fetch failed for %s: %s", ns, e)
        _build_bm25_index(all_chunks)
        log.info("BM25 rebuild complete — %.1fs — %d chunks",
                 time.time()-t0, len(all_chunks))
    except Exception as e:
        log.error("BM25 warm-up error: %s", e)

def _bm25_search(query, top_k):
    with _bm25_lock:
        if not _bm25_built or not _bm25_index:
            return []
        tokens = _tokenise(query)
        scores = _bm25_index.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [_bm25_chunks[i] for i, s in ranked[:top_k] if s > 0]


def _rrf_merge(semantic, lexical, top_k, k=RRF_K):
    """Reciprocal Rank Fusion — merges two ranked lists by position."""
    scores   = {}
    all_docs = {}

    for rank, doc in enumerate(semantic):
        doc_id = doc["id"]
        scores[doc_id]   = scores.get(doc_id, 0.0) + ALPHA / (k + rank + 1)
        all_docs[doc_id] = doc

    for rank, doc in enumerate(lexical):
        doc_id = doc["id"]
        scores[doc_id]   = scores.get(doc_id, 0.0) + (1 - ALPHA) / (k + rank + 1)
        all_docs[doc_id] = doc

    merged_ids = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]

    sem_ids  = {d["id"] for d in semantic}
    lex_ids  = {d["id"] for d in lexical}
    both     = sum(1 for i in merged_ids if i in sem_ids and i in lex_ids)
    sem_only = sum(1 for i in merged_ids if i in sem_ids and i not in lex_ids)
    lex_only = sum(1 for i in merged_ids if i in lex_ids and i not in sem_ids)
    log.debug("RRF: %d total | %d both | %d sem-only | %d lex-only",
              len(merged_ids), both, sem_only, lex_only)

    return [all_docs[i] for i in merged_ids if i in all_docs]


def _parse_match(match, namespace):
    """Parse a Pinecone match into a chunk dict. Returns None if no text."""
    metadata   = match.metadata
    text       = ""
    inner_meta = {}

    node_content = metadata.get("_node_content", "")
    if node_content:
        try:
            node_data  = json.loads(node_content)
            text       = node_data.get("text", "")
            inner_meta = node_data.get("metadata", {})
        except (json.JSONDecodeError, AttributeError):
            pass

    if not text.strip():
        text = metadata.get("text", "")
    if not text.strip():
        return None

    filename = metadata.get("filename") or inner_meta.get("filename", "unknown")
    return {
        "id":              match.id,
        "score":           match.score,
        "text":            text,
        "filename":        filename,
        "clean_name":      (metadata.get("clean_name") or inner_meta.get("clean_name")
                            or filename.replace("_"," ").replace("-"," ")
                                       .replace(".pdf","").title()),
        "resource_type":   metadata.get("resource_type")  or inner_meta.get("resource_type",""),
        "page":            (metadata.get("page_label")    or metadata.get("page")
                            or inner_meta.get("page")     or inner_meta.get("page_label") or "?"),
        "page_url":        metadata.get("page_url")       or inner_meta.get("page_url",""),
        "pdf_url":         metadata.get("pdf_url")        or inner_meta.get("pdf_url",""),
        "namespace":       namespace,
        "document_family": metadata.get("document_family") or inner_meta.get("document_family",""),
        "published_date":  metadata.get("published_date")  or inner_meta.get("published_date",""),
    }


def _query_namespace(vector, namespace, top_k, metadata_filter=None):
    """Query a single Pinecone namespace with is_latest filter + optional intent filter."""
    # Merge is_latest with any intent-based metadata filter
    pinecone_filter = {**_LATEST_FILTER}
    if metadata_filter:
        pinecone_filter = {"$and": [_LATEST_FILTER, metadata_filter]}
    try:
        results = _index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace,
            filter=pinecone_filter,
        )
    except Exception as e:
        log.warning("Pinecone query failed for namespace %s: %s", namespace, e)
        return []

    chunks = []
    for match in results.matches:
        chunk = _parse_match(match, namespace)
        if chunk:
            chunks.append(chunk)
    return chunks


# Start BM25 warm-up in background at module load
threading.Thread(target=_warm_bm25, daemon=True, name="bm25-warmup").start()


@traceable(name="retrieve-chunks", run_type="retriever")
def retrieve_chunks(
    query_embedding,
    query_text="",
    namespace=None,
    metadata_filter=None,   # intent-based Pinecone pre-filter
    top_k=None,             # intent-based top_k override
):
    """
    Hybrid retrieval: Pinecone ANN (semantic) + BM25 (lexical) merged via RRF.

    Args:
        query_embedding : 1024-dim dense vector from embedder.py
        query_text      : raw query string for BM25 (pass retrieval_query)
        namespace       : optional specific namespace; None = all namespaces

    Returns top-K chunks ranked by RRF score, all is_latest=True.
    Falls back to semantic-only if BM25 index not yet warm.
    """
    c   = cache()
    key = MemoryCache.make_key(
        "retrieve_v2", {"dims": query_embedding[:8], "ns": namespace or "all"}
    )
    cached = c.get(key)
    if cached is not None:
        log.debug("retrieval cache HIT")
        return cached

    # Use intent top_k if provided, else settings default
    effective_top_k = top_k if top_k else settings.TOP_K_RETRIEVE

    namespaces_to_query = (
        [namespace.strip()]
        if namespace and namespace.strip().lower() not in ("", "all")
        else ALL_NAMESPACES
    )
    log.debug("Querying namespaces: %s", namespaces_to_query)

    # 1. Semantic — Pinecone ANN
    semantic_chunks = []
    for ns in namespaces_to_query:
        ns_chunks = _query_namespace(query_embedding, ns, effective_top_k, metadata_filter)
        log.debug("Namespace %s: %d chunks (is_latest=True)", ns, len(ns_chunks))
        semantic_chunks.extend(ns_chunks)
    semantic_chunks.sort(key=lambda x: x["score"], reverse=True)
    semantic_chunks = semantic_chunks[:effective_top_k]

    # 2. Lexical — BM25 in-memory
    lexical_chunks = []
    if query_text:
        if _bm25_built:
            lexical_chunks = _bm25_search(query_text, effective_top_k)
            log.debug("BM25 lexical: %d results", len(lexical_chunks))
        else:
            log.debug("BM25 not ready — semantic only this query")

    # 3. RRF merge
    if lexical_chunks:
        top_chunks = _rrf_merge(semantic_chunks, lexical_chunks, effective_top_k)
        log.info("Hybrid retrieval: %d chunks (sem=%d lex=%d) via RRF alpha=%.1f",
                 len(top_chunks), len(semantic_chunks), len(lexical_chunks), ALPHA)
    else:
        top_chunks = semantic_chunks[:effective_top_k]
        log.info("Semantic-only retrieval: %d chunks from %s",
                 len(top_chunks), namespaces_to_query)

    c.set(key, top_chunks, ttl=RETRIEVAL_CACHE_TTL)
    return top_chunks
