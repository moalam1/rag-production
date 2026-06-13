"""
scripts/rebuild_bm25.py — Rebuild BM25 index from Pinecone and persist to Redis.

Run this after:
  - Full seed run completes
  - Nightly crawl adds significant new content
  - Redis cache expires (7 day TTL)

Usage:
  /usr/bin/python3.11 scripts/rebuild_bm25.py
"""
import sys, time, logging
sys.path.insert(0, "/home/ssm-user/rag-production")
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("rebuild_bm25")

from pipeline.retriever import (
    _index, ALL_NAMESPACES, _LATEST_FILTER,
    _parse_match, _build_bm25_index, BM25_REDIS_KEY, _get_redis_binary
)

def main():
    log.info("BM25 rebuild started")
    t0         = time.time()
    all_chunks = []
    dummy      = [0.0] * 1024

    for ns in ALL_NAMESPACES:
        ns_count = 0
        for batch_num in range(50):  # max 50 × 200 = 10,000 per namespace
            results = _index.query(
                vector=dummy,
                top_k=200,
                include_metadata=True,
                namespace=ns,
                filter=_LATEST_FILTER,
            )
            for match in results.matches:
                chunk = _parse_match(match, ns)
                if chunk:
                    all_chunks.append(chunk)
                    ns_count += 1
            if len(results.matches) < 200:
                break
        log.info("  %s: %d chunks", ns, ns_count)

    log.info("Total chunks fetched: %d", len(all_chunks))

    # Build and persist
    _build_bm25_index(all_chunks)

    # Verify Redis
    r    = _get_redis_binary()
    data = r.get(BM25_REDIS_KEY)
    if data:
        log.info("✅ Verified in Redis — %s KB compressed", round(len(data)/1024, 1))
        log.info("TTL: %d seconds", r.ttl(BM25_REDIS_KEY))
    else:
        log.error("❌ Redis verification failed")

    log.info("Done in %.1f seconds", time.time() - t0)

if __name__ == "__main__":
    main()
