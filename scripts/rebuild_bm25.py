"""
scripts/rebuild_bm25.py — Rebuild BM25 index from Pinecone and persist to S3.

Run this after:
  - Full seed run completes
  - Nightly crawl adds significant new content
  - BM25 corpus changes (re-run to refresh the S3 snapshot)

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
    _parse_match, _build_bm25_index, BM25_S3_BUCKET, BM25_S3_KEY, _s3_client
)

def main():
    log.info("BM25 rebuild started")
    t0         = time.time()
    all_chunks = []
    dummy      = [0.0] * 1024

    # FIX: query() returns the same top-K every call (similarity, not a scan) —
    # it only ever fetched ~200/namespace, so BM25 indexed ~2% of the corpus.
    # Use list() to paginate ALL ids, fetch() to get them in batches, and
    # filter is_latest=True in-code (fetch has no metadata filter).
    class _M:  # shim: make a fetched vector look like a query match for _parse_match
        __slots__ = ("id", "metadata", "score")
        def __init__(self, vid, meta):
            self.id, self.metadata, self.score = vid, meta, 0.0

    for ns in ALL_NAMESPACES:
        ns_count = 0
        for id_page in _index.list(namespace=ns):           # walks ALL ids, paginated
            if not id_page:
                continue
            fetched = _index.fetch(ids=id_page, namespace=ns)
            for vid, vec in fetched.vectors.items():
                meta = vec.metadata or {}
                # is_latest filter in-code (boolean True, matching _LATEST_FILTER)
                if meta.get("is_latest") not in (True, "true", "True"):
                    continue
                chunk = _parse_match(_M(vid, meta), ns)
                if chunk:
                    all_chunks.append(chunk)
                    ns_count += 1
        log.info("  %s: %d chunks", ns, ns_count)

    log.info("Total chunks fetched: %d", len(all_chunks))

    # Build and persist
    _build_bm25_index(all_chunks)

    # Verify S3
    try:
        head = _s3_client().head_object(Bucket=BM25_S3_BUCKET, Key=BM25_S3_KEY)
        log.info("✅ Verified in S3 s3://%s/%s — %s KB",
                 BM25_S3_BUCKET, BM25_S3_KEY, round(head["ContentLength"]/1024, 1))
    except Exception as e:
        log.error("❌ S3 verification failed: %s", e)

    log.info("Done in %.1f seconds", time.time() - t0)

if __name__ == "__main__":
    main()
