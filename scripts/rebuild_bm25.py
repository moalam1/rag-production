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
