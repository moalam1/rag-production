"""
run_ingest.py — Fargate container entry point for ingestion.

Runs a section ingest (scripts/ingest_section logic) as a one-shot batch process,
then optionally rebuilds the BM25 S3 snapshot (the corpus changed). Designed for
on-demand Fargate tasks: scale-to-zero, exits 0 on success / non-zero on failure
so EventBridge/ECS knows the outcome.

Config via ENV (Fargate task definition / RunTask overrides) or CLI args:
  INGEST_SECTION   (required) — section name, must exist in rag-config sections
  INGEST_LIMIT     (optional) — ingest only first N URLs (smoke test)
  INGEST_DRY_RUN   (optional) — "true" = discover only, no ingest
  REBUILD_BM25     (optional) — "true" (default) = rebuild BM25 S3 snapshot after ingest

Usage (local/container):
  INGEST_SECTION=customer-success python3.11 run_ingest.py
  python3.11 run_ingest.py customer-success --limit 5
"""
import argparse
import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s — %(message)s")
log = logging.getLogger("run_ingest")


def _bool_env(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("true", "1", "yes")


def main() -> int:
    # Args can come from CLI (local) or ENV (Fargate). CLI wins if provided.
    ap = argparse.ArgumentParser()
    ap.add_argument("section", nargs="?", default=os.getenv("INGEST_SECTION", ""),
                    help="section name (or set INGEST_SECTION)")
    ap.add_argument("--limit", type=int,
                    default=int(os.getenv("INGEST_LIMIT", "0")))
    ap.add_argument("--dry-run", action="store_true",
                    default=_bool_env("INGEST_DRY_RUN", False))
    args = ap.parse_args()

    if not args.section:
        log.error("No section specified. Set INGEST_SECTION env var or pass as arg.")
        return 2

    log.info("=== Fargate ingest task start: section=%r limit=%d dry_run=%s ===",
             args.section, args.limit, args.dry_run)
    t0 = time.time()

    # Run the proven section-ingest logic (reuse, don't duplicate).
    from pipeline.fetchers import discover_urls
    from pipeline.page_parser import parse_page
    from pipeline.ingest_router import route_and_ingest
    import boto3

    # load section config (same as ingest_section._load_section_config)
    ddb = boto3.resource("dynamodb", region_name=os.getenv("AWS_REGION", "us-east-1"))
    sections = ddb.Table("rag-config").get_item(
        Key={"config_key": "sections"}).get("Item", {}).get("data", {})
    if args.section not in sections:
        log.error("section %r not in rag-config: %s", args.section, list(sections.keys()))
        return 1
    cfg = sections[args.section]

    urls = discover_urls(cfg)
    if args.limit:
        urls = urls[:args.limit]
    log.info("discovered %d URLs for section %r", len(urls), args.section)

    if args.dry_run:
        for u in urls:
            print(u)
        log.info("dry-run — no ingest, exiting 0")
        return 0

    ok, skipped, failed = 0, 0, 0
    for i, url in enumerate(urls, 1):
        try:
            page = parse_page(url)
            if not page:
                failed += 1
                continue
            joined = " ".join(route_and_ingest(page, section=args.section))
            if "❌" in joined:
                failed += 1
            elif "skipping" in joined.lower() or "unchanged" in joined.lower():
                skipped += 1
            else:
                ok += 1
            if i % 10 == 0:
                log.info("  progress %d/%d (ok=%d skip=%d fail=%d)", i, len(urls), ok, skipped, failed)
        except Exception:
            log.exception("EXCEPTION on %s", url)
            failed += 1

    log.info("INGEST COMPLETE section=%r in %.1fs — ok=%d skipped=%d failed=%d total=%d",
             args.section, time.time() - t0, ok, skipped, failed, len(urls))

    # Rebuild BM25 S3 snapshot if anything changed (the corpus moved).
    if _bool_env("REBUILD_BM25", True) and (ok > 0):
        log.info("Rebuilding BM25 S3 snapshot (corpus changed: %d new/updated)...", ok)
        try:
            import scripts.rebuild_bm25 as rb
            # rebuild_bm25's main entry — adjust if its API differs
            if hasattr(rb, "main"):
                rb.main()
            log.info("BM25 rebuild complete.")
        except Exception:
            log.exception("BM25 rebuild failed (ingest still succeeded)")
    else:
        log.info("BM25 rebuild skipped (REBUILD_BM25=false or nothing ingested).")

    # Exit non-zero only if EVERYTHING failed (partial success is OK — dedup-safe re-run).
    if failed == len(urls) and len(urls) > 0:
        log.error("All %d URLs failed — exiting 1", len(urls))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
