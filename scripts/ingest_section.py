#!/usr/bin/env python3.11
"""
scripts/ingest_section.py — Bulk-ingest a whole section by name (L1b-source).

Loads the section's config from rag-config DynamoDB, discovers its URLs via the
fetcher abstraction (sitemap + include/exclude filters), then ingests each page
through the proven section-aware write path:

    discover_urls(cfg) -> for each url: route_and_ingest(parse_page(url), section=NAME)

Resilient: one bad page does not abort the batch. Dedup in route_and_ingest
(timestamp + content hash) makes re-runs safe — already-ingested pages skip.

Usage:
    python3.11 -m scripts.ingest_section customer-success
    python3.11 -m scripts.ingest_section customer-success --limit 5     # first N (smoke)
    python3.11 -m scripts.ingest_section customer-success --dry-run     # discover only
"""
import argparse
import logging
import sys
import time

import boto3

from pipeline.fetchers import discover_urls
from pipeline.page_parser import parse_page
from pipeline.ingest_router import route_and_ingest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ingest_section")


def _load_section_config(section: str) -> dict:
    ddb = boto3.resource("dynamodb", region_name="us-east-1")
    table = ddb.Table("rag-config")
    resp = table.get_item(Key={"config_key": "sections"})
    sections = resp.get("Item", {}).get("data", {})
    if section not in sections:
        log.error("section %r not found in rag-config sections: %s",
                  section, list(sections.keys()))
        sys.exit(1)
    return sections[section]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("section", help="section name (must exist in rag-config sections)")
    ap.add_argument("--limit", type=int, default=0, help="ingest only the first N URLs")
    ap.add_argument("--dry-run", action="store_true", help="discover + print URLs, do not ingest")
    args = ap.parse_args()

    cfg = _load_section_config(args.section)
    log.info("section %r config: %s", args.section, cfg)

    urls = discover_urls(cfg)
    if args.limit:
        urls = urls[: args.limit]
    log.info("discovered %d URLs for section %r", len(urls), args.section)

    if args.dry_run:
        for u in urls:
            print(u)
        log.info("dry-run — no ingest")
        return

    ok, skipped, failed = 0, 0, []
    t0 = time.time()
    for i, url in enumerate(urls, 1):
        try:
            page = parse_page(url)
            if not page:
                log.warning("[%d/%d] parse returned None: %s", i, len(urls), url)
                failed.append((url, "parse_none"))
                continue
            logs = route_and_ingest(page, section=args.section)
            joined = " ".join(logs)
            if "❌" in joined:
                log.warning("[%d/%d] ingest error: %s", i, len(urls), url)
                failed.append((url, "ingest_error"))
            elif "skipping" in joined.lower() or "unchanged" in joined.lower():
                skipped += 1
                log.info("[%d/%d] skipped (unchanged): %s", i, len(urls), url)
            else:
                ok += 1
                log.info("[%d/%d] ingested: %s", i, len(urls), url)
        except Exception as e:
            log.exception("[%d/%d] EXCEPTION on %s", i, len(urls), url)
            failed.append((url, str(e)))

    dt = time.time() - t0
    log.info("=" * 60)
    log.info("SECTION %r BULK INGEST COMPLETE in %.1fs", args.section, dt)
    log.info("  ingested: %d | skipped(unchanged): %d | failed: %d | total: %d",
             ok, skipped, len(failed), len(urls))
    if failed:
        log.info("  FAILURES:")
        for url, reason in failed:
            log.info("    %s  (%s)", url, reason)


if __name__ == "__main__":
    main()
