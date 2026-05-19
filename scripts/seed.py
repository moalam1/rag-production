"""
scripts/seed.py — Bulk seed all Equinix resource pages into the RAG pipeline.
"""
import argparse
import logging
import sys
import time
from datetime import datetime

sys.path.insert(0, "/home/ssm-user/rag-production")

from pipeline.crawler import crawl_all, crawl_type
from pipeline.page_parser import parse_page
from pipeline.ingest_router import route_and_ingest

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("seed")


def main():
    parser = argparse.ArgumentParser(description="Bulk seed Equinix resources")
    parser.add_argument("--type",    help="Resource type to seed (e.g. whitepapers)")
    parser.add_argument("--dry-run", action="store_true", help="Discover URLs only")
    parser.add_argument("--limit",   type=int, default=0, help="Max URLs to process")
    parser.add_argument("--delay",   type=float, default=2.0, help="Seconds between requests")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Equinix RAG bulk seed — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")

    print("\n📡 Discovering URLs from sitemap...")
    if args.type:
        urls = crawl_type(args.type)
        print(f"   Type filter: {args.type}")
    else:
        result = crawl_all()
        urls   = result["urls"]
        print(f"   By type: {result['by_type']}")

    print(f"   Found: {len(urls)} URLs")

    if args.limit:
        urls = urls[:args.limit]
        print(f"   Limited to: {len(urls)} URLs")

    if args.dry_run:
        print("\n🔍 Dry run — URLs discovered:")
        for url in urls:
            print(f"  {url}")
        return

    print(f"\n🚀 Starting ingest ({len(urls)} URLs)...\n")

    stats = {"total": len(urls), "success": 0, "skipped": 0, "failed": 0, "errors": []}

    for i, url in enumerate(urls, 1):
        rtype = url.rstrip("/").split("/")[4]
        slug  = url.rstrip("/").split("/")[-1][:40]
        print(f"[{i:4d}/{len(urls)}] {rtype}/{slug}")

        try:
            page = parse_page(url)
            if not page:
                print(f"         ⚠️  Parse failed — skipping")
                stats["failed"] += 1
                continue

            logs = route_and_ingest(page)

            if any("skipping" in l.lower() or "unchanged" in l.lower() for l in logs):
                stats["skipped"] += 1
                print(f"         ⏭️  unchanged")
            elif any("❌" in l for l in logs):
                stats["failed"] += 1
                print(f"         ❌ failed")
            else:
                stats["success"] += 1
                key = [l for l in logs if "🎉" in l or "chunks" in l]
                for l in key[-2:]:
                    print(f"         {l}")

        except Exception as e:
            print(f"         ❌ {e}")
            stats["failed"] += 1
            stats["errors"].append({"url": url, "error": str(e)})

        if i < len(urls):
            time.sleep(args.delay)

    print(f"\n{'='*60}")
    print(f"Total: {stats['total']} | Ingested: {stats['success']} | Skipped: {stats['skipped']} | Failed: {stats['failed']}")
    if stats["errors"]:
        for e in stats["errors"][:5]:
            print(f"  ❌ {e['url']}: {e['error']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
