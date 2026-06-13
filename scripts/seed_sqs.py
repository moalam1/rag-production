"""
scripts/seed_sqs.py — Push all Equinix resource URLs into SQS for async processing.

Usage:
  python3.11 scripts/seed_sqs.py              # all 1,280 URLs
  python3.11 scripts/seed_sqs.py --type whitepapers
  python3.11 scripts/seed_sqs.py --limit 50   # test batch
  python3.11 scripts/seed_sqs.py --dry-run    # count only
"""
import argparse, boto3, json, sys, time
from datetime import datetime
sys.path.insert(0, "/home/ssm-user/rag-production")
from pipeline.crawler import crawl_all, crawl_type

QUEUE_URL  = "https://sqs.us-east-1.amazonaws.com/141927126501/rag-ingest"
REGION     = "us-east-1"
BATCH_SIZE = 10

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type",    help="Resource type e.g. whitepapers")
    parser.add_argument("--limit",   type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delay",   type=float, default=0.1)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Equinix RAG SQS Seed — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
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
        print(f"   Limited to: {len(urls)}")

    if args.dry_run:
        print("\n🔍 Dry run — would enqueue:")
        for u in urls[:10]:
            print(f"  {u}")
        if len(urls) > 10:
            print(f"  ... and {len(urls)-10} more")
        return

    sqs  = boto3.client("sqs", region_name=REGION)
    print(f"\n📬 Queue: {QUEUE_URL}")
    print(f"🚀 Enqueuing {len(urls)} URLs...\n")

    sent = failed = 0
    for i in range(0, len(urls), BATCH_SIZE):
        batch = urls[i:i+BATCH_SIZE]
        entries = [
            {
                "Id":          str(j),
                "MessageBody": json.dumps({
                    "url":         url,
                    "enqueued_at": datetime.utcnow().isoformat(),
                }),
            }
            for j, url in enumerate(batch)
        ]
        try:
            resp    = sqs.send_message_batch(QueueUrl=QUEUE_URL, Entries=entries)
            sent   += len(resp.get("Successful", []))
            failed += len(resp.get("Failed", []))
            for f in resp.get("Failed", []):
                print(f"  ❌ {f}")
        except Exception as e:
            print(f"  ❌ Batch error: {e}")
            failed += len(batch)

        print(f"  [{i+len(batch):4d}/{len(urls)}] Sent={sent} Failed={failed}")
        time.sleep(args.delay)

    print(f"\n✅ Done — Sent: {sent} | Failed: {failed}")

if __name__ == "__main__":
    main()
