"""
scripts/ingest_worker.py — Poll SQS rag-ingest queue and process URLs.

Each message = one URL. On success → message deleted from queue.
On failure → message returns to queue. After 3 failures → DLQ.
Runs alongside FastAPI, does NOT require a restart of the API server.

Usage:
  nohup /usr/bin/python3.11 scripts/ingest_worker.py \
    >> /home/ssm-user/ingest_worker.log 2>&1 &
"""
import boto3, json, logging, sys, time, signal
from datetime import datetime
sys.path.insert(0, "/home/ssm-user/rag-production")
from pipeline.page_parser   import parse_page
from pipeline.ingest_router import route_and_ingest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
log = logging.getLogger("ingest_worker")

QUEUE_URL          = "https://sqs.us-east-1.amazonaws.com/141927126501/rag-ingest"
DLQ_URL            = "https://sqs.us-east-1.amazonaws.com/141927126501/rag-ingest-dlq"
REGION             = "us-east-1"
VISIBILITY_TIMEOUT = 300   # seconds — must be > max processing time per URL
WAIT_TIME          = 20    # long-poll — reduces empty receives
MAX_MESSAGES       = 1     # one at a time — avoids memory spikes on LlamaParse
DELAY_BETWEEN      = 2.0   # seconds between jobs

_running = True

def handle_signal(sig, frame):
    global _running
    log.info("Shutdown signal — finishing current job then stopping")
    _running = False

signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT,  handle_signal)


def process_url(url: str, force: bool = False) -> bool:
    """Parse and ingest one URL. Returns True on success."""
    try:
        log.info("Parsing: %s", url)
        parsed = parse_page(url)
        if not parsed:
            log.warning("parse_page returned None — skipping: %s", url)
            return True  # treat as skip not failure — don't retry

        log.info("Ingesting: type=%s family=%s",
                 getattr(parsed, "resource_type", "?"),
                 getattr(parsed, "document_family", "?"))

        logs = route_and_ingest(parsed, force=force)
        for line in logs:
            log.info("  %s", line)

        log.info("✅ Complete: %s", url)
        return True

    except Exception as e:
        log.error("❌ Failed: %s — %s", url, e, exc_info=True)
        return False


def get_queue_depth(sqs) -> dict:
    """Return current queue depths for monitoring."""
    try:
        r1 = sqs.get_queue_attributes(
            QueueUrl=QUEUE_URL,
            AttributeNames=["ApproximateNumberOfMessages",
                            "ApproximateNumberOfMessagesNotVisible"]
        )
        r2 = sqs.get_queue_attributes(
            QueueUrl=DLQ_URL,
            AttributeNames=["ApproximateNumberOfMessages"]
        )
        return {
            "pending":     int(r1["Attributes"]["ApproximateNumberOfMessages"]),
            "in_flight":   int(r1["Attributes"]["ApproximateNumberOfMessagesNotVisible"]),
            "dlq":         int(r2["Attributes"]["ApproximateNumberOfMessages"]),
        }
    except Exception:
        return {}


def main():
    sqs = boto3.client("sqs", region_name=REGION)
    log.info("Worker started — queue: %s", QUEUE_URL)
    log.info("Settings: visibility=%ds poll=%ds delay=%ds",
             VISIBILITY_TIMEOUT, WAIT_TIME, DELAY_BETWEEN)

    processed = failed = skipped = 0
    last_depth_log = 0

    while _running:
        try:
            # Log queue depth every 10 minutes
            if time.time() - last_depth_log > 600:
                depth = get_queue_depth(sqs)
                log.info("Queue depth — pending=%d in_flight=%d dlq=%d | "
                         "Session: processed=%d failed=%d skipped=%d",
                         depth.get("pending",0), depth.get("in_flight",0),
                         depth.get("dlq",0), processed, failed, skipped)
                if depth.get("dlq", 0) > 0:
                    log.warning("⚠️  %d messages in DLQ — check logs for errors",
                                depth["dlq"])
                last_depth_log = time.time()

            # Poll for message
            resp = sqs.receive_message(
                QueueUrl            = QUEUE_URL,
                MaxNumberOfMessages = MAX_MESSAGES,
                WaitTimeSeconds     = WAIT_TIME,
                VisibilityTimeout   = VISIBILITY_TIMEOUT,
            )

            messages = resp.get("Messages", [])
            if not messages:
                continue  # empty poll — long-poll already waited 20s

            msg  = messages[0]
            body  = json.loads(msg["Body"])
            url   = body.get("url", "")
            force = body.get("force", False)

            if not url:
                log.warning("Message has no URL — deleting: %s", msg["Body"][:100])
                sqs.delete_message(QueueUrl=QUEUE_URL,
                                   ReceiptHandle=msg["ReceiptHandle"])
                skipped += 1
                continue

            # Process
            success = process_url(url, force=force)

            if success:
                # Delete from queue — job done
                sqs.delete_message(
                    QueueUrl      = QUEUE_URL,
                    ReceiptHandle = msg["ReceiptHandle"]
                )
                processed += 1
            else:
                # Leave in queue — visibility timeout will return it for retry
                # After 3 receives it moves to DLQ automatically
                failed += 1
                log.warning("Message left in queue for retry: %s", url)

            time.sleep(DELAY_BETWEEN)

        except KeyboardInterrupt:
            break
        except Exception as e:
            log.error("Worker loop error: %s", e, exc_info=True)
            time.sleep(5)  # back off on unexpected errors

    log.info("Worker stopped — processed=%d failed=%d skipped=%d",
             processed, failed, skipped)


if __name__ == "__main__":
    main()
