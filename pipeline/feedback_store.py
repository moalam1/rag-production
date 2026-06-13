"""
pipeline/feedback_store.py — DynamoDB feedback storage.

Follows the same pattern as pipeline/registry.py.

Table: rag-feedback
  pk: FEEDBACK#<timestamp>   (partition key — unique per rating event)
  sk: <query[:100]>          (sort key — enables query-level lookups)

GSI (add post-demo if needed):
  rating-index: pk=rating, sk=timestamp  → all thumbs down in time order
  cached-index: pk=cached, sk=rating     → cached failures analysis

Why DynamoDB not Redis:
  Redis resets on EC2 restart — feedback must survive restarts.
  DynamoDB PAY_PER_REQUEST — ~$0 at demo volume.
  Permanent record for pre-production eval dataset.
"""
import logging
from datetime import datetime, timezone
from typing import Optional
import boto3
from botocore.exceptions import ClientError

log = logging.getLogger(__name__)

TABLE_NAME = "rag-feedback"
REGION     = "us-east-1"

# ── Table factory (same pattern as registry.py) ───────────────────────────────
def _get_table():
    dynamodb = boto3.resource("dynamodb", region_name=REGION)
    return dynamodb.Table(TABLE_NAME)


def _get_client():
    return boto3.client("dynamodb", region_name=REGION)


# ── Table creation ────────────────────────────────────────────────────────────
def create_table_if_not_exists() -> bool:
    """
    Create rag-feedback table if it does not exist.
    Called once at startup — idempotent.
    Returns True if created, False if already existed.
    """
    client = _get_client()
    try:
        client.describe_table(TableName=TABLE_NAME)
        log.info("rag-feedback table already exists")
        return False
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise

    log.info("Creating rag-feedback DynamoDB table...")
    client.create_table(
        TableName=TABLE_NAME,
        AttributeDefinitions=[
            {"AttributeName": "pk", "AttributeType": "S"},
            {"AttributeName": "sk", "AttributeType": "S"},
        ],
        KeySchema=[
            {"AttributeName": "pk", "KeyType": "HASH"},
            {"AttributeName": "sk", "KeyType": "RANGE"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )
    # Wait until active
    waiter = client.get_waiter("table_exists")
    waiter.wait(TableName=TABLE_NAME)
    log.info("rag-feedback table created and active")
    return True


# ── Write feedback ────────────────────────────────────────────────────────────
def save_feedback(
    query:    str,
    answer:   str,
    rating:   int,           # 1 = thumbs up, -1 = thumbs down
    cached:   bool  = False,
    sources:  list  = None,
    comment:  str   = "",
    lang:     str   = "en",
) -> bool:
    """
    Persist one feedback record to DynamoDB.
    Non-blocking — logs warning and returns False on failure.
    """
    ts    = datetime.now(timezone.utc).isoformat()
    label = "thumbs_up" if rating == 1 else "thumbs_down"

    # URL-decode query in case it came from form submission
    from urllib.parse import unquote
    query = unquote(query)

    item = {
        "pk":        f"FEEDBACK#{ts}",
        "sk":        query[:100],
        "query":     query,
        "answer":    answer[:500],
        "rating":    rating,
        "label":     label,
        "cached":    cached,
        "lang":      lang,
        "comment":   comment,
        "timestamp": ts,
        # Source filenames for retrieval analysis
        "source_files": [
            s.get("filename", "") for s in (sources or [])
        ][:5],
    }

    try:
        _get_table().put_item(Item=item)
        log.info("Feedback saved: %s | cached=%s | query='%s...'",
                 label, cached, query[:50])
        return True
    except ClientError as e:
        log.warning("Feedback DynamoDB write error: %s", e)
        return False
    except Exception as e:
        log.warning("Feedback unexpected error: %s", e)
        return False


# ── Read feedback ─────────────────────────────────────────────────────────────
def get_stats() -> dict:
    """
    Return high-level feedback statistics.
    Scans the full table — acceptable at demo/testing volumes.
    Add GSI + query when table grows beyond 10K records.
    """
    try:
        table    = _get_table()
        response = table.scan(
            FilterExpression="begins_with(pk, :prefix)",
            ExpressionAttributeValues={":prefix": "FEEDBACK#"},
        )
        items = response.get("Items", [])

        total     = len(items)
        thumbs_up = sum(1 for i in items if i.get("rating") == 1)
        thumbs_dn = sum(1 for i in items if i.get("rating") == -1)
        cached_dn = sum(1 for i in items if i.get("rating") == -1 and i.get("cached"))
        pct_up    = round(thumbs_up / total * 100, 1) if total else 0.0

        # Top disliked queries
        from collections import Counter
        dn_queries = [i["query"][:80] for i in items if i.get("rating") == -1]
        top_disliked = Counter(dn_queries).most_common(5)

        return {
            "total":          total,
            "thumbs_up":      thumbs_up,
            "thumbs_down":    thumbs_dn,
            "satisfaction":   pct_up,
            "cached_failures": cached_dn,   # cached=True + thumbs_down → threshold issue
            "top_disliked":   [{"query": q, "count": c} for q, c in top_disliked],
        }
    except Exception as e:
        log.warning("Feedback stats error: %s", e)
        return {"total": 0, "thumbs_up": 0, "thumbs_down": 0, "satisfaction": 0.0}


def get_thumbs_down(limit: int = 50) -> list:
    """
    Return recent thumbs-down records for eval dataset export.
    Used in pre-production analysis to build gold set from real failures.
    """
    try:
        table    = _get_table()
        response = table.scan(
            FilterExpression="rating = :r AND begins_with(pk, :prefix)",
            ExpressionAttributeValues={":r": -1, ":prefix": "FEEDBACK#"},
        )
        items = sorted(
            response.get("Items", []),
            key=lambda x: x.get("timestamp", ""),
            reverse=True
        )[:limit]
        return items
    except Exception as e:
        log.warning("get_thumbs_down error: %s", e)
        return []
