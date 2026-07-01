
"""
pipeline/registry.py — DynamoDB document registry.

Tracks every indexed document with:
  - SHA-256 content hash (detect changes)
  - Version number (increments on update)
  - Chunk counts per index
  - Indexing timestamps
  - is_latest flag

Fixes applied:
  1. page_url added as GSI partition key (page_url-index) — stable PK for web-crawled content
  2. get_record() and is_unchanged() now accept both filename and url lookups
  3. deprecate_old_version() replaced with atomic_version_transition() using
     TransactWriteItems — eliminates race condition where two jobs could both
     write is_latest=True for the same document_family
  4. save_record() accepts url, og_updated_time, published_date as first-class fields
  5. get_version() works by document_family (stable) not just filename
"""
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from config import settings

log = logging.getLogger(__name__)

TABLE_NAME = settings.REGISTRY_TABLE
REGION     = settings.AWS_REGION


def _table():
    dynamodb = boto3.resource("dynamodb", region_name=REGION)
    return dynamodb.Table(TABLE_NAME)


def _client():
    return boto3.client("dynamodb", region_name=REGION)


# ── Hash helpers ──────────────────────────────────────────────────────────────

def compute_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:32]


def compute_hash_bytes(content: bytes) -> str:
    """Compute SHA-256 hash of raw bytes (for web-fetched content)."""
    return hashlib.sha256(content).hexdigest()[:32]


# ── Record lookups ────────────────────────────────────────────────────────────

def get_record(filename: str) -> Optional[dict]:
    """Get registry record by filename (primary key)."""
    try:
        resp = _table().get_item(Key={"filename": filename})
        return resp.get("Item")
    except ClientError as e:
        log.warning("DynamoDB get error: %s", e)
        return None


def get_record_by_url(url: str) -> Optional[dict]:
    """
    Get registry record by page_url using the page_url-index GSI.
    Falls back to scan if GSI not yet created or not authorized.
    """
    try:
        resp = _table().query(
            IndexName="page_url-index",
            KeyConditionExpression="#pu = :u",
            ExpressionAttributeNames={"#pu": "page_url"},
            ExpressionAttributeValues={":u": url},
            Limit=1,
        )
        items = resp.get("Items", [])
        return items[0] if items else None
    except ClientError as e:
        log.debug("page_url-index query error, falling back to scan: %s", e)
        try:
            resp = _table().scan(
                FilterExpression="#pu = :u",
                ExpressionAttributeNames={"#pu": "page_url"},
                ExpressionAttributeValues={":u": url},
                Limit=1,
            )
            items = resp.get("Items", [])
            return items[0] if items else None
        except ClientError as se:
            log.warning("page_url scan fallback error: %s", se)
            return None


def get_record_by_family(document_family: str) -> Optional[dict]:
    """
    Get the latest registry record for a document family.
    Used to find the current version before deprecating it.
    Requires GSI: partition key = document_family on rag-document-registry table.
    """
    try:
        resp = _table().query(
            IndexName="family-index",
            KeyConditionExpression="document_family = :f",
            FilterExpression="is_latest = :t",
            ExpressionAttributeValues={":f": document_family, ":t": True},
            Limit=1,
        )
        items = resp.get("Items", [])
        return items[0] if items else None
    except ClientError as e:
        log.warning("DynamoDB family-index query error: %s", e)
        return None


def is_unchanged(filename: str, content_hash: str, url: str = "") -> bool:
    """
    Returns True if document already indexed with same content hash.
    For web content: checks by filename (=document_family) as PK.
    For uploads: checks by filename as PK.
    URL-based lookup used only when GSI is available.
    """
    # Primary check — use filename (document_family) as PK directly
    record = get_record(filename)
    if not record and url:
        # Fallback to URL-based lookup
        record = get_record_by_url(url)
    if not record:
        return False
    return record.get("content_hash") == content_hash


def is_unchanged_by_timestamp(url: str, og_updated_time: str) -> bool:
    """
    Fast pre-check using og:updated_time meta tag.
    Avoids full content fetch if timestamp unchanged.
    Returns True if timestamp matches stored value (skip ingest).
    """
    if not og_updated_time:
        return False
    record = get_record_by_url(url)
    if not record:
        return False
    return record.get("og_updated_time") == og_updated_time


def get_version(filename: str = "", document_family: str = "") -> int:
    """
    Get current version number. Returns 0 if not indexed.
    Uses filename PK directly — document_family is stored as filename
    for web-crawled content so this works for both cases.
    """
    # Try filename first (covers both uploads and web pages)
    key = filename or document_family
    if not key:
        return 0
    record = get_record(key)
    if not record and document_family and document_family != filename:
        record = get_record_by_family(document_family)
    if not record:
        return 0
    return int(record.get("version", 1))


# ── Atomic version transition ─────────────────────────────────────────────────

def atomic_version_transition(
    old_filename:       str,
    new_filename:       str,
    new_record:         dict,
) -> bool:
    """
    Atomically deprecate old version and write new version in one DynamoDB
    TransactWriteItems call. Eliminates the race condition where two concurrent
    ingest jobs could both write is_latest=True for the same document_family.

    Args:
        old_filename:  Filename (PK) of the record to deprecate.
        new_filename:  Filename (PK) of the new record to write.
        new_record:    Full item dict for the new version.

    Returns:
        True on success, False on failure.
    """
    now = datetime.now(timezone.utc).isoformat()
    client = _client()

    try:
        client.transact_write_items(
            TransactItems=[
                # 1. Deprecate old version
                {
                    "Update": {
                        "TableName": TABLE_NAME,
                        "Key": {"filename": {"S": old_filename}},
                        "UpdateExpression": "SET #s = :s, is_latest = :f, updated_at = :t",
                        "ExpressionAttributeNames": {"#s": "status"},
                        "ExpressionAttributeValues": {
                            ":s":  {"S": "deprecated"},
                            ":f":  {"BOOL": False},
                            ":t":  {"S": now},
                            # Condition: only deprecate if it was is_latest=True
                            # (prevents deprecating something already deprecated)
                        },
                        "ConditionExpression": "is_latest = :was_latest",
                    }
                },
                # 2. Write new version
                {
                    "Put": {
                        "TableName": TABLE_NAME,
                        "Item": _to_dynamo_item(new_record),
                        # Condition: don't overwrite if somehow already exists
                        "ConditionExpression": "attribute_not_exists(filename)",
                    }
                },
            ]
        )
        log.info(
            "Atomic transition: deprecated '%s', wrote '%s' v%s",
            old_filename, new_filename, new_record.get("version")
        )
        return True

    except client.exceptions.TransactionCanceledException as e:
        log.warning("TransactWrite cancelled (likely condition failed): %s", e)
        return False
    except ClientError as e:
        log.error("TransactWrite error: %s", e)
        return False


def _to_dynamo_item(record: dict) -> dict:
    """Convert a Python dict to DynamoDB typed item format for transact_write_items."""
    result = {}
    for k, v in record.items():
        if isinstance(v, bool):
            result[k] = {"BOOL": v}
        elif isinstance(v, int):
            result[k] = {"N": str(v)}
        elif isinstance(v, str):
            result[k] = {"S": v}
        elif v is None:
            result[k] = {"NULL": True}
        else:
            result[k] = {"S": str(v)}
    return result


# ── Save / update ─────────────────────────────────────────────────────────────

def save_record(
    filename:         str,
    clean_name:       str,
    resource_type:    str,
    namespace:        str,
    content_hash:     str,
    version:          int,
    chunks_search:    int,
    chunks_summary:   int,
    document_family:  str  = "",
    page_url:         str  = "",
    url:              str  = "",
    og_updated_time:  str  = "",
    published_date:   str  = "",
    status:           str  = "current",
) -> None:
    """
    Save or update a document registry record.
    Called after successful ingest.

    For web-crawled content, pass url and og_updated_time.
    For file uploads, these default to empty string.
    """
    now = datetime.now(timezone.utc).isoformat()
    item = {
        "filename":         filename,
        "clean_name":       clean_name,
        "resource_type":    resource_type,
        "namespace":        namespace,
        "content_hash":     content_hash,
        "version":          version,
        "is_latest":        True,           # boolean — not string
        "status":           status,
        "chunks_search":    chunks_search,
        "chunks_summary":   chunks_summary,
        "document_family":  document_family,
        "page_url":         page_url,
        "url":              url,             # stable PK for web content
        "og_updated_time":  og_updated_time, # fast pre-check on next crawl
        "published_date":   published_date,  # for freshness decay
        "indexed_at":       now,
        "updated_at":       now,
    }
    try:
        _table().put_item(Item=item)
        log.info(
            "Registry saved: %s v%d (%d search, %d summary chunks)",
            filename, version, chunks_search, chunks_summary
        )
    except ClientError as e:
        log.error("DynamoDB save error: %s", e)


# ── Deprecate (non-atomic fallback) ───────────────────────────────────────────

def deprecate_old_version(filename: str) -> None:
    """
    Mark old version as deprecated.
    Use atomic_version_transition() instead when possible — this is kept
    as a fallback for single-record deprecation without a concurrent write.
    """
    try:
        _table().update_item(
            Key={"filename": filename},
            UpdateExpression="SET #s = :s, is_latest = :f, updated_at = :t",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={
                ":s": "deprecated",
                ":f": False,
                ":t": datetime.now(timezone.utc).isoformat(),
            }
        )
        log.info("Deprecated: %s", filename)
    except ClientError as e:
        log.warning("DynamoDB deprecate error: %s", e)


# ── Query helpers ─────────────────────────────────────────────────────────────

def list_documents(status: str = None) -> list[dict]:
    """List all documents. Optional status filter: 'current' or 'deprecated'."""
    try:
        if status:
            resp = _table().scan(
                FilterExpression="#s = :s",
                ExpressionAttributeNames={"#s": "status"},
                ExpressionAttributeValues={":s": status},
            )
        else:
            resp = _table().scan()
        return sorted(
            resp.get("Items", []),
            key=lambda x: x.get("indexed_at", ""),
            reverse=True,
        )
    except ClientError as e:
        log.warning("DynamoDB list error: %s", e)
        return []


def delete_record(filename: str) -> None:
    """Delete a registry record (when document removed from Pinecone)."""
    try:
        _table().delete_item(Key={"filename": filename})
        log.info("Registry deleted: %s", filename)
    except ClientError as e:
        log.warning("DynamoDB delete error: %s", e)


def list_latest_urls() -> list[str]:
    """
    Return all URLs currently marked is_latest=True.
    Used by the nightly crawl reconciliation step to detect deleted pages.
    """
    try:
        resp = _table().scan(
            FilterExpression="is_latest = :t AND attribute_exists(#u)",
            ExpressionAttributeNames={"#u": "url"},
            ExpressionAttributeValues={":t": True},
            ProjectionExpression="#u",
        )
        return [item["url"] for item in resp.get("Items", []) if item.get("url")]
    except ClientError as e:
        log.warning("DynamoDB list_latest_urls error: %s", e)
        return []
    
