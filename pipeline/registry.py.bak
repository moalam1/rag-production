"""
pipeline/registry.py — DynamoDB document registry.

Tracks every indexed document with:
  - SHA-256 content hash (detect changes)
  - Version number (increments on update)
  - Chunk counts per index
  - Indexing timestamps
  - is_latest flag

Prevents:
  - Duplicate indexing of unchanged documents
  - Stale chunks accumulating in Pinecone
  - Blind indexing with no audit trail
"""
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from config import settings

log = logging.getLogger(__name__)

TABLE_NAME = "rag-document-registry"
REGION     = settings.AWS_REGION


def _table():
    dynamodb = boto3.resource("dynamodb", region_name=REGION)
    return dynamodb.Table(TABLE_NAME)


def compute_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a PDF file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:32]


def get_record(filename: str) -> dict | None:
    """Get existing registry record for a filename."""
    try:
        resp = _table().get_item(Key={"filename": filename})
        return resp.get("Item")
    except ClientError as e:
        log.warning("DynamoDB get error: %s", e)
        return None


def is_unchanged(filename: str, content_hash: str) -> bool:
    """
    Returns True if document is already indexed with same content hash.
    Used to skip re-indexing unchanged documents.
    """
    record = get_record(filename)
    if not record:
        return False
    return record.get("content_hash") == content_hash


def get_version(filename: str) -> int:
    """Get current version number for a document. Returns 0 if not indexed."""
    record = get_record(filename)
    if not record:
        return 0
    return int(record.get("version", 1))


def save_record(
    filename:        str,
    clean_name:      str,
    resource_type:   str,
    namespace:       str,
    content_hash:    str,
    version:         int,
    chunks_search:   int,
    chunks_summary:  int,
    page_url:        str  = "",
    document_family: str  = "",
    status:          str  = "current",
) -> None:
    """
    Save or update a document registry record.
    Called after successful ingest.
    """
    now = datetime.now(timezone.utc).isoformat()
    try:
        _table().put_item(Item={
            "filename":        filename,
            "clean_name":      clean_name,
            "resource_type":   resource_type,
            "namespace":       namespace,
            "content_hash":    content_hash,
            "version":         version,
            "is_latest":       True,
            "status":          status,
            "chunks_search":   chunks_search,
            "chunks_summary":  chunks_summary,
            "page_url":        page_url,
            "document_family": document_family,
            "indexed_at":      now,
            "updated_at":      now,
        })
        log.info("Registry saved: %s v%d (%d search, %d summary chunks)",
                 filename, version, chunks_search, chunks_summary)
    except ClientError as e:
        log.error("DynamoDB save error: %s", e)


def deprecate_old_version(filename: str) -> None:
    """
    Mark old version as deprecated before re-indexing.
    Called when a document update is detected.
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
        log.info("Deprecated old version of: %s", filename)
    except ClientError as e:
        log.warning("DynamoDB deprecate error: %s", e)


def list_documents(status: str = None) -> list[dict]:
    """
    List all documents in the registry.
    Optional status filter: 'current' or 'deprecated'.
    """
    try:
        if status:
            resp = _table().scan(
                FilterExpression="attribute_exists(filename) AND #s = :s",
                ExpressionAttributeNames={"#s": "status"},
                ExpressionAttributeValues={":s": status},
            )
        else:
            resp = _table().scan()
        return sorted(resp.get("Items", []), key=lambda x: x.get("indexed_at", ""), reverse=True)
    except ClientError as e:
        log.warning("DynamoDB list error: %s", e)
        return []


def delete_record(filename: str) -> None:
    """Delete a registry record (called when document is removed from Pinecone)."""
    try:
        _table().delete_item(Key={"filename": filename})
        log.info("Registry deleted: %s", filename)
    except ClientError as e:
        log.warning("DynamoDB delete error: %s", e)
