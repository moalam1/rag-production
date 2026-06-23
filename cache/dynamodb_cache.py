"""
cache/dynamodb_cache.py — DynamoDB-backed cache (serverless, no Redis/VPC).

Best for:  Lambda / Fargate — shared cache with zero infra to manage.
Setup:     CACHE_BACKEND=dynamodb, CACHE_TABLE=rag-<env>-cache (default rag-cache).
TTL:       Stored as an 'expires_at' epoch-seconds attribute. Configure the
           table's native TTL on 'expires_at' so DynamoDB auto-deletes expired
           items (deletion can lag up to ~48h, so we ALSO guard on read).

Item shape:  {cache_key: <str>, value: <json str>, expires_at: <epoch int>}
Mirrors RedisCache: same JSON serialization, same make_key hashing.
"""
import json
import time
import hashlib
import logging
from typing import Any

from cache.base import BaseCache
from config import settings

log = logging.getLogger("cache.dynamodb")


class DynamoDBCache(BaseCache):

    def __init__(self, table_name: str = None,
                 default_ttl: int = settings.CACHE_TTL_SECONDS,
                 region: str = "us-east-1"):
        import boto3
        self.table_name = table_name or getattr(settings, "CACHE_TABLE", None) or "rag-cache"
        self.default_ttl = default_ttl
        self._ddb   = boto3.resource("dynamodb", region_name=region)
        self._table = self._ddb.Table(self.table_name)

    def get(self, key: str) -> Any | None:
        try:
            resp = self._table.get_item(Key={"cache_key": key})
        except Exception as e:
            log.warning("DynamoDB cache get failed (%s) — treating as miss", e)
            return None
        item = resp.get("Item")
        if not item:
            return None
        exp = item.get("expires_at", 0)
        if exp and time.time() > float(exp):
            return None
        raw = item.get("value")
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        ttl = ttl or self.default_ttl
        expires_at = int(time.time() + ttl)
        try:
            self._table.put_item(Item={
                "cache_key":  key,
                "value":      json.dumps(value, default=str),
                "expires_at": expires_at,
            })
        except Exception as e:
            log.warning("DynamoDB cache set failed (%s) — skipping cache write", e)

    def delete(self, key: str) -> None:
        try:
            self._table.delete_item(Key={"cache_key": key})
        except Exception as e:
            log.warning("DynamoDB cache delete failed (%s)", e)

    def clear(self) -> None:
        try:
            scan = self._table.scan(ProjectionExpression="cache_key")
            items = scan.get("Items", [])
            with self._table.batch_writer() as batch:
                for it in items:
                    batch.delete_item(Key={"cache_key": it["cache_key"]})
            log.info("DynamoDB cache cleared — %d keys", len(items))
        except Exception as e:
            log.warning("DynamoDB cache clear failed (%s)", e)

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    def stats(self) -> dict:
        try:
            desc = self._ddb.meta.client.describe_table(TableName=self.table_name)
            return {"backend": "dynamodb", "table": self.table_name,
                    "approx_item_count": desc["Table"].get("ItemCount", "?")}
        except Exception as e:
            return {"backend": "dynamodb", "table": self.table_name, "error": str(e)}

    @staticmethod
    def make_key(namespace: str, data: Any) -> str:
        raw    = json.dumps(data, sort_keys=True, default=str)
        digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return f"rag:{namespace}:{digest}"
