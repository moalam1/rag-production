"""
pipeline/episodic_memory.py — Visitor journey logging and retrieval.

DynamoDB schema:
  PK: visitor_id  SK: timestamp
  TTL: expires_at (90 days)
"""
import json
import logging
from datetime import datetime, timezone, timedelta
from collections import Counter

import boto3
from boto3.dynamodb.conditions import Key

log        = logging.getLogger(__name__)
TABLE_NAME = "rag-episodic"
TTL_DAYS   = 90
MAX_HISTORY = 20

_dynamodb = None

def _get_table():
    global _dynamodb
    if _dynamodb is None:
        _dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
    return _dynamodb.Table(TABLE_NAME)


def log_query(
    visitor_id: str,
    query:      str,
    intent:     str,
    products:   list,
    use_case:   str,
    top_score:  float,
    sources:    list,
    session_id: str = "",
    lead_quality_tag:   str   = "EARLY_EXPLORER",
    resource_types:     list  = None,
    detected_workloads: list  = None,
    detected_competitors: list = None,
) -> bool:
    if not visitor_id or visitor_id == "anonymous":
        return False
    try:
        now        = datetime.now(timezone.utc)
        expires_at = int((now + timedelta(days=TTL_DAYS)).timestamp())
        _get_table().put_item(Item={
            "visitor_id":     visitor_id,
            "timestamp":      now.isoformat(),
            "query":          query[:500],
            "intent":         intent or "general",
            "products":       products or [],
            "use_case":       use_case or "",
            "top_score":      str(round(float(top_score or 0), 4)),
            "resource_types": list({
                s.get("resource_type", "")
                for s in (sources or [])[:5]
                if s.get("resource_type")
            }),
            "session_id":     session_id or "",
            "lead_quality_tag":    lead_quality_tag,
            "resource_types":      __import__('json').dumps(resource_types or []),
            "detected_workloads":  __import__('json').dumps(detected_workloads or []),
            "detected_competitors": detected_competitors or [],
            "expires_at":     expires_at,
        })
        log.debug(f"Episodic: logged for visitor {visitor_id[:8]}")
        return True
    except Exception as e:
        log.warning(f"Episodic log failed: {e}")
        return False


def get_visitor_context(visitor_id: str) -> dict:
    empty = {
        "interests":   [],
        "use_cases":   [],
        "intents":     [],
        "stage":       "awareness",
        "query_count": 0,
        "last_query":  "",
        "last_intent": "",
    }
    if not visitor_id or visitor_id == "anonymous":
        return empty
    try:
        result = _get_table().query(
            KeyConditionExpression = Key("visitor_id").eq(visitor_id),
            ScanIndexForward       = False,
            Limit                  = MAX_HISTORY,
        )
        items = result.get("Items", [])
        if not items:
            return empty

        product_counts  = Counter()
        use_case_counts = Counter()
        intent_list     = []
        for item in items:
            for p in item.get("products", []):
                product_counts[p] += 1
            if item.get("use_case"):
                use_case_counts[item["use_case"]] += 1
            if item.get("intent"):
                intent_list.append(item["intent"])

        return {
            "interests":   [p for p, _ in product_counts.most_common(3)],
            "use_cases":   [u for u, _ in use_case_counts.most_common(3)],
            "intents":     intent_list[:5],
            "stage":       _infer_stage(intent_list),
            "query_count": len(items),
            "last_query":  items[0].get("query", ""),
            "last_intent": items[0].get("intent", ""),
        }
    except Exception as e:
        log.warning(f"Episodic context failed: {e}")
        return empty


def get_visitor_stats(visitor_id: str) -> dict:
    if not visitor_id:
        return {"queries": [], "total": 0}
    try:
        result = _get_table().query(
            KeyConditionExpression = Key("visitor_id").eq(visitor_id),
            ScanIndexForward       = False,
            Limit                  = MAX_HISTORY,
        )
        items = result.get("Items", [])
        return {
            "queries": [
                {
                    "query":    item.get("query", ""),
                    "intent":   item.get("intent", ""),
                    "products": item.get("products", []),
                    "use_case": item.get("use_case", ""),
                    "time":     item.get("timestamp", "")[:16],
                }
                for item in items
            ],
            "total": len(items),
        }
    except Exception as e:
        log.warning(f"Episodic stats failed: {e}")
        return {"queries": [], "total": 0}


def _infer_stage(intents: list) -> str:
    if not intents:
        return "awareness"
    n = len(intents)
    if n >= 4 and "compare" in intents:
        return "intent"
    if n >= 3 and "compare" in intents:
        return "evaluation"
    if n >= 2 and "evaluate_specs" in intents:
        return "evaluation"
    if n >= 2 and any(i in intents for i in ["troubleshoot", "compare"]):
        return "consideration"
    if "evaluate_specs" in intents:
        return "consideration"
    return "awareness"
