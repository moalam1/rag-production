"""pipeline/competitive_rollup.py - competitive-signals aggregation over episodic.
Scans rag-episodic, counts competitor mentions (from the detected_competitors
field captured at search time), broken down by competitor, by co-mentioned
product, and by company (which accounts are comparing us to whom). Mirrors the
population_trends / heatmap_rollup scan+flatten idiom."""
import json
import boto3
from collections import defaultdict


def _flatten(item):
    """DynamoDB item -> plain dict (strip type wrappers)."""
    return {k: list(v.values())[0] for k, v in item.items()}


def _as_list(v):
    """detected_competitors / products may be a DynamoDB list (already a Python
    list after flatten) or a JSON string. Normalise to a list of strings."""
    if isinstance(v, list):
        return [str(x.get("S", x)) if isinstance(x, dict) else str(x) for x in v]
    if isinstance(v, str) and v:
        try:
            j = json.loads(v)
            return [str(x) for x in j] if isinstance(j, list) else []
        except Exception:
            return []
    return []


def build_competitive(table_name="rag-episodic"):
    """Return competitive-signal aggregation:
      by_competitor : mentions + distinct visitors per competitor
      by_company    : which companies mention which competitors (account intel)
      by_product    : which Equinix products co-occur with competitor mentions
      recent        : a few recent (query, competitors, company) examples
    """
    client = boto3.client("dynamodb", region_name="us-east-1")
    pages = client.get_paginator("scan").paginate(TableName=table_name)

    by_competitor = defaultdict(lambda: {"mentions": 0, "visitors": set()})
    by_company    = defaultdict(lambda: defaultdict(int))   # company -> competitor -> count
    by_product    = defaultdict(lambda: defaultdict(int))   # competitor -> product -> count
    recent = []
    total_mentions = 0

    for page in pages:
        for item in page["Items"]:
            q = _flatten(item)
            vid = q.get("visitor_id", "")
            if not vid or vid.startswith(("v_test", "v_debug")):
                continue
            comps = _as_list(q.get("detected_competitors"))
            if not comps:
                continue
            company  = (q.get("company") or "").strip()
            products = _as_list(q.get("products"))
            query    = q.get("query", "")

            for c in comps:
                by_competitor[c]["mentions"] += 1
                by_competitor[c]["visitors"].add(vid)
                total_mentions += 1
                if company:
                    by_company[company][c] += 1
                for p in products:
                    by_product[c][p] += 1

            if len(recent) < 10:
                recent.append({"query": query[:120], "competitors": comps, "company": company})

    by_competitor_out = dict(sorted(
        ({k: {"mentions": v["mentions"], "visitors": len(v["visitors"])}
          for k, v in by_competitor.items()}).items(),
        key=lambda kv: -kv[1]["mentions"]))

    by_company_out = {co: dict(d) for co, d in by_company.items()}
    by_product_out = {c: dict(d) for c, d in by_product.items()}

    return {
        "by_competitor": by_competitor_out,
        "by_company":    by_company_out,
        "by_product":    by_product_out,
        "recent":        recent,
        "total_mentions": total_mentions,
    }
