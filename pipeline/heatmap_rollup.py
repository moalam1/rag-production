"""pipeline/heatmap_rollup.py - regional heatmap aggregation over episodic.
Scans rag-episodic, buckets by metro / country / region / company, counting
distinct visitors AND query volume per bucket. Mirrors the scan+flatten idiom
used by population_trends. Region derived via metro_resolver (IBX-backed)."""
import boto3
from collections import defaultdict

from pipeline.metro_resolver import metro_region


def _flatten(item):
    """DynamoDB item -> plain dict (strip type wrappers: {'S':'US'} -> 'US')."""
    return {k: list(v.values())[0] for k, v in item.items()}


def build_heatmap(table_name="rag-episodic"):
    """Return geo + company aggregation for the regional heatmap.
    visitors = distinct visitor_id per bucket; queries = row count per bucket."""
    client = boto3.client("dynamodb", region_name="us-east-1")
    pages = client.get_paginator("scan").paginate(TableName=table_name)

    # bucket -> {"visitors": set(), "queries": int}
    by_metro, by_country, by_region, by_company = (
        defaultdict(lambda: {"visitors": set(), "queries": 0}) for _ in range(4)
    )
    total_rows = 0
    total_visitors = set()

    for page in pages:
        for item in page["Items"]:
            q = _flatten(item)
            vid = q.get("visitor_id", "")
            if not vid or vid.startswith(("v_test", "v_debug")):
                continue
            total_rows += 1
            total_visitors.add(vid)

            metro   = (q.get("metro") or "").strip()
            country = (q.get("country") or "").strip()
            company = (q.get("company") or "").strip()
            region  = metro_region(metro) if metro else ""

            for bucket, key in ((by_metro, metro), (by_country, country),
                                (by_region, region), (by_company, company)):
                if key:  # skip empty/unknown - don't bucket as a real value
                    bucket[key]["visitors"].add(vid)
                    bucket[key]["queries"] += 1

    def _finalize(b, sort=True):
        out = {k: {"visitors": len(v["visitors"]), "queries": v["queries"]}
               for k, v in b.items()}
        if sort:
            out = dict(sorted(out.items(), key=lambda kv: -kv[1]["visitors"]))
        return out

    return {
        "by_metro":   _finalize(by_metro),
        "by_country": _finalize(by_country),
        "by_region":  _finalize(by_region),
        "by_company": _finalize(by_company),
        "totals":     {"visitors": len(total_visitors), "queries": total_rows},
    }
