"""
api/routes/analytics.py — Analytics endpoints.

Extracted from api/search.py (Tier 1b). All six /analytics/* read-only
aggregation endpoints, hung off their own APIRouter so they're a self-contained
module (and a clean future Lambda boundary). Auth + config come from api.deps;
heavy rollups live in pipeline/*. Endpoint bodies are verbatim moves.
"""
from fastapi import APIRouter, Depends, HTTPException

from api.deps import verify_api_key, log
from config import settings

router = APIRouter(prefix="/api/v1", tags=["analytics"])


@router.get("/analytics/population-trends")
async def population_trends(_: str = Depends(verify_api_key)):
    """Cross-visitor analytics: affinity, conversion velocity, content gaps."""
    try:
        import boto3
        from collections import Counter, defaultdict
        import json as _j

        client  = boto3.client("dynamodb", region_name="us-east-1")
        pages   = client.get_paginator("scan").paginate(TableName="rag-episodic")
        visitors = defaultdict(list)
        all_q    = []

        for page in pages:
            for item in page["Items"]:
                vid = item.get("visitor_id", {}).get("S", "")
                if not vid or vid.startswith(("v_test","v_debug")): continue
                # Parse DynamoDB typed values correctly
                q = {}
                for k, v in item.items():
                    typ = list(v.keys())[0]
                    val = list(v.values())[0]
                    if typ == "L":
                        # List type — extract S values from each element
                        q[k] = [list(x.values())[0] for x in val if isinstance(x, dict)]
                    elif typ == "N":
                        try: q[k] = float(val)
                        except: q[k] = val
                    else:
                        q[k] = val  # S, BOOL, NULL etc
                visitors[vid].append(q)
                all_q.append(q)

        # Product affinity pairs
        pairs = []
        for vid, qs in visitors.items():
            prods = set()
            for q in qs:
                try: prods.update(_j.loads(q.get("products","[]")) if isinstance(q.get("products"),str) else (q.get("products") or []))
                except: pass
            if len(prods) >= 2:
                sp = sorted(prods)
                for i in range(len(sp)):
                    for j in range(i+1, len(sp)):
                        pairs.append(f"{sp[i]} + {sp[j]}")

        tv = len(visitors)
        affinity = {k: round(v/tv*100,1) for k,v in Counter(pairs).most_common(10)} if tv else {}

        # Lead distribution
        tags = [q.get("lead_quality_tag","EARLY_EXPLORER") for q in all_q if q.get("lead_quality_tag")]
        lead_dist = dict(Counter(tags))

        # Content gaps — dead-end queries
        dead_queries = [q.get("query","") for q in all_q if q.get("lead_quality_tag") == "DEAD_END_SUPPORT"]
        gap_kw = Counter()
        stopwords = {"what","does","equinix","about","have","their","with","from","this","that","how","the","and","for"}
        for q in dead_queries:
            for w in q.lower().split():
                if len(w) > 4 and w not in stopwords: gap_kw[w] += 1

        return {
            "total_visitors":    tv,
            "total_queries":     len(all_q),
            "product_affinity":  affinity,
            "lead_distribution": lead_dist,
            "content_gaps":      dict(gap_kw.most_common(15)),
            "dead_end_count":    len(dead_queries),
            "generated_at":      __import__("datetime").datetime.utcnow().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/visitor-profiles")
async def visitor_profiles_analytics(_: str = Depends(verify_api_key)):
    try:
        from pinecone import Pinecone as _PC
        from openai import OpenAI as _OAI
        import json as _j

        _pc    = _PC(api_key=settings.PINECONE_API_KEY)
        _idx   = _pc.Index(settings.PINECONE_INDEX)
        _stats = _idx.describe_index_stats()

        ns_data       = (_stats.get("namespaces") or {})
        profile_count = ns_data.get("visitor-profiles", {}).get("vector_count", 0)

        profiles     = []
        stage_dist   = {}
        tag_dist     = {}

        if profile_count > 0:
            import os as _os
            _client = _OAI(api_key=_os.getenv("OPENAI_API_KEY"))
            _dummy  = _client.embeddings.create(
                model="text-embedding-3-small",
                input="enterprise network infrastructure colocation",
                dimensions=1024,
            ).data[0].embedding

            _res = _idx.query(
                vector=_dummy,
                top_k=min(profile_count, 100),
                include_metadata=True,
                namespace="visitor-profiles",
            )

            for m in (_res.get("matches") or []):
                meta  = getattr(m, "metadata", None) or m.get("metadata", {})
                stage = meta.get("stage", "awareness")
                tag   = meta.get("lead_tag", "EARLY_EXPLORER")
                stage_dist[stage] = stage_dist.get(stage, 0) + 1
                tag_dist[tag]     = tag_dist.get(tag, 0) + 1

                try:    top_prods = _j.loads(meta.get("top_products", "[]"))
                except: top_prods = []

                profiles.append({
                    "visitor_id":   meta.get("visitor_id", m.get("id",""))[:20],
                    "profile":      meta.get("profile", "")[:800],
                    "stage":        stage,
                    "lead_tag":     tag,
                    "top_products": top_prods,
                    "query_count":  int(meta.get("query_count", 0)),
                    "updated_at":   meta.get("updated_at", ""),
                    "name":         meta.get("name", ""),
                    "email":        meta.get("email", ""),
                    "identified":   meta.get("identified", "false"),
                })

        profiles.sort(key=lambda x: x["query_count"], reverse=True)

        return {
            "profile_count":      profile_count,
            "profiles":           profiles,
            "stage_distribution": stage_dist,
            "tag_distribution":   tag_dist,
            "generated_at":       __import__("datetime").datetime.utcnow().isoformat(),
        }

    except Exception as e:
        log.error("visitor-profiles analytics error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/top-queries")
async def top_queries(_: str = Depends(verify_api_key)):
    """Return top 20 most searched queries."""
    from pipeline.analytics import get_top_queries
    return {"queries": get_top_queries(20)}


@router.get("/analytics/stats")
async def analytics_stats(_: str = Depends(verify_api_key)):
    """Return search analytics — volume, cache rate, namespaces, languages."""
    from pipeline.analytics import get_stats
    return get_stats()


@router.get("/analytics/regional-heatmap")
async def regional_heatmap(_: str = Depends(verify_api_key)):
    """Geo + company aggregation by metro/country/region/company."""
    try:
        from pipeline.heatmap_rollup import build_heatmap
        return build_heatmap()
    except Exception as e:
        log.error("regional-heatmap error: %s", e)
        return {"by_metro": {}, "by_country": {}, "by_region": {}, "by_company": {}, "totals": {"visitors": 0, "queries": 0}, "error": str(e)}


@router.get("/analytics/competitive-signals")
async def competitive_signals(_: str = Depends(verify_api_key)):
    """Competitor mentions by competitor / company / product, from episodic."""
    try:
        from pipeline.competitive_rollup import build_competitive
        return build_competitive()
    except Exception as e:
        log.error("competitive-signals analytics error: %s", e)
        return {"by_competitor": {}, "by_company": {}, "by_product": {}, "recent": [], "total_mentions": 0, "error": str(e)}
    
