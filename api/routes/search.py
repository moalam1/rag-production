"""
api/routes/search.py — Search + summarise routes (Tier 2).

The RAG engine lives in pipeline/search_pipeline.py; this is the FastAPI surface:
request handling, guardrails orchestration, response assembly, episodic/analytics
write hooks. Engine imported from pipeline.search_pipeline.
"""
import logging

from fastapi import APIRouter, Depends, Request, HTTPException

from api.models import (SearchRequest, Source, SearchResponse,
                        SummariseRequest, SummariseResponse)
from api.deps import verify_api_key, get_config
from config import settings
from limiter import limiter
import guardrails.input  as input_grad
import guardrails.output as output_guard
from pipeline.reranker import build_context
from pipeline.competitor_detector import detect_competitors
from pipeline.metro_resolver import resolve_metro
from pipeline.search_pipeline import _run_pipeline, _check_greeting

log    = logging.getLogger("api.routes.search")
router = APIRouter(prefix="/api/v1", tags=["search"])


@router.post("/search", response_model=SearchResponse)
@limiter.limit("120/minute")
@limiter.limit("5000/day")
async def search(req: SearchRequest, request: Request, _: str = Depends(verify_api_key)):

    from langsmith import get_current_run_tree
    run = get_current_run_tree()
    if run:
        run.metadata.update({
            "query":      req.query,
            "namespace":  req.namespace or "all",
            "source":     req.source or "unknown",
            "user_agent": req.user_agent or "unknown",
        })

    greeting_response = _check_greeting(req.query)
    if greeting_response:
        return SearchResponse(
            query     = req.query,
            answer    = greeting_response,
            sources   = [],
            followups = [
                "What port speeds does Equinix Fabric support?",
                "How does Equinix Fabric compare to Network Edge?",
                "Show me a blueprint for hybrid multicloud networking",
            ],
            blocked = False,
            cached  = False,
            intent  = "general",
        )

    passed, message = input_grad.run(req.query)
    if not passed:
        return SearchResponse(
            query=req.query, answer=message,
            sources=[], followups=[], blocked=True,
            lead_quality_tag   = "EARLY_EXPLORER",
            resource_types     = [],
            detected_workloads = [],
        )

    source = getattr(req, "source", "") or "api"

    pipeline_out = await _run_pipeline(
        query       = req.query,
        namespace   = getattr(req, "namespace", None),
        source      = source,
        last_query  = req.last_query,
        last_intent = req.last_intent,
        visitor_id  = getattr(req, "visitor_id", ""),
    )
    result        = pipeline_out["result"]
    reranked      = pipeline_out["reranked"]
    detected_lang = pipeline_out["detected_lang"]
    _competitors = detect_competitors(req.query, get_config("competitor_signals", None))
    _metro = resolve_metro(req.country, "")

    if result.get("cache_hit") or result.get("semantic_hit"):
        cached_sources = [
            Source(
                filename=s.get("filename", ""),
                clean_name=s.get("clean_name", ""),
                page=s.get("page", ""),
                pdf_url=s.get("pdf_url", ""),
                page_url=s.get("page_url", ""),
                resource_type=s.get("resource_type", ""),
                preview=s.get("preview", ""),
                relevance_score=s.get("relevance_score", 0.0),
            )
            for s in result.get("sources", [])
        ]
        if getattr(req, "visitor_id", "") and req.visitor_id != "v_prod_guest":
            try:
                import sys
                _p = "/home/ssm-user/rag-production/pipeline"
                if _p not in sys.path: sys.path.insert(0, _p)
                from pipeline.episodic_memory import log_query
                log_query(
                    visitor_id        = req.visitor_id,
                    query             = req.query,
                    intent            = result.get("_intent", result.get("intent", "general")),
                    products          = result.get("_detected_products", result.get("detected_products", [])),
                    use_case          = result.get("_detected_use_case", result.get("detected_use_case", "")),
                    top_score         = float(result.get("similarity", 1.0)),
                    sources           = result.get("sources", [])[:3],
                    lead_quality_tag  = pipeline_out.get("lead_quality_tag", "EARLY_EXPLORER"),
                    resource_types    = pipeline_out.get("resource_types", []),
                    detected_workloads= pipeline_out.get("detected_workloads", []),
                    detected_competitors = _competitors,
                    country = req.country,
                    company = req.company,
                    metro   = _metro,
                )
            except Exception:
                pass
        return SearchResponse(
            query=req.query,
            answer            = result.get("answer", ""),
            sources           = cached_sources,
            followups         = result.get("followups", []),
            cached            = True,
            intent            = pipeline_out.get("intent", "general"),
            detected_products = pipeline_out.get("detected_products", []),
            detected_use_case = pipeline_out.get("detected_use_case", ""),
            rewritten_query   = pipeline_out.get("rewritten_query", ""),
            confidence        = pipeline_out.get("confidence", 0.0),
            inherited         = pipeline_out.get("inherited", False),
            similarity        = pipeline_out.get("similarity", 0.0),
            lead_quality_tag   = pipeline_out.get("lead_quality_tag", "EARLY_EXPLORER"),
            resource_types     = pipeline_out.get("resource_types", []),
            detected_workloads = pipeline_out.get("detected_workloads", []),
        )

    if not reranked:
        return SearchResponse(
            query=req.query,
            answer="No relevant documents found in the index.",
            sources=[], followups=[], blocked=False,
            lead_quality_tag   = pipeline_out.get("lead_quality_tag", "EARLY_EXPLORER"),
            resource_types     = pipeline_out.get("resource_types", []),
            detected_workloads = pipeline_out.get("detected_workloads", []),
        )

    context = build_context(reranked) if reranked else ""
    answer  = result["answer"]

    passed, message = output_guard.run(answer, context, reranked)
    if not passed:
        return SearchResponse(
            query=req.query, answer=message,
            sources=[], followups=[], blocked=True,
        )

    sources = [
        Source(
            filename=c.get("filename", ""),
            clean_name=c.get("clean_name", c.get("filename", "")),
            page=c.get("page", ""),
            pdf_url=c.get("pdf_url", ""),
            page_url=c.get("page_url", ""),
            resource_type=c.get("resource_type", ""),
            preview=c.get("text", "").split("\n\n",1)[-1][:200].strip(),
            relevance_score=round(c.get("rerank_score", 0.0), 4),
        )
        for c in reranked
    ]

    try:
        from pipeline.analytics import log_query as log_stat
        log_stat(
            query=req.query,
            namespace=getattr(req, "namespace", "all") or "all",
            cached=False,
            lang=detected_lang,
        )
    except Exception:
        pass

    if getattr(req, "visitor_id", "") and req.visitor_id != "v_prod_guest":
        try:
            import sys
            _p = "/home/ssm-user/rag-production/pipeline"
            if _p not in sys.path: sys.path.insert(0, _p)
            from pipeline.episodic_memory import log_query as write_to_dynamo
            serialised_sources = [
                {
                    "filename": s.filename,
                    "clean_name": s.clean_name,
                    "page": s.page,
                    "pdf_url": s.pdf_url,
                    "page_url": s.page_url,
                    "resource_type": s.resource_type,
                    "preview": s.preview
                } for s in sources[:3]
            ]
            write_to_dynamo(
                visitor_id = str(req.visitor_id),
                query      = str(req.query),
                intent     = result.get("intent", "general"),
                products   = result.get("detected_products", []),
                use_case   = result.get("detected_use_case", ""),
                top_score  = float(result.get("top_score", 1.0)),
                sources    = serialised_sources
            ,
                lead_quality_tag   = pipeline_out.get("lead_quality_tag", "EARLY_EXPLORER"),
                resource_types     = pipeline_out.get("resource_types", []),
                detected_workloads = pipeline_out.get("detected_workloads", []),
                detected_competitors = _competitors,
                country = req.country,
                company = req.company,
                metro   = _metro,
            )
            print(f"📡 [BACKEND SUCCESS] Committed tracking step to DynamoDB for ID: {req.visitor_id}")
        except Exception as e:
            print(f"⚠️ [BACKEND ERROR] Logging step bypassed: {e}")

    return SearchResponse(
        query              = req.query,
        answer             = answer,
        sources            = sources,
        followups          = result.get("followups", []),
        cached             = False,
        intent             = pipeline_out.get("intent", "general"),
        detected_products  = pipeline_out.get("detected_products", []),
        detected_use_case  = pipeline_out.get("detected_use_case", ""),
        rewritten_query    = pipeline_out.get("rewritten_query", ""),
        confidence         = pipeline_out.get("confidence", 0.0),
        inherited          = pipeline_out.get("inherited", False),
        similarity         = pipeline_out.get("similarity", 0.0),
        lead_quality_tag   = pipeline_out.get("lead_quality_tag", "EARLY_EXPLORER"),
        resource_types     = pipeline_out.get("resource_types", []),
        detected_workloads = pipeline_out.get("detected_workloads", []),
    )


@router.post("/summarise", response_model=SummariseResponse)
@limiter.limit("120/minute")
@limiter.limit("5000/day")
async def summarise(req: SummariseRequest, request: Request, _: str = Depends(verify_api_key)):
    from pipeline.generator import summarise_document
    result = summarise_document(req.filename)
    if "error" in result:
        return SummariseResponse(filename=req.filename, error=result["error"])
    return SummariseResponse(**result)
