"""
api/search.py — FastAPI search endpoint consumed by AEM.
"""
import logging
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from api.models import (SearchRequest, Source, SearchResponse,
                        IdentifyRequest, SummariseRequest, SummariseResponse)
from api.deps import (verify_api_key, api_key_header, get_config,
                      _load_config, invalidate_config)

from config import settings
from limiter import limiter
from starlette.requests import Request
import guardrails.input  as input_grad
import guardrails.output as output_guard
from pipeline.embedder  import embed_text
from pipeline.retriever import retrieve_chunks
from pipeline.reranker  import rerank_chunks, build_context
from pipeline.generator import generate_answer, prepare_query
from pipeline.competitor_detector import detect_competitors
from pipeline.metro_resolver import resolve_metro
from langsmith import traceable

log    = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["search"])

# api_key_header + verify_api_key -> api/deps.py


# ── Request / Response models ─────────────────────────────────────

# ── Endpoints ──────────────────────────────────────────────────────


# ── Lead Quality Classifier + Workload Detector ──────────────────────────────
def _classify_lead(intent, products, resource_types, stage, query):
    COMMERCIAL = {"deployment","provisioning","pricing","cross connect","contract","sla","quote",
                  "trial","poc","pilot","capacity","enterprise","migration",
                  "timeline","cost","rollout","cross-connect","cross connect"}
    q = query.lower()
    has_commercial   = any(kw in q for kw in COMMERCIAL)
    tech_types       = {"whitepaper","blueprint","analyst-report","data-sheet","playbook"}
    has_tech         = bool(set(resource_types or []) & tech_types)
    if stage in ("evaluation","intent") and has_commercial and len(products or []) >= 2:
        return "SOLID_LEAD_COMMERCIAL"
    if intent in ("compare","troubleshoot","evaluate_specs") and len(products or []) >= 1:
        return "TECH_PILOT_ENGAGED"
    if intent == "general" and not products and not has_tech:
        return "DEAD_END_SUPPORT"
    return "EARLY_EXPLORER"


# dynamic config (_load_config/get_config + cache) -> api/deps.py

# WORKLOAD_SIGNALS — loaded from rag-config DynamoDB (refreshed every 5 min)
# To add new workloads: update rag-config table, no code deploy needed
_WORKLOAD_SIGNALS_FALLBACK = {
    "distributed ai":"Distributed AI","ai workload":"Distributed AI",
    "machine learning":"AI & Machine Learning","gpu":"AI & Machine Learning",
    "llm":"AI & Machine Learning","generative ai":"AI & Machine Learning",
    "sd-wan":"SD-WAN","sd wan":"SD-WAN","sdwan":"SD-WAN",
    "hybrid multicloud":"Hybrid Multicloud","multicloud":"Hybrid Multicloud",
    "financial services":"Financial Services","capital markets":"Financial Services",
    "colocation":"Colocation","colo":"Colocation",
    "interconnection":"Interconnection","cross-connect":"Interconnection",
    "network modernization":"Network Modernization","bgp":"Network Modernization",
}

# COMMERCIAL_KEYWORDS_FALLBACK — loaded from rag-config DynamoDB
_COMMERCIAL_KEYWORDS_FALLBACK = [
    "pricing","price","cost","quote","contract","deployment",
    "enterprise agreement","procurement","purchase","buy",
    "billing","subscription","trial","pilot","poc",
]


def _detect_workloads(query, use_case=""):
    # Normalise hyphens so "distributed-ai" matches "distributed ai"
    text    = (query + " " + (use_case or "")).lower().replace("-", " ")
    signals = get_config("workload_signals", _WORKLOAD_SIGNALS_FALLBACK)
    seen, found = set(), []
    for signal, label in signals.items():
        if signal in text and label not in seen:
            found.append(label); seen.add(label)
    return found

@traceable(name="rag-search", run_type="chain")
async def _run_pipeline(
    query:              str  = "",
    namespace:          str  = None,
    source:             str  = "unknown",
    detected_lang_hint: str  = "en",
    last_query:         str  = "",
    last_intent:        str  = "",
    visitor_id:         str  = "",
) -> dict:
    import asyncio as _asyncio
    from pipeline.embedder       import embed_text
    from pipeline.retriever      import retrieve_chunks
    from pipeline.reranker       import rerank_chunks, build_context
    from pipeline.generator      import generate_answer
    from pipeline.intent_detector import detect_intent, get_compare_queries
    import pipeline.semantic_cache as semantic_cache

    # Step 0: Visitor profile personalization
    visitor_profile_summary = ""
    if visitor_id and visitor_id not in ("v_prod_guest", ""):
        try:
            import pinecone as _pin
            from config import settings as _sc
            _pc2  = _pin.Pinecone(api_key=_sc.PINECONE_API_KEY)
            _idx2 = _pc2.Index(_sc.PINECONE_INDEX)
            _res2 = _idx2.fetch(ids=[visitor_id], namespace="visitor-profiles")
            # Pinecone SDK returns FetchResponse object, not dict
            _vecs = getattr(_res2, "vectors", None) or {}
            if visitor_id in _vecs:
                _vmeta = getattr(_vecs[visitor_id], "metadata", None) or {}
                visitor_profile_summary = _vmeta.get("profile", "")
                log.info("Profile loaded for visitor %s", visitor_id[:8])
        except Exception as _pe:
            log.debug("Profile fetch skipped: %s", _pe)

    # Step 1: language detection
    # ── Query normalisation before intent detection ──────────────
    import re as _re
    from urllib.parse import unquote as _unquote
    import unicodedata as _ud

    def _normalise_query(q: str) -> str:
        if not q:
            return q
        # 1. URL decode — "What%20is%20Fabric" → "What is Fabric"
        q = _unquote(q)
        # 2. Remove zero-width + invisible unicode characters
        q = "".join(c for c in q if _ud.category(c) not in ("Cf", "Cc") or c in ("\n", "\t", " "))
        # 3. HTML entities — &amp; &#39; &lt; &gt;
        q = q.replace("&amp;", "&").replace("&#39;", "'").replace(
            "&lt;", "<").replace("&gt;", ">").replace("&nbsp;", " ")
        # 4. Strip markdown formatting — **bold** `code` _italic_
        q = _re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", q)
        q = _re.sub(r"`([^`]+)`", r"\1", q)
        q = _re.sub(r"_{1,2}([^_]+)_{1,2}", r"\1", q)
        # 5. Remove leading step numbers — "1." "2)" "Step 3:" "Q:"
        q = _re.sub(r"^(step\s*)?[\d]+[\s]*[\.\):\-]+\s*", "", q, flags=_re.IGNORECASE)
        q = _re.sub(r"^(q|question|query|user|human|assistant)[:\s]+", "", q, flags=_re.IGNORECASE)
        q = _re.sub(r"^(search|find)[:\s]+", "", q, flags=_re.IGNORECASE)
        # 6. Strip language tag prefixes — "[EN]" "[FR]"
        q = _re.sub(r"^\[[A-Z]{2}\]\s*", "", q)
        # 7. Remove leading/trailing quotes
        q = _re.sub(r'[\"\'“”‘’]', "", q)
        # 8. Normalise greeting prefix — "Hi, what is..." → "what is..."
        q = _re.sub(r"^(hi|hello|hey|good\s*(morning|afternoon|evening))[,!.\s]+",
                    "", q, flags=_re.IGNORECASE)
        # 9. Remove polite filler at start — "Can you please tell me..."
        q = _re.sub(r"^(can you |could you |please |kindly )+(tell me |explain |describe |show me )?",
                    "", q, flags=_re.IGNORECASE)
        # 10. Collapse multiple spaces, tabs, newlines → single space
        q = _re.sub(r"[\s]+", " ", q)
        # 11. Strip trailing punctuation noise — "???" "!!" but keep "?"
        q = _re.sub(r"([?!]){2,}", r"\1", q)
        q = q.strip()
        return q if q else "What does Equinix offer?"  # fallback if query becomes empty

    query = _normalise_query(query)
    retrieval_query, detected_lang = prepare_query(query)

    # Step 2: intent detection + embedding in parallel
    intent, embedding = await _asyncio.gather(
        detect_intent(
            query       = retrieval_query,
            last_query  = last_query,
            last_intent = last_intent,
        ),
        _asyncio.get_event_loop().run_in_executor(None, embed_text, retrieval_query),
    )
    effective_query = intent.rewritten_query

    # Profile-enriched embedding
    if visitor_profile_summary:
        try:
            _enriched = f"Query: {effective_query} | Context: {visitor_profile_summary}"
            embedding  = await _asyncio.get_event_loop().run_in_executor(None, embed_text, _enriched)
            log.info("Profile-enriched embedding for %s", visitor_id[:8])
        except Exception: pass

    # Re-embed if query was rewritten
    if effective_query != retrieval_query:
        embedding = await _asyncio.get_event_loop().run_in_executor(
            None, embed_text, effective_query
        )

    # Step 3: semantic cache check
    cached = semantic_cache.get(embedding, lang=detected_lang)
    if cached:
        cached["detected_lang"] = detected_lang
        cached["source"]        = source
        cached["namespace"]     = namespace or "all"
        cached["chunk_count"]   = 0
        cached["top_score"]     = cached.get("similarity", 0.0)
        # Compute lead fields even for cache hits
        # Use products from cache entry if available, fallback to intent
        _cache_products  = cached.get("_detected_products", intent.detected_products) or intent.detected_products
        _cache_use_case  = cached.get("_detected_use_case", intent.detected_use_case) or ""
        _cache_intent    = cached.get("_intent", intent.intent) or intent.intent
        _cache_workloads = _detect_workloads(query, _cache_use_case)
        _cache_stage     = "evaluation" if _cache_intent in ("compare","troubleshoot") else "consideration" if _cache_intent == "evaluate_specs" else "awareness"
        _cache_lead_tag  = _classify_lead(_cache_intent, _cache_products, [], _cache_stage, query)

        return {
            "result":              cached,
            "reranked":            [],
            "detected_lang":       detected_lang,
            "intent":              cached.get("_intent", intent.intent),
            "confidence":          cached.get("_confidence", intent.confidence),
            "rewritten_query":     cached.get("_rewritten_query", intent.rewritten_query),
            "detected_products":   cached.get("_detected_products", intent.detected_products),
            "detected_use_case":   cached.get("_detected_use_case", intent.detected_use_case),
            "filter_applied":      bool(intent.metadata_filter),
            "inherited":           intent.inherited,
            "lead_quality_tag":    _cache_lead_tag,
            "resource_types":      [],
            "detected_workloads":  _cache_workloads,
            "similarity":        intent.similarity,
        }

    # Step 4: retrieval
    if intent.intent == "compare" and len(intent.detected_products) >= 2:
        compare_result = get_compare_queries(intent, embed_text)
        if compare_result:
            prod_a, emb_a, prod_b, emb_b = compare_result
            chunks_a = retrieve_chunks(emb_a, query_text=effective_query,
                namespace=namespace,
                metadata_filter={"primary_product": {"$in": [prod_a]}, "enriched": {"$eq": True}},
                top_k=5)
            chunks_b = retrieve_chunks(emb_b, query_text=effective_query,
                namespace=namespace,
                metadata_filter={"primary_product": {"$in": [prod_b]}, "enriched": {"$eq": True}},
                top_k=5)
            chunks = chunks_a + chunks_b
        else:
            chunks = retrieve_chunks(embedding, query_text=effective_query, namespace=namespace)
    else:
        use_filter = intent.metadata_filter if intent.metadata_filter else None
        chunks = retrieve_chunks(
            embedding,
            query_text      = effective_query,
            namespace       = namespace,
            metadata_filter = use_filter,
            top_k           = intent.top_k,
        )
        if len(chunks) < 3 and use_filter:
            log.info("Filter returned %d chunks — falling back to unfiltered", len(chunks))
            chunks = retrieve_chunks(
                embedding, query_text=effective_query,
                namespace=namespace, top_k=intent.top_k,
            )

    # Step 5: rerank + fallback
    reranked = rerank_chunks(effective_query, chunks)
    top_score = reranked[0].get("rerank_score", 0) if reranked else 0
    if top_score < 0.10 and intent.metadata_filter:
        fallback  = rerank_chunks(effective_query,
            retrieve_chunks(embedding, query_text=effective_query, namespace=namespace))
        if fallback and fallback[0].get("rerank_score", 0) > top_score:
            reranked = fallback

    # Step 6: generate
    context = build_context(reranked)
    result  = generate_answer(query, context, detected_lang=detected_lang)
    result["detected_lang"] = detected_lang
    result["source"]        = source
    result["namespace"]     = namespace or "all"
    result["chunk_count"]   = len(reranked)
    result["top_score"]     = round(reranked[0].get("rerank_score", 0), 4) if reranked else 0
    result["_intent"]            = intent.intent
    result["_confidence"]        = intent.confidence
    result["_rewritten_query"]   = intent.rewritten_query
    result["_detected_products"] = intent.detected_products
    result["_detected_use_case"] = intent.detected_use_case
    result["_filter_applied"]    = bool(intent.metadata_filter)
    result["_inherited"]         = intent.inherited

    # Step 7: populate semantic cache
    answer_text = result.get("answer", "")
    is_clean = (
        answer_text
        and "Error" not in answer_text
        and "no relevant" not in answer_text.lower()
        and "couldn't find" not in answer_text.lower()
        and not result.get("blocked", False)
        and len(reranked) > 0
    )
    if is_clean:
        result["sources"] = [
            {
                "filename":       c.get("filename", ""),
                "clean_name":     c.get("clean_name", ""),
                "page":           c.get("page", ""),
                "page_url":       c.get("page_url", ""),
                "pdf_url":        c.get("pdf_url", ""),
                "resource_type":  c.get("resource_type", ""),
                "preview":        c.get("text", "")[:200].strip(),
                "relevance_score": round(c.get("rerank_score", 0.0), 4),
            }
            for c in reranked
        ]
        semantic_cache.set(
            query=retrieval_query,
            query_embedding=embedding,
            result=result,
            lang=detected_lang,
        )

    # Lead classification
    _top_rtypes    = list(dict.fromkeys(c.get("resource_type","") for c in reranked[:3] if c.get("resource_type")))
    _workloads     = _detect_workloads(query, intent.detected_use_case or "")
    # Infer stage from full query journey, not just current intent
    _all_intents = []
    try:
        _hist_resp = requests.get(
            f"{settings.API_GATEWAY_URL if hasattr(settings,'API_GATEWAY_URL') else ''}/api/v1/visitor/{visitor_id}/history",
            headers={"X-API-Key": os.getenv("API_KEY","")}, timeout=2
        )
        if _hist_resp.status_code == 200:
            _all_intents = [q.get("intent","") for q in _hist_resp.json().get("queries",[])]
    except Exception:
        pass
    _all_intents.append(intent.intent)
    _n = len(_all_intents)
    if _n >= 4 and "compare" in _all_intents: _stage = "intent"
    elif _n >= 3 and "compare" in _all_intents: _stage = "evaluation"
    elif _n >= 2 and any(i in _all_intents for i in ["troubleshoot","compare","evaluate_specs"]): _stage = "consideration"
    else: _stage = "awareness"
    _lead_tag      = _classify_lead(intent.intent, intent.detected_products, _top_rtypes, _stage, query)
    result["_lead_quality_tag"]    = _lead_tag
    result["_resource_types"]      = _top_rtypes
    result["_detected_workloads"]  = _workloads

    # Write to semantic cache AFTER all enrichment — ensures workloads + lead tag stored
    semantic_cache.set(
        query=retrieval_query,
        query_embedding=embedding,
        result=result,
        lang=detected_lang,
    )

    return {
        "result":            result,
        "reranked":          reranked,
        "detected_lang":     detected_lang,
        "intent":            intent.intent,
        "confidence":        intent.confidence,
        "rewritten_query":   intent.rewritten_query,
        "detected_products": intent.detected_products,
        "detected_use_case": intent.detected_use_case,
        "filter_applied":    bool(intent.metadata_filter),
        "inherited":         intent.inherited,
        "similarity":          intent.similarity,
        "lead_quality_tag":    result.get("_lead_quality_tag", "EARLY_EXPLORER"),
        "resource_types":      result.get("_resource_types", []),
        "detected_workloads":  result.get("_detected_workloads", []),
    }

# ── Greeting detection ───────────────────────────────────────────────────────
import re as _re

_GREETING_PATTERNS = [
    r"^(hi|hey|hello|hiya|howdy|heya|yo|sup)[\s!?.]*$",
    r"^good\s+(morning|afternoon|evening|day)[\s!?.]*$",
    r"^(hi|hey|hello),?\s*(there|friend)?[\s!?.]*$",
    r"^(can you help|help me|i need help)[\s!?.]*$",
    r"^(what can you do|how does this work)[\s!?.]*$",
    r"^(who are you|what are you)[\s!?.]*$",
    r"^(thanks|thank you|cheers|thx|ty)[\s!?.]*$",
    r"^(bye|goodbye|see you|cya)[\s!?.]*$",
]

def _check_greeting(query: str) -> str | None:
    q = query.strip().lower()
    if not q:
        return None
    for pattern in _GREETING_PATTERNS:
        if _re.match(pattern, q, _re.IGNORECASE):
            if any(w in q for w in ["help", "can you"]):
                return "Of course! I can help you search across Equinix's full resource library — including technical blueprints, product data sheets, analyst reports, and case studies. Try asking about a specific product or use case."
            if any(w in q for w in ["what", "who", "how"]):
                return "I'm an AI assistant for the Equinix resource library. I can answer questions about Equinix products like Fabric, Network Edge, and Metal — and help you find the right documentation. What are you looking for?"
            if any(w in q for w in ["thanks", "thank", "thx", "cheers", "ty"]):
                return "You're welcome! Feel free to ask if you have more questions about Equinix products or resources."
            if any(w in q for w in ["bye", "goodbye", "cya"]):
                return "Goodbye! Come back anytime you need help finding Equinix resources."
            return "Hi! I'm the Equinix resource assistant. I can help you find whitepapers, blueprints, data sheets, and case studies. What would you like to know?"
    return None


@router.post("/search", response_model=SearchResponse)
@limiter.limit("120/minute")
@limiter.limit("5000/day")
async def search(req: SearchRequest, request: Request, _: str = Depends(verify_api_key)):

    # ── Log query metadata to LangSmith ──────────────────────────
    from langsmith import get_current_run_tree
    run = get_current_run_tree()
    if run:
        run.metadata.update({
            "query":      req.query,
            "namespace":  req.namespace or "all",
            "source":     req.source or "unknown",
            "user_agent": req.user_agent or "unknown",
        })

    # ── Greeting check — before guardrails so "Hi" isn't blocked ────────────
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

    # 1. Input guardrails
    passed, message = input_grad.run(req.query)
    if not passed:
        # Input blocked before any pipeline ran — return defaults (no pipeline_out yet)
        return SearchResponse(
            query=req.query, answer=message,
            sources=[], followups=[], blocked=True,
            lead_quality_tag   = "EARLY_EXPLORER",
            resource_types     = [],
            detected_workloads = [],
        )

    source = getattr(req, "source", "") or "api"

    # 2-3. Language detection + RAG pipeline via _run_pipeline
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

    # Cache hit path
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
        
        # Async episodic tracking hook for cache hits
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

    # 4. Output guardrails
    passed, message = output_guard.run(answer, context, reranked)
    if not passed:
        return SearchResponse(
            query=req.query, answer=message,
            sources=[], followups=[], blocked=True,
        )

    # 5. Build source list for response validation mapping
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

    # ── Log analytics stats ──────────────────────────────────────
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

    # ── Production-Grade DynamoDB Write Hook ──────────────────────
    if getattr(req, "visitor_id", "") and req.visitor_id != "v_prod_guest":
        try:
            import sys
            _p = "/home/ssm-user/rag-production/pipeline"
            if _p not in sys.path: sys.path.insert(0, _p)
            from pipeline.episodic_memory import log_query as write_to_dynamo
            
            # Map Pydantic structures to flat dictionaries cleanly
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




# /visitor/identify -> api/routes/visitor.py



# ════════ Admin console endpoints ════════════════════════════
# admin endpoints + _ADMIN_CONFIG_KEYS + _PROMPT_META -> api/routes/admin.py


# /visitor/{id}/suggestions + _default/_generate_suggestions -> api/routes/visitor.py



# ── Summarise endpoint ────────────────────────────────────────────

@router.post("/summarise", response_model=SummariseResponse)
@limiter.limit("120/minute")
@limiter.limit("5000/day")
async def summarise(req: SummariseRequest, request: Request, _: str = Depends(verify_api_key)):
    from pipeline.generator import summarise_document
    result = summarise_document(req.filename)
    if "error" in result:
        return SummariseResponse(filename=req.filename, error=result["error"])
    return SummariseResponse(**result)


# /visitor/{id}/history -> api/routes/visitor.py
