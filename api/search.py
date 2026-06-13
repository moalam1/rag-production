"""
api/search.py — FastAPI search endpoint consumed by AEM.
"""
import logging
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field

from config import settings
from limiter import limiter
from starlette.requests import Request
import guardrails.input  as input_grad
import guardrails.output as output_guard
from pipeline.embedder  import embed_text
from pipeline.retriever import retrieve_chunks
from pipeline.reranker  import rerank_chunks, build_context
from pipeline.generator import generate_answer, prepare_query
from langsmith import traceable

log    = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["search"])

# ── API key auth ──────────────────────────────────────────────────
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(key: str = Depends(api_key_header)):
    if settings.API_KEY and key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return key


# ── Request / Response models ─────────────────────────────────────

class SearchRequest(BaseModel):
    query:      str       = Field(..., min_length=1, max_length=1000)
    top_k:      int       = Field(5, ge=1, le=10)
    visitor_id: str       = Field(default="v_prod_guest")
    namespace:  str       = Field(default="all")
    source:     str       = Field(default="api")
    user_agent: str       = Field(default="unknown")
    last_query:  str       = Field(default="")
    last_intent: str       = Field(default="")

class Source(BaseModel):
    filename:        str
    clean_name:      str
    page:            str
    pdf_url:         str
    page_url:        str = ""
    resource_type:   str = ""
    preview:         str
    relevance_score: float

class SearchResponse(BaseModel):
    query:     str
    answer:    str
    sources:   list[Source]
    followups: list[str]
    blocked:   bool = False
    cached:    bool = False
    # Intent fields
    intent:            str       = "general"
    detected_products: list[str] = []
    detected_use_case: str       = ""
    rewritten_query:   str       = ""
    confidence:        float     = 0.0
    inherited:         bool      = False
    similarity:        float     = 0.0
    visitor_history:   list      = []
    lead_quality_tag:  str       = "EARLY_EXPLORER"
    resource_types:    list[str] = []
    detected_workloads: list[str] = []


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


# ── Dynamic config from DynamoDB rag-config ─────────────────
import threading as _threading
import time as _time

_config_cache      = {}
_config_lock       = _threading.Lock()
_config_loaded_at  = 0
_CONFIG_TTL        = 300  # refresh every 5 minutes

def _load_config() -> dict:
    """Load all 4 config categories from rag-config DynamoDB table."""
    global _config_cache, _config_loaded_at
    now = _time.time()
    if _config_cache and (now - _config_loaded_at) < _CONFIG_TTL:
        return _config_cache
    with _config_lock:
        if _config_cache and (now - _config_loaded_at) < _CONFIG_TTL:
            return _config_cache
        try:
            import boto3 as _b3
            _ddb   = _b3.resource("dynamodb", region_name="us-east-1")
            _table = _ddb.Table("rag-config")
            keys   = ["workload_signals","product_signals",
                      "commercial_keywords","workload_badge_styles"]
            loaded = {}
            for k in keys:
                resp = _table.get_item(Key={"config_key": k})
                if "Item" in resp:
                    loaded[k] = resp["Item"].get("data", {})
            if loaded:
                _config_cache     = loaded
                _config_loaded_at = now
                log.info("rag-config loaded from DynamoDB: %s", list(loaded.keys()))
        except Exception as _ce:
            log.warning("rag-config load failed — using hardcoded fallback: %s", _ce)
    return _config_cache

def get_config(key: str, fallback):
    """Get a config value, falling back to hardcoded if DynamoDB unavailable."""
    cfg = _load_config()
    return cfg.get(key, fallback)

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
@limiter.limit("20/minute")
@limiter.limit("100/day")
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
        return SearchResponse(
            query=req.query, answer=message,
            sources=[], followups=[], blocked=True,
            lead_quality_tag   = pipeline_out.get("lead_quality_tag", "EARLY_EXPLORER"),
            resource_types     = pipeline_out.get("resource_types", []),
            detected_workloads = pipeline_out.get("detected_workloads", []),
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



class IdentifyRequest(BaseModel):
    visitor_id: str
    email:      str
    name:       str = ""
    source:     str = "commercial_nudge"
    products:   str = ""

@router.post("/visitor/identify")
async def identify_visitor(req: IdentifyRequest, _: str = Depends(verify_api_key)):
    try:
        import boto3 as _b3
        from datetime import datetime, timezone

        _ddb   = _b3.resource("dynamodb", region_name="us-east-1")
        _table = _ddb.Table("rag-episodic")

        _table.put_item(Item={
            "visitor_id":       req.visitor_id,
            "timestamp":        datetime.now(timezone.utc).isoformat(),
            "intent":           "identity_capture",
            "query":            f"IDENTITY — {req.email}",
            "email":            req.email,
            "name":             req.name,
            "source":           req.source,
            "products":         req.products,
            "lead_quality_tag": "SOLID_LEAD_COMMERCIAL",
        })

        log.info("Identity captured: visitor=%s email=%s", req.visitor_id, req.email)
        return {"status": "ok", "visitor_id": req.visitor_id, "email": req.email}

    except Exception as e:
        log.error("identify_visitor error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))



@router.get("/config/badge-styles")
async def badge_styles(_: str = Depends(verify_api_key)):
    """Return workload badge styles from rag-config — used by HF Space UI."""
    styles = get_config("workload_badge_styles", {
        "Distributed AI":        {"icon":"🤖","bg":"#0c2340","color":"#93c5fd"},
        "AI & Machine Learning": {"icon":"🤖","bg":"#0c2340","color":"#93c5fd"},
        "SD-WAN":                {"icon":"🔀","bg":"#1a1200","color":"#fcd34d"},
        "Hybrid Multicloud":     {"icon":"☁️", "bg":"#0a1628","color":"#60a5fa"},
        "Financial Services":    {"icon":"🏦","bg":"#0a1a0a","color":"#86efac"},
        "Network Modernization": {"icon":"🔧","bg":"#1a0a1a","color":"#d8b4fe"},
        "Colocation":            {"icon":"🏢","bg":"#1a1a0a","color":"#fde68a"},
        "Interconnection":       {"icon":"🔗","bg":"#0f1a1a","color":"#5eead4"},
    })
    return {"badge_styles": styles}



# ════════ Admin console endpoints ════════════════════════════
_ADMIN_CONFIG_KEYS = ["workload_signals","product_signals","commercial_keywords","workload_badge_styles","equinix_products","equinix_use_cases","competitor_signals"]

@router.get("/admin/config")
async def admin_get_config(_: str = Depends(verify_api_key)):
    """All editable rag-config keys for the admin console."""
    import boto3 as _b3
    _t = _b3.resource("dynamodb", region_name="us-east-1").Table("rag-config")
    out = {}
    for k in _ADMIN_CONFIG_KEYS:
        resp = _t.get_item(Key={"config_key": k})
        if "Item" in resp:
            out[k] = resp["Item"].get("data", {})
    return {"config": out}


@router.put("/admin/config/{config_key}")
async def admin_put_config(config_key: str, body: dict,
                           _: str = Depends(verify_api_key)):
    """Update one rag-config key from the admin console. 5-min TTL applies."""
    if config_key not in _ADMIN_CONFIG_KEYS:
        raise HTTPException(400, f"Unknown config key: {config_key}")
    data = body.get("data")
    if data is None:
        raise HTTPException(400, "Missing 'data' in body")
    import boto3 as _b3, datetime as _dt
    _t = _b3.resource("dynamodb", region_name="us-east-1").Table("rag-config")
    _t.update_item(
        Key={"config_key": config_key},
        UpdateExpression="SET #d = :d, updated_at = :u",
        ExpressionAttributeNames={"#d": "data"},
        ExpressionAttributeValues={
            ":d": data,
            ":u": _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
        },
    )
    # Bust the in-process config cache so it reloads next request
    global _config_loaded_at
    _config_loaded_at = 0
    log.info("admin: rag-config %s updated (%s items)", config_key,
             len(data) if isinstance(data, (list, dict)) else 1)
    return {"ok": True, "config_key": config_key}


@router.get("/admin/prompt")
async def admin_get_prompt(_: str = Depends(verify_api_key)):
    """Current system prompt + version. Reads rag-config override or code default."""
    import boto3 as _b3
    _t = _b3.resource("dynamodb", region_name="us-east-1").Table("rag-config")
    resp = _t.get_item(Key={"config_key": "system_prompt"})
    from pipeline.generator import SYSTEM_PROMPT as _code_prompt
    from pipeline.prompt_registry import get_prompt_version
    _pv = get_prompt_version('generation', 2)
    if "Item" in resp:
        item = resp["Item"]
        return {"prompt": item.get("data", _code_prompt),
                "prompt_version": int(item.get("prompt_version", _pv)),
                "source": "rag-config"}
    return {"prompt": _code_prompt, "prompt_version": _pv, "source": "code"}


@router.put("/admin/prompt")
async def admin_put_prompt(body: dict, _: str = Depends(verify_api_key)):
    """Save prompt to rag-config, bump version, clear answer caches."""
    prompt = (body.get("prompt") or "").strip()
    if len(prompt) < 50:
        raise HTTPException(400, "Prompt too short — refusing to save")
    import boto3 as _b3, datetime as _dt
    _t = _b3.resource("dynamodb", region_name="us-east-1").Table("rag-config")
    resp = _t.get_item(Key={"config_key": "system_prompt"})
    from pipeline.prompt_registry import get_prompt_version
    _pv_code = get_prompt_version('generation', 2)
    current_pv = int(resp["Item"].get("prompt_version", _pv_code)) if "Item" in resp else _pv_code
    new_pv = current_pv + 1
    _t.put_item(Item={
        "config_key": "system_prompt",
        "data": prompt,
        "prompt_version": new_pv,
        "updated_at": _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
    })
    log.warning("admin: system prompt updated → pv=%s. NOTE: generator must read "
                "rag-config prompt for this to take effect (see deploy notes).", new_pv)
    return {"ok": True, "prompt_version": new_pv,
            "note": "Restart rag-api + clear semantic cache to fully apply"}




_PROMPT_META = {
    "generation": {"model": "gpt-4o",      "label": "Answer generation",
                   "note": "Saving bumps version → semantic + memory caches auto-invalidate"},
    "intent":     {"model": "gpt-4o-mini", "label": "Intent detection",
                   "note": "Keep {products} and {use_cases} placeholders intact"},
    "profiles":   {"model": "gpt-4o-mini", "label": "Nightly buyer briefs",
                   "note": "Applies on next consolidation run"},
}

@router.get("/admin/prompts")
async def admin_list_prompts(_: str = Depends(verify_api_key)):
    import boto3 as _b3
    t = _b3.resource("dynamodb", region_name="us-east-1").Table("rag-config")
    out = []
    for pid, meta in _PROMPT_META.items():
        r = t.get_item(Key={"config_key": f"prompt#{pid}"})
        item = r.get("Item", {})
        out.append({"id": pid, **meta,
                    "version": int(item.get("prompt_version", 0)),
                    "chars": len(item.get("data", "")),
                    "source": "registry" if item else "code-fallback",
                    "updated_at": item.get("updated_at", "")})
    return {"prompts": out}

@router.get("/admin/prompts/{pid}")
async def admin_get_prompt_v2(pid: str, _: str = Depends(verify_api_key)):
    if pid not in _PROMPT_META:
        raise HTTPException(404, f"Unknown prompt id: {pid}")
    import boto3 as _b3
    t = _b3.resource("dynamodb", region_name="us-east-1").Table("rag-config")
    r = t.get_item(Key={"config_key": f"prompt#{pid}"})
    if "Item" in r:
        return {"id": pid, "prompt": r["Item"].get("data", ""),
                "version": int(r["Item"].get("prompt_version", 1)), "source": "registry"}
    return {"id": pid, "prompt": "", "version": 0, "source": "code-fallback",
            "note": "Not yet in registry — save once to take control"}

@router.put("/admin/prompts/{pid}")
async def admin_put_prompt_v2(pid: str, body: dict, _: str = Depends(verify_api_key)):
    if pid not in _PROMPT_META:
        raise HTTPException(404, f"Unknown prompt id: {pid}")
    prompt = (body.get("prompt") or "").strip()
    if len(prompt) < 50:
        raise HTTPException(400, "Prompt too short — refusing to save")
    if pid == "intent" and ("{products}" not in prompt or "{use_cases}" not in prompt):
        raise HTTPException(400, "Intent prompt must keep {products} and {use_cases} placeholders")
    import boto3 as _b3, datetime as _dt
    t = _b3.resource("dynamodb", region_name="us-east-1").Table("rag-config")
    r = t.get_item(Key={"config_key": f"prompt#{pid}"})
    new_v = (int(r["Item"].get("prompt_version", 0)) if "Item" in r else 0) + 1
    t.put_item(Item={"config_key": f"prompt#{pid}", "data": prompt,
                     "prompt_version": new_v,
                     "updated_at": _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
                     "description": _PROMPT_META[pid]["label"]})
    from pipeline.prompt_registry import bust as _bust; _bust()
    global _config_loaded_at; _config_loaded_at = 0
    log.warning("admin: prompt#%s saved → v%s (%s chars)", pid, new_v, len(prompt))
    return {"ok": True, "id": pid, "version": new_v, "note": _PROMPT_META[pid]["note"]}


@router.get("/health")
async def health():
    return {"status": "ok", "environment": settings.ENVIRONMENT}


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
                q = {k: list(v.values())[0] for k, v in item.items()}
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

    return {"status": "ok", "environment": settings.ENVIRONMENT}

@router.get("/visitor/{visitor_id}/suggestions")
async def visitor_suggestions(visitor_id: str, _: str = Depends(verify_api_key)):
    """Return personalised query suggestions based on visitor history."""
    try:
        from pipeline.episodic_memory import get_visitor_context
        context = get_visitor_context(visitor_id)
        suggestions = _generate_suggestions(context)
        return {
            "suggestions":   suggestions,
            "personalised":  len(context.get("interests", [])) > 0,
            "stage":         context.get("stage", "awareness"),
            "query_count":   context.get("query_count", 0),
        }
    except Exception as e:
        log.warning(f"Suggestions error: {e}")
        return {"suggestions": _default_suggestions(), "personalised": False}


def _default_suggestions() -> list[dict]:
    return [
        {"icon": "💡", "text": "What is Equinix Fabric and how does it work?",       "intent": "learn_concept"},
        {"icon": "📊", "text": "What port speeds and SLAs does Equinix Fabric support?", "intent": "evaluate_specs"},
        {"icon": "⚖️",  "text": "Equinix Fabric vs Network Edge for SD-WAN deployment", "intent": "compare"},
        {"icon": "📄", "text": "Show me a hybrid multicloud networking blueprint",    "intent": "find_resource"},
    ]


def _generate_suggestions(context: dict) -> list[dict]:
    intents     = context.get("intents", [])
    products    = context.get("interests", [])
    stage       = context.get("stage", "awareness")
    last_intent = context.get("last_intent", "")
    last_query  = context.get("last_query", "")

    # New visitor — return defaults
    if not intents or not products:
        return _default_suggestions()

    suggestions = []
    used_intents = set()
    p0 = products[0] if products else "Equinix Fabric"
    # Shorten long product names for card display
    p0_short = p0.replace("Equinix Fabric Cloud Router", "Fabric Cloud Router")

    # ── Rule 1: Continue the thread ───────────────────────────────────────────
    if last_intent == "learn_concept" and "evaluate_specs" not in intents:
        suggestions.append({
            "icon": "📊",
            "text": f"What are the technical specifications for {p0_short}?",
            "intent": "evaluate_specs",
            "rule": "continue"
        })
        used_intents.add("evaluate_specs")

    elif last_intent == "evaluate_specs" and "compare" not in intents:
        suggestions.append({
            "icon": "⚖️",
            "text": f"How does {p0_short} compare to Network Edge?",
            "intent": "compare",
            "rule": "continue"
        })
        used_intents.add("compare")

    elif last_intent == "compare" and "find_resource" not in intents:
        suggestions.append({
            "icon": "📄",
            "text": f"Show me a {p0_short} deployment blueprint",
            "intent": "find_resource",
            "rule": "continue"
        })
        used_intents.add("find_resource")

    elif last_intent == "find_resource" and stage in ("evaluation", "intent"):
        suggestions.append({
            "icon": "🎯",
            "text": f"What does an enterprise {p0_short} rollout look like?",
            "intent": "find_resource",
            "rule": "commercial"
        })
        used_intents.add("find_resource")

    # ── Rule 2: Fill the product gap ──────────────────────────────────────────
    ALL_CORE = ["Equinix Fabric", "Network Edge", "Equinix Metal",
                "Equinix Fabric Cloud Router", "Internet Access"]
    unseen = [p for p in ALL_CORE if p not in products]
    if unseen and len(suggestions) < 3 and "compare" not in used_intents:
        p1_short = unseen[0].replace("Equinix Fabric Cloud Router", "Fabric Cloud Router")
        suggestions.append({
            "icon": "⚖️",
            "text": f"{p0_short} vs {p1_short} — what's the difference?",
            "intent": "compare",
            "rule": "gap"
        })
        used_intents.add("compare")

    # ── Rule 3: Stage advancement ─────────────────────────────────────────────
    if stage == "consideration" and len(suggestions) < 3:
        if "evaluate_specs" not in used_intents:
            suggestions.append({
                "icon": "📊",
                "text": f"What SLAs and performance guarantees does {p0_short} offer?",
                "intent": "evaluate_specs",
                "rule": "stage"
            })
            used_intents.add("evaluate_specs")

    elif stage in ("evaluation", "intent") and len(suggestions) < 3:
        if "find_resource" not in used_intents:
            suggestions.append({
                "icon": "🏗️",
                "text": f"Show me a {p0_short} enterprise deployment case study",
                "intent": "find_resource",
                "rule": "stage"
            })
            used_intents.add("find_resource")

    # ── Rule 4: Always one generic broadening card ────────────────────────────
    # Pick a topic the visitor hasn't explored
    broadening = [
        {"icon": "🏢", "text": "What is colocation and why use Equinix IBX?",     "intent": "learn_concept"},
        {"icon": "☁️",  "text": "How does Equinix Fabric connect to AWS and Azure?", "intent": "learn_concept"},
        {"icon": "🔀", "text": "What are Equinix Fabric Cloud Router capabilities?", "intent": "evaluate_specs"},
        {"icon": "🏦", "text": "Show me a financial services case study for Equinix", "intent": "find_resource"},
        {"icon": "💡", "text": "What is Equinix Metal bare-metal infrastructure?",  "intent": "learn_concept"},
    ]
    # Pick one not already covered by visitor history
    for b in broadening:
        text_lower = b["text"].lower()
        already_seen = any(
            p.lower().replace("equinix ", "") in text_lower
            for p in products)
        if not already_seen or len(suggestions) >= 3:
            suggestions.append({**b, "rule": "broadening"})
            break

    # Pad to 4 with defaults if needed
    defaults = _default_suggestions()
    for d in defaults:
        if len(suggestions) >= 4:
            break
        if d["intent"] not in used_intents:
            suggestions.append(d)

    return suggestions[:4]


@router.get("/visitor/{visitor_id}/suggestions")
async def visitor_suggestions(visitor_id: str, _: str = Depends(verify_api_key)):
    """Return personalised query suggestions based on visitor history."""
    try:
        from pipeline.episodic_memory import get_visitor_context
        context = get_visitor_context(visitor_id)
        suggestions = _generate_suggestions(context)
        return {
            "suggestions":   suggestions,
            "personalised":  len(context.get("interests", [])) > 0,
            "stage":         context.get("stage", "awareness"),
            "query_count":   context.get("query_count", 0),
        }
    except Exception as e:
        log.warning(f"Suggestions error: {e}")
        return {"suggestions": _default_suggestions(), "personalised": False}


def _default_suggestions() -> list[dict]:
    return [
        {"icon": "💡", "text": "What is Equinix Fabric and how does it work?",       "intent": "learn_concept"},
        {"icon": "📊", "text": "What port speeds and SLAs does Equinix Fabric support?", "intent": "evaluate_specs"},
        {"icon": "⚖️",  "text": "Equinix Fabric vs Network Edge for SD-WAN deployment", "intent": "compare"},
        {"icon": "📄", "text": "Show me a hybrid multicloud networking blueprint",    "intent": "find_resource"},
    ]


def _generate_suggestions(context: dict) -> list[dict]:
    intents     = context.get("intents", [])
    products    = context.get("interests", [])
    stage       = context.get("stage", "awareness")
    last_intent = context.get("last_intent", "")
    last_query  = context.get("last_query", "")

    # New visitor — return defaults
    if not intents or not products:
        return _default_suggestions()

    suggestions = []
    used_intents = set()
    p0 = products[0] if products else "Equinix Fabric"
    # Shorten long product names for card display
    p0_short = p0.replace("Equinix Fabric Cloud Router", "Fabric Cloud Router")

    # ── Rule 1: Continue the thread ───────────────────────────────────────────
    if last_intent == "learn_concept" and "evaluate_specs" not in intents:
        suggestions.append({
            "icon": "📊",
            "text": f"What are the technical specifications for {p0_short}?",
            "intent": "evaluate_specs",
            "rule": "continue"
        })
        used_intents.add("evaluate_specs")

    elif last_intent == "evaluate_specs" and "compare" not in intents:
        suggestions.append({
            "icon": "⚖️",
            "text": f"How does {p0_short} compare to Network Edge?",
            "intent": "compare",
            "rule": "continue"
        })
        used_intents.add("compare")

    elif last_intent == "compare" and "find_resource" not in intents:
        suggestions.append({
            "icon": "📄",
            "text": f"Show me a {p0_short} deployment blueprint",
            "intent": "find_resource",
            "rule": "continue"
        })
        used_intents.add("find_resource")

    elif last_intent == "find_resource" and stage in ("evaluation", "intent"):
        suggestions.append({
            "icon": "🎯",
            "text": f"What does an enterprise {p0_short} rollout look like?",
            "intent": "find_resource",
            "rule": "commercial"
        })
        used_intents.add("find_resource")

    # ── Rule 2: Fill the product gap ──────────────────────────────────────────
    ALL_CORE = ["Equinix Fabric", "Network Edge", "Equinix Metal",
                "Equinix Fabric Cloud Router", "Internet Access"]
    unseen = [p for p in ALL_CORE if p not in products]
    if unseen and len(suggestions) < 3 and "compare" not in used_intents:
        p1_short = unseen[0].replace("Equinix Fabric Cloud Router", "Fabric Cloud Router")
        suggestions.append({
            "icon": "⚖️",
            "text": f"{p0_short} vs {p1_short} — what's the difference?",
            "intent": "compare",
            "rule": "gap"
        })
        used_intents.add("compare")

    # ── Rule 3: Stage advancement ─────────────────────────────────────────────
    if stage == "consideration" and len(suggestions) < 3:
        if "evaluate_specs" not in used_intents:
            suggestions.append({
                "icon": "📊",
                "text": f"What SLAs and performance guarantees does {p0_short} offer?",
                "intent": "evaluate_specs",
                "rule": "stage"
            })
            used_intents.add("evaluate_specs")

    elif stage in ("evaluation", "intent") and len(suggestions) < 3:
        if "find_resource" not in used_intents:
            suggestions.append({
                "icon": "🏗️",
                "text": f"Show me a {p0_short} enterprise deployment case study",
                "intent": "find_resource",
                "rule": "stage"
            })
            used_intents.add("find_resource")

    # ── Rule 4: Always one generic broadening card ────────────────────────────
    # Pick a topic the visitor hasn't explored
    broadening = [
        {"icon": "🏢", "text": "What is colocation and why use Equinix IBX?",     "intent": "learn_concept"},
        {"icon": "☁️",  "text": "How does Equinix Fabric connect to AWS and Azure?", "intent": "learn_concept"},
        {"icon": "🔀", "text": "What are Equinix Fabric Cloud Router capabilities?", "intent": "evaluate_specs"},
        {"icon": "🏦", "text": "Show me a financial services case study for Equinix", "intent": "find_resource"},
        {"icon": "💡", "text": "What is Equinix Metal bare-metal infrastructure?",  "intent": "learn_concept"},
    ]
    # Pick one not already covered by visitor history
    for b in broadening:
        text_lower = b["text"].lower()
        already_seen = any(
            p.lower().replace("equinix ", "") in text_lower
            for p in products
        )
        if not already_seen or len(suggestions) >= 3:
            suggestions.append({**b, "rule": "broadening"})
            break

    # Pad to 4 with defaults if needed
    defaults = _default_suggestions()
    for d in defaults:
        if len(suggestions) >= 4:
            break
        if d["intent"] not in used_intents:
            suggestions.append(d)

    return suggestions[:4]



@router.get("/cache/stats")
async def cache_stats(_: str = Depends(verify_api_key)):
    from cache.factory import cache
    c = cache()
    if hasattr(c, "stats"):
        return c.stats()
    return {"message": "Stats not available for this backend"}


@router.delete("/cache")
async def clear_cache(_: str = Depends(verify_api_key)):
    from cache.factory import cache
    cache().clear()
    return {"message": "Cache cleared"}


# ── Summarise endpoint ────────────────────────────────────────────

class SummariseRequest(BaseModel):
    filename: str = Field(..., min_length=1, max_length=500)

class SummariseResponse(BaseModel):
    filename:       str
    summary:        str = ""
    key_topics:     list[str] = []
    suggested_name: str = ""
    suggested_type: str = ""
    cached:         bool = False
    error:          str = ""

@router.post("/summarise", response_model=SummariseResponse)
@limiter.limit("20/minute")
@limiter.limit("100/day")
async def summarise(req: SummariseRequest, request: Request, _: str = Depends(verify_api_key)):
    from pipeline.generator import summarise_document
    result = summarise_document(req.filename)
    if "error" in result:
        return SummariseResponse(filename=req.filename, error=result["error"])
    return SummariseResponse(**result)


@router.get("/registry")
async def list_registry(_: str = Depends(verify_api_key)):
    """List all documents in the DynamoDB registry."""
    from pipeline.registry import list_documents
    docs = list_documents()
    return {
        "total": len(docs),
        "documents": docs,
    }


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


@router.get("/visitor/{visitor_id}/history")
async def visitor_history(visitor_id: str, _: str = Depends(verify_api_key)):
    """Return visitor query history from DynamoDB episodic memory."""
    import sys
    _p = "/home/ssm-user/rag-production/pipeline"
    if _p not in sys.path: sys.path.insert(0, _p)

    try:
        from pipeline.episodic_memory import get_visitor_stats
        stats = get_visitor_stats(visitor_id)
        return stats
    except Exception as e:
        import logging
        logging.getLogger("api.search").warning(f"Visitor history endpoint error: {e}")
        return {"queries": [], "total": 0}
