"""
pipeline/search_pipeline.py — The RAG search engine (Tier 2 extraction).

Core pipeline (_run_pipeline) + helpers (_classify_lead, _detect_workloads,
_check_greeting) + fallback constants + greeting patterns. Thin routes live in
api/routes/search.py and import from here.

PRESERVED BEHAVIOUR: _run_pipeline's stage-inference block references `requests`
and `os` which are intentionally NOT imported (as in the original) — that block
raises NameError swallowed by bare except (dead code). Logged for separate redesign.
"""
import logging
import re as _re

from config import settings
from api.deps import get_config
from pipeline.generator import prepare_query, generate_answer
from langsmith import traceable

log = logging.getLogger("pipeline.search_pipeline")


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

_COMMERCIAL_KEYWORDS_FALLBACK = [
    "pricing","price","cost","quote","contract","deployment",
    "enterprise agreement","procurement","purchase","buy",
    "billing","subscription","trial","pilot","poc",
]


def _detect_workloads(query, use_case=""):
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

    visitor_profile_summary = ""
    if visitor_id and visitor_id not in ("v_prod_guest", ""):
        try:
            import pinecone as _pin
            from config import settings as _sc
            _pc2  = _pin.Pinecone(api_key=_sc.PINECONE_API_KEY)
            _idx2 = _pc2.Index(_sc.PINECONE_INDEX)
            _res2 = _idx2.fetch(ids=[visitor_id], namespace="visitor-profiles")
            _vecs = getattr(_res2, "vectors", None) or {}
            if visitor_id in _vecs:
                _vmeta = getattr(_vecs[visitor_id], "metadata", None) or {}
                visitor_profile_summary = _vmeta.get("profile", "")
                log.info("Profile loaded for visitor %s", visitor_id[:8])
        except Exception as _pe:
            log.debug("Profile fetch skipped: %s", _pe)

    import re as _re
    from urllib.parse import unquote as _unquote
    import unicodedata as _ud

    def _normalise_query(q: str) -> str:
        if not q:
            return q
        q = _unquote(q)
        q = "".join(c for c in q if _ud.category(c) not in ("Cf", "Cc") or c in ("\n", "\t", " "))
        q = q.replace("&amp;", "&").replace("&#39;", "'").replace(
            "&lt;", "<").replace("&gt;", ">").replace("&nbsp;", " ")
        q = _re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", q)
        q = _re.sub(r"`([^`]+)`", r"\1", q)
        q = _re.sub(r"_{1,2}([^_]+)_{1,2}", r"\1", q)
        q = _re.sub(r"^(step\s*)?[\d]+[\s]*[\.\):\-]+\s*", "", q, flags=_re.IGNORECASE)
        q = _re.sub(r"^(q|question|query|user|human|assistant)[:\s]+", "", q, flags=_re.IGNORECASE)
        q = _re.sub(r"^(search|find)[:\s]+", "", q, flags=_re.IGNORECASE)
        q = _re.sub(r"^\[[A-Z]{2}\]\s*", "", q)
        q = _re.sub(r'[\"\'“”‘’]', "", q)
        q = _re.sub(r"^(hi|hello|hey|good\s*(morning|afternoon|evening))[,!.\s]+",
                    "", q, flags=_re.IGNORECASE)
        q = _re.sub(r"^(can you |could you |please |kindly )+(tell me |explain |describe |show me )?",
                    "", q, flags=_re.IGNORECASE)
        q = _re.sub(r"[\s]+", " ", q)
        q = _re.sub(r"([?!]){2,}", r"\1", q)
        q = q.strip()
        return q if q else "What does Equinix offer?"

    query = _normalise_query(query)
    retrieval_query, detected_lang = prepare_query(query)

    intent, embedding = await _asyncio.gather(
        detect_intent(
            query       = retrieval_query,
            last_query  = last_query,
            last_intent = last_intent,
        ),
        _asyncio.get_event_loop().run_in_executor(None, embed_text, retrieval_query),
    )
    effective_query = intent.rewritten_query

    if visitor_profile_summary:
        try:
            _enriched = f"Query: {effective_query} | Context: {visitor_profile_summary}"
            embedding  = await _asyncio.get_event_loop().run_in_executor(None, embed_text, _enriched)
            log.info("Profile-enriched embedding for %s", visitor_id[:8])
        except Exception: pass

    if effective_query != retrieval_query:
        embedding = await _asyncio.get_event_loop().run_in_executor(
            None, embed_text, effective_query
        )

    cached = semantic_cache.get(embedding, lang=detected_lang)
    if cached:
        cached["detected_lang"] = detected_lang
        cached["source"]        = source
        cached["namespace"]     = namespace or "all"
        cached["chunk_count"]   = 0
        cached["top_score"]     = cached.get("similarity", 0.0)
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

    reranked = rerank_chunks(effective_query, chunks)
    top_score = reranked[0].get("rerank_score", 0) if reranked else 0
    if top_score < 0.10 and intent.metadata_filter:
        fallback  = rerank_chunks(effective_query,
            retrieve_chunks(embedding, query_text=effective_query, namespace=namespace))
        if fallback and fallback[0].get("rerank_score", 0) > top_score:
            reranked = fallback

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

    _top_rtypes    = list(dict.fromkeys(c.get("resource_type","") for c in reranked[:3] if c.get("resource_type")))
    _workloads     = _detect_workloads(query, intent.detected_use_case or "")
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
                return "Of course! I can help you search across Equinix's full resource library — including technical blueprints, product data sheets, analyst reports, and case studies. Tryasking about a specific product or use case."
            if any(w in q for w in ["what", "who", "how"]):
                return "I'm an AI assistant for the Equinix resource library. I can answer questions about Equinix products like Fabric, Network Edge, and Metal — and help you find the right documentation. What are you looking for?"
            if any(w in q for w in ["thanks", "thank", "thx", "cheers", "ty"]):
                return "You're welcome! Feel free to ask if you have more questions about Equinix products or resources."
            if any(w in q for w in ["bye", "goodbye", "cya"]):
                return "Goodbye! Come back anytime you need help finding Equinix resources."
            return "Hi! I'm the Equinix resource assistant. I can help you find whitepapers, blueprints, data sheets, and case studies. What would you like to know?"
    return None
