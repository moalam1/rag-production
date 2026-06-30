from pipeline.prompt_registry import get_prompt as _gp
"""
pipeline/intent_detector.py — Query intent detection with coherence gate.

6 intents:
  find_resource   — wants a specific document format
  evaluate_specs  — wants technical specs, numbers, SLAs, speeds
  compare         — comparing two or more products
  troubleshoot    — has an active break-fix problem
  learn_concept   — wants to understand a concept
  general         — vague or too broad

Two-layer classification:
  Layer 1 — Coherence gate: if cosine(current, previous) ≥ 0.85
             AND no friction keywords → inherit previous intent
             Saves 300ms + $0.0002 per inherited query
  Layer 2 — LLM classification: gpt-4o-mini with disambiguation rules

Cost:  ~$0.0002/query (LLM path)
       ~$0.000001/query (coherence gate path — embed only)
"""

import asyncio
import json
import logging
import re as _re_module
_CASESTUDY_RE = _re_module.compile(
    r"\b(case[\s-]?stud(?:y|ies)|customer[\s-]?stor(?:y|ies)|"
    r"success[\s-]?stor(?:y|ies))\b", _re_module.I)
_METAWORD_RE = _re_module.compile(
    r"\b(case[\s-]?stud(?:y|ies)|customer[\s-]?stor(?:y|ies)|"
    r"success[\s-]?stor(?:y|ies)|whitepapers?|blueprints?|data[\s-]?sheets?|"
    r"solution[\s-]?briefs?|find|show\s+me|i\s+need|related\s+to|about|for)\b",
    _re_module.I)
_WS_RE = _re_module.compile(r"\s+")
import os
from dataclasses import dataclass, field
from typing import Optional

from openai import AsyncOpenAI

log    = logging.getLogger(__name__)
client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ── Official Equinix products ─────────────────────────────────────────────────
_EQUINIX_PRODUCTS_FALLBACK = [
    "Equinix Fabric", "Equinix Fabric Cloud Router",
    "Equinix Metal", "Equinix Precision Time",
    "Internet Access", "Managed Services",
    "Network Edge", "Platform Equinix",
    "xScale", "IBX", "Internet Exchange",
    "SmartKey", "Equinix Connect",
]

_EQUINIX_USE_CASES_FALLBACK = [
    "application-exchange", "application-optimization", "cloud-adjacency",
    "colocation", "digital-transformation", "distributed-ai",
    "distributed-data", "distributed-security", "edge-computing",
    "edge-infrastructure", "high-performance-data-center",
    "hybrid-multicloud-networking", "ioa-network-architecture",
    "interconnection", "network-optimization", "network-modernization",
    "sustainability",
]

def _load_equinix_config():
    """Load product + use case lists from rag-config DynamoDB. Cached per process."""
    try:
        import boto3 as _b3, time as _t
        _ddb   = _b3.resource("dynamodb", region_name="us-east-1")
        _table = _ddb.Table("rag-config")
        prods  = _table.get_item(Key={"config_key":"equinix_products"})
        uses   = _table.get_item(Key={"config_key":"equinix_use_cases"})
        return (
            prods["Item"]["data"] if "Item" in prods else _EQUINIX_PRODUCTS_FALLBACK,
            uses["Item"]["data"]  if "Item" in uses  else _EQUINIX_USE_CASES_FALLBACK,
        )
    except Exception as e:
        return _EQUINIX_PRODUCTS_FALLBACK, _EQUINIX_USE_CASES_FALLBACK

_EQUINIX_PRODUCTS, _EQUINIX_USE_CASES = _load_equinix_config()

# Keep EQUINIX_PRODUCTS as alias for backwards compatibility
EQUINIX_PRODUCTS  = _EQUINIX_PRODUCTS
EQUINIX_USE_CASES = _EQUINIX_USE_CASES

# EQUINIX_USE_CASES loaded from DynamoDB above

# ── Per-intent retrieval parameters ──────────────────────────────────────────
RETRIEVAL_PARAMS = {
    "find_resource": {
        "top_k":       5,
        "rrf_alpha":   0.6,
        "namespaces":  ["technical", "business", "media", "customer-success"],
        "description": "User wants a specific document format",
    },
    "evaluate_specs": {
        "top_k":       6,
        "rrf_alpha":   0.5,    # heavy BM25 — exact terms like Gbps, SLA, ms
        "namespaces":  ["technical"],
        "description": "User wants technical specifications and numbers",
    },
    "compare": {
        "top_k":       10,
        "rrf_alpha":   0.7,
        "namespaces":  ["technical", "business"],
        "description": "User comparing two or more products",
    },
    "troubleshoot": {
        "top_k":       8,
        "rrf_alpha":   0.8,
        "namespaces":  ["technical"],
        "description": "User has an active break-fix problem",
    },
    "learn_concept": {
        "top_k":       6,
        "rrf_alpha":   0.7,
        "namespaces":  ["technical", "business"],
        "description": "User wants to understand a concept",
    },
    "general": {
        "top_k":       5,
        "rrf_alpha":   0.7,
        "namespaces":  ["technical", "business", "media", "customer-success"],
        "description": "Vague or multi-topic query",
    },
}

# ── Enrichment tag filters per intent ─────────────────────────────────────────
INTENT_FILTERS = {
    "find_resource":  {},
    "evaluate_specs": {"has_specs": {"$eq": True}},
    "compare":        {},
    "troubleshoot":   {
        "technical_depth":  {"$in": ["practitioner", "engineer"]},
    },
    "learn_concept":  {"technical_depth": {"$in": ["executive", "practitioner"]}},
    "general":        {},
}

# ── Coherence gate ─────────────────────────────────────────────────────────────
COHERENCE_THRESHOLD = 0.85   # cosine similarity threshold to inherit intent

# Friction keywords — trigger fresh classification regardless of similarity
FRICTION_KEYWORDS = {
    "error", "drop", "dropping", "slow", "timeout", "fail", "failing",
    "not working", "packet loss", "cannot", "issue", "broken", "keeps",
    "disconnecting", "down", "unreachable", "spike", "latency spike",
    "outage", "degraded", "unstable", "intermittent", "flapping",
}

# Intents that should never be inherited — always re-classify
NON_INHERITABLE = {"general", "unknown", "find_resource"}


def _has_friction_keywords(query: str) -> bool:
    """Return True if query contains active problem indicators."""
    q_lower = query.lower()
    return any(kw in q_lower for kw in FRICTION_KEYWORDS)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Fast cosine similarity between two embedding vectors."""
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


async def _embed(text: str) -> list[float]:
    """Embed a single string using the same model as the retriever."""
    resp = await client.embeddings.create(
        model      = "text-embedding-3-small",
        input      = text,
        dimensions = 1024,
    )
    return resp.data[0].embedding


# ── Intent prompt ─────────────────────────────────────────────────────────────
INTENT_PROMPT = """You are classifying a search query for the Equinix resource library.

Equinix products: {products}
Equinix use cases: {use_cases}

Classify into exactly one intent:
- find_resource:   user wants a specific document — whitepaper, blueprint, case study, data sheet
- evaluate_specs:  user wants technical specifications, numbers, speeds, SLAs, capacity, pricing
- compare:         user is comparing two or more Equinix products or approaches
- troubleshoot:    user has an ACTIVE problem — something is broken, failing, or not working
- learn_concept:   user wants to understand a concept or how something works
- general:         vague, too broad, or doesn't fit the above

Return ONLY valid JSON:
{{
  "intent":            "<intent>",
  "confidence":        <0.0-1.0>,
  "rewritten_query":   "<retrieval-optimised version>",
  "detected_products": [],
  "detected_use_case": "",
  "content_type_hint": "<data-sheets|whitepapers|blueprints|case-studies|solution-briefs|any>"
}}

Disambiguation rules:
evaluate_specs triggers — query wants to MEASURE or COMPARE a spec:
  "how fast", "how much bandwidth", "what are the options", "supported speeds",
  "throughput specs", "bandwidth tiers", "show me specs", "what Gbps options",
  "maximum throughput", transmit, Gbps, Mbps — WITHOUT "what is" framing

evaluate_specs also triggers for commercial/pricing queries regardless of "what is" framing:
  "what is the pricing", "what is the cost", "what is the SLA", "what is the contract",
  "what is the cross-connect pricing", "enterprise deployment contract", "pricing and contract",
  "what does it cost", "how much does" → always evaluate_specs NOT learn_concept
learn_concept triggers for conceptual "what is" queries ONLY when no pricing/spec/cost signal:
  "what is Equinix Fabric", "what is colocation", "what is Network Edge" → learn_concept
  EXCEPTION: if query contains pricing, cost, SLA, contract, speed, Gbps → evaluate_specs

troubleshoot triggers — query describes an ACTIVE problem:
  error, drop, dropping, slow, timeout, fail, not working, packet loss,
  cannot, issue, broken, keeps, disconnecting, down, unreachable, spike

Other rules:
- "what is X / how does X work / explain X" → learn_concept
- "show me / find me / I need a [doc type]" → find_resource
- CRITICAL for find_resource rewritten_query: STRIP document-type words
  ("case study", "case studies", "customer story", "customer stories", "success story",
   "whitepaper", "blueprint", "data sheet", "find", "show me", "related to", "about")
  and keep ONLY the topic/industry/subject. The resource_type is captured separately in
  content_type_hint, so the rewritten_query must be the bare topic for relevance matching.
  Examples:
    "case studies related to banking" -> rewritten_query: "banking financial services"
    "customer stories for AI" -> rewritten_query: "artificial intelligence AI"
    "case studies for media and healthcare" -> rewritten_query: "media healthcare"
- "X vs Y / compare / difference between"   → compare
- Asking WHAT specs ARE → evaluate_specs, never troubleshoot
- Only troubleshoot when user describes active failure they are experiencing
- If confidence < 0.70, use general

Query: {query}"""


# ── Result dataclass ──────────────────────────────────────────────────────────
@dataclass
class IntentResult:
    intent:             str
    confidence:         float
    rewritten_query:    str
    original_query:     str
    detected_products:  list[str]  = field(default_factory=list)
    detected_use_case:  str        = ""
    content_type_hint:  str        = "any"
    metadata_filter:    dict       = field(default_factory=dict)
    top_k:              int        = 5
    rrf_alpha:          float      = 0.7
    namespaces:         list[str]  = field(default_factory=lambda: ["technical", "business"])
    inherited:          bool       = False   # True = coherence gate fired
    similarity:         float      = 0.0     # cosine similarity to previous query


def _build_metadata_filter(
    intent:            str,
    detected_products: list[str],
    content_type_hint: str,
    confidence:        float,
) -> dict:
    """Build Pinecone pre-filter. Max 2 hard filters for recall safety."""
    filters = {}

    base = INTENT_FILTERS.get(intent, {})
    filters.update(base)

    # Product filter — only when confident and specific product named
    if detected_products and confidence >= 0.80:
        if intent in ("find_resource", "evaluate_specs", "troubleshoot", "learn_concept"):
            filters["primary_product"] = {"$in": detected_products}

    # Content type filter — only for find_resource with high confidence
    if (content_type_hint != "any"
            and confidence >= 0.85
            and intent == "find_resource"):
        # Normalise plural → singular. rstrip("s") breaks on -ies words
        # ("case-studies" -> "case-studie"), so map known types explicitly.
        _RTYPE_MAP = {
            "case-studies":   "case-study",
            "success-stories":"success-story",
            "data-sheets":    "data-sheet",
            "whitepapers":    "whitepaper",
            "blueprints":     "blueprint",
            "solution-briefs":"solution-brief",
        }
        rtype = _RTYPE_MAP.get(content_type_hint, content_type_hint.rstrip("s"))
        filters["resource_type"] = {"$eq": rtype}

    # Only filter on enriched chunks when applying metadata filters
    if filters:
        filters["enriched"] = {"$eq": True}

    return filters


def _make_default(query: str) -> IntentResult:
    """Safe fallback — general intent, no filters."""
    params = RETRIEVAL_PARAMS["general"]
    return IntentResult(
        intent          = "general",
        confidence      = 1.0,
        rewritten_query = query,
        original_query  = query,
        top_k           = params["top_k"],
        rrf_alpha       = params["rrf_alpha"],
        namespaces      = params["namespaces"],
    )


# ── Core detect_intent ────────────────────────────────────────────────────────
async def detect_intent(
    query:         str,
    last_query:    str        = "",
    last_intent:   str        = "",
    last_embedding: list[float] = None,
    retries:       int        = 1,
) -> IntentResult:
    """
    Classify query intent with coherence gate.

    Args:
        query:          Current query (already translated to English)
        last_query:     Previous query in session (for coherence gate)
        last_intent:    Previous intent (to inherit if similar enough)
        last_embedding: Previous query embedding (avoids re-embed if cached)
        retries:        LLM retry attempts

    Returns:
        IntentResult — always returns, falls back to general on failure.
        result.inherited=True means coherence gate fired (no LLM call).
    """
    # ── Default for short/empty queries ──────────────────────────────────────
    if len(query.split()) <= 2:
        log.debug(f"Short query ({len(query.split())} words) — skipping classification")
        return _make_default(query)

    # ── Layer 1: Coherence gate ───────────────────────────────────────────────
    if (last_query
            and last_intent
            and last_intent not in NON_INHERITABLE
            and not _has_friction_keywords(query)):
        try:
            # Embed current query
            current_emb = await _embed(query)

            # Use cached embedding if available, else embed previous query
            prev_emb = last_embedding or await _embed(last_query)

            similarity = _cosine_similarity(current_emb, prev_emb)

            if similarity >= COHERENCE_THRESHOLD:
                params = RETRIEVAL_PARAMS.get(last_intent, RETRIEVAL_PARAMS["general"])

                log.info(
                    f"Intent inherited: {last_intent} "
                    f"(similarity={similarity:.3f} ≥ {COHERENCE_THRESHOLD}, "
                    f"no friction keywords)"
                )

                return IntentResult(
                    intent          = last_intent,
                    confidence      = 0.90,
                    rewritten_query = query,   # no rewrite for inherited queries
                    original_query  = query,
                    top_k           = params["top_k"],
                    rrf_alpha       = params["rrf_alpha"],
                    namespaces      = params["namespaces"],
                    metadata_filter = INTENT_FILTERS.get(last_intent, {}),
                    inherited       = True,
                    similarity      = similarity,
                )
        except Exception as e:
            log.warning(f"Coherence gate failed — falling through to LLM: {e}")

    # ── Layer 2: LLM classification ───────────────────────────────────────────
    prompt = (_gp("intent", "") or INTENT_PROMPT).format(
        products  = json.dumps(EQUINIX_PRODUCTS),
        use_cases = json.dumps(EQUINIX_USE_CASES),
        query     = query,
    )

    for attempt in range(retries + 1):
        try:
            resp = await client.chat.completions.create(
                model           = "gpt-4o-mini",
                temperature     = 0,
                max_tokens      = 250,
                response_format = {"type": "json_object"},
                messages        = [{"role": "user", "content": prompt}]
            )

            raw        = json.loads(resp.choices[0].message.content)
            intent     = raw.get("intent", "general")
            confidence = float(raw.get("confidence", 0.8))

            # Deterministic override: case study / customer story / success story
            # queries are ALWAYS find_resource (LLM waffles by phrasing).
            if _CASESTUDY_RE.search(query):
                intent = "find_resource"
                confidence = max(confidence, 0.90)
                raw["content_type_hint"] = ("success-stories"
                    if "success" in query.lower() else "case-studies")
            # Validate
            if intent not in RETRIEVAL_PARAMS:
                intent     = "general"
                confidence = 1.0

            if confidence < 0.70:
                intent = "general"

            detected_products = [
                p for p in raw.get("detected_products", [])
                if p in EQUINIX_PRODUCTS
            ]

            detected_use_case = raw.get("detected_use_case", "")
            if detected_use_case not in EQUINIX_USE_CASES:
                detected_use_case = ""

            content_type_hint = raw.get("content_type_hint", "any")

            rewritten = (
                raw.get("rewritten_query", query)
                if confidence >= 0.75 else query
            )

            metadata_filter = _build_metadata_filter(
                intent, detected_products, content_type_hint, confidence
            )

            params = RETRIEVAL_PARAMS[intent]

            result = IntentResult(
                intent            = intent,
                confidence        = confidence,
                rewritten_query   = rewritten,
                original_query    = query,
                detected_products = detected_products,
                detected_use_case = detected_use_case,
                content_type_hint = content_type_hint,
                metadata_filter   = metadata_filter,
                top_k             = params["top_k"],
                rrf_alpha         = params["rrf_alpha"],
                namespaces        = params["namespaces"],
                inherited         = False,
                similarity        = 0.0,
            )

            log.info(
                f"Intent: {intent} ({confidence:.0%}) | "
                f"products={detected_products} | "
                f"rewritten='{rewritten[:50]}' | "
                f"inherited=False"
            )

            return result

        except Exception as e:
            if attempt == retries:
                log.warning(f"Intent detection failed: {e}")
                return _make_default(query)
            await asyncio.sleep(0.5)

    return _make_default(query)


# ── Compare intent helpers ────────────────────────────────────────────────────
def get_compare_queries(
    intent_result: IntentResult,
    embedding_fn,
) -> Optional[tuple]:
    """
    For compare intent with 2+ detected products — return parallel embeddings.
    Returns (product_a, emb_a, product_b, emb_b) or None.
    """
    if intent_result.intent != "compare":
        return None
    if len(intent_result.detected_products) < 2:
        return None

    prod_a = intent_result.detected_products[0]
    prod_b = intent_result.detected_products[1]
    query  = intent_result.rewritten_query

    emb_a = embedding_fn(f"{query} {prod_a}")
    emb_b = embedding_fn(f"{query} {prod_b}")

    return (prod_a, emb_a, prod_b, emb_b)
