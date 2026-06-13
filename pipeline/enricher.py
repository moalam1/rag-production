"""
pipeline/enricher.py — LLM-powered chunk enrichment for Equinix RAG.

Sits between page_parser.py and ingester.py in the ingest pipeline.
Tags each chunk with structured metadata before embedding and upsert.

Architecture: tiered trust model
  Tier 1 — structural overrides  (100% accurate, free, no LLM)
  Tier 2 — high-confidence tags  (85-95%, hard filter safe)
  Tier 3 — medium-confidence     (75-80%, soft boost only)
  Tier 4 — store but don't filter (~55-65%, future use)

Interface:
    from pipeline.enricher import enrich_chunks_batch

    enriched = await enrich_chunks_batch(
        chunks        = chunks,           # list[dict] with 'text' key
        title         = parsed.title,
        resource_type = parsed.resource_type,
        url           = parsed.url,
        aem_tags      = list(parsed.tags),
    )
    # Each chunk gets an 'enrichment' key added

Cost:  ~$0.0001/chunk  (gpt-4o-mini, ~300 tokens)
       ~$1.00 full corpus re-ingest (16,850 vectors, one-time)
       ~$0.02/night nightly delta (changed docs only)
"""

import asyncio
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from openai import AsyncOpenAI

log = logging.getLogger(__name__)
client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ── Registry ──────────────────────────────────────────────────────────────────
REGISTRY_PATH = Path("/home/ssm-user/rag-production/config/products.json")

# ── Equinix official taxonomy (nav menu — source of truth) ───────────────────

OFFICIAL_PRODUCTS = [
    "Equinix Fabric",
    "Equinix Fabric Cloud Router",
    "Equinix Metal",
    "Equinix Precision Time",
    "Internet Access",
    "Managed Services",
    "Network Edge",
    "Platform Equinix",
]

# FCR is a sub-feature of Fabric but also standalone — LLM needs explicit guidance
FCR_CLARIFICATION = (
    "'Equinix Fabric Cloud Router' is a separate product from 'Equinix Fabric'. "
    "If a chunk is primarily about Cloud Router / Layer 3 routing via Fabric, "
    "tag it as 'Equinix Fabric Cloud Router', not 'Equinix Fabric'."
)

OFFICIAL_USE_CASES = [
    "application-exchange",
    "application-optimization",
    "cloud-adjacency",
    "colocation",
    "digital-transformation",
    "distributed-ai",
    "distributed-data",
    "distributed-security",
    "edge-computing",
    "edge-infrastructure",
    "high-performance-data-center",
    "hybrid-multicloud-networking",
    "ioa-network-architecture",
    "interconnection",
    "network-optimization",
    "network-modernization",
    "sustainability",
]

# Use case priority rules to resolve overlap
# Applied in prompt to reduce LLM ambiguity on overlapping cases
USE_CASE_PRIORITY_RULES = (
    "Use case priority rules (resolve ambiguity):\n"
    "- Chunk mentions specific cloud providers (AWS/Azure/GCP) → prefer 'cloud-adjacency'\n"
    "- Chunk mentions SD-WAN or branch connectivity → prefer 'network-modernization'\n"
    "- Chunk mentions AI workloads or GPU → prefer 'distributed-ai'\n"
    "- Chunk mentions sustainability/ESG/carbon → use 'sustainability' only\n"
    "- Chunk covers multiple use cases → pick the 2-3 most specific, not 'interconnection' as catch-all"
)

# ── Structural override rules (Tier 1 — 100% accurate, no LLM needed) ────────

STRUCTURAL_OVERRIDES = {
    # resource_type → forced tag values
    "blueprint": {
        "has_architecture": True,
        "technical_depth":  "engineer",
        "content_role":     "architecture",
    },
    "data-sheet": {
        "has_specs":       True,
        "content_role":    "technical-detail",
        "technical_depth": "practitioner",
    },
    "playbook": {
        "content_role":    "technical-detail",
        "technical_depth": "engineer",
    },
    "case-study": {
        "content_role":    "case-example",
        "technical_depth": "practitioner",
    },
    "analyst-report": {
        "technical_depth": "practitioner",  # analyst reports look exec but have depth
        "content_role":    "overview",
    },
    "whitepaper": {
        "technical_depth": "practitioner",
    },
    "infographic": {
        "technical_depth": "executive",
        "content_role":    "overview",
    },
    "solution-brief": {
        "technical_depth": "practitioner",
        "content_role":    "overview",
    },
    "infopaper": {
        "technical_depth": "executive",
        "content_role":    "overview",
    },
    "success-story": {
        "content_role":    "case-example",
        "technical_depth": "executive",
    },
    "video": {
        "technical_depth": "executive",
    },
    "webinar": {
        "technical_depth": "practitioner",
    },
}

# ── Audience groupings for broad filtering ────────────────────────────────────

AUDIENCE_GROUPS = {
    "finance": {
        "finance", "financial-services", "retail-banking",
        "wealth-management", "insurance", "payments",
        "e-trade", "electronic-trading",
    },
    "technology": {
        "ai-machine-learning", "cloud-service-providers", "cloud-services",
        "it-security", "big-data", "managed-service-providers",
        "network-service-providers", "networks", "iot",
    },
    "government": {"federal-government", "government"},
    "industry": {
        "healthcare", "manufacturing", "energy-oil-gas",
        "pharmaceutical", "automotive-cvst", "transportation",
        "legal", "professional-services",
    },
    "retail":   {"retail", "consumer-retail", "e-commerce", "gaming"},
    "media":    {"content-digital-media", "5g-telecommunications"},
}

# ── Validation sets ───────────────────────────────────────────────────────────

VALID_PRODUCTS    = set(OFFICIAL_PRODUCTS)
VALID_USE_CASES   = set(OFFICIAL_USE_CASES)
VALID_DEPTHS      = {"executive", "practitioner", "engineer"}
VALID_ROLES       = {
    "overview", "technical-detail", "case-example",
    "getting-started", "pricing-info", "architecture",
}
VALID_MATURITY    = {"awareness", "evaluation", "implementation", "operations"}
VALID_AUDIENCES   = {
    "5g-telecommunications", "ai-machine-learning", "automotive-cvst",
    "big-data", "cloud-service-providers", "cloud-services",
    "consumer-retail", "content-digital-media", "e-commerce",
    "e-trade", "electronic-trading", "energy-oil-gas",
    "federal-government", "finance", "financial-services",
    "gaming", "government", "healthcare", "it-security",
    "insurance", "iot", "legal", "managed-service-providers",
    "manufacturing", "network-service-providers", "networks",
    "payments", "pharmaceutical", "professional-services",
    "retail", "retail-banking", "transportation", "wealth-management",
}


# ── Registry loader ───────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_registry() -> dict:
    """Load products.json once. Falls back to hardcoded list if missing."""
    if not REGISTRY_PATH.exists():
        log.warning(
            f"Registry not found at {REGISTRY_PATH} — "
            "using hardcoded product list. Run build_product_list.py"
        )
        return {"_alias_to_canonical": {}}
    with open(REGISTRY_PATH) as f:
        data = json.load(f)
    log.info(
        f"Registry loaded: {data.get('product_count',0)} products, "
        f"{data.get('total_names',0)} names"
    )
    return data


def _get_alias_map() -> dict[str, str]:
    return _load_registry().get("_alias_to_canonical", {})


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(
    chunk_text:    str,
    title:         str,
    resource_type: str,
    url:           str,
    aem_tags:      list[str],
    structural:    dict,
) -> str:
    """
    Build enrichment prompt. Tells the LLM which fields are already
    determined by structural overrides so it doesn't waste tokens on them.
    """
    aem_str = ", ".join(aem_tags) if aem_tags else "none"

    # Tell LLM which fields are already determined — reduces hallucination
    already_known = []
    if structural.get("has_architecture"):
        already_known.append("has_architecture=true (blueprint type)")
    if structural.get("has_specs"):
        already_known.append("has_specs=true (data-sheet type)")
    if structural.get("technical_depth"):
        already_known.append(
            f"technical_depth={structural['technical_depth']} (set by document type)"
        )
    if structural.get("content_role"):
        already_known.append(
            f"content_role={structural['content_role']} (set by document type)"
        )
    already_str = (
        "Already determined by document type (do not change these):\n  " +
        "\n  ".join(already_known)
    ) if already_known else ""

    return f"""Tag this Equinix resource chunk using their official taxonomy.
Return ONLY valid JSON — no markdown, no explanation.

Document context:
  Title:         {title}
  Resource type: {resource_type}
  URL:           {url}
  AEM keywords:  {aem_str}
{already_str}

Official Equinix products — use exact names only:
{json.dumps(OFFICIAL_PRODUCTS, indent=2)}

Important: {FCR_CLARIFICATION}

Official use cases — use exact slugs only, max 3:
{json.dumps(OFFICIAL_USE_CASES, indent=2)}

{USE_CASE_PRIORITY_RULES}

Chunk text:
{chunk_text[:1500]}

Return exactly this JSON:
{{
  "primary_product":      [],
  "mentioned_products":   [],
  "use_case":             [],
  "target_audience":      [],
  "content_role":         "",
  "technical_depth":      "",
  "product_maturity":     "",
  "has_specs":            false,
  "has_architecture":     false,
  "integration_partners": []
}}

Tagging rules:
- primary_product:      products that are the MAIN focus — not just mentioned
- mentioned_products:   products referenced briefly — must not overlap with primary
- use_case:             from official list only, max 3, exact slugs
- target_audience:      industry/segment this content specifically targets
- content_role:         overview | technical-detail | case-example | getting-started | pricing-info | architecture
- technical_depth:      executive (C-suite) | practitioner (IT manager) | engineer (architect/developer)
- product_maturity:     awareness | evaluation | implementation | operations
- has_specs:            true if chunk has port speeds, SLAs, latency, pricing, throughput numbers
- has_architecture:     true if chunk describes system topology, component relationships, data flows
- integration_partners: named tech partners only — AWS, Azure, GCP, Cisco, VMware, Palo Alto etc.
"""


# ── Validation and merging ────────────────────────────────────────────────────

def _normalise_product(name: str) -> Optional[str]:
    """Map alias to canonical product name. Returns None if unknown."""
    alias_map = _get_alias_map()
    canonical = alias_map.get(name.lower())
    if canonical:
        return canonical
    if name in VALID_PRODUCTS:
        return name
    return None


def _validate_llm_output(raw: dict) -> dict:
    """
    Validate and clean raw LLM output.
    Unknown values silently dropped — never corrupt Pinecone metadata.
    Normalises product aliases to canonical names.
    """
    # Normalise and deduplicate products
    primary = list({
        p for p in
        [_normalise_product(p) for p in raw.get("primary_product", [])]
        if p
    })
    mentioned = list({
        p for p in
        [_normalise_product(p) for p in raw.get("mentioned_products", [])]
        if p and p not in primary  # never overlap with primary
    })

    # Validate audiences and derive groups
    raw_audiences   = raw.get("target_audience", [])
    valid_audiences = [a for a in raw_audiences if a in VALID_AUDIENCES]
    audience_groups = [
        group for group, members in AUDIENCE_GROUPS.items()
        if any(a in members for a in valid_audiences)
    ]

    return {
        "primary_product":      sorted(primary),
        "mentioned_products":   sorted(mentioned),
        "use_case":             [
            u for u in raw.get("use_case", [])[:3]
            if u in VALID_USE_CASES
        ],
        "target_audience":      valid_audiences,
        "audience_groups":      audience_groups,
        "content_role":         raw.get("content_role", "overview")
                                if raw.get("content_role") in VALID_ROLES
                                else "overview",
        "technical_depth":      raw.get("technical_depth", "practitioner")
                                if raw.get("technical_depth") in VALID_DEPTHS
                                else "practitioner",
        "product_maturity":     raw.get("product_maturity", "awareness")
                                if raw.get("product_maturity") in VALID_MATURITY
                                else "awareness",
        "has_specs":            bool(raw.get("has_specs", False)),
        "has_architecture":     bool(raw.get("has_architecture", False)),
        "integration_partners": [
            p for p in raw.get("integration_partners", [])
            if isinstance(p, str) and 1 < len(p) < 50
        ],
    }


def _apply_structural_overrides(
    llm_result:    dict,
    resource_type: str,
) -> dict:
    """
    Tier 1 — apply structural overrides after LLM output.
    100% accurate, derived from resource_type which is always known.
    Overrides always win over LLM inference on the same field.
    """
    overrides = STRUCTURAL_OVERRIDES.get(resource_type, {})
    merged = {**llm_result}

    for field, value in overrides.items():
        if field in ("has_specs", "has_architecture"):
            # Boolean: override only if structural says True
            # LLM can still set True on non-override types
            if value is True:
                merged[field] = True
        else:
            # String fields: structural always wins
            merged[field] = value

    return merged


def _default_metadata() -> dict:
    """
    Safe fallback when enrichment fails entirely.
    enriched=False flags chunk for re-enrichment on next ingest.
    Ingest pipeline never blocks on enrichment failure.
    """
    return {
        "primary_product":      [],
        "mentioned_products":   [],
        "use_case":             [],
        "target_audience":      [],
        "audience_groups":      [],
        "content_role":         "overview",
        "technical_depth":      "practitioner",
        "product_maturity":     "awareness",
        "has_specs":            False,
        "has_architecture":     False,
        "integration_partners": [],
        "enriched":             False,
        "enrichment_error":     "all_retries_failed",
    }


# ── Core enrichment ───────────────────────────────────────────────────────────

async def enrich_chunk(
    chunk_text:    str,
    title:         str       = "",
    resource_type: str       = "",
    url:           str       = "",
    aem_tags:      list[str] = None,
    retries:       int       = 2,
) -> dict:
    """
    Enrich a single chunk. Always returns — never raises.

    Args:
        chunk_text:    Raw text content of the chunk
        title:         parsed.title from page_parser
        resource_type: parsed.resource_type (whitepaper, blueprint etc.)
        url:           parsed.url
        aem_tags:      list(parsed.tags) — AEM keyword meta tags
        retries:       API retry attempts

    Returns:
        Enrichment dict with all tags + enriched=True/False flag.
        enriched=False means the LLM call failed — structural overrides
        are still applied, so the chunk is partially enriched.
    """
    aem_tags = aem_tags or []

    # ── Tier 1: structural overrides (always run, even on LLM failure) ──
    structural = STRUCTURAL_OVERRIDES.get(resource_type, {})

    # Skip LLM for very short chunks — not enough signal
    if len(chunk_text.split()) < 15:
        log.debug(f"Short chunk ({len(chunk_text.split())} words) — structural only")
        result = _default_metadata()
        result.update(structural)
        result["enriched"]         = True   # structural overrides count as enriched
        result["enrichment_error"] = "short_chunk_structural_only"
        return result

    # ── Tier 2-4: LLM enrichment ─────────────────────────────────────────
    prompt = _build_prompt(
        chunk_text, title, resource_type,
        url, aem_tags, structural
    )

    llm_result = None
    last_error = None

    for attempt in range(retries + 1):
        try:
            resp = await client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                max_tokens=350,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}]
            )

            raw        = json.loads(resp.choices[0].message.content)
            llm_result = _validate_llm_output(raw)
            break

        except Exception as e:
            last_error = str(e)
            if attempt < retries:
                wait = 1.5 ** attempt
                log.debug(
                    f"Enrichment attempt {attempt+1} failed "
                    f"({url[:50]}): {e} — retrying in {wait:.1f}s"
                )
                await asyncio.sleep(wait)
            else:
                log.warning(
                    f"Enrichment failed after {retries+1} attempts "
                    f"({url[:60]}): {e}"
                )

    # ── Merge: structural overrides always win over LLM ──────────────────
    if llm_result is not None:
        merged = _apply_structural_overrides(llm_result, resource_type)
        merged["enriched"]         = True
        merged["enrichment_error"] = None

        log.debug(
            f"Enriched ✓  primary={merged['primary_product']}  "
            f"use_case={merged['use_case']}  "
            f"depth={merged['technical_depth']}  "
            f"has_specs={merged['has_specs']}"
        )
        return merged

    else:
        # LLM failed — return structural overrides + defaults
        fallback = _default_metadata()
        fallback.update(structural)
        fallback["enriched"]         = False
        fallback["enrichment_error"] = last_error or "unknown"
        return fallback


async def enrich_chunks_batch(
    chunks:        list[dict],
    title:         str       = "",
    resource_type: str       = "",
    url:           str       = "",
    aem_tags:      list[str] = None,
    concurrency:   int       = 5,
) -> list[dict]:
    """
    Enrich a batch of chunks concurrently with rate limiting.

    Args:
        chunks:       list of chunk dicts — must have 'text' key
        title:        Document title (parsed.title)
        resource_type: Document type (parsed.resource_type)
        url:          Source URL (parsed.url)
        aem_tags:     AEM keyword tags (list(parsed.tags))
        concurrency:  Max parallel LLM calls (5 is safe for gpt-4o-mini)

    Returns:
        Same chunks with 'enrichment' key added to each.
    """
    aem_tags = aem_tags or []
    sem      = asyncio.Semaphore(concurrency)

    async def _enrich_one(chunk: dict) -> dict:
        async with sem:
            enrichment = await enrich_chunk(
                chunk_text    = chunk.get("text", ""),
                title         = title,
                resource_type = resource_type,
                url           = url,
                aem_tags      = aem_tags,
            )
            return {**chunk, "enrichment": enrichment}

    results = await asyncio.gather(*[_enrich_one(c) for c in chunks])

    # ── Summary log ───────────────────────────────────────────────────────
    total         = len(results)
    fully_enriched = sum(1 for r in results
                         if r["enrichment"].get("enriched")
                         and not r["enrichment"].get("enrichment_error"))
    structural_only = sum(1 for r in results
                          if r["enrichment"].get("enrichment_error") ==
                          "short_chunk_structural_only")
    failed         = sum(1 for r in results
                         if not r["enrichment"].get("enriched"))

    log.info(
        f"Batch enrichment — "
        f"fully enriched: {fully_enriched}/{total}  "
        f"structural only: {structural_only}/{total}  "
        f"failed: {failed}/{total}"
    )

    if failed > 0:
        log.warning(
            f"{failed} chunks have enriched=False — "
            "will retry on next ingest of this document"
        )

    # Log product coverage for monitoring
    all_products = []
    for r in results:
        all_products.extend(r["enrichment"].get("primary_product", []))
    if all_products:
        from collections import Counter
        top = Counter(all_products).most_common(3)
        log.info(f"Top products in batch: {top}")

    return results


# ── Utility — Pinecone metadata merge ────────────────────────────────────────

def merge_enrichment_into_metadata(
    existing_metadata: dict,
    enrichment:        dict,
) -> dict:
    """
    Merge enrichment dict into existing Pinecone chunk metadata.
    Call this in ingester.py before upsert.

    Args:
        existing_metadata: Current metadata dict for the chunk
        enrichment:        Output from enrich_chunk()

    Returns:
        Merged metadata dict ready for Pinecone upsert.
    """
    # Fields to exclude from Pinecone metadata
    # (enrichment_error is internal — don't store in vector DB)
    exclude = {"enrichment_error"}

    enrichment_clean = {
        k: v for k, v in enrichment.items()
        if k not in exclude
    }

    return {**existing_metadata, **enrichment_clean}


# ── Utility — audit unenriched chunks ────────────────────────────────────────

async def find_unenriched_chunks(
    pinecone_index,
    namespaces:  list[str] = None,
    sample_size: int       = 100,
) -> dict[str, list[str]]:
    """
    Query Pinecone for chunks where enriched=False or enriched missing.
    Returns dict of namespace → list of page_urls needing re-ingest.

    Use after initial deployment to find chunks that failed enrichment.
    Run weekly to catch any new failures.

    Args:
        pinecone_index: Initialised Pinecone Index object
        namespaces:     Namespaces to check (default: technical, business)
        sample_size:    Chunks to sample per namespace

    Returns:
        {"technical": ["url1", "url2"], "business": [...]}
    """
    import random
    namespaces = namespaces or ["technical", "business"]
    results    = {}

    for ns in namespaces:
        try:
            dim = 1024
            raw = [random.gauss(0, 1) for _ in range(dim)]
            mag = sum(x**2 for x in raw) ** 0.5
            vec = [x / mag for x in raw]

            result = pinecone_index.query(
                vector=vec,
                top_k=sample_size,
                include_metadata=True,
                namespace=ns,
                filter={"enriched": {"$ne": True}},
            )

            urls = list({
                m["metadata"].get("page_url", "")
                for m in result.get("matches", [])
                if m["metadata"].get("page_url")
            })

            results[ns] = urls
            log.info(
                f"Namespace '{ns}': {len(urls)} URLs with "
                f"unenriched chunks"
            )

        except Exception as e:
            log.error(f"Could not query namespace '{ns}': {e}")
            results[ns] = []

    total_urls = sum(len(v) for v in results.values())
    log.info(f"Total URLs needing re-ingest: {total_urls}")
    return results
