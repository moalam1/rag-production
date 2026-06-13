"""
Option 1 — Scrape equinix.com product/services pages for product names.

How it works:
  1. Fetches key Equinix product/services pages
  2. Parses headings, nav items, product cards for name signals
  3. LLM cleans the noisy HTML-extracted text into product names
  4. Returns raw_products: list[str]

Cost:  ~$0.015–0.020 per run (GPT-4o for noise filtering)
Time:  ~30–45 seconds
Run:   Quarterly as a sanity check — catches brand-new products
       before they appear in indexed content
"""

import asyncio
import json
import logging
import os
import re

import httpx
from bs4 import BeautifulSoup
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)
client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Pages most likely to list all Equinix products
PRODUCT_PAGES = [
    "https://www.equinix.com/services",
    "https://www.equinix.com/interconnection-services",
    "https://www.equinix.com/data-centers",
    "https://www.equinix.com/digital-infrastructure-services",
    "https://www.equinix.com/network-edge",
]

# CSS selectors that typically contain product names on marketing sites
PRODUCT_SELECTORS = [
    "h1", "h2", "h3", "h4",
    "nav a", ".product-name", ".service-name",
    ".card-title", ".tile-title", "[data-product]",
    ".hero-title", ".section-title",
]

# Rough filter — Equinix products usually contain these signals
EQUINIX_SIGNALS = {
    "equinix", "fabric", "metal", "ibx", "xscale",
    "network edge", "smartkey", "precision", "connect",
    "internet exchange", "ose", "infrastructure"
}

CLEANING_PROMPT = """You are cleaning a raw list of text fragments scraped from 
Equinix's website. Many items are navigation labels, CTAs, or generic phrases.

Your job: extract ONLY genuine Equinix product or service names.

Rules:
- Keep: Equinix-branded products (Equinix Fabric, Metal, IBX, Network Edge, etc.)
- Keep: Short names that are clearly product references (Fabric, IBX, xScale)
- Remove: Generic phrases (Learn more, Get started, Data Center, Cloud)  
- Remove: Competitor names (AWS, Azure, Google Cloud)
- Remove: Page titles and navigation labels unrelated to specific products
- Remove: Partner or technology names (BGP, MPLS, SD-WAN) unless Equinix-branded

Raw fragments:
{fragments}

Return ONLY a valid JSON array of clean product name strings. No markdown.
Example: ["Equinix Fabric", "Fabric", "Network Edge", "IBX Data Centers"]"""


async def _fetch_page(http: httpx.AsyncClient, url: str) -> str:
    """Fetch a single page. Returns empty string on failure."""
    try:
        resp = await http.get(url, timeout=20, follow_redirects=True)
        if resp.status_code == 200:
            logger.info(f"  Fetched: {url} ({len(resp.text):,} chars)")
            return resp.text
        logger.warning(f"  {url} → HTTP {resp.status_code}")
        return ""
    except Exception as e:
        logger.warning(f"  {url} → failed: {e}")
        return ""


def _extract_text_candidates(html: str, source_url: str) -> list[str]:
    """
    Parse HTML and extract text fragments that might be product names.
    Returns de-noised list of short text strings.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove script/style noise before parsing
    for tag in soup(["script", "style", "footer", "meta"]):
        tag.decompose()

    candidates = set()

    for selector in PRODUCT_SELECTORS:
        for element in soup.select(selector):
            text = element.get_text(separator=" ", strip=True)
            text = re.sub(r"\s+", " ", text).strip()

            # Filter by length — product names are short
            if 2 <= len(text.split()) <= 7 and len(text) < 60:
                # Must contain an Equinix signal word
                if any(sig in text.lower() for sig in EQUINIX_SIGNALS):
                    candidates.add(text)

    logger.info(f"  Extracted {len(candidates)} candidates from {source_url}")
    return sorted(candidates)


async def _llm_clean_candidates(candidates: list[str]) -> list[str]:
    """
    Use GPT-4o to filter noise and return clean product names.
    GPT-4o (not mini) here — this is a quarterly job where quality matters.
    """
    if not candidates:
        return []

    try:
        resp = await client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            max_tokens=600,
            messages=[{
                "role": "user",
                "content": CLEANING_PROMPT.format(
                    fragments=json.dumps(candidates, indent=2)
                )
            }]
        )

        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        products = json.loads(raw)

        if not isinstance(products, list):
            logger.warning("LLM cleaning returned unexpected shape")
            return []

        cleaned = [
            p.strip() for p in products
            if isinstance(p, str) and 1 <= len(p.split()) <= 7
        ]

        logger.info(f"LLM cleaning: {len(candidates)} fragments → {len(cleaned)} products")
        return cleaned

    except Exception as e:
        logger.warning(f"LLM cleaning failed: {e} — returning raw candidates")
        return candidates  # graceful fallback


async def run_option1() -> list[str]:
    """
    Main entry point for Option 1.

    Returns:
        raw_products: deduplicated list of product name strings
    """
    logger.info("Option 1: Scraping equinix.com product pages")
    logger.info("Note: Run quarterly — this is the most expensive option (~$0.017/run)")

    # ── Step 1: Fetch all product pages concurrently ──────────────────────
    async with httpx.AsyncClient(headers=HEADERS) as http:
        pages = await asyncio.gather(*[
            _fetch_page(http, url) for url in PRODUCT_PAGES
        ])

    # ── Step 2: Extract candidates from each page ─────────────────────────
    all_candidates: set[str] = set()
    for html, url in zip(pages, PRODUCT_PAGES):
        if html:
            candidates = _extract_text_candidates(html, url)
            all_candidates.update(candidates)

    logger.info(f"Total raw candidates across all pages: {len(all_candidates)}")

    if not all_candidates:
        logger.warning("No candidates found — equinix.com may have changed structure")
        return []

    # ── Step 3: LLM cleaning pass ─────────────────────────────────────────
    sorted_candidates = sorted(all_candidates)
    cleaned = await _llm_clean_candidates(sorted_candidates)

    sorted_products = sorted(set(cleaned))

    logger.info(
        f"Option 1 complete: {len(sorted_products)} clean product names"
    )

    return sorted_products
