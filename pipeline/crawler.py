"""
pipeline/crawler.py — Discover Equinix resource URLs via sitemap.

Replaces Playwright approach (blocked by Akamai on EC2) with sitemap parsing.
equinix.com publishes sitemap-resources.xml containing all 1,281+ resource URLs.
Clean, fast, no JS rendering, no bot detection issues.

Two modes:
  - crawl_all()        — all resource types from sitemap
  - crawl_type(rtype)  — filter to a specific resource type
"""
import logging
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import date
from typing import Optional

import httpx

log = logging.getLogger(__name__)

SITEMAP_URL  = "https://www.equinix.com/sitemap-resources.xml"
BASE_URL     = "https://www.equinix.com"
SITEMAP_NS   = {"sm": "https://www.sitemaps.org/schemas/sitemap/0.9"}
CRAWL_PREFIX = "crawl"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# All known resource types — expanded from sitemap discovery
RESOURCE_TYPES = {
    "whitepapers",
    "analyst-reports",
    "data-sheets",
    "case-studies",
    "solution-briefs",
    "blueprints",
    "playbooks",
    "videos",
    "webinars",
    "infopapers",
    "product-documents",
    "infographics",
    "success-stories",
}

# Pages to exclude — gated/thank-you/listing pages with no real content
EXCLUDE_SLUGS = {"thank-you", "thank_you", "thankyou", "download", "gated"}


def crawl_all(run_id: str = "", redis_client=None) -> dict:
    """
    Fetch all resource URLs from sitemap and return deduplicated list.

    Args:
        run_id:       Unique ID for this crawl run (defaults to today's date)
        redis_client: Optional Redis client for within-run dedup

    Returns:
        {
          "run_id": "2026-05-18",
          "total_in_sitemap": 1281,
          "queued": 245,
          "skipped_dedup": 0,
          "urls": [...],
          "by_type": {"whitepapers": 159, ...},
          "errors": []
        }
    """
    run_id = run_id or date.today().isoformat()
    log.info("Starting sitemap crawl — run_id=%s", run_id)

    all_urls, errors = _fetch_sitemap_urls()
    if not all_urls:
        return {
            "run_id": run_id, "total_in_sitemap": 0,
            "queued": 0, "skipped_dedup": 0,
            "urls": [], "by_type": {}, "errors": errors,
        }

    total_in_sitemap = len(all_urls)

    # Filter to valid resource pages
    filtered = _filter_urls(all_urls)
    log.info("Sitemap: %d total → %d after filtering", total_in_sitemap, len(filtered))

    # Redis dedup — skip URLs already seen this run
    new_urls   = _redis_dedup(filtered, run_id, redis_client)
    skipped    = len(filtered) - len(new_urls)

    by_type = Counter(
        u.rstrip("/").split("/")[4]
        for u in new_urls
        if len(u.rstrip("/").split("/")) >= 6
    )

    log.info(
        "Crawl complete — queued=%d skipped=%d errors=%d",
        len(new_urls), skipped, len(errors)
    )

    return {
        "run_id":           run_id,
        "total_in_sitemap": total_in_sitemap,
        "queued":           len(new_urls),
        "skipped_dedup":    skipped,
        "urls":             new_urls,
        "by_type":          dict(by_type),
        "errors":           errors,
    }


def crawl_type(resource_type: str, run_id: str = "", redis_client=None) -> list[str]:
    """
    Return URLs for a specific resource type only.

    Args:
        resource_type: e.g. "whitepapers", "videos", "analyst-reports"
        run_id:        Crawl run ID for dedup
        redis_client:  Optional Redis client

    Returns:
        List of resource URLs for that type.
    """
    run_id = run_id or date.today().isoformat()
    all_urls, _ = _fetch_sitemap_urls()
    filtered    = _filter_urls(all_urls, resource_type=resource_type)
    return _redis_dedup(filtered, run_id, redis_client)


def get_sitemap_stats() -> dict:
    """
    Return breakdown of all resource types in sitemap without filtering.
    Useful for discovery and monitoring.
    """
    all_urls, errors = _fetch_sitemap_urls()
    resource_urls    = [
        u for u in all_urls
        if u.startswith(f"{BASE_URL}/resources/")
        and len(u.rstrip("/").split("/")) >= 6
    ]
    by_type = Counter(
        u.rstrip("/").split("/")[4] for u in resource_urls
    )
    return {
        "total":    len(all_urls),
        "by_type":  dict(sorted(by_type.items(), key=lambda x: -x[1])),
        "errors":   errors,
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _fetch_sitemap_urls() -> tuple[list[str], list[str]]:
    """Fetch and parse sitemap-resources.xml. Returns (urls, errors)."""
    try:
        with httpx.Client(headers=HEADERS, timeout=30, follow_redirects=True) as client:
            resp = client.get(SITEMAP_URL)
            resp.raise_for_status()

        root     = ET.fromstring(resp.text)
        all_urls = [
            loc.text.strip()
            for loc in root.findall(".//sm:loc", SITEMAP_NS)
            if loc.text and loc.text.strip()
        ]
        log.info("Fetched sitemap: %d URLs", len(all_urls))
        return all_urls, []

    except httpx.HTTPError as e:
        log.error("Sitemap fetch failed: %s", e)
        return [], [str(e)]
    except ET.ParseError as e:
        log.error("Sitemap parse failed: %s", e)
        return [], [str(e)]


def _filter_urls(
    urls:          list[str],
    resource_type: Optional[str] = None,
) -> list[str]:
    """
    Filter sitemap URLs to valid English resource pages.

    Rules:
      - Must start with https://www.equinix.com/resources/
      - Must have at least 3 path segments (resources/type/slug)
      - Type must be in RESOURCE_TYPES
      - Slug must not be in EXCLUDE_SLUGS
      - If resource_type specified, filter to that type only
    """
    results = []
    for url in urls:
        url = url.rstrip("/")

        # English only — skip localised URLs
        if not url.startswith(f"{BASE_URL}/resources/"):
            continue

        parts = url.split("/")
        # parts: ['https:', '', 'www.equinix.com', 'resources', 'type', 'slug']
        if len(parts) < 6:
            continue

        rtype = parts[4]
        slug  = parts[5]

        # Must be a known resource type
        if rtype not in RESOURCE_TYPES:
            continue

        # Skip gated/thank-you pages
        if slug.lower() in EXCLUDE_SLUGS:
            continue

        # Filter by type if specified
        if resource_type and rtype != resource_type:
            continue

        results.append(url)

    return results


def _redis_dedup(urls: list[str], run_id: str, redis_client) -> list[str]:
    """
    Filter URLs already seen in this crawl run using Redis SADD.
    Returns only new URLs. If no Redis client, returns all URLs.
    """
    if not redis_client or not urls:
        return urls

    key      = f"{CRAWL_PREFIX}:{run_id}"
    new_urls = [url for url in urls if redis_client.sadd(key, url)]
    redis_client.expire(key, 90000)  # 25hr TTL
    return new_urls
