"""
pipeline/fetchers.py — Source abstraction for section content discovery (L1b-source).

discover_urls(section_config) returns the list of page URLs to ingest for a
section, dispatching on the section's `source_type`:

  - "sitemap" : fetch the section's sitemap_url, filter by url_include/url_exclude
                substring patterns. (Equinix publishes proper sitemaps for all
                content trees, so this is the primary, robust path.)
  - "crawl"   : currently aliases to sitemap discovery (Equinix content is in
                sitemaps). A link-scraping crawler is a future fallback for
                sections genuinely not covered by any sitemap.

This sits ON TOP of the proven L1b-write path: callers do
    for url in discover_urls(cfg):
        route_and_ingest(parse_page(url), section=<name>)
so discovery feeds the existing section-aware writer.
"""
import logging

from pipeline.crawler import _fetch_sitemap_urls, SITEMAP_URL

log = logging.getLogger(__name__)


def discover_urls(section_config: dict) -> list[str]:
    """Discover page URLs for a section, per its source_type + include/exclude filters.

    section_config keys used:
      source_type : "sitemap" | "crawl"  (default "sitemap")
      sitemap_url : which sitemap to fetch (default = resources sitemap)
      url_include : list of substrings; a URL is kept if it contains ANY
                    (empty list = keep all)
      url_exclude : list of substrings; a URL is dropped if it contains ANY

    Returns a filtered, de-duplicated list of URLs (order preserved).
    """
    st = section_config.get("source_type", "sitemap")
    if st not in ("sitemap", "crawl"):
        raise ValueError(f"unknown source_type: {st!r}")

    sitemap = section_config.get("sitemap_url") or SITEMAP_URL
    inc = section_config.get("url_include", []) or []
    exc = section_config.get("url_exclude", []) or []

    urls, errors = _fetch_sitemap_urls(sitemap)
    if errors:
        log.warning("discover_urls: sitemap fetch errors for %s: %s", sitemap, errors)

    seen, filtered = set(), []
    for u in urls:
        if inc and not any(p in u for p in inc):
            continue
        if exc and any(p in u for p in exc):
            continue
        if u not in seen:
            seen.add(u)
            filtered.append(u)

    log.info(
        "discover_urls: %s -> %d total, %d after include=%s exclude=%s",
        sitemap, len(urls), len(filtered), inc, exc,
    )
    return filtered
