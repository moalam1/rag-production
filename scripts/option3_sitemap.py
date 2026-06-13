"""
Option 3 — Extract product names from Equinix sitemap.
Uses the same crawler logic already working in pipeline/crawler.py
"""
import logging
import re
import sys
sys.path.insert(0, "/home/ssm-user/rag-production")

from pipeline.crawler import crawl_all

logger = logging.getLogger(__name__)

SLUG_PATTERNS = [
    (r"equinix-fabric|/fabric/",      "Equinix Fabric",            ["Fabric", "EF"]),
    (r"equinix-metal|/metal/",        "Equinix Metal",             ["Metal", "Bare Metal"]),
    (r"network-edge",                 "Network Edge",              ["NE"]),
    (r"equinix-connect",              "Equinix Connect",           ["Connect"]),
    (r"ibx|colocation",              "IBX Data Centers",          ["IBX"]),
    (r"xscale",                       "xScale Data Centers",       ["xScale"]),
    (r"precision-time|ptaas",         "Equinix Precision Time",    ["PTaaS"]),
    (r"smartkey",                     "SmartKey",                  []),
    (r"internet-exchange|equinix-ix", "Equinix Internet Exchange", ["IX"]),
    (r"network-hub",                  "Equinix Network Hub",       ["Network Hub"]),
    (r"cloud-exchange|equinix-cx",    "Cloud Exchange",            ["ECX"]),
]
COMPILED = [(re.compile(p, re.IGNORECASE), c, a) for p, c, a in SLUG_PATTERNS]


async def run_option3() -> list[str]:
    """
    Extract product names from the 1,282 URLs already fetched by crawl_all().
    Reuses the exact same Chrome UA headers that bypass Akamai on EC2.
    No 403s — this is your existing working crawler.
    """
    logger.info("Option 3: Extracting product signals from sitemap via crawler")

    # crawl_all() is synchronous — call directly
    result = crawl_all()
    urls = result["urls"]
    logger.info(f"  crawl_all() returned {len(urls)} URLs")
    logger.info(f"  by_type: {result['by_type']}")

    found: dict[str, set] = {}
    for url in urls:
        for pattern, canonical, aliases in COMPILED:
            if pattern.search(url.lower()):
                if canonical not in found:
                    found[canonical] = set()
                found[canonical].update(aliases)

    # Build flat list — canonical + all aliases
    all_names: set[str] = set()
    for canonical, aliases in found.items():
        all_names.add(canonical)
        all_names.update(aliases)

    logger.info(f"Option 3 complete: {len(found)} products "
                f"({len(all_names)} names including aliases)")
    for canonical, aliases in sorted(found.items()):
        logger.info(f"  ✓ {canonical:<35} {sorted(aliases)}")

    return sorted(all_names)
