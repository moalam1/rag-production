"""
build_product_list.py — Master runner for Equinix product registry.

Usage:
  # Full run (all three options) — recommended first time:
  python scripts/build_product_list.py

  # Weekly refresh (Option 2 only — fast and cheap):
  python scripts/build_product_list.py --weekly

  # Quarterly sanity check (all options including scraper):
  python scripts/build_product_list.py --quarterly

  # Specific options only:
  python scripts/build_product_list.py --options 2 3

Cost summary:
  --weekly    ~$0.005   (Option 2 only)
  Full run    ~$0.012   (Options 2 + 3, skip Option 1)
  --quarterly ~$0.030   (All three options)
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from option1_scraper import run_option1
from option2_corpus  import run_option2
from option3_sitemap import run_option3
from normaliser      import normalise, save_registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _get_pinecone_index():
    """
    Initialise Pinecone index using your existing environment variables.
    Matches your current setup in the RAG codebase.
    """
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index = pc.Index(os.environ.get("PINECONE_INDEX", "rag-poc"))
        logger.info(f"Pinecone connected: {os.environ.get('PINECONE_INDEX','rag-poc')}")
        return index
    except Exception as e:
        logger.error(f"Pinecone init failed: {e}")
        raise


async def run(options: list[int]) -> None:
    """
    Orchestrate selected options, merge results, normalise, save.
    """
    logger.info("=" * 55)
    logger.info("Equinix Product Registry Builder")
    logger.info(f"Running options: {options}")
    logger.info("=" * 55)

    all_raw: list[str] = []
    sources_used: list[str] = []

    # ── Option 2 first (cheapest, highest quality) ────────────────────────
    if 2 in options:
        logger.info("\n── Option 2: Mine Pinecone corpus ──")
        try:
            pinecone_index = _get_pinecone_index()
            raw2 = await run_option2(pinecone_index)
            all_raw.extend(raw2)
            sources_used.append(f"corpus ({len(raw2)} raw names)")
            logger.info(f"Option 2 contributed {len(raw2)} raw names")
        except Exception as e:
            logger.error(f"Option 2 failed: {e} — continuing with other options")

    # ── Option 3 next (near-free, low noise) ─────────────────────────────
    if 3 in options:
        logger.info("\n── Option 3: Parse sitemaps ──")
        try:
            raw3 = await run_option3()
            all_raw.extend(raw3)
            sources_used.append(f"sitemap ({len(raw3)} raw names)")
            logger.info(f"Option 3 contributed {len(raw3)} raw names")
        except Exception as e:
            logger.error(f"Option 3 failed: {e} — continuing with other options")

    # ── Option 1 last (most expensive, run quarterly) ─────────────────────
    if 1 in options:
        logger.info("\n── Option 1: Scrape equinix.com ──")
        logger.info("Note: This uses GPT-4o (~$0.017). Recommended quarterly only.")
        try:
            raw1 = await run_option1()
            all_raw.extend(raw1)
            sources_used.append(f"scraper ({len(raw1)} raw names)")
            logger.info(f"Option 1 contributed {len(raw1)} raw names")
        except Exception as e:
            logger.error(f"Option 1 failed: {e} — continuing")

    if not all_raw:
        logger.error("All options failed or returned no results. Aborting.")
        sys.exit(1)

    logger.info(f"\n── Normalisation pass ──")
    logger.info(f"Total raw names before dedup: {len(all_raw)}")

    # ── Normalise and save ────────────────────────────────────────────────
    registry = await normalise(all_raw)
    output_path = save_registry(registry, sources_used)

    # ── Print summary ─────────────────────────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info("Registry build complete")
    logger.info(f"Output: {output_path}")
    logger.info(f"Sources: {', '.join(sources_used)}")
    logger.info(
        f"Result:  {registry['product_count']} canonical products"
        if 'product_count' in registry
        else f"Result:  {len(registry.get('products',[]))} canonical products"
    )
    logger.info("\nCanonical products:")
    for p in registry.get("products", []):
        aliases = p.get("aliases", [])
        alias_str = f"  [{', '.join(aliases[:3])}]" if aliases else ""
        logger.info(f"  ✓ {p['canonical']}{alias_str}  ({p.get('category','')})")
    logger.info("=" * 55)


def main():
    parser = argparse.ArgumentParser(
        description="Build Equinix product registry from corpus, sitemap, and/or scraper"
    )
    parser.add_argument(
        "--weekly",
        action="store_true",
        help="Weekly refresh — Option 2 only (~$0.005)"
    )
    parser.add_argument(
        "--quarterly",
        action="store_true",
        help="Quarterly full run — all three options (~$0.030)"
    )
    parser.add_argument(
        "--options",
        nargs="+",
        type=int,
        choices=[1, 2, 3],
        help="Specific options to run (e.g. --options 2 3)"
    )
    args = parser.parse_args()

    if args.weekly:
        options = [2]
    elif args.quarterly:
        options = [2, 3, 1]
    elif args.options:
        options = args.options
    else:
        # Default: Options 2 + 3 (best quality-to-cost ratio)
        options = [2, 3]

    asyncio.run(run(options))


if __name__ == "__main__":
    main()
