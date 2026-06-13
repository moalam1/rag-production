"""
Normaliser — merges raw product lists from Options 1, 2, 3 into
a clean, versioned products.json with canonical names and aliases.

Run after any combination of options complete.
Output: config/products.json
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)
client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

OUTPUT_PATH = Path("config/products.json")

NORMALISE_PROMPT = """You are building a canonical product registry for Equinix's 
RAG system. You have a raw list of product name strings that may contain:
- Duplicates with different casing (fabric, Fabric, FABRIC)
- Aliases for the same product (EF = Equinix Fabric)  
- Abbreviations (NE = Network Edge, IBX = IBX Data Centers)
- Near-duplicates (Metal, Equinix Metal, Bare Metal)
- Noise that slipped through (remove these)

Build a clean registry. For each distinct product:
- Choose the most descriptive name as canonical
- List ALL aliases, abbreviations, short names
- Assign a category

Valid categories:
  interconnection, colocation, compute, networking, 
  security, timing, cloud, edge, platform

Return ONLY valid JSON, no markdown:
{{
  "products": [
    {{
      "canonical": "Equinix Fabric",
      "aliases": ["Fabric", "EF", "equinix fabric"],
      "category": "interconnection",
      "description": "Software-defined interconnection platform"
    }}
  ]
}}

Raw product name list:
{raw_list}"""


async def normalise(raw_products: list[str]) -> dict:
    """
    Send the merged raw list to GPT-4o-mini for normalisation.
    Returns the structured registry dict.
    """
    if not raw_products:
        logger.error("No raw products to normalise")
        return {"products": []}

    # Deduplicate case-insensitively before sending
    seen_lower: set[str] = set()
    deduped: list[str] = []
    for p in sorted(raw_products):
        if p.lower() not in seen_lower:
            seen_lower.add(p.lower())
            deduped.append(p)

    logger.info(f"Normalising {len(deduped)} unique raw names")

    try:
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",     # normalisation is structured — mini handles it well
            temperature=0,
            max_tokens=2000,
            response_format={"type": "json_object"},
            messages=[{
                "role": "user",
                "content": NORMALISE_PROMPT.format(
                    raw_list=json.dumps(deduped, indent=2)
                )
            }]
        )

        result = json.loads(resp.choices[0].message.content)

        if "products" not in result:
            logger.error("Normaliser returned unexpected shape")
            return {"products": []}

        logger.info(
            f"Normalisation complete: "
            f"{len(result['products'])} canonical products"
        )
        return result

    except Exception as e:
        logger.error(f"Normalisation failed: {e}")
        # Return a minimal fallback so the pipeline doesn't break
        return {
            "products": [
                {"canonical": p, "aliases": [], "category": "platform",
                 "description": ""}
                for p in deduped[:30]  # safety cap
            ]
        }


def save_registry(registry: dict, sources: list[str]) -> Path:
    """
    Save the final registry to config/products.json.
    Includes metadata about when it was built and which sources contributed.
    """
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Build flat lookup sets for enricher.py to use at runtime
    all_names: list[str] = []
    alias_map: dict[str, str] = {}   # alias.lower() → canonical

    for product in registry["products"]:
        canonical = product["canonical"]
        all_names.append(canonical)
        alias_map[canonical.lower()] = canonical

        for alias in product.get("aliases", []):
            all_names.append(alias)
            alias_map[alias.lower()] = canonical

    output = {
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        "sources":        sources,
        "product_count":  len(registry["products"]),
        "total_names":    len(all_names),
        "products":       registry["products"],
        # Precomputed flat list for enricher.py prompt injection
        "_all_names":     sorted(set(all_names)),
        # Precomputed alias map for query-time lookup
        "_alias_to_canonical": alias_map,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved registry → {OUTPUT_PATH}")
    logger.info(f"  {len(registry['products'])} canonical products")
    logger.info(f"  {len(all_names)} total names including aliases")

    return OUTPUT_PATH


def load_registry() -> dict:
    """
    Load products.json at runtime. Call this from enricher.py.
    Raises RuntimeError if file not found — prevents silent failures.
    """
    if not OUTPUT_PATH.exists():
        raise RuntimeError(
            f"Product registry not found at {OUTPUT_PATH}. "
            "Run: python scripts/build_product_list.py"
        )

    with open(OUTPUT_PATH) as f:
        data = json.load(f)

    age_hours = (
        datetime.now(timezone.utc) -
        datetime.fromisoformat(data["generated_at"])
    ).total_seconds() / 3600

    if age_hours > 24 * 8:  # warn if older than 8 days
        logger.warning(
            f"Product registry is {age_hours:.0f}h old — "
            "consider refreshing with build_product_list.py"
        )

    return data
