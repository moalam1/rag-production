"""
Option 2 — Mine your own Pinecone corpus for Equinix product names.
Text is stored inside _node_content as a JSON string field.
"""
import asyncio
import json
import logging
import os
import random
from typing import Any

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)
client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

NAMESPACES = ["technical", "business", "media"]
CHUNKS_PER_NAMESPACE = 40
BATCH_SIZE = 15

EXTRACTION_PROMPT = """You are extracting Equinix product and service names from 
technical documentation chunks.

Extract EVERY distinct Equinix product, service, or platform name mentioned.
Include full names, short names, and abbreviations.
Exclude generic tech terms, competitor names, partner names like AWS/Azure/Google.

Return ONLY a valid JSON array of strings. No markdown. No explanation.
Example: ["Equinix Fabric", "Fabric", "EF", "IBX", "Equinix Metal"]

Chunks:
{chunks}"""


def _extract_text(match: dict) -> str:
    """
    Extract readable text from a Pinecone match.
    Text is stored inside _node_content as a JSON string — not a top-level field.
    Falls back to clean_name if _node_content has no text.
    """
    meta = match.get("metadata", {})

    # Primary: parse _node_content JSON
    raw_node = meta.get("_node_content", "")
    if raw_node:
        try:
            node = json.loads(raw_node)
            text = node.get("text", "").strip()
            if text:
                return text
        except (json.JSONDecodeError, AttributeError):
            pass

    # Fallback 1: clean_name (always present, short but useful)
    clean_name = meta.get("clean_name", "").strip()
    if clean_name:
        return clean_name

    # Fallback 2: page_url slug — still carries product signal
    url = meta.get("page_url", meta.get("url", "")).strip()
    return url


async def _query_pinecone_namespace(
    pinecone_index: Any,
    namespace: str,
    n: int,
) -> list[str]:
    """Sample n chunks from a Pinecone namespace using random query vectors."""
    texts = []
    try:
        dim = 1024
        for _ in range(3):
            raw = [random.gauss(0, 1) for _ in range(dim)]
            magnitude = sum(x ** 2 for x in raw) ** 0.5
            query_vec = [x / magnitude for x in raw]

            result = pinecone_index.query(
                vector=query_vec,
                top_k=max(1, n // 3),
                include_metadata=True,
                namespace=namespace,
            )

            for match in result.get("matches", []):
                text = _extract_text(match)
                if text and len(text.split()) >= 3:
                    texts.append(text)

        logger.info(f"  Namespace '{namespace}': sampled {len(texts)} chunks")

    except Exception as e:
        logger.warning(f"  Namespace '{namespace}' query failed: {e}")

    return texts


async def _extract_products_from_batch(batch: list[str], batch_num: int) -> list[str]:
    """Send one batch of chunks to GPT-4o-mini and extract product names."""
    combined = "\n---\n".join(batch[:BATCH_SIZE])
    try:
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=400,
            messages=[{
                "role": "user",
                "content": EXTRACTION_PROMPT.format(chunks=combined[:5000])
            }]
        )
        raw_text = resp.choices[0].message.content.strip()
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()
        products = json.loads(raw_text)
        if not isinstance(products, list):
            return []
        cleaned = [
            p.strip() for p in products
            if isinstance(p, str)
            and 1 <= len(p.split()) <= 6
            and len(p) < 60
        ]
        logger.info(f"  Batch {batch_num}: extracted {len(cleaned)} products")
        return cleaned
    except Exception as e:
        logger.warning(f"  Batch {batch_num}: failed — {e}")
        return []


async def run_option2(pinecone_index: Any) -> list[str]:
    """
    Main entry point for Option 2.
    Returns deduplicated list of raw product name strings.
    """
    logger.info("Option 2: Mining Pinecone corpus for Equinix product names")

    # Sample chunks from all namespaces
    tasks = [
        _query_pinecone_namespace(pinecone_index, ns, CHUNKS_PER_NAMESPACE)
        for ns in NAMESPACES
    ]
    results = await asyncio.gather(*tasks)

    # Deduplicate
    seen: set[str] = set()
    all_texts: list[str] = []
    for texts in results:
        for t in texts:
            key = t[:120]
            if key not in seen:
                seen.add(key)
                all_texts.append(t)

    logger.info(f"Total unique chunks to process: {len(all_texts)}")

    if not all_texts:
        logger.error("No chunks retrieved — check Pinecone connection")
        return []

    # Extract in batches
    batches = [all_texts[i:i+BATCH_SIZE] for i in range(0, len(all_texts), BATCH_SIZE)]
    sem = asyncio.Semaphore(3)

    async def _extract_with_sem(batch, idx):
        async with sem:
            return await _extract_products_from_batch(batch, idx+1)

    batch_results = await asyncio.gather(*[
        _extract_with_sem(b, i) for i, b in enumerate(batches)
    ])

    raw_products: set[str] = set()
    for product_list in batch_results:
        raw_products.update(product_list)

    sorted_products = sorted(raw_products)
    logger.info(f"Option 2 complete: {len(sorted_products)} unique raw names")
    logger.info(f"Preview: {sorted_products[:10]}")
    return sorted_products
