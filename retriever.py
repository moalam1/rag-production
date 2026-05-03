"""
pipeline/retriever.py — Pinecone vector retrieval with cache.

Cache key: hash of the query vector (so same query = same cache key).
Cache TTL: shorter than answer cache since new docs may be indexed.
"""
import json
import logging
from pinecone import Pinecone

from config import settings
from cache.factory import cache
from cache.memory import MemoryCache

log = logging.getLogger(__name__)

_pc    = Pinecone(api_key=settings.PINECONE_API_KEY)
_index = _pc.Index(settings.PINECONE_INDEX)

# Retrieval cache TTL — shorter so newly indexed docs show up faster
RETRIEVAL_CACHE_TTL = min(settings.CACHE_TTL_SECONDS, 600)   # max 10 min


def retrieve_chunks(query_embedding: list[float]) -> list[dict]:
    """
    Query Pinecone for the top-K most similar chunks.
    Results cached by embedding vector hash.
    """
    c   = cache()
    key = MemoryCache.make_key("retrieve", query_embedding[:8])  # first 8 dims as key fingerprint

    cached = c.get(key)
    if cached is not None:
        log.debug("retrieval cache HIT")
        return cached

    results = _index.query(
        vector=query_embedding,
        top_k=settings.TOP_K_RETRIEVE,
        include_metadata=True,
    )

    chunks = []
    for match in results.matches:
        metadata   = match.metadata
        text       = ""
        inner_meta = {}

        node_content = metadata.get("_node_content", "")
        if node_content:
            try:
                node_data  = json.loads(node_content)
                text       = node_data.get("text", "")
                inner_meta = node_data.get("metadata", {})
            except (json.JSONDecodeError, AttributeError):
                pass

        if not text.strip():
            text = metadata.get("text", "")
        if not text.strip():
            continue

        filename   = metadata.get("filename")   or inner_meta.get("filename", "unknown")
        clean_name = (metadata.get("clean_name") or inner_meta.get("clean_name")
                      or filename.replace("_", " ").replace("-", " ").replace(".pdf", "").title())
        page       = (metadata.get("page_label") or metadata.get("page")
                      or inner_meta.get("page")  or inner_meta.get("page_label") or "?")
        pdf_url    = metadata.get("pdf_url") or inner_meta.get("pdf_url", "")

        chunks.append({
            "id":         match.id,
            "score":      match.score,
            "text":       text,
            "filename":   filename,
            "clean_name": clean_name,
            "page":       page,
            "pdf_url":    pdf_url,
        })

    c.set(key, chunks, ttl=RETRIEVAL_CACHE_TTL)
    return chunks
