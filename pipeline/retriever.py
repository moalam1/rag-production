"""
pipeline/retriever.py — Pinecone retrieval across all namespaces.
Pinecone serverless does NOT search across namespaces automatically.
We query each namespace separately and merge by score.
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

RETRIEVAL_CACHE_TTL = min(settings.CACHE_TTL_SECONDS, 600)
ALL_NAMESPACES      = ["technical", "business", "media"]


def _query_namespace(vector: list, namespace: str, top_k: int) -> list[dict]:
    try:
        results = _index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace,
            filter={"is_latest": {"$eq": "true"}},
        )
        if not results.matches:
            results = _index.query(
                vector=vector,
                top_k=top_k,
                include_metadata=True,
                namespace=namespace,
            )
    except Exception as e:
        log.warning("Pinecone query failed for namespace '%s': %s", namespace, e)
        return []

    chunks = []
    for match in results.matches:
        metadata     = match.metadata
        text         = ""
        inner_meta   = {}
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

        filename      = metadata.get("filename")       or inner_meta.get("filename", "unknown")
        clean_name    = (metadata.get("clean_name")    or inner_meta.get("clean_name")
                         or filename.replace("_", " ").replace("-", " ").replace(".pdf", "").title())
        page          = (metadata.get("page_label")    or metadata.get("page")
                         or inner_meta.get("page")     or inner_meta.get("page_label") or "?")
        resource_type = (metadata.get("resource_type") or inner_meta.get("resource_type", ""))
        page_url      = (metadata.get("page_url")      or inner_meta.get("page_url", ""))
        pdf_url       = (metadata.get("pdf_url")       or inner_meta.get("pdf_url", ""))

        chunks.append({
            "id":            match.id,
            "score":         match.score,
            "text":          text,
            "filename":      filename,
            "clean_name":    clean_name,
            "resource_type": resource_type,
            "page":          page,
            "page_url":      page_url,
            "pdf_url":       pdf_url,
            "namespace":     namespace,
        })
    return chunks


def retrieve_chunks(query_embedding: list[float], namespace: str = None) -> list[dict]:
    c   = cache()
    key = MemoryCache.make_key("retrieve", {"dims": query_embedding[:8], "ns": namespace or "all"})

    cached = c.get(key)
    if cached is not None:
        log.debug("retrieval cache HIT")
        return cached

    namespaces_to_query = (
        [namespace.strip()] if namespace and namespace.strip().lower() not in ("", "all")
        else ALL_NAMESPACES
    )

    log.debug("Querying namespaces: %s", namespaces_to_query)

    all_chunks = []
    for ns in namespaces_to_query:
        ns_chunks = _query_namespace(query_embedding, ns, settings.TOP_K_RETRIEVE)
        log.debug("Namespace '%s': %d chunks", ns, len(ns_chunks))
        all_chunks.extend(ns_chunks)

    all_chunks.sort(key=lambda x: x["score"], reverse=True)
    top_chunks = all_chunks[:settings.TOP_K_RETRIEVE]

    log.info("Retrieved %d chunks from %s", len(top_chunks), namespaces_to_query)

    c.set(key, top_chunks, ttl=RETRIEVAL_CACHE_TTL)
    return top_chunks
