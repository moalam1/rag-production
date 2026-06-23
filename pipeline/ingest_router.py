"""
pipeline/ingest_router.py — Route parsed pages to correct ingestion path.

Takes a ParsedPage and:
  1. Always ingests the page teaser as a lightweight chunk
  2. For PDF types — fetches PDF and runs through ingester.py (LlamaParse)
  3. For video types — ingests transcript if available, else teaser only

Both page teaser and PDF chunks share the same document_family so they
are linked in retrieval and versioning.
"""
import logging
import os
import re
import tempfile
import hashlib
from datetime import datetime
from typing import Optional

import httpx
from pinecone import Pinecone
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.vector_stores.pinecone import PineconeVectorStore

from config import settings
from pipeline.page_parser import ParsedPage, PDF_TYPES, VIDEO_TYPES
from pipeline.ingester import NAMESPACE_MAP
from config_dynamic import resolve_write_namespace, resolve_section_namespaces
from pipeline.registry import (
    is_unchanged,
    is_unchanged_by_timestamp,
    get_version,
    save_record,
    deprecate_old_version,
)

# ── NEW: enricher import ──────────────────────────────────────────────────────
import asyncio
from pipeline.enricher import enrich_chunks_batch, merge_enrichment_into_metadata

log = logging.getLogger(__name__)

# ── Pinecone clients ──────────────────────────────────────────────────────────
_pc      = Pinecone(api_key=settings.PINECONE_API_KEY)
_index   = _pc.Index(settings.PINECONE_INDEX)
_summary = _pc.Index(settings.PINECONE_SUMMARY_INDEX)

# NAMESPACE_MAP imported from pipeline.ingester (single source of truth)

FETCH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/pdf,*/*",
}


def route_and_ingest(page: ParsedPage, force: bool = False, section: str = "") -> list[str]:
    """
    Main entry point. Takes a ParsedPage and runs the full ingest.

    Returns list of log strings.
    """
    logs  = []
    rtype = page.resource_type
    ns    = resolve_write_namespace(section, rtype, NAMESPACE_MAP)

    logs.append(f"🔀 Routing: {page.url}")
    logs.append(f"   type={rtype} | namespace={ns} | family={page.document_family}")

    # ── Fast pre-check: skip if timestamp unchanged ───────────────────────────
    if not force and is_unchanged_by_timestamp(page.url, page.og_updated_time):
        logs.append("⏭️  og:updated_time unchanged — skipping.")
        return logs

    # ── Hash check: skip if content unchanged ─────────────────────────────────
    if not force and is_unchanged(filename=page.document_family, content_hash=page.content_hash, url=page.url):
        logs.append("⏭️  Content hash unchanged — skipping.")
        return logs

    # ── Version from registry ─────────────────────────────────────────────────
    existing_version = get_version(document_family=page.document_family)
    version          = existing_version + 1 if existing_version > 0 else 1

    # ── Deprecate old version ─────────────────────────────────────────────────
    if existing_version > 0:
        logs.append(f"🔄 v{existing_version} → v{version} — deprecating old chunks...")
        for ns_del in resolve_section_namespaces(section or "resources"):
            try:
                _index.delete(filter={"document_family": {"$eq": page.document_family}}, namespace=ns_del)
                _summary.delete(filter={"document_family": {"$eq": page.document_family}}, namespace=ns_del)
            except Exception as e:
                logs.append(f"⚠️  Delete failed in {ns_del}: {e}")
        deprecate_old_version(page.document_family)
        logs.append("✅ Old chunks removed.")

    # ── Embed model ───────────────────────────────────────────────────────────
    embed_model = OpenAIEmbedding(
        model=settings.EMBED_MODEL,
        api_key=settings.OPENAI_API_KEY,
        dimensions=settings.EMBED_DIMS,
    )

    total_search  = 0
    total_summary = 0

    # ── 1. Always ingest page teaser ──────────────────────────────────────────
    logs.append(f"📄 Ingesting page teaser ({page.word_count} words)...")
    s, su = _ingest_text(
        text            = page.teaser,
        title           = page.title,
        url             = page.url,
        resource_type   = rtype,
        namespace       = ns,
        document_family = page.document_family,
        version         = version,
        published_date  = page.published_date,
        pdf_url         = page.pdf_url,
        chunk_type      = "page_teaser",
        embed_model     = embed_model,
        # ── NEW: pass AEM tags for enrichment ────────────────────────
        aem_tags        = list(page.tags),
    )
    total_search  += s
    total_summary += su
    logs.append(f"✅ Teaser: {s} search + {su} summary chunks → '{ns}'")

    # ── 2. PDF ingestion (PDF types only) ─────────────────────────────────────
    if page.has_pdf and rtype in PDF_TYPES:
        logs.append(f"📥 Fetching PDF: {page.pdf_url[:60]}...")
        pdf_logs = _ingest_pdf(
            pdf_url         = page.pdf_url,
            title           = page.title,
            page_url        = page.url,
            resource_type   = rtype,
            namespace       = ns,
            document_family = page.document_family,
            version         = version,
            published_date  = page.published_date,
            embed_model     = embed_model,
            aem_tags        = list(page.tags),
            force           = force,
        )
        logs.extend(pdf_logs)

    # ── 3. Transcript ingestion (video types) ─────────────────────────────────
    elif page.has_transcript and rtype in VIDEO_TYPES:
        logs.append(f"🎬 Ingesting transcript ({len(page.transcript.split())} words)...")
        s, su = _ingest_text(
            text            = page.transcript,
            title           = page.title,
            url             = page.url,
            resource_type   = rtype,
            namespace       = ns,
            document_family = page.document_family,
            version         = version,
            published_date  = page.published_date,
            pdf_url         = "",
            chunk_type      = "transcript",
            embed_model     = embed_model,
            # ── NEW: pass AEM tags for enrichment ────────────────────
            aem_tags        = list(page.tags),
        )
        total_search  += s
        total_summary += su
        logs.append(f"✅ Transcript: {s} search + {su} summary chunks → '{ns}'")

    # ── 4. Save to DynamoDB registry ──────────────────────────────────────────
    try:
        save_record(
            filename        = page.document_family,
            clean_name      = page.title,
            resource_type   = rtype,
            namespace       = ns,
            content_hash    = page.content_hash,
            version         = version,
            chunks_search   = total_search,
            chunks_summary  = total_summary,
            document_family = page.document_family,
            page_url        = page.url,
            url             = page.url,
            og_updated_time = page.og_updated_time,
            published_date  = page.published_date,
        )
        logs.append("📋 Registry updated in DynamoDB.")
    except Exception as e:
        logs.append(f"⚠️  Registry update failed: {e}")

    logs.append(f"🎉 Done — '{page.title[:50]}' v{version} searchable.")
    return logs


# ── Text ingestion (teaser + transcript) ──────────────────────────────────────

def _ingest_text(
    text:            str,
    title:           str,
    url:             str,
    resource_type:   str,
    namespace:       str,
    document_family: str,
    version:         int,
    published_date:  str,
    pdf_url:         str,
    chunk_type:      str,
    embed_model,
    aem_tags:        list = None,   # NEW
) -> tuple[int, int]:
    """Chunk, embed and upsert a text string. Returns (search_chunks, summary_chunks)."""
    aem_tags = aem_tags or []       # NEW

    header = f"# {title}\nType: {resource_type} | Source: {url}\n---\n"
    metadata = {
        "url":             url,
        "page_url":        url,
        "pdf_url":         pdf_url,
        "clean_name":      title,
        "resource_type":   resource_type,
        "namespace":       namespace,
        "document_family": document_family,
        "version":         str(version),
        "is_latest":       True,
        "status":          "current",
        "published_date":  published_date,
        "indexed_at":      datetime.now().isoformat(),
        "chunk_type":      chunk_type,
        "filename":        document_family,
        "page":            "1",
    }

    doc = Document(
        text=header + text,
        metadata=metadata,
        excluded_llm_metadata_keys=["indexed_at", "pdf_url"],
        excluded_embed_metadata_keys=["indexed_at", "pdf_url"],
    )

    def _make_splitter(buf, threshold):
        return SemanticSplitterNodeParser(
            buffer_size=buf,
            breakpoint_percentile_threshold=threshold,
            embed_model=embed_model,
        )

    # Search chunks — always indexed
    search_nodes = _make_splitter(1, 80).get_nodes_from_documents([doc], show_progress=False)
    for node in search_nodes:
        node.excluded_llm_metadata_keys   = ["indexed_at", "pdf_url"]
        node.excluded_embed_metadata_keys  = ["indexed_at", "pdf_url"]

    # ── NEW: enrich search nodes ──────────────────────────────────────────────
    search_nodes = _enrich_nodes(
        nodes         = search_nodes,
        title         = title,
        resource_type = resource_type,
        url           = url,
        aem_tags      = aem_tags,
    )
    # ─────────────────────────────────────────────────────────────────────────

    search_vs  = PineconeVectorStore(pinecone_index=_index, namespace=namespace)
    search_ctx = StorageContext.from_defaults(vector_store=search_vs)
    VectorStoreIndex(search_nodes, storage_context=search_ctx, embed_model=embed_model)

    # Summary chunks — PDF and transcript only
    if chunk_type in ("pdf", "transcript"):
        summary_nodes = _make_splitter(3, 95).get_nodes_from_documents([doc], show_progress=False)
        total = len(summary_nodes)
        for idx, node in enumerate(summary_nodes):
            node.metadata["chunk_index"]  = str(idx)
            node.metadata["total_chunks"] = str(total)
            node.excluded_llm_metadata_keys   = ["indexed_at", "pdf_url"]
            node.excluded_embed_metadata_keys  = ["indexed_at", "pdf_url"]

        # ── NEW: enrich summary nodes ─────────────────────────────────────────
        summary_nodes = _enrich_nodes(
            nodes         = summary_nodes,
            title         = title,
            resource_type = resource_type,
            url           = url,
            aem_tags      = aem_tags,
        )
        # ─────────────────────────────────────────────────────────────────────

        summary_vs  = PineconeVectorStore(pinecone_index=_summary, namespace=namespace)
        summary_ctx = StorageContext.from_defaults(vector_store=summary_vs)
        VectorStoreIndex(summary_nodes, storage_context=summary_ctx, embed_model=embed_model)
        return len(search_nodes), len(summary_nodes)

    return len(search_nodes), 0


# ── PDF fetch + LlamaParse ingestion ──────────────────────────────────────────

def _ingest_pdf(
    pdf_url:         str,
    title:           str,
    page_url:        str,
    resource_type:   str,
    namespace:       str,
    document_family: str,
    version:         int,
    published_date:  str,
    embed_model,
    aem_tags:        list = None,   # NEW
    force:           bool = False,  # NEW
) -> list[str]:
    """Fetch PDF from CDN URL and ingest via LlamaParse."""
    aem_tags = aem_tags or []       # NEW
    logs = []
    try:
        with httpx.Client(headers=FETCH_HEADERS, timeout=60, follow_redirects=True) as client:
            resp = client.get(pdf_url)
            resp.raise_for_status()
            pdf_bytes = resp.content
            logs.append(f"✅ PDF fetched: {len(pdf_bytes):,} bytes")
    except Exception as e:
        logs.append(f"❌ PDF fetch failed: {e}")
        return logs

    with tempfile.TemporaryDirectory() as tmp_dir:
        filename = document_family + ".pdf"
        pdf_path = os.path.join(tmp_dir, filename)
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)

        try:
            from pipeline.ingester import ingest
            pdf_logs = ingest(
                tmp_dir             = tmp_dir,
                resource_type       = resource_type,
                clean_name_override = title,
                page_url_override   = page_url,
                document_family     = document_family,
                published_date      = published_date,
                aem_tags            = aem_tags,
                force               = force,
                namespace_override  = namespace,
            )
            logs.extend(pdf_logs)
        except Exception as e:
            logs.append(f"❌ PDF ingest failed: {e}")

    return logs


# ── NEW: enrichment helper ────────────────────────────────────────────────────

def _enrich_nodes(
    nodes:         list,
    title:         str,
    resource_type: str,
    url:           str,
    aem_tags:      list,
) -> list:
    """
    Run enrichment on a list of LlamaIndex nodes.
    Merges enrichment metadata into each node's metadata dict.

    Runs synchronously by creating an event loop — safe because
    ingest_router.py is called from ingest_worker.py which is synchronous.

    Returns the same nodes with enrichment metadata added.
    Never raises — enrichment failure is logged but doesn't block ingest.
    """
    if not nodes:
        return nodes

    try:
        # Convert nodes to chunk dicts for enricher
        chunks = [
            {"text": node.text or node.get_content(), "idx": i}
            for i, node in enumerate(nodes)
        ]

        # Run async enrichment in a new event loop
        # (ingest_worker is sync — no existing loop to conflict with)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If somehow called from async context — use run_coroutine_threadsafe
                import concurrent.futures
                future = asyncio.run_coroutine_threadsafe(
                    enrich_chunks_batch(
                        chunks        = chunks,
                        title         = title,
                        resource_type = resource_type,
                        url           = url,
                        aem_tags      = aem_tags,
                    ),
                    loop
                )
                enriched_chunks = future.result(timeout=120)
            else:
                enriched_chunks = loop.run_until_complete(
                    enrich_chunks_batch(
                        chunks        = chunks,
                        title         = title,
                        resource_type = resource_type,
                        url           = url,
                        aem_tags      = aem_tags,
                    )
                )
        except RuntimeError:
            # No event loop — create one
            enriched_chunks = asyncio.run(
                enrich_chunks_batch(
                    chunks        = chunks,
                    title         = title,
                    resource_type = resource_type,
                    url           = url,
                    aem_tags      = aem_tags,
                )
            )

        # Merge enrichment metadata back into nodes
        for node, enriched in zip(nodes, enriched_chunks):
            enrichment = enriched.get("enrichment", {})
            merged = merge_enrichment_into_metadata(
                node.metadata,
                enrichment,
            )
            node.metadata = merged

        enriched_count = sum(
            1 for c in enriched_chunks
            if c.get("enrichment", {}).get("enriched")
        )
        log.info(
            f"Enrichment: {enriched_count}/{len(nodes)} nodes enriched "
            f"for '{title[:40]}' ({resource_type})"
        )

    except Exception as e:
        # Enrichment failure must never block ingest
        log.warning(
            f"Enrichment failed for '{title[:40]}' — "
            f"ingesting without enrichment: {e}"
        )

    return nodes
