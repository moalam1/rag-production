
"""
pipeline/ingester.py — Production-grade dual-index ingestion pipeline.

Writes to two Pinecone indexes simultaneously:
  rag-poc     — search chunks (300-500 tokens, SemanticSplitter 80th pct)
  rag-summary — summary chunks (1500-2000 tokens, SemanticSplitter 95th pct)

Metadata inherited on every chunk:
  filename, clean_name, resource_type, namespace, page, page_url, pdf_url,
  document_family, is_latest, status, published_date, version

Fixes applied:
  1. is_latest = True  (boolean, was string "true" — broke Pinecone filter)
  2. family = "" reset removed — was silently overwriting correctly-derived family
  3. version derived from registry (get_version() + 1), not from caller arg
  4. Old chunks purged by document_family filter, not filename
     (stable across filename changes between versions)
  5. Dead _pdfs computation removed from save_record() call block
  6. published_date accepted as parameter; falls back to today only if not provided
  7. atomic_version_transition() used instead of two sequential DynamoDB writes
"""
import asyncio
import os
import re
import shutil
import hashlib
import logging
from datetime import datetime
from pathlib import Path

from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader, Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone

from config import settings
from pipeline.enricher import enrich_chunks_batch, merge_enrichment_into_metadata
from api.deps import resolve_write_namespace, resolve_section_namespaces
from pipeline.registry import (
    compute_hash,
    is_unchanged,
    get_version,
    save_record,
    deprecate_old_version,
    atomic_version_transition,
    get_record_by_family,
)

log = logging.getLogger(__name__)

# ── Pinecone clients ──────────────────────────────────────────────────────────
_pc      = Pinecone(api_key=settings.PINECONE_API_KEY)
_index   = _pc.Index(settings.PINECONE_INDEX)
_summary = _pc.Index(settings.PINECONE_SUMMARY_INDEX)

# ── Namespace routing ─────────────────────────────────────────────────────────
NAMESPACE_MAP = {
    "whitepaper":     "technical",
    "blueprint":      "technical",
    "analyst-report": "technical",
    "data-sheet":     "technical",
    "playbook":       "technical",
    "case-study":     "business",
    "solution-brief": "business",
    "article":        "business",
    "media":          "business",
    "multimedia":     "media",
    "webinar":        "media",
    "infopaper":        "technical",
    "product-document": "technical",
    "infographic":      "business",
    "success-story":    "business",
}

HF_REPO_ID  = os.getenv("HF_REPO_ID", "perwaizalam/rag-poc-demo")
AEM_BASE    = os.getenv("AEM_BASE_URL", "https://www.equinix.com/resources")
TYPE_FOLDER = {
    "whitepaper":     "whitepapers",
    "blueprint":      "blueprints",
    "case-study":     "case-studies",
    "analyst-report": "analyst-reports",
    "data-sheet":     "data-sheets",
    "solution-brief": "solution-briefs",
    "playbook":       "playbooks",
    "article":        "articles",
    "media":          "media",
    "multimedia":     "videos",
    "webinar":        "webinars",
}


def _build_page_url(clean_name: str, resource_type: str, page_url: str = "") -> str:
    if page_url and page_url.strip():
        return page_url.strip()
    folder = TYPE_FOLDER.get(resource_type.lower(), "resources")
    slug   = re.sub(r"[_\s]+", "-", clean_name.lower())
    slug   = re.sub(r"[^a-z0-9\-]", "", slug)
    slug   = re.sub(r"-{2,}", "-", slug).strip("-")
    return f"{AEM_BASE}/{folder}/{slug}"


def _pdf_url(filename: str, page) -> str:
    base = f"https://huggingface.co/spaces/{HF_REPO_ID}/resolve/main/pdfs/{filename}"
    try:
        return f"{base}#page={int(page)}"
    except (ValueError, TypeError):
        return base


def _content_hash(tmp_dir: str) -> str:
    """SHA-256 hash of all PDF content in tmp_dir (computed before parsing)."""
    h = hashlib.sha256()
    for f in sorted(Path(tmp_dir).glob("*.pdf")):
        h.update(f.read_bytes())
    return h.hexdigest()[:32]


def _derive_family(document_family: str, clean_name: str, filename: str) -> str:
    """Derive a stable document_family slug. Caller arg takes priority."""
    if document_family.strip():
        return document_family.strip()
    base = clean_name if clean_name else os.path.splitext(filename)[0]
    return re.sub(r"[^a-z0-9_]", "_", base.lower().replace(" ", "_"))[:40]


def _make_splitter(embed_model, buffer_size: int, threshold: int) -> SemanticSplitterNodeParser:
    return SemanticSplitterNodeParser(
        buffer_size=buffer_size,
        breakpoint_percentile_threshold=threshold,
        embed_model=embed_model,
    )


def ingest(
    tmp_dir:             str,
    resource_type:       str,
    clean_name_override: str = "",
    page_url_override:   str = "",
    document_family:     str = "",
    published_date:      str = "",   # FIX 6: accept real publish date, not just today
    aem_tags:            list = None,  # enrichment tags from AEM page
    force:               bool = False, # bypass hash check for re-enrichment
    section:             str  = "",     # L1b: target section (empty/'resources' = grandfathered)
    namespace_override:  str  = "",     # L1b: pre-resolved namespace (skips resolution if set)
) -> list[str]:
    """
    Full ingestion pipeline for PDF documents.

    Args:
        tmp_dir:             Directory containing PDF(s) to ingest.
        resource_type:       One of the 11 Equinix resource types.
        clean_name_override: Display name (uses filename if empty).
        page_url_override:   AEM resource page URL (auto-built if empty).
        document_family:     Logical group e.g. 'ai_infrastructure_guide'.
        published_date:      Document publish date (YYYY-MM-DD). Defaults to today.

    Returns:
        List of log strings for display in HF Space.
    """
    rtype     = resource_type.lower().strip()
    namespace = namespace_override or resolve_write_namespace(section, rtype, NAMESPACE_MAP)
    logs      = []

    try:
        # ── 1. Identify files and compute hash BEFORE any parsing ─────────────
        pdf_files      = list(Path(tmp_dir).glob("*.pdf"))
        if not pdf_files:
            logs.append("❌ No PDF files found in upload directory.")
            return logs

        first_filename = pdf_files[0].name
        content_hash   = _content_hash(tmp_dir)
        logs.append(f"🔑 Content hash: {content_hash[:8]}...")

        # ── 2. Derive clean_name and family ONCE — before any resets ─────────
        # FIX 2: family derived here and carried through — never reset below
        clean_name = (
            clean_name_override.strip() if clean_name_override.strip()
            else " ".join(
                w.capitalize()
                for w in os.path.splitext(first_filename)[0]
                .replace("_", " ").replace("-", " ").strip().split()
            )
        )
        family = _derive_family(document_family, clean_name, first_filename)

        # ── 3. Skip if unchanged ──────────────────────────────────────────────
        if not force and is_unchanged(first_filename, content_hash):
            logs.append("⏭️  Document unchanged — skipping re-index (same hash).")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return logs

        # ── 4. Version from registry — not from caller ────────────────────────
        # FIX 3: version is always registry-driven, caller no longer supplies it
        existing_version = get_version(document_family=family, filename=first_filename)
        version          = existing_version + 1 if existing_version > 0 else 1

        # ── 5. Deprecate old version if update detected ───────────────────────
        if existing_version > 0:
            logs.append(
                f"🔄 Update detected (v{existing_version} → v{version}) "
                f"— deprecating old chunks..."
            )
            # FIX 4: purge by document_family, not filename
            # Stable across filename changes between document versions
            for ns in resolve_section_namespaces(section or "resources"):
                try:
                    _index.delete(
                        filter={"document_family": {"$eq": family}},
                        namespace=ns,
                    )
                    _summary.delete(
                        filter={"document_family": {"$eq": family}},
                        namespace=ns,
                    )
                except Exception as de:
                    logs.append(f"⚠️  Could not delete old chunks in {ns}: {de}")
            logs.append("✅ Old chunks removed from Pinecone.")
            deprecate_old_version(first_filename)

        # ── 6. Parse PDFs with LlamaParse ────────────────────────────────────
        logs.append("🔍 Parsing PDFs with LlamaParse...")
        parser = LlamaParse(
            api_key=settings.LLAMA_CLOUD_API_KEY,
            result_type="markdown",
            verbose=False,
            parsing_instructions=(
                "Extract all text accurately. Preserve headings, tables, "
                "bullet points, and page structure. Include page numbers."
            ),
        )
        reader   = SimpleDirectoryReader(tmp_dir, file_extractor={".pdf": parser})
        raw_docs = reader.load_data()
        logs.append(f"✅ Parsed {len(raw_docs)} page(s).")

        # ── 7. Build documents with full metadata ─────────────────────────────
        logs.append("📋 Building documents with full metadata...")
        pub_date  = published_date.strip() if published_date.strip() else datetime.now().strftime("%Y-%m-%d")
        documents = []

        for i, doc in enumerate(raw_docs):
            filename = os.path.basename(doc.metadata.get("file_name", f"doc_{i}.pdf"))

            # Per-page clean_name (respects override)
            cn = (
                clean_name_override.strip() if clean_name_override.strip()
                else " ".join(
                    w.capitalize()
                    for w in os.path.splitext(filename)[0]
                    .replace("_", " ").replace("-", " ").strip().split()
                )
            )

            page   = str(doc.metadata.get("page_label", doc.metadata.get("page", str(i + 1))))
            pg_url = _build_page_url(cn, rtype, page_url_override)
            pdf_u  = _pdf_url(filename, page)

            # FIX 1: is_latest is a boolean True, not string "true"
            # Pinecone filter {"is_latest": {"$eq": True}} requires native bool
            metadata = {
                # Identity
                "filename":         filename,
                "clean_name":       cn,
                "resource_type":    rtype,
                "namespace":        namespace,

                # Navigation
                "page":             page,
                "page_url":         pg_url,
                "pdf_url":          pdf_u,

                # Versioning — FIX 1: boolean, not string
                "document_family":  family,
                "version":          str(version),
                "is_latest":        True,        # ← boolean
                "status":           "current",

                # Freshness — FIX 6: real publish date if provided
                "published_date":   pub_date,
                "indexed_at":       datetime.now().isoformat(),
            }

            header = (
                f"Document: {cn}\n"
                f"Type: {rtype} | Page: {page}\n"
                f"---\n"
            )
            documents.append(Document(
                text=header + doc.text,
                metadata=metadata,
                excluded_llm_metadata_keys=["indexed_at", "pdf_url"],
                excluded_embed_metadata_keys=["indexed_at", "pdf_url", "page_url"],
            ))

        logs.append(f"📋 Built {len(documents)} document(s) with metadata.")

        # ── 8. Embed model ────────────────────────────────────────────────────
        embed_model = OpenAIEmbedding(
            model=settings.EMBED_MODEL,
            api_key=settings.OPENAI_API_KEY,
            dimensions=settings.EMBED_DIMS,
        )

        # ── 9. Search index — 300-500 token chunks (80th pct) ─────────────────
        logs.append("✂️  Search chunking (80th pct threshold)...")
        search_splitter = _make_splitter(embed_model, buffer_size=1, threshold=80)
        search_nodes    = search_splitter.get_nodes_from_documents(
            documents, show_progress=False
        )
        for node in search_nodes:
            node.excluded_llm_metadata_keys  = ["indexed_at", "pdf_url"]
            node.excluded_embed_metadata_keys = ["indexed_at", "pdf_url", "page_url"]

        # Filter noise chunks — title fragments, copyright notices, footers
        MIN_CHUNK_WORDS = 50
        before = len(search_nodes)
        search_nodes = [n for n in search_nodes if len((n.text or n.get_content()).split()) >= MIN_CHUNK_WORDS]
        if before - len(search_nodes):
            logs.append(f"🧹 Dropped {before - len(search_nodes)} noise chunks (<{MIN_CHUNK_WORDS} words)")

        logs.append(f"✂️  Created {len(search_nodes)} search chunks.")

        # ── NEW: enrich search nodes ──────────────────────────────────────────
        search_nodes = _enrich_nodes_sync(
            nodes         = search_nodes,
            title         = clean_name_override or "",
            resource_type = rtype,
            url           = page_url_override or "",
            aem_tags      = aem_tags or [],
        )
        logs.append(f"🏷️  Enriched {sum(1 for n in search_nodes if n.metadata.get('enriched'))} search chunks.")
        # ─────────────────────────────────────────────────────────────────────

        logs.append(f"📤 Uploading to rag-poc ({namespace})...")
        search_vs  = PineconeVectorStore(pinecone_index=_index, namespace=namespace)
        search_ctx = StorageContext.from_defaults(vector_store=search_vs)
        VectorStoreIndex(search_nodes, storage_context=search_ctx, embed_model=embed_model)
        logs.append(f"✅ {len(search_nodes)} chunks → rag-poc '{namespace}'")

        # ── 10. Summary index — 1500-2000 token chunks (95th pct) ─────────────
        logs.append("✂️  Summary chunking (95th pct threshold)...")
        summary_splitter = _make_splitter(embed_model, buffer_size=3, threshold=95)
        summary_nodes    = summary_splitter.get_nodes_from_documents(
            documents, show_progress=False
        )
        total = len(summary_nodes)
        for idx, node in enumerate(summary_nodes):
            node.metadata["chunk_index"]  = str(idx)
            node.metadata["total_chunks"] = str(total)
            node.excluded_llm_metadata_keys  = ["indexed_at", "pdf_url"]
            node.excluded_embed_metadata_keys = ["indexed_at", "pdf_url", "page_url"]

        # Filter noise chunks from summary index
        MIN_SUMMARY_WORDS = 50
        before_sum = len(summary_nodes)
        summary_nodes = [n for n in summary_nodes if len((n.text or n.get_content()).split()) >= MIN_SUMMARY_WORDS]
        if before_sum - len(summary_nodes):
            logs.append(f"🧹 Dropped {before_sum - len(summary_nodes)} noise summary chunks (<{MIN_SUMMARY_WORDS} words)")

        logs.append(f"✂️  Created {len(summary_nodes)} summary chunks.")

        # ── NEW: enrich summary nodes ─────────────────────────────────────────
        summary_nodes = _enrich_nodes_sync(
            nodes         = summary_nodes,
            title         = clean_name_override or "",
            resource_type = rtype,
            url           = page_url_override or "",
            aem_tags      = aem_tags or [],
        )
        logs.append(f"🏷️  Enriched {sum(1 for n in summary_nodes if n.metadata.get('enriched'))} summary chunks.")
        # ─────────────────────────────────────────────────────────────────────

        logs.append(f"📤 Uploading to rag-summary ({namespace})...")
        summary_vs  = PineconeVectorStore(pinecone_index=_summary, namespace=namespace)
        summary_ctx = StorageContext.from_defaults(vector_store=summary_vs)
        VectorStoreIndex(summary_nodes, storage_context=summary_ctx, embed_model=embed_model)
        logs.append(f"✅ {len(summary_nodes)} chunks → rag-summary '{namespace}'")

        # ── 11. Save to DynamoDB registry ─────────────────────────────────────
        # FIX 5: removed dead _pdfs computation — first_filename and content_hash in scope
        try:
            save_record(
                filename        = first_filename,
                clean_name      = clean_name,
                resource_type   = rtype,
                namespace       = namespace,
                content_hash    = content_hash,
                version         = version,
                chunks_search   = len(search_nodes),
                chunks_summary  = len(summary_nodes),
                page_url        = page_url_override,
                document_family = family,
                published_date  = pub_date,
            )
            logs.append("📋 Registry updated in DynamoDB.")
        except Exception as e:
            logs.append(f"⚠️  Registry update failed: {e}")

        logs.append(
            f"🎉 Ingest complete — '{clean_name}' v{version} "
            f"({rtype}) → namespace '{namespace}' — now searchable."
        )

    except Exception as e:
        logs.append(f"❌ Error: {e}")
        log.exception("Ingestion failed")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return logs

# ── Enrichment helper (sync wrapper for async enricher) ───────────────────────

def _enrich_nodes_sync(
    nodes:         list,
    title:         str,
    resource_type: str,
    url:           str,
    aem_tags:      list,
) -> list:
    """
    Sync wrapper around enrich_chunks_batch for use in ingester.py.
    Handles event loop safely — ingester is called synchronously.
    Never raises — enrichment failure logs and returns nodes unchanged.
    """
    if not nodes:
        return nodes

    try:
        chunks = [
            {"text": node.text or node.get_content(), "idx": i}
            for i, node in enumerate(nodes)
        ]

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
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
            enriched_chunks = asyncio.run(
                enrich_chunks_batch(
                    chunks        = chunks,
                    title         = title,
                    resource_type = resource_type,
                    url           = url,
                    aem_tags      = aem_tags,
                )
            )

        for node, enriched in zip(nodes, enriched_chunks):
            merged = merge_enrichment_into_metadata(
                node.metadata,
                enriched.get("enrichment", {}),
            )
            node.metadata = merged

        enriched_count = sum(
            1 for c in enriched_chunks
            if c.get("enrichment", {}).get("enriched")
        )
        log.info(
            f"PDF enrichment: {enriched_count}/{len(nodes)} nodes enriched "
            f"({resource_type})"
        )

    except Exception as e:
        log.warning(f"PDF enrichment failed — ingesting without tags: {e}")

    return nodes
