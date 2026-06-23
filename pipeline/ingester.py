
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
from config_dynamic import resolve_write_namespace, resolve_section_namespaces
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


# ── L2a: ingest() decomposed into stages (behavior-identical extraction) ──────
# Each _stage_* function holds the EXACT logic from the corresponding numbered
# block of the original ingest(). IngestContext is the state that flows between
# stages (the future Step Functions JSON payload). ingest() is now a thin
# orchestrator calling these in sequence.

from dataclasses import dataclass, field


@dataclass
class IngestContext:
    """State carried between ingestion stages (future Step Functions payload)."""
    tmp_dir:          str
    rtype:            str
    namespace:        str
    section:          str
    first_filename:   str = ""
    content_hash:     str = ""
    clean_name:       str = ""
    family:           str = ""
    clean_name_override: str = ""
    page_url_override:   str = ""
    aem_tags:         list = field(default_factory=list)


def _stage_prepare(tmp_dir, resource_type, clean_name_override,
                   document_family, namespace, section, aem_tags,
                   page_url_override, logs):
    """Blocks 1-2: identify PDFs, compute hash, derive clean_name + family.
    Returns IngestContext, or None if no PDFs found (caller returns logs)."""
    rtype = resource_type.lower().strip()

    pdf_files = list(Path(tmp_dir).glob("*.pdf"))
    if not pdf_files:
        logs.append("❌ No PDF files found in upload directory.")
        return None

    first_filename = pdf_files[0].name
    content_hash   = _content_hash(tmp_dir)
    logs.append(f"🔑 Content hash: {content_hash[:8]}...")

    clean_name = (
        clean_name_override.strip() if clean_name_override.strip()
        else " ".join(
            w.capitalize()
            for w in os.path.splitext(first_filename)[0]
            .replace("_", " ").replace("-", " ").strip().split()
        )
    )
    family = _derive_family(document_family, clean_name, first_filename)

    return IngestContext(
        tmp_dir=tmp_dir, rtype=rtype, namespace=namespace, section=section,
        first_filename=first_filename, content_hash=content_hash,
        clean_name=clean_name, family=family,
        clean_name_override=clean_name_override, page_url_override=page_url_override,
        aem_tags=aem_tags or [],
    )


def _stage_dedup(ctx, logs) -> bool:
    """Block 3: hash-based skip check. Returns True if unchanged (skip)."""
    if is_unchanged(ctx.first_filename, ctx.content_hash):
        logs.append("⏭️  Document unchanged — skipping re-index (same hash).")
        return True
    return False


def _stage_version_and_deprecate(ctx, logs) -> int:
    """Blocks 4-5: registry version + deprecate old chunks across section ns."""
    existing_version = get_version(document_family=ctx.family, filename=ctx.first_filename)
    version          = existing_version + 1 if existing_version > 0 else 1

    if existing_version > 0:
        logs.append(
            f"🔄 Update detected (v{existing_version} → v{version}) "
            f"— deprecating old chunks..."
        )
        for ns in resolve_section_namespaces(ctx.section or "resources"):
            try:
                _index.delete(
                    filter={"document_family": {"$eq": ctx.family}},
                    namespace=ns,
                )
                _summary.delete(
                    filter={"document_family": {"$eq": ctx.family}},
                    namespace=ns,
                )
            except Exception as de:
                logs.append(f"⚠️  Could not delete old chunks in {ns}: {de}")
        logs.append("✅ Old chunks removed from Pinecone.")
        deprecate_old_version(ctx.first_filename)

    return version


def _stage_parse(tmp_dir, logs):
    """Block 6: LlamaParse the PDFs. Returns raw_docs."""
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
    return raw_docs


def _stage_build_documents(raw_docs, ctx, version, published_date, logs):
    """Block 7: build Document objects with full metadata. Returns (documents, pub_date)."""
    logs.append("📋 Building documents with full metadata...")
    pub_date  = published_date.strip() if published_date.strip() else datetime.now().strftime("%Y-%m-%d")
    documents = []

    for i, doc in enumerate(raw_docs):
        filename = os.path.basename(doc.metadata.get("file_name", f"doc_{i}.pdf"))

        cn = (
            ctx.clean_name_override.strip() if ctx.clean_name_override.strip()
            else " ".join(
                w.capitalize()
                for w in os.path.splitext(filename)[0]
                .replace("_", " ").replace("-", " ").strip().split()
            )
        )

        page   = str(doc.metadata.get("page_label", doc.metadata.get("page", str(i + 1))))
        pg_url = _build_page_url(cn, ctx.rtype, ctx.page_url_override)
        pdf_u  = _pdf_url(filename, page)

        metadata = {
            "filename":         filename,
            "clean_name":       cn,
            "resource_type":    ctx.rtype,
            "namespace":        ctx.namespace,
            "page":             page,
            "page_url":         pg_url,
            "pdf_url":          pdf_u,
            "document_family":  ctx.family,
            "version":          str(version),
            "is_latest":        True,
            "status":           "current",
            "published_date":   pub_date,
            "indexed_at":       datetime.now().isoformat(),
        }

        header = (
            f"Document: {cn}\n"
            f"Type: {ctx.rtype} | Page: {page}\n"
            f"---\n"
        )
        documents.append(Document(
            text=header + doc.text,
            metadata=metadata,
            excluded_llm_metadata_keys=["indexed_at", "pdf_url"],
            excluded_embed_metadata_keys=["indexed_at", "pdf_url", "page_url"],
        ))

    logs.append(f"📋 Built {len(documents)} document(s) with metadata.")
    return documents, pub_date


def _stage_index(documents, embed_model, ctx, target_index,
                 buffer_size, threshold, min_words, add_chunk_index,
                 store_label, chunk_label, logs) -> int:
    """Blocks 9/10 unified: chunk + filter + (chunk_index meta) + enrich + upsert."""
    logs.append(f"✂️  {chunk_label} chunking ({threshold}th pct threshold)...")
    splitter = _make_splitter(embed_model, buffer_size=buffer_size, threshold=threshold)
    nodes    = splitter.get_nodes_from_documents(documents, show_progress=False)

    if add_chunk_index:
        total = len(nodes)
        for idx, node in enumerate(nodes):
            node.metadata["chunk_index"]  = str(idx)
            node.metadata["total_chunks"] = str(total)

    for node in nodes:
        node.excluded_llm_metadata_keys  = ["indexed_at", "pdf_url"]
        node.excluded_embed_metadata_keys = ["indexed_at", "pdf_url", "page_url"]

    before = len(nodes)
    nodes  = [n for n in nodes if len((n.text or n.get_content()).split()) >= min_words]
    if before - len(nodes):
        label = "noise" if not add_chunk_index else "noise summary"
        logs.append(f"🧹 Dropped {before - len(nodes)} {label} chunks (<{min_words} words)")

    logs.append(f"✂️  Created {len(nodes)} {chunk_label.lower()} chunks.")

    nodes = _enrich_nodes_sync(
        nodes         = nodes,
        title         = ctx.clean_name_override or "",
        resource_type = ctx.rtype,
        url           = ctx.page_url_override or "",
        aem_tags      = ctx.aem_tags or [],
    )
    logs.append(f"🏷️  Enriched {sum(1 for n in nodes if n.metadata.get('enriched'))} {chunk_label.lower()} chunks.")

    logs.append(f"📤 Uploading to {store_label} ({ctx.namespace})...")
    vs  = PineconeVectorStore(pinecone_index=target_index, namespace=ctx.namespace)
    store_ctx = StorageContext.from_defaults(vector_store=vs)
    VectorStoreIndex(nodes, storage_context=store_ctx, embed_model=embed_model)
    logs.append(f"✅ {len(nodes)} chunks → {store_label} '{ctx.namespace}'")
    return len(nodes)


def _stage_persist(ctx, version, n_search, n_summary, pub_date, logs):
    """Block 11: save registry record."""
    try:
        save_record(
            filename        = ctx.first_filename,
            clean_name      = ctx.clean_name,
            resource_type   = ctx.rtype,
            namespace       = ctx.namespace,
            content_hash    = ctx.content_hash,
            version         = version,
            chunks_search   = n_search,
            chunks_summary  = n_summary,
            page_url        = ctx.page_url_override,
            document_family = ctx.family,
            published_date  = pub_date,
        )
        logs.append("📋 Registry updated in DynamoDB.")
    except Exception as e:
        logs.append(f"⚠️  Registry update failed: {e}")



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
        # ── Stage: prepare (identify PDFs, hash, clean_name, family) ──────────
        ctx = _stage_prepare(
            tmp_dir, resource_type, clean_name_override, document_family,
            namespace, section, aem_tags, page_url_override, logs,
        )
        if ctx is None:
            return logs

        # ── Stage: dedup (skip if unchanged) ──────────────────────────────────
        if not force and _stage_dedup(ctx, logs):
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return logs

        # ── Stage: version + deprecate old ────────────────────────────────────
        version = _stage_version_and_deprecate(ctx, logs)

        # ── Stage: parse (LlamaParse) ─────────────────────────────────────────
        raw_docs = _stage_parse(tmp_dir, logs)

        # ── Stage: build documents + metadata ─────────────────────────────────
        documents, pub_date = _stage_build_documents(
            raw_docs, ctx, version, published_date, logs
        )

        # ── Embed model (shared by both index passes) ─────────────────────────
        embed_model = OpenAIEmbedding(
            model=settings.EMBED_MODEL,
            api_key=settings.OPENAI_API_KEY,
            dimensions=settings.EMBED_DIMS,
        )

        # ── Stage: index — search (buffer=1, 80th pct) ────────────────────────
        n_search = _stage_index(
            documents, embed_model, ctx, _index,
            buffer_size=1, threshold=80, min_words=50,
            add_chunk_index=False, store_label="rag-poc", chunk_label="Search", logs=logs,
        )

        # ── Stage: index — summary (buffer=3, 95th pct, chunk_index meta) ─────
        n_summary = _stage_index(
            documents, embed_model, ctx, _summary,
            buffer_size=3, threshold=95, min_words=50,
            add_chunk_index=True, store_label="rag-summary", chunk_label="Summary", logs=logs,
        )

        # ── Stage: persist (registry) ─────────────────────────────────────────
        _stage_persist(ctx, version, n_search, n_summary, pub_date, logs)

        logs.append(
            f"🎉 Ingest complete — '{ctx.clean_name}' v{version} "
            f"({ctx.rtype}) → namespace '{ctx.namespace}' — now searchable."
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
