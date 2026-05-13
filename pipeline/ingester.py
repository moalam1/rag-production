"""
pipeline/ingester.py — Production-grade dual-index ingestion pipeline.

Writes to two Pinecone indexes simultaneously:
  rag-poc     — search chunks (300-500 tokens, SemanticSplitter 80th pct)
  rag-summary — summary chunks (1500-2000 tokens, SemanticSplitter 95th pct)

Metadata inherited on every chunk:
  filename, clean_name, resource_type, namespace, page, page_url, pdf_url,
  document_family, is_latest, status, published_date, version
"""
import os
import re
import json
import shutil
import hashlib
import logging
from datetime import datetime
from pathlib import Path

from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader, Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import MetadataMode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone

from config import settings
from pipeline.registry import (
    compute_hash, is_unchanged, get_version,
    save_record, deprecate_old_version
)

log = logging.getLogger(__name__)

# ── Pinecone clients ──────────────────────────────────────────────
_pc      = Pinecone(api_key=settings.PINECONE_API_KEY)
_index   = _pc.Index(settings.PINECONE_INDEX)
_summary = _pc.Index(settings.PINECONE_SUMMARY_INDEX)

# ── Namespace routing ─────────────────────────────────────────────
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
}

HF_REPO_ID = os.getenv("HF_REPO_ID", "perwaizalam/rag-poc-demo")
AEM_BASE   = os.getenv("AEM_BASE_URL", "https://www.equinix.com/resources")
TYPE_FOLDER = {
    "whitepaper": "whitepapers", "blueprint": "blueprints",
    "case-study": "case-studies", "analyst-report": "analyst-reports",
    "data-sheet": "data-sheets", "solution-brief": "solution-briefs",
    "playbook": "playbooks", "article": "articles",
    "media": "media", "multimedia": "videos", "webinar": "webinars",
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
    """SHA-256 hash of all PDF content in tmp_dir."""
    h = hashlib.sha256()
    for f in sorted(Path(tmp_dir).glob("*.pdf")):
        h.update(f.read_bytes())
    return h.hexdigest()[:32]


def _make_splitter(embed_model, buffer_size: int, threshold: int) -> SemanticSplitterNodeParser:
    return SemanticSplitterNodeParser(
        buffer_size=buffer_size,
        breakpoint_percentile_threshold=threshold,
        embed_model=embed_model,
    )


def ingest(
    tmp_dir: str,
    resource_type: str,
    clean_name_override: str = "",
    page_url_override: str   = "",
    document_family: str     = "",
    version: int             = 1,
) -> list[str]:
    """
    Full ingestion pipeline.

    Args:
        tmp_dir:             Directory containing PDF(s) to ingest.
        resource_type:       One of the 11 Equinix resource types.
        clean_name_override: Display name (uses filename if empty).
        page_url_override:   AEM resource page URL (auto-built if empty).
        document_family:     Logical group e.g. 'ai_infrastructure_guide'.
        version:             Document version number (1 = first index).

    Returns:
        List of log strings for display in HF Space.
    """
    rtype     = resource_type.lower().strip()
    namespace = NAMESPACE_MAP.get(rtype, "technical")
    logs      = []

    try:
        # ── Parse PDFs ────────────────────────────────────────────
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

        # ── Build documents with full metadata ────────────────────
        logs.append("📋 Building documents with full metadata...")
        family    = ""
        clean_name = clean_name_override.strip() if clean_name_override.strip() else first_filename
        today = datetime.now().strftime("%Y-%m-%d")
        documents = []

        for i, doc in enumerate(raw_docs):
            filename   = os.path.basename(doc.metadata.get("file_name", f"doc_{i}.pdf"))
            clean_name = clean_name_override.strip() if clean_name_override.strip() else (
                " ".join(w.capitalize() for w in
                         os.path.splitext(filename)[0]
                         .replace("_", " ").replace("-", " ").strip().split())
            )
            page   = str(doc.metadata.get("page_label", doc.metadata.get("page", str(i + 1))))
            pg_url = _build_page_url(clean_name, rtype, page_url_override)
            pdf_u  = _pdf_url(filename, page)

            # Document family slug — auto-derive from clean_name if not provided
            family = document_family.strip() if document_family.strip() else (
                re.sub(r"[^a-z0-9_]", "_",
                       clean_name.lower().replace(" ", "_"))[:40]
            )

            # Full metadata — inherited by every chunk via MetadataMode.ALL
            metadata = {
                # Identity
                "filename":        filename,
                "clean_name":      clean_name,
                "resource_type":   rtype,
                "namespace":       namespace,

                # Navigation
                "page":            page,
                "page_url":        pg_url,
                "pdf_url":         pdf_u,

                # Versioning
                "document_family": family,
                "version":         str(version),
                "is_latest":       "true",      # string — Pinecone metadata filter
                "status":          "current",

                # Freshness
                "published_date":  today,
                "indexed_at":      datetime.now().isoformat(),
            }

            # Prepend document header so chunks retain full context
            header = (
                f"Document: {clean_name}\n"
                f"Type: {rtype} | Page: {page}\n"
                f"---\n"
            )
            documents.append(Document(
                text=header + doc.text,
                metadata=metadata,
                excluded_llm_metadata_keys=[
                    "indexed_at", "pdf_url",
                ],
                excluded_embed_metadata_keys=[
                    "indexed_at", "pdf_url", "page_url",
                ],
            ))

        logs.append(f"📋 Built {len(documents)} document(s) with full metadata.")

        # ── Embed model ───────────────────────────────────────────
        embed_model = OpenAIEmbedding(
            model=settings.EMBED_MODEL,
            api_key=settings.OPENAI_API_KEY,
            dimensions=settings.EMBED_DIMS,
        )

        # ── Search index: 300-500 token chunks ────────────────────
        logs.append(f"✂️ Search chunking (80th pct threshold)...")
        search_splitter = _make_splitter(embed_model, buffer_size=1, threshold=80)
        search_nodes    = search_splitter.get_nodes_from_documents(
            documents, show_progress=False
        )

        # Ensure all metadata is present on every chunk
        for node in search_nodes:
            node.excluded_llm_metadata_keys   = ["indexed_at", "pdf_url"]
            node.excluded_embed_metadata_keys  = ["indexed_at", "pdf_url", "page_url"]

        logs.append(f"✂️ Created {len(search_nodes)} search chunks.")

        logs.append(f"📤 Uploading to rag-poc ({namespace})...")
        search_vs  = PineconeVectorStore(pinecone_index=_index, namespace=namespace)
        search_ctx = StorageContext.from_defaults(vector_store=search_vs)
        VectorStoreIndex(search_nodes, storage_context=search_ctx, embed_model=embed_model)
        logs.append(f"✅ {len(search_nodes)} chunks → rag-poc '{namespace}'")

        # ── Summary index: 1500-2000 token chunks ─────────────────
        logs.append(f"✂️ Summary chunking (95th pct threshold)...")
        summary_splitter = _make_splitter(embed_model, buffer_size=3, threshold=95)
        summary_nodes    = summary_splitter.get_nodes_from_documents(
            documents, show_progress=False
        )

        # Add summary-specific metadata
        total = len(summary_nodes)
        for idx, node in enumerate(summary_nodes):
            node.metadata["chunk_index"]  = str(idx)
            node.metadata["total_chunks"] = str(total)
            node.excluded_llm_metadata_keys  = ["indexed_at", "pdf_url"]
            node.excluded_embed_metadata_keys = ["indexed_at", "pdf_url", "page_url"]

        logs.append(f"✂️ Created {len(summary_nodes)} summary chunks.")

        logs.append(f"📤 Uploading to rag-summary ({namespace})...")
        summary_vs  = PineconeVectorStore(pinecone_index=_summary, namespace=namespace)
        summary_ctx = StorageContext.from_defaults(vector_store=summary_vs)
        VectorStoreIndex(summary_nodes, storage_context=summary_ctx, embed_model=embed_model)
        logs.append(f"✅ {len(summary_nodes)} chunks → rag-summary '{namespace}'")

        # ── Save to DynamoDB registry ─────────────────────────────
        try:
            # Derive first_filename and content_hash if hash check block not reached
            import os as _os
            _pdfs = [f for f in _os.listdir(tmp_dir) if f.endswith('.pdf')] if _os.path.exists(tmp_dir) else []
            first_filename = documents[0].metadata.get('filename', 'unknown.pdf') if documents else 'unknown.pdf'
            try:
                content_hash
            except NameError:
                content_hash = 'manual'
            try:
                family
            except NameError:
                family = document_family.strip() if document_family.strip() else 'unknown'
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
                document_family = document_family or family,
            )
            logs.append(f"📋 Registry updated in DynamoDB.")
        except Exception as e:
            logs.append(f"⚠️  Registry update failed: {e}")

        logs.append(
            f"🎉 Ingest complete — '{clean_name}' v{version} "
            f"({rtype}) is now searchable."
        )

    except Exception as e:
        logs.append(f"❌ Error: {e}")
        log.exception("Ingestion failed")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return logs
