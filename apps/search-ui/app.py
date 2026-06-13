import sys
import asyncio.base_events

# Suppress Python 3.13 asyncio cleanup noise
if sys.version_info >= (3, 13):
    _orig_del = asyncio.base_events.BaseEventLoop.__del__
    def _safe_del(self):
        try:
            _orig_del(self)
        except Exception:
            pass
    asyncio.base_events.BaseEventLoop.__del__ = _safe_del

import os
import re
import json
import shutil
import tempfile
import traceback
import requests
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# ── EC2 API Gateway — all search handled here ─────────────────────
API_GATEWAY_URL = os.getenv("API_GATEWAY_URL", "https://lxhxqqh3r8.execute-api.us-east-1.amazonaws.com")
API_KEY         = os.getenv("API_KEY", "")

# Badge styles — loaded from API on first use, cached for session
_BADGE_STYLES_CACHE: dict = {}
_BADGE_STYLES_FALLBACK = {
    "Distributed AI":        {"icon":"🤖","bg":"#0c2340","color":"#93c5fd"},
    "AI & Machine Learning": {"icon":"🤖","bg":"#0c2340","color":"#93c5fd"},
    "SD-WAN":                {"icon":"🔀","bg":"#1a1200","color":"#fcd34d"},
    "Hybrid Multicloud":     {"icon":"☁️", "bg":"#0a1628","color":"#60a5fa"},
    "Financial Services":    {"icon":"🏦","bg":"#0a1a0a","color":"#86efac"},
    "Network Modernization": {"icon":"🔧","bg":"#1a0a1a","color":"#d8b4fe"},
    "Colocation":            {"icon":"🏢","bg":"#1a1a0a","color":"#fde68a"},
    "Interconnection":       {"icon":"🔗","bg":"#0f1a1a","color":"#5eead4"},
}

def _get_badge_styles() -> dict:
    """Fetch badge styles from API — falls back to hardcoded if unavailable."""
    global _BADGE_STYLES_CACHE
    if _BADGE_STYLES_CACHE:
        return _BADGE_STYLES_CACHE
    try:
        import requests as _req
        resp = _req.get(
            f"{API_GATEWAY_URL}/api/v1/config/badge-styles",
            headers={"X-API-Key": API_KEY},
            timeout=3
        )
        if resp.status_code == 200:
            styles = resp.json().get("badge_styles", {})
            if styles:
                _BADGE_STYLES_CACHE = styles
                print(f"✓ Badge styles loaded from API: {list(styles.keys())}")
                return _BADGE_STYLES_CACHE
    except Exception as e:
        print(f"Badge styles fetch failed — using fallback: {e}")
    _BADGE_STYLES_CACHE = _BADGE_STYLES_FALLBACK
    return _BADGE_STYLES_CACHE


import gradio as gr
from openai import OpenAI
from pinecone import Pinecone
import cohere
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from huggingface_hub import HfApi

# ================================================================
# Clients
# ================================================================
openai_client  = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc             = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index         = pc.Index("rag-poc")
pinecone_summary_index = pc.Index(os.getenv("PINECONE_SUMMARY_INDEX", "rag-summary"))
co             = cohere.Client(os.getenv("COHERE_API_KEY"))
hf_api         = HfApi(token=os.getenv("HF_TOKEN"))

EMBED_MODEL    = "text-embedding-3-small"
EMBED_DIMS     = 1024
TOP_K_RETRIEVE = 20
TOP_K_RERANK   = 5

HF_REPO_ID   = os.getenv("HF_REPO_ID", "perwaizalam/rag-poc-demo")
HF_REPO_TYPE = "space"
PDF_DIR      = "pdfs"

LOCAL_PDF_DIR = Path("/app/pdfs")
LOCAL_PDF_DIR.mkdir(parents=True, exist_ok=True)

DOC_TOPIC = os.getenv("DOC_TOPIC", "enterprise technology, data centers, and digital infrastructure")

# ── Namespace routing ─────────────────────────────────────────────
NAMESPACE_MAP = {
    "whitepaper":      "technical",
    "blueprint":       "technical",
    "analyst-report":  "technical",
    "data-sheet":      "technical",
    "playbook":        "technical",
    "case-study":      "business",
    "solution-brief":  "business",
    "article":         "business",
    "media":           "business",
    "multimedia":      "media",
    "webinar":         "media",
}

# ── AEM page URL builder ──────────────────────────────────────────
AEM_BASE_URL = os.getenv("AEM_BASE_URL", "https://www.equinix.com/resources")

TYPE_FOLDER = {
    "whitepaper":      "whitepapers",
    "blueprint":       "blueprints",
    "case-study":      "case-studies",
    "analyst-report":  "analyst-reports",
    "data-sheet":      "data-sheets",
    "solution-brief":  "solution-briefs",
    "playbook":        "playbooks",
    "article":         "articles",
    "media":           "media",
    "multimedia":      "videos",
    "webinar":         "webinars",
}

RESOURCE_TYPE_LABELS = [
    "whitepaper", "blueprint", "case-study", "analyst-report",
    "data-sheet", "solution-brief", "playbook", "article",
    "media", "multimedia", "webinar",
]


def build_page_url(clean_name: str, resource_type: str, page_url: str = None) -> str:
    """Build AEM resource page URL. Uses explicit page_url if provided."""
    if page_url and page_url.strip():
        return page_url.strip()
    folder = TYPE_FOLDER.get(resource_type.lower(), "resources")
    slug   = clean_name.lower()
    slug   = re.sub(r"[_\s]+", "-", slug)
    slug   = re.sub(r"[^a-z0-9\-]", "", slug)
    slug   = re.sub(r"-{2,}", "-", slug).strip("-")
    return f"{AEM_BASE_URL}/{folder}/{slug}"

SYSTEM_PROMPT = """You are a multilingual research assistant. Answer the query using ONLY the provided context chunks.

Rules:
- LANGUAGE: You will be told the exact language to respond in via the user message. Always follow it strictly.
- Write a clear, flowing answer of 2-4 sentences.
- Each chunk starts with "Document: <name>" — use that name when citing.
- Cite sources inline using [1], [2], etc. matching the chunk numbers provided.
- Be factual and concise.
- Do NOT make up information not in the context.
- If nothing relevant found, say the equivalent of "I couldn't find relevant information in the documents." in the specified language.
- Generate follow-up questions in the SAME specified language.

Return ONLY a JSON object with exactly these two fields:
{
  "answer": "Your answer here with inline [1] citations [2].",
  "followups": ["Follow-up question 1?", "Follow-up question 2?", "Follow-up question 3?"]
}
No markdown, no explanation outside the JSON.
"""


def pdf_url(filename: str, page) -> str:
    base = f"https://huggingface.co/spaces/{HF_REPO_ID}/resolve/main/{PDF_DIR}/{filename}"
    try:
        p = int(page)
        return f"{base}#page={p}"
    except (ValueError, TypeError):
        return base


# ================================================================
# GUARDRAILS
# ================================================================

# ── Patterns that indicate prompt injection attempts ──
INJECTION_PATTERNS = [
    r"ignore (previous|all|above) instructions",
    r"you are now",
    r"act as (if you are|a|an)",
    r"jailbreak",
    r"system prompt",
    r"forget everything",
    r"disregard (all|previous|your)",
    r"new persona",
    r"pretend (you are|to be)",
]

# ── PII patterns to block from leaking in answers ──
PII_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",                                      # SSN
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",      # email
    r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",                   # credit card
    r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",                          # phone number
]


def guardrail_query_length(query: str) -> tuple[bool, str]:
    if len(query.strip()) < 3:
        return False, "⚠️ Query too short. Please ask a complete question."
    if len(query) > 1000:
        return False, "⚠️ Query too long. Please keep it under 1000 characters."
    return True, ""


def guardrail_injection(query: str) -> tuple[bool, str]:
    q = query.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, q):
            return False, "⚠️ Invalid query detected. Please ask a genuine question."
    return True, ""


def guardrail_relevance(query: str) -> tuple[bool, str]:
    """Use a cheap GPT call to check if query is on-topic."""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": (
                    f"Is this query relevant to {DOC_TOPIC}?\n"
                    f"Query: \"{query}\"\n"
                    f"Answer with only YES or NO."
                )
            }],
            max_tokens=5,
            temperature=0,
        )
        answer = response.choices[0].message.content.strip().upper()
        if "NO" in answer:
            return False, f"⚠️ This question is outside the scope of the document library. Try asking about {DOC_TOPIC}."
    except Exception:
        pass  # if the check fails, allow the query through
    return True, ""


def run_input_guardrails(query: str) -> tuple[bool, str]:
    """Run all input checks. Returns (passed, message)."""
    # ── ALL GUARDRAILS DISABLED ──
    return True, ""


def guardrail_pii(answer: str) -> tuple[bool, str]:
    """Block answers that contain PII."""
    # ── DISABLED ──
    return True, ""


def guardrail_citations(answer: str, sources: list) -> tuple[bool, str]:
    """Ensure the answer has at least one citation."""
    # ── DISABLED ──
    return True, ""


def guardrail_grounding(answer: str, context: str) -> tuple[bool, str]:
    """Check that the answer is grounded in the retrieved context."""
    # ── DISABLED ──
    return True, ""


def run_output_guardrails(answer: str, context: str, sources: list) -> tuple[bool, str]:
    """Run all output checks. Returns (passed, message)."""
    # ── ALL GUARDRAILS DISABLED ──
    return True, ""


def blocked_html(query: str, message: str) -> str:
    """Render a friendly blocked-result card."""
    return f"""
    <div style="background:linear-gradient(135deg,#e8d5f5 0%,#f5e0d0 50%,#fde8d0 100%);
                border:1px solid rgba(0,0,0,0.08);border-radius:16px;
                padding:24px 28px;margin-top:16px;font-family:'Inter',sans-serif;">
        <div style="font-size:0.72rem;color:#888;text-transform:uppercase;
                    letter-spacing:0.1em;margin-bottom:14px;">🔍 &nbsp;{query}</div>
        <div style="display:flex;align-items:center;gap:12px;
                    background:rgba(255,255,255,0.7);border:1px solid rgba(194,65,12,0.2);
                    border-radius:10px;padding:16px 20px;">
            <span style="font-size:1.4rem;">🛡️</span>
            <div>
                <div style="font-size:0.85rem;font-weight:600;color:#c2410c;margin-bottom:4px;">
                    Query blocked by guardrail
                </div>
                <div style="font-size:0.9rem;color:#333;">{message}</div>
            </div>
        </div>
    </div>"""


# ================================================================
# INGEST + INDEX pipeline
# ================================================================

def ingest_and_index(files, resource_type, clean_name_input="", page_url_input="", progress=gr.Progress()):
    if not files:
        return "⚠️ No files uploaded."
    rtype     = resource_type.lower().strip()
    namespace = NAMESPACE_MAP.get(rtype, "technical")
    logs = []
    try:
        progress(0.05, desc="Preparing files...")
        tmp_dir = tempfile.mkdtemp()

        # Normalise — Gradio may pass a string path or a file object
        if not isinstance(files, list):
            files = [files]

        for file in files:
            file_path = file if isinstance(file, str) else file.name
            filename  = os.path.basename(file_path)
            dest      = os.path.join(tmp_dir, filename)
            shutil.copy(file_path, dest)
            local_dest = LOCAL_PDF_DIR / filename
            shutil.copy(file_path, local_dest)
            try:
                hf_api.upload_file(
                    path_or_fileobj=str(local_dest),
                    path_in_repo=f"{PDF_DIR}/{filename}",
                    repo_id=HF_REPO_ID,
                    repo_type=HF_REPO_TYPE,
                )
                logs.append(f"☁️  Uploaded to HF repo: {filename}")
            except Exception as e:
                logs.append(f"⚠️  HF upload skipped ({e}) — links may not work")
            logs.append(f"📁 Received: {filename}")

        progress(0.2, desc="Parsing PDFs with LlamaParse...")
        logs.append("🔍 Parsing PDFs...")
        parser = LlamaParse(
            api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
            result_type="markdown",
            verbose=False,
            language="en",
            parsing_instructions="Extract headings and tables accurately.",
        )
        file_extractor = {".pdf": parser}
        reader = SimpleDirectoryReader(tmp_dir, file_extractor=file_extractor)
        raw_docs = reader.load_data()
        logs.append(f"✅ Parsed {len(raw_docs)} document(s).")

        progress(0.4, desc="Preparing documents...")
        documents = []
        for i, doc in enumerate(raw_docs):
            filename   = os.path.basename(doc.metadata.get("file_name", f"uploaded_doc_{i}.pdf"))

            # Use user-provided display name if given, else derive from filename
            if clean_name_input and clean_name_input.strip():
                clean_name = clean_name_input.strip()
            else:
                clean_name = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ").strip()
                clean_name = " ".join(w.capitalize() for w in clean_name.split())

            page   = doc.metadata.get("page_label", doc.metadata.get("page", str(i + 1)))
            pdf_u  = pdf_url(filename, page)

            # Use user-provided AEM page URL if given, else derive from clean_name + type
            pg_url = page_url_input.strip() if page_url_input.strip() else build_page_url(clean_name, rtype)

            logs.append(f"📋 {clean_name} → {pg_url}")

            # Prepend document name so GPT-4o knows source while reading context
            text_with_name = f"Document: {clean_name}\n\n{doc.text}"

            documents.append(Document(
                text=text_with_name,
                metadata={
                    "filename":      filename,
                    "clean_name":    clean_name,
                    "resource_type": rtype,
                    "namespace":     namespace,
                    "page":          str(page),
                    "page_url":      pg_url,
                    "pdf_url":       pdf_u,
                }
            ))

        progress(0.55, desc="Chunking for rag-poc (search)...")
        logs.append(f"✂️ Chunking for search → namespace: {namespace}...")
        embed_model = OpenAIEmbedding(
            model=EMBED_MODEL,
            api_key=os.getenv("OPENAI_API_KEY"),
            dimensions=EMBED_DIMS,
        )

        # ── Search index: 300-500 token chunks ────────────────────
        search_splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=80,
            embed_model=embed_model,
        )
        search_nodes = search_splitter.get_nodes_from_documents(documents)
        logs.append(f"✂️ Created {len(search_nodes)} search chunks.")

        progress(0.65, desc="Uploading to rag-poc (search)...")
        logs.append(f"📤 Uploading to rag-poc namespace: {namespace}...")
        search_vs  = PineconeVectorStore(pinecone_index=pinecone_index, namespace=namespace)
        search_ctx = StorageContext.from_defaults(vector_store=search_vs)
        VectorStoreIndex(search_nodes, storage_context=search_ctx, embed_model=embed_model)
        logs.append(f"✅ Indexed {len(search_nodes)} chunks into rag-poc '{namespace}'.")

        # ── Summary index: 1500-2000 token chunks ─────────────────
        progress(0.75, desc="Chunking for rag-summary (AI summary)...")
        logs.append(f"✂️ Chunking for summary → larger chunks...")
        summary_splitter = SemanticSplitterNodeParser(
            buffer_size=3,
            breakpoint_percentile_threshold=95,
            embed_model=embed_model,
        )
        summary_nodes = summary_splitter.get_nodes_from_documents(documents)
        logs.append(f"✂️ Created {len(summary_nodes)} summary chunks.")

        for idx, node in enumerate(summary_nodes):
            node.metadata["chunk_index"]  = idx
            node.metadata["total_chunks"] = len(summary_nodes)

        progress(0.88, desc="Uploading to rag-summary (AI summary)...")
        logs.append(f"📤 Uploading to rag-summary namespace: {namespace}...")
        summary_vs  = PineconeVectorStore(pinecone_index=pinecone_summary_index, namespace=namespace)
        summary_ctx = StorageContext.from_defaults(vector_store=summary_vs)
        VectorStoreIndex(summary_nodes, storage_context=summary_ctx, embed_model=embed_model)
        logs.append(f"✅ Indexed {len(summary_nodes)} chunks into rag-summary '{namespace}'.")

        shutil.rmtree(tmp_dir, ignore_errors=True)
        progress(1.0, desc="Done!")
        logs.append("🎉 Ingest complete — document available in search and AI summary.")
    except Exception as e:
        logs.append(f"❌ Error: {str(e)}\n{traceback.format_exc()}")
    return "\n".join(logs)


def ingest_via_url(page_url_input: str, progress=gr.Progress()) -> str:
    """
    Ingest an Equinix resource page URL via the EC2 ingest API.
    Handles page teaser + PDF detection automatically.
    """
    url = page_url_input.strip() if page_url_input else ""
    if not url:
        return "⚠️ Please enter a URL."
    if not url.startswith("https://www.equinix.com/resources/"):
        return "⚠️ URL must start with https://www.equinix.com/resources/"

    logs = []
    try:
        progress(0.1, desc="Submitting ingest job...")
        logs.append(f"🔗 Submitting: {url}")

        # Submit ingest job
        resp = requests.post(
            f"{API_GATEWAY_URL}/api/v1/ingest",
            json={"page_url": url},
            headers={"Content-Type": "application/json", "X-API-Key": API_KEY},
            timeout=30,
        )
        resp.raise_for_status()
        data   = resp.json()
        job_id = data.get("job_id")
        logs.append(f"✅ Job created: {job_id}")
        logs.append(f"⏳ Polling for status...")

        # Poll for completion
        import time
        for attempt in range(30):
            time.sleep(5)
            progress(0.1 + (attempt / 30) * 0.85, desc=f"Processing... ({attempt * 5}s)")

            poll = requests.get(
                f"{API_GATEWAY_URL}/api/v1/ingest/{job_id}",
                headers={"X-API-Key": API_KEY},
                timeout=15,
            )
            poll.raise_for_status()
            status_data = poll.json()
            status      = status_data.get("status", "")

            if status == "complete":
                progress(1.0, desc="Done!")
                logs.append(f"✅ {status_data.get('message', 'Ingest complete')}")
                # Show key log lines from the job
                job_logs = status_data.get("logs", [])
                for line in job_logs:
                    if any(x in line for x in ["✅", "⏭️", "❌", "🎉", "chunks", "→"]):
                        logs.append(f"   {line}")
                break
            elif status == "failed":
                logs.append(f"❌ {status_data.get('message', 'Ingest failed')}")
                break
        else:
            logs.append("⚠️ Timed out waiting for job — check status manually.")

    except requests.exceptions.ConnectionError:
        logs.append("❌ Cannot reach ingest API. Check EC2 is running.")
    except Exception as e:
        logs.append(f"❌ Error: {str(e)}")

    return "\n".join(logs)


# ================================================================
# QUERY / SEARCH pipeline
# ================================================================

def embed_query(query: str) -> list:
    response = openai_client.embeddings.create(
        input=query,
        model=EMBED_MODEL,
        dimensions=EMBED_DIMS,
    )
    return response.data[0].embedding


def retrieve_chunks(query_embedding: list, namespace: str = None) -> list:
    """Retrieve chunks from Pinecone. If namespace is None, query all namespaces."""
    chunks = []

    # If no namespace specified, query all namespaces
    namespaces_to_query = [None, "technical", "business", "media"] if namespace is None else [namespace]

    for ns in namespaces_to_query:
        query_kwargs = dict(
            vector=query_embedding,
            top_k=TOP_K_RETRIEVE,
            include_metadata=True,
        )

        # Only add namespace if it's not None
        if ns is not None:
            query_kwargs["namespace"] = ns

        results = pinecone_index.query(**query_kwargs)

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

            filename      = metadata.get("filename")      or inner_meta.get("filename", "unknown")
            clean_name    = (metadata.get("clean_name")   or inner_meta.get("clean_name")
                             or filename.replace("_", " ").replace("-", " ").replace(".pdf", "").title())
            page          = (metadata.get("page_label")   or metadata.get("page")
                             or inner_meta.get("page")    or inner_meta.get("page_label") or "?")
            resource_type = (metadata.get("resource_type") or inner_meta.get("resource_type", ""))
            page_url      = (metadata.get("page_url")     or inner_meta.get("page_url", ""))
            stored_pdf    = (metadata.get("pdf_url")      or inner_meta.get("pdf_url")
                             or pdf_url(filename, page))

            chunks.append({
                "id":            match.id,
                "score":         match.score,
                "text":          text,
                "filename":      filename,
                "clean_name":    clean_name,
                "resource_type": resource_type,
                "page":          page,
                "page_url":      page_url,
                "pdf_url":       stored_pdf,
            })

    # Remove duplicates and sort by score (highest first)
    seen = set()
    unique_chunks = []
    for chunk in chunks:
        if chunk["id"] not in seen:
            seen.add(chunk["id"])
            unique_chunks.append(chunk)

    unique_chunks.sort(key=lambda x: x["score"], reverse=True)
    return unique_chunks[:TOP_K_RETRIEVE]


def rerank_chunks(query: str, chunks: list) -> list:
    if not chunks:
        return []
    docs = [c["text"] for c in chunks]
    response = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=docs,
        top_n=TOP_K_RERANK,
    )
    reranked = []
    for result in response.results:
        chunk = chunks[result.index].copy()
        chunk["rerank_score"] = result.relevance_score
        reranked.append(chunk)
    return reranked


def build_context(chunks: list) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Chunk {i}]\n"
            f"Document: {chunk.get('clean_name', chunk['filename'])}\n"
            f"Resource type: {chunk.get('resource_type', 'unknown')}\n"
            f"Page: {chunk['page']}\n\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def detect_and_translate(query: str) -> tuple[str, str]:
    """
    Detect the language of the query and translate to English for retrieval.
    Returns (translated_query, detected_language_code).
    English queries are returned as-is.
    Uses gpt-4o-mini — cheap and fast.
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": (
                    "Detect the language of this text and translate it to English if it is not already English.\n"
                    "Return ONLY a JSON object with two fields:\n"
                    "  \"language\": ISO 639-1 language code (e.g. \"en\", \"fr\", \"ja\", \"ar\", \"es\")\n"
                    "  \"translated\": the text translated to English (same text if already English)\n\n"
                    f"Text: {query}"
                )
            }],
            max_tokens=300,
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())
        lang        = result.get("language", "en")
        translated  = result.get("translated", query)
        return translated, lang
    except Exception as e:
        print(f"detect_and_translate error: {e}")
        return query, "en"   # fallback — use original query, assume English


def generate_answer(query: str, context: str, detected_lang: str = "en") -> dict:
    # Map language code to full name for clearer instruction to GPT-4o
    LANG_NAMES = {
        "en": "English", "fr": "French", "es": "Spanish", "de": "German",
        "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ja": "Japanese",
        "zh": "Chinese", "ko": "Korean", "ar": "Arabic", "hi": "Hindi",
        "ru": "Russian", "tr": "Turkish", "pl": "Polish", "sv": "Swedish",
        "da": "Danish", "fi": "Finnish", "no": "Norwegian", "id": "Indonesian",
    }
    lang_name = LANG_NAMES.get(detected_lang, "English")

    user_message = (
        f"IMPORTANT: Respond entirely in {lang_name}. "
        f"The answer, citations, and all follow-up questions must be in {lang_name}.\n\n"
        f"Query: {query}\n\nContext:\n{context}"
    )

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.2,
            max_tokens=600,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        parsed = json.loads(raw)
        return {
            "answer":    parsed.get("answer", ""),
            "followups": parsed.get("followups", []),
        }
    except Exception as e:
        print(f"generate_answer error: {e}")
        return {"answer": "Error generating answer.", "followups": []}


# ================================================================
# RENDER
# ================================================================

def render_result_html(query: str, answer: str, sources: list, followups: list, cached: bool = False) -> str:
    # Inline citation badges
    styled_answer = answer
    # Replace [N] citations — whether or not source cards exist
    import re as _re
    def make_badge(m):
        n = m.group(1)
        return (
            f'<sup style="display:inline-flex;align-items:center;justify-content:center;'
            f'width:16px;height:16px;background:#c2410c;color:#fff;border-radius:50%;'
            f'font-size:0.6rem;font-weight:700;margin:0 2px;vertical-align:super;line-height:1;">{n}</sup>'
        )
    styled_answer = _re.sub(r'\[(\d+)\]', make_badge, styled_answer)

    # Source cards
    source_cards = []
    for i, chunk in enumerate(sources, 1):
        filename      = chunk.get("filename", "unknown")
        fname         = chunk.get("clean_name") or filename.replace("_", " ").replace(".pdf", "").title()
        page          = chunk.get("page", "?")
        page_url      = chunk.get("page_url", "")
        pdf_u         = chunk.get("pdf_url", "#")
        resource_type = chunk.get("resource_type", "")
        text_value    = chunk.get("text") or chunk.get("preview") or ""
        preview       = text_value[:130].strip().replace("\n", " ")
        if len(text_value) > 130:
            preview += "…"
        page_label    = "Open PDF"

        # Resource type badge colour
        badge_colors = {
            "whitepaper": "#185FA5", "blueprint": "#185FA5", "analyst-report": "#185FA5",
            "data-sheet": "#185FA5", "playbook": "#185FA5",
            "case-study": "#0F6E56", "solution-brief": "#0F6E56", "article": "#0F6E56",
            "media": "#0F6E56",
            "multimedia": "#854F0B", "webinar": "#854F0B",
        }
        badge_color = badge_colors.get(resource_type, "#888")
        badge_html  = (
            f'<span style="font-size:0.65rem;font-weight:700;color:#fff;'
            f'background:{badge_color};border-radius:4px;padding:1px 6px;'
            f'text-transform:uppercase;letter-spacing:0.05em;">'
            f'{resource_type or "doc"}</span>'
        ) if resource_type else ""

        # Buttons: Read full resource (AEM page) + View PDF page
        btn_read = (
            f'<a href="{page_url}" target="_blank" rel="noopener noreferrer" '
            f'style="text-decoration:none;font-size:0.68rem;font-weight:600;color:#fff;'
            f'background:#c2410c;border-radius:4px;padding:3px 8px;white-space:nowrap;">'
            f'Read resource →</a>'
        ) if page_url else ""

        # PDF button intentionally removed per UI request.
        btn_pdf = ""

        # AI Summary trigger — textarea pattern (same as follow-up questions)
        trigger       = f"__summarise__:{filename}"
        btn_summary   = (
            f'<div onclick="'
            f"var ta=document.querySelector(\'textarea\');"
            f"var nv=Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype,\'value\').set;"
            f"nv.call(ta,\'{trigger}\');"
            f"ta.dispatchEvent(new Event(\'input\',{{bubbles:true}}));"
            f"ta.dispatchEvent(new Event(\'change\',{{bubbles:true}}));"
            f"ta.focus();"
            f"setTimeout(function(){{"
            f"ta.dispatchEvent(new KeyboardEvent(\'keydown\',{{key:\'Enter\',code:\'Enter\',keyCode:13,bubbles:true}}));"
            f"ta.dispatchEvent(new KeyboardEvent(\'keypress\',{{key:\'Enter\',code:\'Enter\',keyCode:13,bubbles:true}}));"
            f"ta.dispatchEvent(new KeyboardEvent(\'keyup\',{{key:\'Enter\',code:\'Enter\',keyCode:13,bubbles:true}}));"
            f"}},300);"
            f'" '
            f'style="display:inline-flex;align-items:center;justify-content:center;'
            f'font-size:0.68rem;font-weight:600;line-height:1;color:#7c3aed;'
            f'border:1px solid #7c3aed;border-radius:4px;background:transparent;'
            f'padding:3px 8px;height:24px;box-sizing:border-box;white-space:nowrap;'
            f'vertical-align:middle;cursor:pointer;">'
            f'✨ AI Summary</div>'
        )

        source_cards.append(f"""
        <div class="source-card" style="background:rgba(255,255,255,0.7);border:1px solid rgba(0,0,0,0.08);
                    border-radius:10px;padding:14px;min-width:220px;max-width:260px;flex-shrink:0;"
             onmouseover="this.style.borderColor='#c2410c'"
             onmouseout="this.style.borderColor='rgba(0,0,0,0.08)'">
            <div style="display:flex;align-items:center;gap:6px;margin-bottom:8px;flex-wrap:wrap;">
                <span style="background:#c2410c;color:#fff;border-radius:50%;width:20px;height:20px;
                             min-width:20px;display:flex;align-items:center;justify-content:center;
                             font-size:0.65rem;font-weight:700;">{i}</span>
                <span style="font-size:0.78rem;font-weight:600;color:#111;white-space:nowrap;
                             overflow:hidden;text-overflow:ellipsis;flex:1;">{fname}</span>
                {badge_html}
            </div>
            <div style="font-size:0.72rem;color:#555;line-height:1.5;margin-bottom:10px;">{preview}</div>
            <div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center;">
                {btn_summary}
                {btn_read}
                {btn_pdf}
            </div>
        </div>""")

    sources_html = f"""
    <div style="margin-top:20px;">
        <div style="font-size:0.72rem;font-weight:700;color:#888;
                    text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px;">Sources</div>
        <div class="mobile-sources" style="display:flex;gap:10px;overflow-x:auto;padding-bottom:6px;max-width:100%;">
            {"".join(source_cards)}
        </div>
    </div>""" if source_cards else ""

    # Follow-up questions
    followup_html = ""
    if followups:
        fq_items = []
        for fq in followups[:3]:
            fq_js = fq.replace("\\", "\\\\").replace("'", "\\'")
            fq_items.append(f"""
            <div onclick="
                var ta = document.querySelector('textarea');
                var nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
                nativeInputValueSetter.call(ta, '{fq_js}');
                ta.dispatchEvent(new Event('input', {{bubbles: true}}));
                ta.dispatchEvent(new Event('change', {{bubbles: true}}));
                ta.focus();
                setTimeout(function() {{
                    ta.dispatchEvent(new KeyboardEvent('keydown', {{key:'Enter',code:'Enter',keyCode:13,bubbles:true}}));
                    ta.dispatchEvent(new KeyboardEvent('keypress', {{key:'Enter',code:'Enter',keyCode:13,bubbles:true}}));
                    ta.dispatchEvent(new KeyboardEvent('keyup', {{key:'Enter',code:'Enter',keyCode:13,bubbles:true}}));
                }}, 300);
            " style="border:1px solid rgba(0,0,0,0.1);border-radius:8px;padding:11px 16px;
                     font-size:0.87rem;color:#333;display:flex;align-items:center;
                     gap:10px;background:rgba(255,255,255,0.6);margin-bottom:8px;cursor:pointer;"
               onmouseover="this.style.borderColor='#c2410c';this.style.color='#c2410c';this.style.background='rgba(255,255,255,0.9)';"
               onmouseout="this.style.borderColor='rgba(0,0,0,0.1)';this.style.color='#333';this.style.background='rgba(255,255,255,0.6)';">
                <span style="color:#c2410c;font-size:1rem;">↳</span> {fq}
            </div>""")

        followup_html = f"""
        <div style="margin-top:20px;">
            <div style="font-size:0.72rem;font-weight:700;color:#888;
                        text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px;">
                Related questions
            </div>
            {"".join(fq_items)}
        </div>"""

    if cached:
        cache_badge = (
            '<span style="display:inline-flex;align-items:center;gap:4px;'
            'font-size:0.65rem;font-weight:700;color:#166534;'
            'background:#dcfce7;border:1px solid #86efac;'
            'border-radius:20px;padding:2px 8px;margin-left:8px;'
            'vertical-align:middle;letter-spacing:0.03em;">'
            '⚡ Cached</span>'
        )
    else:
        cache_badge = (
            '<span style="display:inline-flex;align-items:center;gap:4px;'
            'font-size:0.65rem;font-weight:700;color:#1e40af;'
            'background:#dbeafe;border:1px solid #93c5fd;'
            'border-radius:20px;padding:2px 8px;margin-left:8px;'
            'vertical-align:middle;letter-spacing:0.03em;">'
            '🔍 Live</span>'
        )

    feedback_bar = ''

    return f"""
    <div style="background:linear-gradient(135deg,#e8d5f5 0%,#f5e0d0 50%,#fde8d0 100%);
                border:1px solid rgba(0,0,0,0.08);border-radius:16px;
                padding:24px 28px;margin-top:16px;font-family:'Inter',sans-serif;">
        <div style="display:flex;align-items:center;margin-bottom:14px;flex-wrap:wrap;gap:6px;">
            <span style="font-size:0.72rem;color:#888;text-transform:uppercase;
                         letter-spacing:0.1em;">🔍 &nbsp;{query}</span>
            {cache_badge}
        </div>
        <div style="font-size:1rem;color:#0f0f0f;line-height:1.85;font-weight:400;
                    border-left:3px solid #c2410c;padding-left:16px;">{styled_answer}</div>
        {sources_html}
        {followup_html}

    </div>"""


from collections import Counter
from datetime import datetime

# ================================================================
# LEAD GENERATION NUDGES — contextual support cards
# ================================================================
# Never blocks search. Appears alongside results when signals fire.
# Three patterns: dead-end, circular loop, commercial shift.
# ================================================================

COMMERCIAL_KEYWORDS = {
    "deployment", "provisioning", "cross-connect", "pricing",
    "contract", "sla", "quote", "trial", "poc", "pilot",
    "capacity", "enterprise", "dedicated", "migration",
    "timeline", "lead time", "cost", "bandwidth tier",
    "account manager", "solutions architect", "rollout",
}

FALLBACK_PHRASES = [
    "couldn't find", "no relevant", "not find",
    "no information", "unable to find", "i don't have",
]


def _is_dead_end(top_score: float, answer: str) -> bool:
    low_score   = isinstance(top_score, (int, float)) and top_score < 0.80
    is_fallback = any(p in answer.lower() for p in FALLBACK_PHRASES)
    return low_score or is_fallback


def _is_commercial_shift(memory: list, stage: str, query: str) -> bool:
    # Require at least 5 queries before CTA fires — visitor must show sustained intent
    if len(memory or []) < 5:
        return False
    if stage not in ("evaluation", "intent"):
        return False
    # 2+ distinct products across history
    all_products = set()
    for m in (memory or []):
        all_products.update(m.get("products", []))
    if len(all_products) < 2:
        return False
    # Commercial keyword in current query
    q_lower = query.lower()
    return any(kw in q_lower for kw in COMMERCIAL_KEYWORDS)


def _is_circular_loop(memory: list, similarity: float, meta: dict) -> bool:
    if len(memory or []) < 3:
        return False
    if similarity < 0.88:
        return False
    # Check time window — last 3 queries within 3 minutes
    try:
        from datetime import datetime
        recent_times = []
        for m in memory[-3:]:
            if not isinstance(m, dict):
                continue
            ts = m.get("timestamp")
            if not ts or not isinstance(ts, str):
                continue
            try:
                recent_times.append(datetime.fromisoformat(ts))
            except (ValueError, TypeError):
                pass
        if len(recent_times) >= 2:
            span = (recent_times[-1] - recent_times[0]).total_seconds() / 60
            if span > 3:
                return False
    except Exception:
        pass
    # No citation clicks since loop started
    return meta.get("citation_clicks", 0) == 0


def render_nudge_card(nudge_type: str, products: list = None, visitor_id: str = "unknown", cta_index: int = 0) -> str:
    """Render a contextual support nudge card. Non-blocking — appears below result.
    cta_index cycles 0→1→2 across queries to rotate which CTA button is highlighted.
    """

    products_str = " + ".join((products or [])[:2])

    # 3 distinct CTA definitions — one shown at a time, others hidden
    CTAS = [
        {
            "icon": "🏗️",
            "label": "Customer Care",
            "url":   "https://www.equinix.com/contact",
            "bg":    "#1D9E7522",
            "border":"#1D9E7544",
            "color": "#059669",
        },
        {
            "icon": "📅",
            "label": "Customer Care",
            "url":   "https://equinix.com/services/consulting/",
            "bg":    "#185FA511",
            "border":"#185FA544",
            "color": "#185FA5",
        },
        {
            "icon": "🎧",
            "label": "Customer Care",
            "url":   "https://www.equinix.com/support",
            "bg":    "#7c3aed11",
            "border":"#7c3aed44",
            "color": "#7c3aed",
        },
    ]

    if nudge_type == "commercial":
        product_line = (
            f'<div style="font-size:11px;color:#555;margin-bottom:8px;">'
            f'Based on your interest in <strong style="color:#1D7A5F;">{products_str}</strong></div>'
            if products_str else ""
        )

        # Active CTA — the one shown prominently
        active = CTAS[cta_index % 3]

        # Other two shown as small discrete links below
        others = [c for i, c in enumerate(CTAS) if i != cta_index % 3]

        active_btn = f"""
                <a href="{active['url']}" target="_blank" rel="noopener"
                   style="text-decoration:none;background:{active['bg']};border:1.5px solid {active['border']};
                          color:{active['color']};padding:7px 16px;border-radius:7px;font-size:0.75rem;
                          font-weight:700;display:inline-flex;align-items:center;gap:6px;">
                    {active['icon']} {active['label']}
                </a>"""

        other_links = " · ".join(
            f'<a href="{c["url"]}" target="_blank" rel="noopener" '
            f'style="color:#9ca3af;font-size:0.65rem;text-decoration:none;">'
            f'{c["icon"]} {c["label"]}</a>'
            for c in others
        )

        return f"""
        <div style="margin-bottom:14px;background:#f0fdf4;border:2px solid #059669;
                    border-radius:10px;padding:14px 16px;font-family:'DM Sans',sans-serif;
                    animation:nudgeFadeIn 0.4s ease;">
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
                <span style="font-size:16px;">🎯</span>
                <span style="font-size:0.78rem;font-weight:600;color:#059669;">
                    Ready to talk to an expert?
                </span>
            </div>
            {product_line}
            <div style="font-size:0.72rem;color:#374151;margin-bottom:10px;line-height:1.5;">
                You're evaluating a serious infrastructure decision.
                Our Customer Care can design a custom deployment for your needs.
            </div>
            <div style="display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin-bottom:8px;">
                {active_btn}
                <span style="font-size:0.65rem;color:#6b7280;margin-left:auto;">
                    Same-day response · No commitment
                </span>
            </div>
            <div style="padding-top:6px;border-top:1px solid #d1fae5;">
                {other_links}
            </div>
            <div id="contact-capture" style="border-top:1px solid #d1fae5;padding-top:10px;">
                <div style="font-size:0.68rem;color:#059669;font-weight:600;margin-bottom:7px;">
                    📬 Or leave your details — we&#39;ll reach out
                </div>
                <div style="display:flex;gap:7px;flex-wrap:wrap;">
                    <input id="nudge-name" type="text" placeholder="Your name"
                           style="flex:1;min-width:110px;padding:5px 9px;border-radius:6px;
                                  border:1px solid #a7f3d0;background:#ffffff;
                                  font-size:0.7rem;color:#111;outline:none;
                                  font-family:'DM Sans',sans-serif;"
                           onfocus="this.style.borderColor='#059669'"
                           onblur="this.style.borderColor='#a7f3d0'"/>
                    <input id="nudge-email" type="email" placeholder="Work email"
                           style="flex:2;min-width:160px;padding:5px 9px;border-radius:6px;
                                  border:1px solid #a7f3d0;background:#ffffff;
                                  font-size:0.7rem;color:#111;outline:none;
                                  font-family:'DM Sans',sans-serif;"
                           onfocus="this.style.borderColor='#059669'"
                           onblur="this.style.borderColor='#a7f3d0'"/>
                    <button id="nudge-submit"
                            onclick="(function(){{
                                var n=document.getElementById('nudge-name');
                                var e=document.getElementById('nudge-email');
                                var b=document.getElementById('nudge-submit');
                                var s=document.getElementById('nudge-success');
                                if(!e||!e.value||!e.value.includes('@')){{
                                    e.style.borderColor='#ef4444';
                                    e.placeholder='Enter a valid email';
                                    return;
                                }}
                                b.textContent='Sending…';
                                b.disabled=true;
                                fetch('{API_GATEWAY_URL}/api/v1/visitor/identify',{{
                                    method:'POST',
                                    headers:{{'Content-Type':'application/json','X-API-Key':'{API_KEY}'}},
                                    body:JSON.stringify({{
                                        visitor_id: '{visitor_id}',
                                        name: n?n.value:'',
                                        email: e.value,
                                        source: 'commercial_nudge',
                                        products: '{products_str}'
                                    }})
                                }}).catch(()=>{{}}).finally(()=>{{
                                    document.getElementById('contact-capture').style.display='none';
                                    s.style.display='flex';
                                }});
                            }})()"
                            style="padding:5px 13px;border-radius:6px;border:none;
                                   background:#059669;color:#fff;font-size:0.7rem;
                                   font-weight:600;cursor:pointer;white-space:nowrap;
                                   font-family:'DM Sans',sans-serif;">
                        Get in touch →
                    </button>
                </div>
            </div>
            <div id="nudge-success" style="display:none;border-top:1px solid #d1fae5;
                 padding-top:10px;align-items:center;gap:8px;">
                <span style="font-size:14px;">✅</span>
                <span style="font-size:0.7rem;color:#059669;font-weight:600;">
                    Thanks! A Customer Care will reach out today.
                </span>
            </div>
        </div>"""

    elif nudge_type == "dead_end":
        return f"""
        <div style="margin-bottom:14px;background:#fff5f5;border:2px solid #EC1C24;
                    border-radius:10px;padding:14px 16px;font-family:'DM Sans',sans-serif;
                    animation:nudgeFadeIn 0.4s ease;">
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
                <span style="font-size:16px;">📋</span>
                <span style="font-size:0.78rem;font-weight:600;color:#dc2626;">
                    Not finding what you need?
                </span>
            </div>
            <div style="font-size:0.72rem;color:#374151;margin-bottom:10px;line-height:1.5;">
                Our documentation may not cover this specific topic yet.
                Our support team can answer directly.
            </div>
            <div style="display:flex;gap:8px;flex-wrap:wrap;">
                <a href="https://www.equinix.com/support" target="_blank" rel="noopener"
                   style="text-decoration:none;background:#c0505022;border:1px solid #c0505044;
                          color:#dc2626;padding:5px 12px;border-radius:6px;font-size:0.7rem;
                          font-weight:600;">
                    📧 Contact Support
                </a>
                <a href="https://www.equinix.com/contact" target="_blank" rel="noopener"
                   style="text-decoration:none;background:transparent;border:1px solid #333;
                          color:#374151;padding:5px 12px;border-radius:6px;font-size:0.7rem;">
                    💬 Live Chat
                </a>
            </div>
        </div>"""

    elif nudge_type == "loop":
        return f"""
        <div style="margin-bottom:14px;background:#fffbeb;border:2px solid #f59e0b;
                    border-radius:10px;padding:14px 16px;font-family:'DM Sans',sans-serif;
                    animation:nudgeFadeIn 0.4s ease;">
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
                <span style="font-size:16px;">💬</span>
                <span style="font-size:0.78rem;font-weight:600;color:#d97706;">
                    Let an expert answer directly
                </span>
            </div>
            <div style="font-size:0.72rem;color:#374151;margin-bottom:10px;line-height:1.5;">
                You've searched this several times. Our team can give you
                a precise answer faster than documentation search.
            </div>
            <div style="display:flex;gap:8px;flex-wrap:wrap;">
                <a href="https://www.equinix.com/contact" target="_blank" rel="noopener"
                   style="text-decoration:none;background:#ba751722;border:1px solid #ba751744;
                          color:#d97706;padding:5px 12px;border-radius:6px;font-size:0.7rem;
                          font-weight:600;">
                    💬 Chat with an Expert
                </a>
                <a href="https://www.equinix.com/contact" target="_blank" rel="noopener"
                   style="text-decoration:none;background:transparent;border:1px solid #333;
                          color:#374151;padding:5px 12px;border-radius:6px;font-size:0.7rem;">
                    📞 Contact Support
                </a>
            </div>
        </div>
        <style>
        @keyframes nudgeFadeIn {{
            from {{ opacity:0; transform:translateY(6px); }}
            to   {{ opacity:1; transform:translateY(0); }}
        }}
        </style>"""

    return ""


# ================================================================
# EPISODIC MEMORY — session-scoped visitor journey tracking
# ================================================================

INTENT_LABELS = {
    "find_resource":  ("📄", "Find resource",  "#185FA5"),
    "evaluate_specs": ("📊", "Evaluate specs", "#1D7A5F"),
    "compare":        ("⚖️", "Compare",        "#7c3aed"),
    "troubleshoot":   ("🔧", "Troubleshoot",   "#c2410c"),
    "learn_concept":  ("💡", "Learn concept",  "#0F6E56"),
    "general":        ("🔍", "General",        "#555"),
    "unknown":        ("🔍", "Search",         "#555"),
}

STAGE_MAP = {
    "awareness":     ("🌱", "Awareness",     "Just exploring"),
    "consideration": ("🔭", "Considering",   "Evaluating options"),
    "evaluation":    ("⚖️", "Evaluating",    "Comparing products"),
    "intent":        ("🎯", "High intent",   "Ready to engage"),
}

# Global to capture latest API response intent data
_last_api_response: dict = {}


def _infer_stage(memory: list) -> str:
    if not memory:
        return "awareness"
    # Handle both list of dicts and list of intent strings
    # Filter out non-dict items and extract intents safely
    intents = []
    for m in memory:
        if isinstance(m, dict):
            intents.append(m.get("intent", "general"))
        elif isinstance(m, str):
            intents.append(m)  # assume it's an intent string
    n = len(intents)
    if n >= 4 and "compare" in intents:
        return "intent"
    if n >= 3 and "compare" in intents:
        return "evaluation"
    if n >= 2 and any(i in intents for i in ["troubleshoot", "compare"]):
        return "consideration"
    return "awareness"


def update_memory(memory, query, intent, products, use_case, sources, workloads=None):
    from datetime import datetime as _dt, timezone
    entry = {
        "query":     query,
        "intent":    intent or "general",
        "products":  products or [],
        "use_case":  use_case or "",
        "workloads": workloads or [],
        "sources":   [s.get("resource_type", "") for s in (sources or [])[:3]],
        "timestamp": _dt.now(timezone.utc).strftime("%H:%M"),
    }
    return (memory or []) + [entry]


def render_memory_panel(memory: list) -> str:
    if not memory:
        return ""

    # Product interest: ALL memory (full journey), not just last 5
    all_products = []
    for m in memory:
        all_products.extend(m.get("products", []))
        # Also count workloads as product interest signals
        for wl in (m.get("workloads") or []):
            all_products.append(wl)
    product_counts = Counter(all_products)

    stage_key = _infer_stage(memory)
    stage_icon, stage_label, stage_desc = STAGE_MAP.get(
        stage_key, ("🌱", "Awareness", "Just exploring")
    )

    rows = []
    for m in memory[-5:]:
        intent_key = m.get("intent", "general")
        icon, label, color = INTENT_LABELS.get(intent_key, ("🔍", "Search", "#555"))
        # Defensive access for products field
        products = m.get("products", [])
        products_str = " · ".join(products[:2]) if products else ""
        # Defensive access for query field
        query = m.get("query", "")
        q_short = query[:50] + ("…" if len(query) > 50 else "")
        # Pre-compute workload tags — styles from DynamoDB via API
        def _workload_badge(w):
            styles = _get_badge_styles()
            # Match by label name (exact) or by keyword substring
            style  = styles.get(w)
            if not style:
                # Fallback: match by keyword in label
                wl = w.lower()
                for label, s in styles.items():
                    if label.lower() in wl or wl in label.lower():
                        style = s
                        break
            if style:
                ico = style.get("icon","⚡")
                bg  = style.get("bg","#1a1a1a")
                fg  = style.get("color","#9ca3af")
            else:
                ico, bg, fg = "⚡", "#1a1a1a", "#9ca3af"
            return (
                f'<span style="background:{bg};color:{fg};font-size:0.62rem;'
                f'font-weight:600;padding:1px 6px;border-radius:3px;'
                f'margin-right:3px;white-space:nowrap;">'
                f'{ico} {w}</span>'
            )
        workload_tag = "".join(
            _workload_badge(w) for w in (m.get("workloads") or [])[:2]
        )
        # Clean timestamp — handles ISO format and HH:MM format
        raw_ts = m.get("timestamp", m.get("time", m.get("created_at", "")))
        try:
            from datetime import datetime as _dt2, timezone as _tz
            if raw_ts and len(raw_ts) > 5:
                _parsed = _dt2.fromisoformat(raw_ts.replace("Z", "+00:00"))
                timestamp = _parsed.strftime("%-I:%M %p")
            elif raw_ts:
                timestamp = raw_ts
            else:
                timestamp = "—"
        except Exception:
            timestamp = raw_ts[:5] if raw_ts and len(raw_ts) >= 5 else (raw_ts or "—")
        rows.append(f"""
        <div style="display:flex;align-items:flex-start;gap:10px;
                    padding:8px 0;border-bottom:1px solid #e5e7eb;">
            <div style="width:20px;height:20px;border-radius:50%;
                        background:{color}22;border:1px solid {color}44;
                        display:flex;align-items:center;justify-content:center;
                        font-size:10px;flex-shrink:0;margin-top:2px;">{icon}</div>
            <div style="min-width:0;flex:1;">
                <div style="font-size:0.78rem;color:#111827;line-height:1.4;
                            overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{q_short}</div>
                <div style="font-size:0.64rem;color:#374151;margin-top:2px;display:flex;gap:6px;flex-wrap:wrap;align-items:center;">
                    <span style="color:{color};font-weight:600;">{label}</span>
                    {"<span>·</span><span style=\'color:#777;\'>" + products_str + "</span>" if products_str else ""}
                    {workload_tag}
                    <span style="color:#6b7280;">{timestamp}</span>
                </div>
            </div>
        </div>""")

    product_bars = ""
    if product_counts:
        max_c = max(product_counts.values())
        bar_rows = []
        for product, count in product_counts.most_common(4):
            pct   = int(count / max_c * 100)
            short = product.replace("Equinix ", "").replace(" Cloud Router", " CR")
            bar_rows.append(f"""
            <div style="margin-bottom:7px;">
                <div style="display:flex;justify-content:space-between;
                            font-size:0.64rem;color:#666;margin-bottom:3px;">
                    <span>{short}</span><span style="color:#444;">{count}×</span>
                </div>
                <div style="background:#e4e7ed;border-radius:2px;height:3px;">
                    <div style="background:linear-gradient(90deg,#ec1c24,#ff6060);
                                height:100%;width:{pct}%;border-radius:2px;"></div>
                </div>
            </div>""")
        product_bars = (
            '<div style="margin-bottom:14px;padding-top:10px;border-top:1px solid #e5e7eb;">' +
            '<div style="font-size:0.62rem;font-weight:700;color:#444;text-transform:uppercase;'
            'letter-spacing:0.1em;margin-bottom:8px;">Product interest</div>' +
            "".join(bar_rows) + '</div>'
        )

    n = len(memory)
    personalized = (
        '&nbsp;·&nbsp;<span style="color:#ec1c2466;font-size:0.64rem;font-weight:600;">Personalizing</span>'
        if n >= 2 else ""
    )

    return f"""
    <div style="background:#f7f8fa;border:1px solid #e5e7eb;border-radius:12px;
                padding:14px 16px;margin-bottom:14px;font-family:'DM Sans',sans-serif;">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;">
            <div style="display:flex;align-items:center;gap:7px;">
                <div style="width:7px;height:7px;border-radius:50%;background:#ec1c24;
                            box-shadow:0 0 5px #ec1c2466;animation:mpulse 2s infinite;"></div>
                <span style="font-size:0.68rem;font-weight:700;color:#666;
                             text-transform:uppercase;letter-spacing:0.1em;">Session memory</span>
            </div>
            <span style="font-size:0.64rem;color:#444;">{stage_icon} {stage_label} · {n} {"query" if n==1 else "queries"}{personalized}</span>
        </div>
        <div style="font-size:0.7rem;color:#444;margin-bottom:10px;padding:5px 9px;
                    background:#f0f2f5;border-radius:5px;">
            {stage_desc}
        </div>
        {"".join(rows)}
        {product_bars}
        <style>@keyframes mpulse{{0%,100%{{opacity:1}}50%{{opacity:0.3}}}}</style>
    </div>"""


# ================================================================
# SEARCH handler — guardrails wired in
# ================================================================

def do_search(query, history_html, prev_slot0="", search_namespace="", memory=None, visitor_id="v_prod_guest"):
    if not query or not query.strip():
        import gradio as _gr
        return ("", history_html, "", "", "", False, [], _gr.update(visible=False), _gr.update(visible=False), "")

    # Move previous slot0 into history before showing new result
    if prev_slot0:
        divider_sep = "<hr style='border:none;border-top:1px solid rgba(0,0,0,0.08);margin:24px 0;'>"
        history_html = prev_slot0 + divider_sep + history_html if history_html else prev_slot0
    divider = "<hr style='border:none;border-top:1px solid rgba(0,0,0,0.08);margin:24px 0;'>" if history_html else ""

    ns = None
    if search_namespace:
        ns_val = search_namespace.strip().lower() if isinstance(search_namespace, str) else ""
        if ns_val and ns_val != "all":
            ns = ns_val

    # ── AI Summary trigger ────────────────────────────────────────
    if query.strip().startswith("__summarise__:"):
        filename = query.strip().replace("__summarise__:", "", 1).strip()
        try:
            resp = requests.post(
                f"{API_GATEWAY_URL}/api/v1/summarise",
                json={"filename": filename},
                headers={"Content-Type": "application/json", "X-API-Key": API_KEY},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("error"):
                import gradio as _gr
                return ("", blocked_html(filename, data["error"]) + divider + history_html, "", "", "", False, [], _gr.update(visible=False), _gr.update(visible=False), "")

            topics_html = " ".join([
                f'<span style="background:#ede9fe;color:#6d28d9;border-radius:4px;'
                f'padding:1px 8px;font-size:0.72rem;display:inline-block;margin:2px;">{t}</span>'
                for t in data.get("key_topics", [])
            ])
            cached_badge = '<span style="font-size:0.65rem;color:#16a34a;font-weight:600;background:#dcfce7;border-radius:4px;padding:1px 5px;margin-left:8px;">⚡ cached</span>' if data.get("cached") else ""

            summary_html = f"""
            <div style="background:linear-gradient(135deg,#f5f3ff,#ede9fe);
                        border:1px solid #c4b5fd;border-radius:16px;
                        padding:24px 28px;margin-top:16px;font-family:'Inter',sans-serif;">
                <div style="font-size:0.72rem;color:#7c3aed;font-weight:700;
                            text-transform:uppercase;letter-spacing:0.1em;margin-bottom:12px;">
                    ✨ AI Summary {cached_badge}
                </div>
                <div style="font-size:0.85rem;font-weight:600;color:#4c1d95;margin-bottom:8px;">
                    {data.get('suggested_name', filename)}
                </div>
                <div style="font-size:1rem;color:#2e1065;line-height:1.85;
                            border-left:3px solid #7c3aed;padding-left:16px;margin-bottom:16px;">
                    {data.get('summary', 'No summary available.')}
                </div>
                <div style="font-size:0.72rem;color:#6d28d9;margin-bottom:8px;">
                    🏷️ {topics_html}
                </div>
                <div style="font-size:0.72rem;color:#888;margin-top:8px;">
                    Suggested type: <b>{data.get('suggested_type', '—')}</b>
                </div>
            </div>"""
            import gradio as _gr
            return ("", summary_html + divider + history_html, "", "", "", False, [], _gr.update(visible=False), _gr.update(visible=False), "")

        except Exception as e:
            import gradio as _gr
            return ("", blocked_html(filename, f"Could not load summary: {str(e)}") + divider + history_html, "", "", "", False, [], _gr.update(visible=False), _gr.update(visible=False), "")

    # ── Greeting check — intercept before API call (avoids 422 on short queries) ──
    _GREETINGS = {
        "hi", "hello", "hey", "hiya", "howdy", "heya", "yo", "sup",
        "good morning", "good afternoon", "good evening", "good day",
        "thanks", "thank you", "thx", "cheers", "ty",
        "bye", "goodbye", "cya", "see you",
        "what can you do", "how does this work", "who are you", "what are you",
        "can you help", "help me", "i need help",
    }
    _q_norm = query.strip().lower().rstrip("!?.,")
    if _q_norm in _GREETINGS or len(query.strip()) <= 3:
        import gradio as _gr
        _greeting_html = """
        <div style="background:#181818;border:1px solid #2e2e2e;border-radius:14px;
                    padding:22px 24px;font-family:'DM Sans',sans-serif;margin-bottom:12px;">
            <div style="font-size:1rem;font-weight:600;color:#f0f0f0;margin-bottom:10px;">
                👋 Hi! Ask any Questions on Equinix resource.
            </div>
            <div style="font-size:0.83rem;color:#aaa;line-height:1.7;">
                I can help you search across Equinix's full resource library — including
                technical blueprints, product data sheets, analyst reports, and case studies.<br><br>
                Try asking about a specific product like <strong style="color:#ccc;">Equinix Fabric</strong>
                or a use case like <strong style="color:#ccc;">hybrid multicloud networking</strong>.
            </div>
        </div>"""
        return (_greeting_html, history_html, "",
                query, "", False, [],
                _gr.update(visible=False), _gr.update(visible=False), "")

    # ── Normal search — delegates to EC2 API ─────────────────────
    try:
        # Build payload with visitor identity + coherence gate context
        last = (memory[-1] if memory else {})
        payload = {
            "query":       query,
            "visitor_id":  visitor_id,
            "last_query":  last.get("query", ""),
            "last_intent": last.get("intent", ""),
        }
        if ns:
            payload["namespace"] = ns

        resp = requests.post(
            f"{API_GATEWAY_URL}/api/v1/search",
            json=payload,
            headers={"Content-Type": "application/json", "X-API-Key": API_KEY},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("blocked"):
            import gradio as _gr
            return ("", blocked_html(query, data.get("answer", "Query blocked by security policy.")) + divider + history_html, "", "", "", False, [], _gr.update(visible=False), _gr.update(visible=False), "")

        # Capture intent fields for episodic memory
        _last_api_response.update({
            "intent":             data.get("intent", "general"),
            "detected_products":  data.get("detected_products", []),
            "detected_use_case":  data.get("detected_use_case", ""),
            "detected_workloads": data.get("detected_workloads", []),
            "sources":            data.get("sources", []),
            "rewritten_query":    data.get("rewritten_query", ""),
            "lead_quality_tag":   data.get("lead_quality_tag", ""),
            "similarity":         data.get("similarity", 0.0),
            "inherited":          data.get("inherited", False),
            "top_score":          data.get("top_score", 0.0),
            "cached":             data.get("cached", False),
        })
        answer    = data.get("answer", "")
        sources   = data.get("sources", [])
        followups = data.get("followups", [])
        cached    = data.get("cached", False)

        # Only treat as no-results if the answer text itself says nothing was found.
        # Do NOT use `not sources` here; cached responses may intentionally omit sources.
        no_results = (
            not answer or
            "couldn't find" in answer.lower() or
            "no relevant" in answer.lower() or
            "not find" in answer.lower()
        )
        if no_results:
            result_html = render_result_html(query, answer, [], [], cached=False)
            import gradio as _gr
            return (result_html, divider + history_html, "", "", "", False, [], _gr.update(visible=False), _gr.update(visible=False), "")

        result_html = render_result_html(query, answer, sources, followups, cached=cached)
        # Store last result for feedback buttons
        _last_result["query"]   = query
        _last_result["answer"]  = answer[:400]
        _last_result["cached"]  = cached
        _last_result["sources"] = [
            {"filename": s.get("filename",""), "clean_name": s.get("clean_name","")}
            for s in sources[:5]
        ]
        src_list = [{"filename": s.get("filename",""), "clean_name": s.get("clean_name","")} for s in sources[:5]]
        import gradio as _gr
        # Return: slot0_html, history_html, msg, lq, la, lc, ls, up_vis, dn_vis, msg0
        return (result_html, divider + history_html, "",
                query, answer[:400], cached, src_list,
                _gr.update(visible=True), _gr.update(visible=True), "")
    except requests.exceptions.Timeout:
        import gradio as _gr
        return ("", blocked_html(query, "Request timed out.") + divider + history_html, query,
                "", "", False, [], _gr.update(visible=False), _gr.update(visible=False), "")
    except requests.exceptions.ConnectionError:
        import gradio as _gr
        return ("", blocked_html(query, "Cannot reach search API.") + divider + history_html, query,
                "", "", False, [], _gr.update(visible=False), _gr.update(visible=False), "")
    except Exception as e:
        import gradio as _gr
        error_html = f'''
        <div style="padding:20px;background:#fff5f5;border:1px solid #fca5a5;
                    border-radius:12px;color:#dc2626;font-family:monospace;font-size:0.82rem;margin-top:16px;">
            <strong>Error:</strong> {str(e)}
        </div>'''
        return ("", error_html + divider + history_html, query,
                "", "", False, [], _gr.update(visible=False), _gr.update(visible=False), "")


def do_clear():
    return "", ""



def do_search_with_memory(query, history_html, prev_slot0, memory_meta=None, search_namespace="", visitor_id_from_storage="v_prod_guest", session_memory_state=None, request: gr.Request = None):
    """Wraps do_search — reads visitor ID from browser local storage, pulls history from DynamoDB, adds analytics/nudges."""
    from datetime import datetime as _dt, timezone
    import requests
    global _current_visitor_id

    print(f"\n{'='*80}")
    print(f"🔍 NEW SEARCH SESSION INITIATED")
    print(f"{'='*80}\n")

    # 1. Use visitor_id from browser local storage (via BrowserState)
    visitor_id = "v_prod_guest"

    if visitor_id_from_storage and visitor_id_from_storage != "v_prod_guest":
        visitor_id = visitor_id_from_storage
        print(f"📦 Using visitor_id from browser storage: {visitor_id}")
    else:
        print(f"⚠️ No visitor_id in browser storage, using default")

    # Store visitor_id globally for feedback submission
    _current_visitor_id = visitor_id
    print(f"🔐 Current visitor session set to: {visitor_id}\n")

    # 2. Fetch ground-truth history from DynamoDB right now to prevent state loss
    base_history = []
    try:
        hist_resp = requests.get(
            f"{API_GATEWAY_URL}/api/v1/visitor/{visitor_id}/history",
            headers={"X-API-Key": API_KEY},
            timeout=4
        )
        if hist_resp.status_code == 200:
            base_history = hist_resp.json().get("queries", [])
            print(f"📥 Retrieved {len(base_history)} history records from DynamoDB")
    except Exception as e:
        print(f"❌ History pull failed: {e}")

    # Use Gradio session state if it has more entries (fresher within session)
    # Always prefer session_memory_state — it has workloads which DynamoDB history lacks
    # Only fall back to DynamoDB if session state is empty (fresh tab/reload)
    if session_memory_state and len(session_memory_state) >= len(base_history):
        base_history = session_memory_state
        print(f"📥 Using session state ({len(base_history)} entries, workloads preserved)")
    elif base_history:
        # DynamoDB records — ensure workloads field exists (may be missing)
        for entry in base_history:
            if "workloads" not in entry:
                entry["workloads"] = []
        print(f"📥 Using DynamoDB history ({len(base_history)} entries)")

    # 3. Run the core RAG search execution path on EC2 backend
    result = do_search(query, history_html, prev_slot0, search_namespace, base_history, visitor_id=visitor_id)

    # ── Extract API response signals ──────────────────────────────────────────
    current_products = _last_api_response.get("detected_products", [])
    inherited        = _last_api_response.get("inherited", False)
    similarity       = float(_last_api_response.get("similarity", 0.0))
    top_score        = float(_last_api_response.get("top_score", 0.0))
    answer_text      = result[4] if len(result) > 4 else ""

    # Carry forward products on inherited (coherence gate) queries
    if inherited and not current_products and base_history:
        current_products = base_history[-1].get("products", [])

    # 4. Append the current interaction to the timeline
    updated_memory = update_memory(
        memory    = base_history,
        query     = query,
        intent    = _last_api_response.get("intent", "general"),
        products  = current_products,
        use_case  = _last_api_response.get("detected_use_case", ""),
        sources   = _last_api_response.get("sources", []),
        workloads = _last_api_response.get("detected_workloads", []),
    )

    # Stamp timestamp on last entry for loop detection
    now_iso = _dt.now(timezone.utc).isoformat()
    if updated_memory:
        updated_memory[-1]["timestamp"] = now_iso
        # Normalise stored query — strip numbers/prefixes so memory panel is clean
        import re as _re2
        _raw_q = updated_memory[-1].get("query", "")
        _clean_q = _re2.sub(r'^[0-9]+[.):-]+\s*', '', _raw_q.strip()).strip()
        _clean_q = _re2.sub(r'^(q|question|hi|hello)[,: ]+', '', _clean_q, flags=_re2.IGNORECASE).strip()
        if _clean_q:
            updated_memory[-1]["query"] = _clean_q

    # 5. ❌ REMOVED THE BROKEN /query POST CALL ❌
    # Your backend handles the DynamoDB write automatically when do_search invokes POST /api/v1/search
    # The visitor_id is passed directly in the payload to /api/v1/search and the backend saves it seamlessly.

    # ── Update meta counters ─────────────────────────────────────────────────
    meta = dict(memory_meta or {})
    meta.setdefault("consecutive_failures", 0)
    meta.setdefault("citation_clicks", 0)
    meta.setdefault("last_query_time", None)
    meta.setdefault("nudge_shown", set())

    meta["citation_clicks"]  = 0        # reset on every new search
    meta["last_query_time"]  = now_iso

    if _is_dead_end(top_score, answer_text):
        meta["consecutive_failures"] += 1
    else:
        meta["consecutive_failures"] = 0

    # ── Nudge detection — additive, never blocks search ───────────────────────
    nudge_html = ""
    stage      = _infer_stage(updated_memory)
    shown      = set(meta.get("nudge_shown", set()))

    # Nudge logic:
    # - Once fired, sticky for entire session (minimum 5 prompts visible)
    # - CTA rotates every prompt so all 3 options get seen
    # - query count tracked to enforce minimum visibility

    cta_idx = meta.get("cta_index", 0)
    nudge_count = meta.get("nudge_query_count", 0)

    if "commercial" in shown:
        # Already fired — keep showing, rotate CTA, track count
        nudge_html = render_nudge_card("commercial", current_products,
                                       visitor_id=visitor_id, cta_index=cta_idx)
        meta["nudge_query_count"] = nudge_count + 1
        meta["cta_index"]         = (cta_idx + 1) % 3   # rotate 0→1→2→0

    elif "loop" in shown:
        nudge_html = render_nudge_card("loop")
        meta["nudge_query_count"] = nudge_count + 1

    elif "dead_end" in shown:
        nudge_html = render_nudge_card("dead_end")
        meta["nudge_query_count"] = nudge_count + 1

    elif not shown:
        # First time — check triggers
        if _is_commercial_shift(updated_memory, stage, query):
            nudge_html = render_nudge_card("commercial", current_products,
                                           visitor_id=visitor_id, cta_index=cta_idx)
            shown.add("commercial")
            meta["nudge_query_count"] = 1
            meta["cta_index"]         = 1   # next query shows CTA #2
        elif _is_circular_loop(updated_memory, similarity, meta):
            nudge_html = render_nudge_card("loop")
            shown.add("loop")
            meta["nudge_query_count"] = 1
        elif meta["consecutive_failures"] >= 8:
            nudge_html = render_nudge_card("dead_end")
            shown.add("dead_end")
            meta["nudge_query_count"] = 1

    meta["nudge_shown"] = shown

    # ── Append nudge below result (search result always shown first) ──────────
    slot0_html = nudge_html + (result[0] or "") if nudge_html else (result[0] or "")

    # ── Render memory panel ───────────────────────────────────────────────────
    memory_html = render_memory_panel(updated_memory)

    # Summary log
    print(f"\n{'='*80}")
    print(f"✅ SEARCH COMPLETED")
    print(f"{'='*80}")
    print(f"  Query: {query[:60]}")
    print(f"  Visitor: {visitor_id}")
    print(f"  Memory entries: {len(updated_memory)}")
    print(f"  Intent: {_last_api_response.get('intent', 'N/A')}")
    print(f"  Products detected: {_last_api_response.get('detected_products', [])}")
    print(f"{'='*80}\n")

    return (slot0_html,) + result[1:] + (updated_memory, memory_html, meta, visitor_id)


# ================================================================
# UI
# ================================================================

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700;900&family=DM+Sans:wght@400;500;600;700&display=swap');

/* ── Width constraint — the ONLY reason we need CSS ─────────── */
.gradio-container {
    max-width: 980px !important;
    margin: 0 auto !important;
    padding: 0 28px 80px !important;
    box-shadow: none !important;
    border: none !important;
}
footer { display: none !important; }

/* ── Fonts ───────────────────────────────────────────────────── */
:root {
  --eq-red: #EC1C24; --eq-red-h: #c9181f;
  --fg1: #111827; --fg2: #374151; --fg3: #6b7280; --fg-m: #9ca3af;
  --bd: #e5e7eb; --bd-s: #d1d5db;
  --font: 'DM Sans', sans-serif;
  --font-d: 'Source Sans 3', 'DM Sans', sans-serif;
  --r-md: 8px;
}

/* ── Hero ────────────────────────────────────────────────────── */
#hero-eyebrow { font-family: var(--font-d) !important; font-size: 12px !important; font-weight: 700 !important; letter-spacing: .12em !important; text-transform: uppercase !important; color: var(--eq-red) !important; margin-bottom: 14px !important; display: block !important; }
#hero-h1 { font-family: var(--font-d) !important; font-weight: 900 !important; font-size: clamp(32px,4.4vw,46px) !important; line-height: 1.06 !important; letter-spacing: -.02em !important; color: var(--fg1) !important; margin: 0 0 14px !important; }
#hero-sub { font-size: 17px !important; line-height: 1.6 !important; color: var(--fg3) !important; max-width: 540px !important; margin: 0 !important; }

/* ── Search ──────────────────────────────────────────────────── */
#search-input textarea, #search-input input { background: #fff !important; color: var(--fg1) !important; border: 1.5px solid var(--bd-s) !important; border-radius: var(--r-md) !important; font-size: 16px !important; padding: 15px 16px !important; min-height: 52px !important; caret-color: var(--eq-red) !important; }
#search-input textarea:focus, #search-input input:focus { border-color: var(--eq-red) !important; box-shadow: 0 0 0 3px rgba(236,28,36,.12) !important; }
#search-input textarea::placeholder, #search-input input::placeholder { color: var(--fg-m) !important; }
#search-btn button { background: var(--eq-red) !important; color: #fff !important; border: none !important; border-radius: var(--r-md) !important; font-family: var(--font-d) !important; font-weight: 700 !important; font-size: 13px !important; letter-spacing: .08em !important; text-transform: uppercase !important; height: 52px !important; padding: 0 28px !important; }
#search-btn button:hover { background: var(--eq-red-h) !important; }

/* ── Card row ────────────────────────────────────────────────── */
#sq-row-1 { gap: 12px !important; margin: 0 0 24px !important; }
#sq-row-1 > div { flex: 1 1 0 !important; min-width: 0 !important; max-width: 25% !important; }

/* ── Clear btn ───────────────────────────────────────────────── */
#clear-btn button { background: transparent !important; border: 1px solid var(--bd-s) !important; border-radius: var(--r-md) !important; color: var(--fg3) !important; font-size: 13px !important; height: 36px !important; padding: 0 14px !important; }
#clear-btn button:hover { border-color: var(--eq-red) !important; color: var(--eq-red) !important; }

/* ── Tabs ────────────────────────────────────────────────────── */
.tab-nav button, .tabs > .tab-nav > button { font-family: var(--font-d) !important; font-size: 15px !important; font-weight: 600 !important; padding: 13px 18px !important; border-bottom: 2px solid transparent !important; }
.tab-nav button.selected, .tabs > .tab-nav > button.selected { color: var(--eq-red) !important; border-bottom-color: var(--eq-red) !important; }

@media (max-width: 640px) {
  .gradio-container { padding: 0 16px 60px !important; }
  #sq-row-1 { flex-wrap: wrap !important; }
  #sq-row-1 > div { min-width: calc(50% - 6px) !important; max-width: 50% !important; }
  #hero-h1 { font-size: 28px !important; }
}
"""





# ── Feedback state (last search result) ──────────────────────────────────────
_last_result: dict = {"query": "", "answer": "", "cached": False, "sources": []}
_current_visitor_id: str = "v_prod_guest"  # Track current visitor for feedback


def submit_feedback(rating: int) -> str:
    """Send thumbs up/down for the most recent search result to EC2."""
    if not _last_result.get("query"):
        return '<span style="font-size:0.78rem;color:#9ca3af;">Run a search first.</span>'
    try:
        payload = {
            **_last_result,
            "rating": rating,
            "visitor_id": _current_visitor_id
        }
        print(f"\n{'='*80}")
        print(f"💬 SUBMITTING FEEDBACK")
        print(f"{'='*80}")
        print(f"  URL: {API_GATEWAY_URL}/api/v1/feedback")
        print(f"  Visitor ID: {_current_visitor_id}")
        print(f"  Rating: {'👍 UPVOTE' if rating == 1 else '👎 DOWNVOTE'}")
        print(f"  Query: {_last_result.get('query', '')[:60]}")
        print(f"  Payload: {json.dumps(payload, indent=2)}")
        print(f"{'='*80}\n")

        resp = requests.post(
            f"{API_GATEWAY_URL}/api/v1/feedback",
            json=payload,
            headers={"Content-Type": "application/json", "X-API-Key": API_KEY},
            timeout=10,
        )
        print(f"✓ Feedback response - Status: {resp.status_code}")
        print(f"  Response Body: {resp.text[:300]}")
        print(f"  Response Headers: {dict(resp.headers)}\n")

        resp.raise_for_status()
        if rating == 1:
            q_short = _last_result.get("query","")[:40]
            return f'<span style="font-size:0.78rem;color:#16a34a;font-weight:600;">👍 Thanks! Rating saved for: {q_short}...</span>'
        else:
            return '<span style="font-size:0.78rem;color:#dc2626;font-weight:600;">👎 Got it — we\'ll use this to improve.</span>'
    except requests.exceptions.Timeout as e:
        print(f"❌ TIMEOUT submitting feedback: {e}")
        return f"Feedback timeout: {e}"
    except requests.exceptions.ConnectionError as e:
        print(f"❌ CONNECTION ERROR submitting feedback: {e}")
        return f"Cannot reach feedback API"
    except Exception as e:
        print(f"❌ ERROR submitting feedback: {e}")
        import traceback
        traceback.print_exc()
        return f"Could not save feedback: {e}"


_theme = gr.themes.Base(
    primary_hue=gr.themes.colors.red,
    neutral_hue=gr.themes.colors.gray,
    font=[gr.themes.GoogleFont("DM Sans"), "ui-sans-serif", "sans-serif"],
).set(
    body_background_fill="#ffffff",     body_background_fill_dark="#ffffff",
    body_text_color="#111827",          body_text_color_dark="#111827",
    background_fill_primary="#ffffff",  background_fill_primary_dark="#ffffff",
    background_fill_secondary="#f7f8fa",background_fill_secondary_dark="#f7f8fa",
    input_background_fill="#ffffff",    input_background_fill_dark="#ffffff",
    input_border_color="#d1d5db",       input_border_color_dark="#d1d5db",
    input_placeholder_color="#9ca3af",  input_placeholder_color_dark="#9ca3af",
    button_secondary_background_fill="#f7f8fa",
    button_secondary_background_fill_dark="#f7f8fa",
    button_secondary_text_color="#374151",
    button_secondary_text_color_dark="#374151",
    button_secondary_border_color="#d1d5db",
    button_secondary_border_color_dark="#d1d5db",
    block_background_fill="#ffffff",    block_background_fill_dark="#ffffff",
    block_border_color="#e5e7eb",       block_border_color_dark="#e5e7eb",
)
with gr.Blocks(title="Equinix Resource Assistant", theme=_theme) as demo:

    results_html_state = gr.State("")
    session_memory     = gr.State([])   # episodic memory list
    memory_meta        = gr.State({     # nudge trigger counters
        "consecutive_failures": 0,
        "citation_clicks":      0,
        "last_query_time":      None,
        "nudge_shown":          set(),  # which nudges already shown
        "nudge_query_count":    0,      # how many queries nudge has been shown
        "cta_index":            0,      # which CTA button to highlight (rotates 0→1→2)
    })
    # Browser-based visitor ID storage using local storage (survives page refreshes in HF Spaces)
    visitor_storage_key = gr.BrowserState(storage_key="rag_visitor_id")

    # Global CSS Variables for Equinix Brand Colors
    gr.HTML("""
    <style>
    :root, html, body {
        --eq-red:    #EC1C24;
        --eq-blue:   #003087;
        --eq-teal:   #006B5E;
        --eq-purple: #5B2D8E;
        --eq-mid:    #0070C0;
    }
    </style>
    """, visible=False)

    with gr.Tab("Search"):
        gr.HTML("""
        <div style="padding:22px 0 0;display:flex;align-items:center;justify-content:space-between;margin-bottom:0;">
          <div style="display:flex;align-items:center;gap:11px;">
            <svg style="width:24px;height:24px;" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="4.4" r="2.8" fill="#EC1C24"/>
              <circle cx="4.8" cy="16.5" r="2.8" fill="#EC1C24"/>
              <circle cx="19.2" cy="16.5" r="2.8" fill="#EC1C24"/>
              <path d="M12 4.4 L4.8 16.5 M12 4.4 L19.2 16.5 M4.8 16.5 L19.2 16.5" stroke="#EC1C24" stroke-width="1.3" opacity="0.5"/>
            </svg>
            <span style="font-family:'Source Sans 3','DM Sans',sans-serif;font-weight:900;font-size:19px;color:#EC1C24;text-transform:uppercase;letter-spacing:-0.01em;">Equinix</span>
            <div style="width:1px;height:18px;background:#d1d5db;"></div>
            <span style="font-family:'Source Sans 3','DM Sans',sans-serif;font-size:16px;color:#374151;font-weight:600;">Resource Assistant</span>
          </div>
          <span style="font-family:'Source Sans 3','DM Sans',sans-serif;font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#4b5563;border:1px solid #d1d5db;border-radius:9999px;padding:4px 12px;">RAG · Beta</span>
        </div>
        <div id="hero-wrap" style="padding:46px 0 30px;">
          <div id="hero-eyebrow">Equinix Resource Library</div>
          <h1 id="hero-h1">Ask anything about Equinix</h1>
          <p id="hero-sub">Search across whitepapers, blueprints, data sheets, analyst reports and case studies. Get cited, 2–4 sentence answers — with the sources to back them up.</p>
        </div>
        """)

        # ── Suggested queries — Equinix branded gr.Button cards ─────────
        gr.HTML("""
        <style>
        .sq-section { margin: 0 0 20px 0; }
        .sq-label {
            font-size: 0.68rem; font-weight: 700; color: #444;
            text-transform: uppercase; letter-spacing: 0.1em;
            margin-bottom: 10px; font-family: 'DM Sans', sans-serif;
        }
        </style>
        <div class="sq-section"><div class="sq-label">Suggested questions</div></div>
        """)

        gr.HTML(value="""<script>
(function(){
  var C=[
    {id:"sq-btn-0",bg:"linear-gradient(160deg,#eff5fc,#f8fbfe)",bl:"4px solid #2563eb",bc:"1px solid #c7dcf0",col:"#1e3a5f",ic:"#185FA5",lbl:"LEARN"},
    {id:"sq-btn-1",bg:"linear-gradient(160deg,#ebf6f1,#f7fcfa)",bl:"4px solid #059669",bc:"1px solid #b8e0d0",col:"#064e3b",ic:"#1D7A5F",lbl:"SPECS"},
    {id:"sq-btn-2",bg:"linear-gradient(160deg,#f4eefb,#fbf8fe)",bl:"4px solid #7c3aed",bc:"1px solid #d4b8f0",col:"#3b0764",ic:"#6B3AA0",lbl:"COMPARE"},
    {id:"sq-btn-3",bg:"linear-gradient(160deg,#fef2f2,#fff8f8)",bl:"4px solid #EC1C24",bc:"1px solid #fcc8c8",col:"#7f1d1d",ic:"#EC1C24",lbl:"FIND RESOURCE"}
  ];
  function run(){
    // White background
    [document.body,document.querySelector(".gradio-container"),document.querySelector(".main")].forEach(function(e){if(!e)return;e.style.setProperty("background","#fff","important");e.style.setProperty("background-color","#fff","important");e.style.setProperty("color","#111827","important");});
    // Search input
    var inp=document.querySelector("#search-input textarea")||document.querySelector("#search-input input");
    if(inp){inp.style.setProperty("background","#fff","important");inp.style.setProperty("background-color","#fff","important");inp.style.setProperty("color","#111827","important");inp.style.setProperty("border","1.5px solid #d1d5db","important");}
    // Cards
    C.forEach(function(c){
      var el=document.getElementById(c.id);if(!el)return;
      var btn=el.querySelector("button");if(!btn)return;
      btn.style.cssText="background:"+c.bg+"!important;border:"+c.bc+"!important;border-left:"+c.bl+"!important;color:"+c.col+"!important;text-align:left!important;min-height:116px!important;padding:16px 16px 18px!important;font-size:14px!important;font-weight:600!important;line-height:1.4!important;border-radius:12px!important;white-space:normal!important;width:100%!important;display:flex!important;flex-direction:column!important;gap:10px!important;cursor:pointer!important;box-shadow:none!important;";
      if(!btn.querySelector(".sq-ico")){var t=btn.textContent.trim();btn.innerHTML="<div class=\"sq-ico\" style=\"width:34px;height:34px;border-radius:8px;background:"+c.ic+"22;color:"+c.ic+";display:flex;align-items:center;justify-content:center;font-size:16px;flex-shrink:0;\">◉</div><span style=\"flex:1;font-size:14px;font-weight:600;color:"+c.col+";line-height:1.4;\">"+t+"</span><span style=\"font-size:10.5px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:"+c.ic+";">"+c.lbl+"</span>";}
    });
    // Hide greeting
    var greet=document.getElementById("eq-greeting");
    if(greet){var hasR=false;document.querySelectorAll(".gradio-html").forEach(function(e){if(e.innerHTML.length>200)hasR=true;});if(hasR)greet.style.display="none";}
  }
  run();[200,500,1000,2000,4000].forEach(function(t){setTimeout(run,t);});
  new MutationObserver(run).observe(document.body,{childList:true,subtree:true});
})();
</script>""")

        with gr.Row(elem_id="sq-row-1"):
            sq_btn_0 = gr.Button(
                value="""💡  What is Equinix Fabric and how does it work?""",
                elem_id="sq-btn-0",
                elem_classes=["sq-btn", "sq-learn-concept"],
            )
            sq_btn_1 = gr.Button(
                value="""📊  What port speeds and SLAs does Equinix Fabric support?""",
                elem_id="sq-btn-1",
                elem_classes=["sq-btn", "sq-evaluate-specs"],
            )
            sq_btn_2 = gr.Button(
                value="""⚖️  Equinix Fabric vs Network Edge for SD-WAN""",
                elem_id="sq-btn-2",
                elem_classes=["sq-btn", "sq-compare"],
            )
            sq_btn_3 = gr.Button(
                value="""📄  Show me a hybrid multicloud networking blueprint""",
                elem_id="sq-btn-3",
                elem_classes=["sq-btn", "sq-find-resource"],
            )



        # CSS for Equinix-branded sq-btn cards
        gr.HTML("""
        <style>
        /* Equinix brand palette */
        :root {
            --eq-red:    #EC1C24;
            --eq-blue:   #003087;
            --eq-teal:   #006B5E;
            --eq-purple: #5B2D8E;
            --eq-mid:    #0070C0;
        }
        #sq-row-1 {
            gap: 10px !important;
            margin-bottom: 0 !important;
        }
        #sq-row-1 > div > div {
            flex: 1 1 0 !important;
            min-width: 0 !important;
        }
        .sq-btn button {
            width: 100% !important;
            background: #1a1a2e !important;
            border: 1px solid #2e2e4a !important;
            border-radius: 12px !important;
            padding: 16px 16px 16px 20px !important;
            text-align: left !important;
            font-family: 'DM Sans', sans-serif !important;
            font-size: 0.83rem !important;
            font-weight: 500 !important;
            color: #e8e8ff !important;
            line-height: 1.5 !important;
            cursor: pointer !important;
            transition: all 0.15s ease !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.4) !important;
            min-height: 80px !important;
            white-space: normal !important;
        }
        .sq-btn button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 24px rgba(0,0,0,0.5) !important;
            color: #fff !important;
        }
        /* Learn concept — Deep Blue tint using Equinix color */
        #sq-btn-0 button, #sq-btn-4 button, #sq-btn-5 button {
            background-color: #0a0f1f !important;
            background: #0a0f1f !important;
            border-color: #003087 !important;
            border-left-color: #003087 !important;
            border-top-color: #003087 !important;
            border-left: 4px solid #003087 !important;
            border-top: 2px solid #003087 !important;
            color: #87ceeb !important;
        }
        #sq-btn-0 button:hover, #sq-btn-4 button:hover, #sq-btn-5 button:hover {
            background-color: #0d1a2e !important;
            background: #0d1a2e !important;
            border-color: #0070C0 !important;
            border-left-color: #0070C0 !important;
            border-top-color: #0070C0 !important;
        }
        /* Evaluate specs — Teal tint using Equinix color */
        #sq-btn-1 button, #sq-btn-6 button {
            background-color: #051615 !important;
            background: #051615 !important;
            border-color: #006B5E !important;
            border-left-color: #006B5E !important;
            border-top-color: #006B5E !important;
            border-left: 4px solid #006B5E !important;
            border-top: 2px solid #006B5E !important;
            color: #7dd3c0 !important;
        }
        #sq-btn-1 button:hover, #sq-btn-6 button:hover {
            background-color: #0a1e1a !important;
            background: #0a1e1a !important;
            border-color: #26ddb3 !important;
            border-left-color: #26ddb3 !important;
            border-top-color: #26ddb3 !important;
        }
        /* Compare — Purple tint using Equinix color */
        #sq-btn-2 button {
            background-color: #0f050f !important;
            background: #0f050f !important;
            border-color: #5B2D8E !important;
            border-left-color: #5B2D8E !important;
            border-top-color: #5B2D8E !important;
            border-left: 4px solid #5B2D8E !important;
            border-top: 2px solid #5B2D8E !important;
            color: #d8a8de !important;
        }
        #sq-btn-2 button:hover {
            background-color: #18102a !important;
            background: #18102a !important;
            border-color: #a78bfa !important;
            border-left-color: #a78bfa !important;
            border-top-color: #a78bfa !important;
        }
        /* Find resource — Red tint using Equinix color */
        #sq-btn-3 button, #sq-btn-7 button {
            background-color: #1a0505 !important;
            background: #1a0505 !important;
            border-color: #EC1C24 !important;
            border-left-color: #EC1C24 !important;
            border-top-color: #EC1C24 !important;
            border-left: 4px solid #EC1C24 !important;
            border-top: 2px solid #EC1C24 !important;
            color: #ff9999 !important;
        }
        #sq-btn-3 button:hover, #sq-btn-7 button:hover {
            background-color: #2a0e0e !important;
            background: #2a0e0e !important;
            border-color: #ff4550 !important;
            border-left-color: #ff4550 !important;
            border-top-color: #ff4550 !important;
        }
        #sq-row-1 { gap: 8px !important; margin-bottom: 8px !important; }
        #sq-row-1 > div { flex: 1 1 0 !important; min-width: 0 !important; max-width: 25% !important; }
        @media (max-width: 640px) {
            #sq-row-1 { flex-wrap: wrap !important; }
            #sq-row-1 > div { min-width: calc(50% - 5px) !important; max-width: 50% !important; }
        }
        </style>
        """)


        gr.HTML(value="""<script>
(function applyColors(){
    var cfg=[
        {id:'sq-btn-0',bg:'#0d1a2e',bl:'4px solid #2980d9',col:'#c8dff7',bd:'1px solid #1e3a5f'},
        {id:'sq-btn-1',bg:'#0a1e1a',bl:'4px solid #1ab894',col:'#b8e8de',bd:'1px solid #1a3a34'},
        {id:'sq-btn-2',bg:'#18102a',bl:'4px solid #8b5cf6',col:'#d4c8f7',bd:'1px solid #2e1a4a'},
        {id:'sq-btn-3',bg:'#1a0a0a',bl:'4px solid #EC1C24',col:'#f7c8c8',bd:'1px solid #3a1515'}
    ];
    cfg.forEach(function(c){
        var el=document.getElementById(c.id);
        if(!el)return;
        var btn=el.querySelector('button');
        if(!btn)return;
        btn.style.cssText='background:'+c.bg+'!important;background-color:'+c.bg+'!important;border:'+c.bd+'!important;border-left:'+c.bl+'!important;color:'+c.col+'!important;text-align:left!important;min-height:80px!important;white-space:normal!important;padding:16px 16px 16px 18px!important;font-size:0.83rem!important;border-radius:10px!important;width:100%!important;cursor:pointer!important;transition:all 0.15s ease!important;';
    });
}
applyColors();
setTimeout(applyColors,300);
setTimeout(applyColors,1000);
setTimeout(applyColors,2500);
var ob=new MutationObserver(applyColors);
ob.observe(document.body,{childList:true,subtree:true,attributes:true});
</script>""")

        with gr.Row(elem_id="search-row"):
            msg = gr.Textbox(
                placeholder="Ask anything about documents...",
                show_label=False, scale=5, container=False, lines=1,
                elem_id="search-input",
            )
            search_btn = gr.Button("Search ➤", scale=1, variant="primary", elem_id="search-btn")

        # Greeting card
        gr.HTML(value="""
        <div id="eq-greeting" style="background:#f7f8fa;border:1px solid #e5e7eb;border-radius:12px;
                    padding:22px 24px;display:flex;gap:16px;align-items:flex-start;margin-top:16px;">
          <div style="width:40px;height:40px;border-radius:8px;background:#fef2f2;display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:20px;">✨</div>
          <div>
            <div style="font-family:'Source Sans 3','DM Sans',sans-serif;font-size:16px;font-weight:700;color:#111827;margin-bottom:7px;">Ask Questions on Equinix's Resource</div>
            <div style="font-size:14.5px;color:#6b7280;line-height:1.65;">Search across Equinix&#39;s full resource library — technical blueprints, product data sheets, analyst reports and case studies. Try a product like <strong style="color:#EC1C24;font-weight:700;">Equinix Fabric</strong> or a use case like <strong style="color:#EC1C24;font-weight:700;">hybrid multicloud networking</strong>.</div>
          </div>
        </div>""", elem_id="greeting-card")

        # Slot 0 — most recent result (shown first, has feedback buttons)
        slot0_html   = gr.HTML(value="")
        with gr.Row():
            gr.HTML('<span style="font-size:0.72rem;color:#9ca3af;line-height:32px;">Was this helpful?</span>')
            up0   = gr.Button("👍", size="sm", scale=0, min_width=48, visible=False)
            dn0   = gr.Button("👎", size="sm", scale=0, min_width=48, visible=False)
            msg0  = gr.HTML(value="")

        # Older results (no per-card buttons — history)
        results_display = gr.HTML(value="")

        clear_btn = gr.Button("✕  Clear", size="sm", elem_id="clear-btn")

        # Memory panel — below results
        memory_panel = gr.HTML(value="", elem_id="memory-panel")

        # State to hold latest result data for feedback
        latest_query  = gr.State("")
        latest_answer = gr.State("")
        latest_cached = gr.State(False)
        latest_sources= gr.State([])

        with gr.Accordion("📤 Upload & Index", open=False, elem_id="upload-accordion"):

            with gr.Tab("🔗 Ingest URL"):
                gr.Markdown("<span style='color:#9ca3af;font-size:0.9rem;'>Paste any equinix.com/resources page URL — page content and PDF are ingested automatically.</span>")
                ingest_url_input = gr.Textbox(
                    label="Equinix resource page URL",
                    placeholder="https://www.equinix.com/resources/whitepapers/cloud-first-to-private-first",
                    info="Paste any equinix.com/resources/... URL",
                    lines=1,
                )
                url_ingest_btn = gr.Button("⚡ Ingest URL", variant="primary", elem_id="index-btn")
                url_ingest_log = gr.Textbox(
                    label="Progress log", lines=8, interactive=False,
                    elem_id="upload-log", placeholder="Logs will appear here...",
                )
                url_ingest_btn.click(
                    ingest_via_url,
                    inputs=[ingest_url_input],
                    outputs=[url_ingest_log],
                )

            with gr.Tab("📄 Upload PDF"):
                gr.Markdown("<span style='color:#9ca3af;font-size:0.9rem;'>Upload one PDF at a time. Fill in the AEM page URL so source cards link back to the correct resource page.</span>")

                with gr.Row():
                    pdf_upload    = gr.File(label="Select PDF", file_types=[".pdf"], file_count="single", scale=3)
                    resource_type = gr.Dropdown(
                        label="Resource type",
                        choices=RESOURCE_TYPE_LABELS,
                        value="whitepaper",
                        scale=1,
                    )

                with gr.Row():
                    doc_clean_name = gr.Textbox(
                        label="Display name",
                        placeholder="e.g. High Performance Data Centers for Dummies",
                        info="How the document name appears in search results",
                        scale=2,
                    )
                    doc_page_url = gr.Textbox(
                        label="AEM resource page URL",
                        placeholder="https://www.equinix.com/resources/whitepapers/high-performance-data-centers-for-dummies",
                        info="Paste the full equinix.com page URL — source cards will link here",
                        scale=3,
                    )

                index_btn  = gr.Button("⚡ Ingest & Index", variant="primary", elem_id="index-btn")
                upload_log = gr.Textbox(
                    label="Progress log", lines=8, interactive=False,
                    elem_id="upload-log", placeholder="Logs will appear here...",
                )
                index_btn.click(
                    ingest_and_index,
                    inputs=[pdf_upload, resource_type, doc_clean_name, doc_page_url],
                    outputs=[upload_log],
                )

        # Event handlers correctly excluding session_memory input to prevent state overwrite race condition
        search_btn.click(
            do_search_with_memory,
            [msg, results_html_state, slot0_html, memory_meta, gr.State("all"), visitor_storage_key, session_memory],
            [slot0_html, results_html_state, msg,
             latest_query, latest_answer, latest_cached, latest_sources,
             up0, dn0, msg0, session_memory, memory_panel, memory_meta, visitor_storage_key])

        msg.submit(
            do_search_with_memory,
            [msg, results_html_state, slot0_html, memory_meta, gr.State("all"), visitor_storage_key, session_memory],
            [slot0_html, results_html_state, msg,
             latest_query, latest_answer, latest_cached, latest_sources,
             up0, dn0, msg0, session_memory, memory_panel, memory_meta, visitor_storage_key])


        # ── Suggested query buttons — prefill + trigger search ──────────────
        sq_btn_0.click(fn=lambda q="What is Equinix Fabric and how does it work?": q, outputs=[msg]).then(
            do_search_with_memory,
            [msg, results_html_state, slot0_html, memory_meta, gr.State("all"), visitor_storage_key, session_memory],
            [slot0_html, results_html_state, msg, latest_query, latest_answer,
             latest_cached, latest_sources, up0, dn0, msg0,
             session_memory, memory_panel, memory_meta, visitor_storage_key])
        sq_btn_1.click(fn=lambda q="What port speeds and SLAs does Equinix Fabric support?": q, outputs=[msg]).then(
            do_search_with_memory,
            [msg, results_html_state, slot0_html, memory_meta, gr.State("all"), visitor_storage_key, session_memory],
            [slot0_html, results_html_state, msg, latest_query, latest_answer,
             latest_cached, latest_sources, up0, dn0, msg0,
             session_memory, memory_panel, memory_meta, visitor_storage_key])
        sq_btn_2.click(fn=lambda q="Equinix Fabric vs Network Edge for SD-WAN deployment": q, outputs=[msg]).then(
            do_search_with_memory,
            [msg, results_html_state, slot0_html, memory_meta, gr.State("all"), visitor_storage_key, session_memory],
            [slot0_html, results_html_state, msg, latest_query, latest_answer,
             latest_cached, latest_sources, up0, dn0, msg0,
             session_memory, memory_panel, memory_meta, visitor_storage_key])
        sq_btn_3.click(fn=lambda q="Show me a hybrid multicloud networking blueprint": q, outputs=[msg]).then(
            do_search_with_memory,
            [msg, results_html_state, slot0_html, memory_meta, gr.State("all"), visitor_storage_key, session_memory],
            [slot0_html, results_html_state, msg, latest_query, latest_answer,
             latest_cached, latest_sources, up0, dn0, msg0,
             session_memory, memory_panel, memory_meta, visitor_storage_key])
        results_html_state.change(lambda h: h, [results_html_state], [results_display])
        clear_btn.click(do_clear, outputs=[results_html_state, msg])
        clear_btn.click(lambda: ("", "", False, [], gr.update(visible=False), gr.update(visible=False), "", [], ""),
                        outputs=[slot0_html, latest_query, latest_cached, latest_sources, up0, dn0, msg0, session_memory, memory_panel])

        def _fb(rating, q, a, c, s):
            if not q:
                return '<span style="font-size:0.72rem;color:#9ca3af;">Run a search first.</span>'
            _last_result["query"]   = q
            _last_result["answer"]  = a
            _last_result["cached"]  = c
            _last_result["sources"] = s
            return submit_feedback(rating)

        up0.click(_fb, inputs=[gr.State(1),  latest_query, latest_answer, latest_cached, latest_sources], outputs=[msg0])
        dn0.click(_fb, inputs=[gr.State(-1), latest_query, latest_answer, latest_cached, latest_sources], outputs=[msg0])

    with gr.Tab("Analytics"):
        gr.HTML('''
        <div style="display: flex; align-items: center; justify-content: center; min-height: 920px; background: #000;">
            <div style="text-align: center;">
                <p style="color: #ffffff; font-size: 18px; margin-bottom: 20px;">Open Analytics Dashboard</p>
                <a href="https://huggingface.co/spaces/perwaizalam/rag-analytics" target="_blank" rel="noopener noreferrer"
                   style="display: inline-block; background: #ec1c24; color: #fff; padding: 14px 32px;
                           border-radius: 4px; text-decoration: none; font-weight: 600; font-size: 16px;
                           transition: background 0.3s ease; cursor: pointer;"
                   onmouseover="this.style.background='#c11219'"
                   onmouseout="this.style.background='#ec1c24'">
                     Go to Analytics →
                 </a>
            </div>
        </div>
        ''', elem_id="analytics-panel")

    # ── Safe Browser LocalStorage Engine (Bypasses Iframe Cookie Blocking) ──
    def sync_visitor_session(browser_stored_id):
        """
        Initialize visitor session using browser local storage instead of HTTP cookies.
        This works reliably in Hugging Face Space iframes where cookies are blocked.
        """
        import uuid

        # 1. Look for an existing ID in the browser's local storage first
        visitor_id = browser_stored_id if browser_stored_id else None

        # 2. If browser storage is empty, initialize a fresh token
        if not visitor_id:
            visitor_id = f"v_prod_{uuid.uuid4().hex[:12]}"
            print(f"✨ [INIT] Generated new visitor ID: {visitor_id}")
        else:
            print(f"📡 [INIT] Restored visitor ID from storage: {visitor_id}")

        # 3. Pull ground-truth records directly from your working backend history route
        history_data = []
        try:
            resp = requests.get(
                f"{API_GATEWAY_URL}/api/v1/visitor/{visitor_id}/history",
                headers={"X-API-Key": API_KEY},
                timeout=5
            )
            if resp.status_code == 200:
                history_data = resp.json().get("queries", [])
                print(f"📥 [INIT] Restored {len(history_data)} queries from DynamoDB.")
            else:
                print(f"⚠️ [INIT] History fetch returned {resp.status_code}")
        except Exception as e:
            print(f"❌ [INIT] Could not restore history: {e}")

        memory_html = render_memory_panel(history_data)

        # 4. Return history to state, update the HTML panel, and save the ID back to browser storage
        return gr.update(value=history_data), memory_html, visitor_id

    # ── Personalised card loader ─────────────────────────────────────────────
    def load_session_and_cards(browser_stored_id):
        """Load session + personalised suggested cards in one call."""
        # Get session data
        session_data, mem_html, vid = sync_visitor_session(browser_stored_id)

        # Get personalised suggestions
        card_updates = [gr.update()] * 4
        try:
            if vid and vid != "v_prod_guest":
                r = requests.get(
                    f"{API_GATEWAY_URL}/api/v1/visitor/{vid}/suggestions",
                    headers={"X-API-Key": API_KEY}, timeout=3
                )
                if r.status_code == 200:
                    data = r.json()
                    if data.get("personalised", False):
                        suggestions = data.get("suggestions", [])
                        card_updates = []
                        for s in suggestions[:4]:
                            icon = s.get("icon", "🔍")
                            text = s.get("text", "")
                            card_updates.append(gr.update(value=f"{icon}  {text}"))
                        while len(card_updates) < 4:
                            card_updates.append(gr.update())
                        print(f"✓ Personalised cards loaded for {vid[:12]}")
        except Exception as e:
            print(f"Card personalisation failed: {e}")

        return [session_data, mem_html, vid] + card_updates

    demo.load(
        load_session_and_cards,
        inputs=[visitor_storage_key],
        outputs=[session_memory, memory_panel, visitor_storage_key,
                 sq_btn_0, sq_btn_1, sq_btn_2, sq_btn_3]
    )

if __name__ == "__main__":
    demo.launch(css=CSS)