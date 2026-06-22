"""
api/ingest.py — Document ingestion endpoint.

POST /api/v1/ingest   — ingest a PDF (url/base64) or a web page URL
GET  /api/v1/ingest/{job_id} — poll job status
DELETE /api/v1/ingest/{filename} — remove document from indexes

Three ingest modes:
  1. page_url   — web page URL (equinix.com/resources/...)
                  → page_parser + ingest_router (teaser + PDF if present)
  2. pdf_url    — direct PDF URL
                  → ingester (LlamaParse)
  3. pdf_base64 — base64-encoded PDF bytes
                  → ingester (LlamaParse)
"""
import uuid
import logging
import tempfile
import os
import base64
import httpx
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.deps import verify_api_key
from pipeline.ingester import ingest

log    = logging.getLogger(__name__)
router = APIRouter()

# In-memory job store — replace with DynamoDB in Phase 2
_jobs: dict[str, dict] = {}


class IngestRequest(BaseModel):
    # ── Mode 1: web page URL ──────────────────────────────────────
    page_url:        Optional[str] = Field(None, description="Equinix resource page URL")

    # ── Mode 2/3: direct PDF ─────────────────────────────────────
    pdf_url:         Optional[str] = None
    pdf_base64:      Optional[str] = None
    filename:        Optional[str] = Field(None, min_length=1, max_length=500)

    # ── Metadata (optional for page_url — auto-detected) ─────────
    resource_type:   Optional[str] = Field(None, description="One of the Equinix resource types")
    clean_name:      str           = ""
    document_family: str           = ""
    published_date:  str           = ""


class IngestResponse(BaseModel):
    job_id:   str
    status:   str
    source:   str
    message:  str


class JobStatusResponse(BaseModel):
    job_id:  str
    status:  str        # queued | processing | complete | failed
    source:  str
    logs:    list[str] = []
    message: str       = ""


@router.post("/ingest", response_model=IngestResponse)
async def start_ingest(req: IngestRequest, _: str = Depends(verify_api_key)):
    """
    Start a document ingestion job.

    Pass one of:
      - page_url   → ingest an Equinix resource page (auto-detects PDF)
      - pdf_url    → ingest a PDF from a direct URL
      - pdf_base64 → ingest a base64-encoded PDF

    Returns job_id immediately. Poll GET /api/v1/ingest/{job_id} for status.
    """
    # Validate — exactly one source required
    sources = [s for s in [req.page_url, req.pdf_url, req.pdf_base64] if s]
    if not sources:
        raise HTTPException(
            status_code=400,
            detail="Provide one of: page_url, pdf_url, or pdf_base64"
        )
    if len(sources) > 1:
        raise HTTPException(
            status_code=400,
            detail="Provide only one of: page_url, pdf_url, or pdf_base64"
        )

    # PDF modes require filename
    if (req.pdf_url or req.pdf_base64) and not req.filename:
        raise HTTPException(
            status_code=400,
            detail="filename is required for pdf_url and pdf_base64 modes"
        )

    # PDF modes require resource_type
    if (req.pdf_url or req.pdf_base64) and not req.resource_type:
        raise HTTPException(
            status_code=400,
            detail="resource_type is required for pdf_url and pdf_base64 modes"
        )

    job_id = str(uuid.uuid4())
    source = req.page_url or req.pdf_url or "base64-upload"

    _jobs[job_id] = {
        "status":  "queued",
        "source":  source,
        "logs":    [],
        "message": "Job queued",
    }

    import asyncio
    if req.page_url:
        asyncio.create_task(_run_page_ingest(job_id, req))
    else:
        asyncio.create_task(_run_pdf_ingest(job_id, req))

    return IngestResponse(
        job_id  = job_id,
        status  = "queued",
        source  = source,
        message = f"Ingestion started — poll /api/v1/ingest/{job_id} for status",
    )


# ── Mode 1: page URL ingest ───────────────────────────────────────────────────

async def _run_page_ingest(job_id: str, req: IngestRequest):
    """
    Ingest an Equinix resource page URL.
    Runs page_parser → ingest_router (handles teaser + PDF automatically).
    """
    import asyncio
    from pipeline.page_parser import parse_page
    from pipeline.ingest_router import route_and_ingest

    _jobs[job_id]["status"] = "processing"
    _jobs[job_id]["logs"].append(f"🔍 Parsing page: {req.page_url}")

    try:
        # parse_page is sync — run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        page = await loop.run_in_executor(None, parse_page, req.page_url)

        if not page:
            _jobs[job_id]["status"]  = "failed"
            _jobs[job_id]["message"] = "Page parse returned no content"
            _jobs[job_id]["logs"].append("❌ Could not extract content from page")
            return

        _jobs[job_id]["logs"].append(
            f"✅ Parsed — {page.word_count} words | type={page.resource_type} | "
            f"pdf={'yes' if page.has_pdf else 'no'}"
        )

        # Override metadata if provided
        if req.document_family:
            page.document_family = req.document_family
        if req.published_date:
            page.published_date = req.published_date

        # Route and ingest
        logs = await loop.run_in_executor(None, route_and_ingest, page)
        _jobs[job_id]["logs"].extend(logs)

        # Determine outcome
        if any("❌" in l for l in logs):
            _jobs[job_id]["status"]  = "failed"
            _jobs[job_id]["message"] = "Ingest failed — see logs"
        elif any("skipping" in l.lower() or "unchanged" in l.lower() for l in logs):
            _jobs[job_id]["status"]  = "complete"
            _jobs[job_id]["message"] = "Document unchanged — skipped"
        else:
            _jobs[job_id]["status"]  = "complete"
            _jobs[job_id]["message"] = f"Page ingested successfully (v{page.document_family})"

        log.info("Job %s complete — %s", job_id, req.page_url)

    except Exception as e:
        _jobs[job_id]["status"]  = "failed"
        _jobs[job_id]["message"] = str(e)
        _jobs[job_id]["logs"].append(f"❌ Error: {e}")
        log.exception("Page ingest job %s failed", job_id)


# ── Mode 2/3: PDF ingest ──────────────────────────────────────────────────────

async def _run_pdf_ingest(job_id: str, req: IngestRequest):
    """
    Ingest a PDF from URL or base64.
    Runs ingester directly (LlamaParse → chunk → embed → upsert).
    """
    _jobs[job_id]["status"] = "processing"
    tmp_dir = tempfile.mkdtemp()

    try:
        dest = os.path.join(tmp_dir, req.filename)

        if req.pdf_url:
            _jobs[job_id]["logs"].append(f"📥 Downloading PDF: {req.pdf_url}")
            async with httpx.AsyncClient(
                timeout=60,
                follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (compatible; EquinixRAG/1.0)"},
            ) as client:
                r = await client.get(req.pdf_url)
                r.raise_for_status()
                Path(dest).write_bytes(r.content)
            _jobs[job_id]["logs"].append(f"✅ Downloaded ({len(r.content):,} bytes)")
        else:
            _jobs[job_id]["logs"].append("📥 Decoding base64 PDF...")
            Path(dest).write_bytes(base64.b64decode(req.pdf_base64))
            _jobs[job_id]["logs"].append("✅ Decoded")

        # Run ingestion pipeline
        logs = ingest(
            tmp_dir             = tmp_dir,
            resource_type       = req.resource_type,
            clean_name_override = req.clean_name,
            page_url_override   = "",
            document_family     = req.document_family,
            published_date      = req.published_date,
        )

        _jobs[job_id]["logs"].extend(logs)

        if any("❌" in l for l in logs):
            _jobs[job_id]["status"]  = "failed"
            _jobs[job_id]["message"] = "Ingest failed — see logs"
        elif any("unchanged" in l.lower() for l in logs):
            _jobs[job_id]["status"]  = "complete"
            _jobs[job_id]["message"] = "Document unchanged — skipped"
        else:
            _jobs[job_id]["status"]  = "complete"
            _jobs[job_id]["message"] = "PDF ingested successfully"

        log.info("Job %s complete — %s", job_id, req.filename)

    except Exception as e:
        _jobs[job_id]["status"]  = "failed"
        _jobs[job_id]["message"] = str(e)
        _jobs[job_id]["logs"].append(f"❌ Error: {e}")
        log.exception("PDF ingest job %s failed", job_id)


# ── Job status polling ────────────────────────────────────────────────────────

@router.get("/ingest/{job_id}", response_model=JobStatusResponse)
async def get_ingest_status(job_id: str, _: str = Depends(verify_api_key)):
    """Poll ingestion job status."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return JobStatusResponse(
        job_id  = job_id,
        status  = job["status"],
        source  = job.get("source", "unknown"),
        logs    = job["logs"],
        message = job["message"],
    )


# ── Delete document ───────────────────────────────────────────────────────────

@router.delete("/ingest/{identifier}")
async def delete_document(identifier: str, _: str = Depends(verify_api_key)):
    """
    Remove all chunks for a document from both Pinecone indexes.
    Pass filename or document_family as identifier.
    """
    from pipeline.ingester import _index, _summary
    ALL_NS  = ["technical", "business", "media"]
    deleted = 0

    for ns in ALL_NS:
        for idx in [_index, _summary]:
            for filter_key in ["filename", "document_family"]:
                try:
                    idx.delete(
                        filter    = {filter_key: {"$eq": identifier}},
                        namespace = ns,
                    )
                    deleted += 1
                except Exception as e:
                    log.warning("Delete error ns=%s key=%s: %s", ns, filter_key, e)

    return {
        "identifier": identifier,
        "message":    f"Deleted from {deleted} namespace/index combinations",
    }
