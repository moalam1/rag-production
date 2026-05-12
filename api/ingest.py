"""
api/ingest.py — Document ingestion endpoint.

POST /api/v1/ingest   — ingest a PDF from URL or base64
GET  /api/v1/ingest/{job_id} — poll job status
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

from api.search import verify_api_key
from pipeline.ingester import ingest

log    = logging.getLogger(__name__)
router = APIRouter()

# In-memory job store — replace with DynamoDB in Phase 2
_jobs: dict[str, dict] = {}


class IngestRequest(BaseModel):
    # Document source — one of pdf_url or pdf_base64 required
    pdf_url:         Optional[str] = None
    pdf_base64:      Optional[str] = None
    filename:        str           = Field(..., min_length=1, max_length=500)

    # Document metadata
    resource_type:   str           = Field(..., description="One of the 11 Equinix resource types")
    clean_name:      str           = ""
    page_url:        str           = ""
    document_family: str           = ""
    version:         int           = 1


class IngestResponse(BaseModel):
    job_id:   str
    status:   str
    filename: str
    message:  str


class JobStatusResponse(BaseModel):
    job_id:   str
    status:   str           # queued | processing | complete | failed
    filename: str
    logs:     list[str]     = []
    message:  str           = ""


@router.post("/ingest", response_model=IngestResponse)
async def start_ingest(req: IngestRequest, _: str = Depends(verify_api_key)):
    """
    Start a document ingestion job.
    Returns job_id immediately — poll GET /api/v1/ingest/{job_id} for status.
    """
    if not req.pdf_url and not req.pdf_base64:
        raise HTTPException(status_code=400, detail="Provide either pdf_url or pdf_base64")

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status":   "queued",
        "filename": req.filename,
        "logs":     [],
        "message":  "Job queued",
    }

    # Run synchronously for now (move to background task in Phase 2)
    import asyncio
    asyncio.create_task(_run_ingest(job_id, req))

    return IngestResponse(
        job_id=job_id,
        status="queued",
        filename=req.filename,
        message="Ingestion started — poll /api/v1/ingest/{job_id} for status",
    )


async def _run_ingest(job_id: str, req: IngestRequest):
    """Background ingestion task."""
    _jobs[job_id]["status"] = "processing"
    tmp_dir = tempfile.mkdtemp()

    try:
        dest = os.path.join(tmp_dir, req.filename)

        # Download or decode PDF
        if req.pdf_url:
            _jobs[job_id]["logs"].append(f"📥 Downloading {req.pdf_url}...")
            async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
                r = await client.get(req.pdf_url)
                r.raise_for_status()
                Path(dest).write_bytes(r.content)
            _jobs[job_id]["logs"].append("✅ Downloaded.")
        else:
            _jobs[job_id]["logs"].append("📥 Decoding base64 PDF...")
            Path(dest).write_bytes(base64.b64decode(req.pdf_base64))
            _jobs[job_id]["logs"].append("✅ Decoded.")

        # Run ingestion pipeline
        logs = ingest(
            tmp_dir=tmp_dir,
            resource_type=req.resource_type,
            clean_name_override=req.clean_name,
            page_url_override=req.page_url,
            document_family=req.document_family,
            version=req.version,
        )

        _jobs[job_id]["logs"].extend(logs)
        _jobs[job_id]["status"]  = "complete"
        _jobs[job_id]["message"] = "Ingestion complete"
        log.info("Job %s complete — %s", job_id, req.filename)

    except Exception as e:
        _jobs[job_id]["status"]  = "failed"
        _jobs[job_id]["message"] = str(e)
        _jobs[job_id]["logs"].append(f"❌ Error: {e}")
        log.exception("Job %s failed", job_id)


@router.get("/ingest/{job_id}", response_model=JobStatusResponse)
async def get_ingest_status(job_id: str, _: str = Depends(verify_api_key)):
    """Poll ingestion job status."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        filename=job["filename"],
        logs=job["logs"],
        message=job["message"],
    )


@router.delete("/ingest/{filename}")
async def delete_document(filename: str, _: str = Depends(verify_api_key)):
    """Remove all chunks for a document from both Pinecone indexes."""
    from pipeline.ingester import _index, _summary
    ALL_NS = ["technical", "business", "media"]
    deleted = 0
    for ns in ALL_NS:
        for idx in [_index, _summary]:
            try:
                idx.delete(
                    filter={"filename": {"$eq": filename}},
                    namespace=ns,
                )
                deleted += 1
            except Exception as e:
                log.warning("Delete error ns=%s: %s", ns, e)
    return {"filename": filename, "message": f"Deleted from {deleted} namespaces"}
