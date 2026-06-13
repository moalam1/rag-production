"""
api/feedback.py — Feedback endpoints.
POST /api/v1/feedback        — submit thumbs up/down
GET  /api/v1/feedback/stats  — satisfaction rate + top disliked queries
GET  /api/v1/feedback/export — export thumbs-down for eval dataset
"""
import logging
from fastapi import APIRouter, Request, Form
from fastapi.responses import JSONResponse
from typing import Optional
from pydantic import BaseModel
from pipeline.feedback_store import save_feedback, get_stats, get_thumbs_down

log    = logging.getLogger(__name__)
router = APIRouter()


class FeedbackRequest(BaseModel):
    query:   str
    answer:  str
    rating:  int
    cached:  bool = False
    sources: list = []
    comment: str  = ""
    lang:    str  = "en"


@router.post("/feedback")
async def submit_feedback(
    request: Request,
    # Form fields (from iframe form submission)
    query:   Optional[str] = Form(None),
    answer:  Optional[str] = Form(None),
    rating:  Optional[int] = Form(None),
    cached:  Optional[str] = Form(None),
):
    """Accept both JSON and form-encoded feedback."""
    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        # JSON path — from direct API calls / Gradio native buttons
        body = await request.json()
        q      = body.get("query", "")
        a      = body.get("answer", "")
        r      = int(body.get("rating", 0))
        c      = bool(body.get("cached", False))
        sources= body.get("sources", [])
        lang   = body.get("lang", "en")
    else:
        # Form-encoded path — from iframe form submission (HF Space buttons)
        q      = query  or ""
        a      = answer or ""
        try:
            r = int(rating) if rating is not None else 0
        except (ValueError, TypeError):
            r = 0
        c      = (cached or "").lower() == "true"
        sources= []
        lang   = "en"

    if r not in (1, -1):
        return JSONResponse({"status": "error", "message": "rating must be 1 or -1"})

    ok = save_feedback(query=q, answer=a, rating=r, cached=c, sources=sources, lang=lang)
    label = "thumbs_up" if r == 1 else "thumbs_down"
    return JSONResponse({"status": "ok" if ok else "partial", "label": label})


@router.get("/feedback/stats")
async def feedback_stats():
    return get_stats()


@router.get("/feedback/export")
async def export_thumbs_down(limit: int = 50):
    items = get_thumbs_down(limit=limit)
    return {"total": len(items), "records": items}
