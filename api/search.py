"""
api/search.py — FastAPI search endpoint consumed by AEM.
"""
import logging
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field

from config import settings
import guardrails.input  as input_guard
import guardrails.output as output_guard
from pipeline.embedder  import embed_text
from pipeline.retriever import retrieve_chunks
from pipeline.reranker  import rerank_chunks, build_context
from pipeline.generator import generate_answer, prepare_query
from langsmith import traceable

log    = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["search"])

# ── API key auth ──────────────────────────────────────────────────
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(key: str = Depends(api_key_header)):
    if settings.API_KEY and key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return key


# ── Request / Response models ─────────────────────────────────────

class SearchRequest(BaseModel):
    query:     str       = Field(..., min_length=3, max_length=1000)
    top_k:     int       = Field(5, ge=1, le=10)

class Source(BaseModel):
    filename:        str
    clean_name:      str
    page:            str
    pdf_url:         str
    page_url:        str = ""
    resource_type:   str = ""
    preview:         str
    relevance_score: float

class SearchResponse(BaseModel):
    query:     str
    answer:    str
    sources:   list[Source]
    followups: list[str]
    blocked:   bool = False
    cached:    bool = False


# ── Endpoints ──────────────────────────────────────────────────────

@traceable(name="rag-search", run_type="chain")
async def _run_pipeline(query: str, namespace: str = None) -> dict:
    from pipeline.embedder import embed_text
    from pipeline.retriever import retrieve_chunks
    from pipeline.reranker import rerank_chunks, build_context
    retrieval_query, detected_lang = prepare_query(query)
    embedding = embed_text(retrieval_query)
    chunks    = retrieve_chunks(embedding, namespace=namespace)
    reranked  = rerank_chunks(retrieval_query, chunks)
    context   = build_context(reranked)
    result    = generate_answer(query, context, detected_lang=detected_lang)
    return {"result": result, "reranked": reranked, "detected_lang": detected_lang}

@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest, _: str = Depends(verify_api_key)):

    # 1. Input guardrails
    passed, message = input_guard.run(req.query)
    if not passed:
        return SearchResponse(
            query=req.query, answer=message,
            sources=[], followups=[], blocked=True,
        )

    # 2. Language detection + translation
    # detect_language uses Comprehend ($0.0001/call, stays in AWS)
    # translate_to_english uses gpt-4o-mini only if non-English detected
    # English documents need English embeddings for accurate Pinecone retrieval
    retrieval_query, detected_lang = prepare_query(req.query)

    # 3. RAG pipeline
    embedding = embed_text(retrieval_query)
    chunks    = retrieve_chunks(embedding)

    if not chunks:
        return SearchResponse(
            query=req.query,
            answer="No relevant documents found in the index.",
            sources=[], followups=[], blocked=False,
        )

    reranked = rerank_chunks(retrieval_query, chunks)
    context  = build_context(reranked)
    # Pass original query so GPT-4o responds in user's language
    result   = generate_answer(req.query, context, detected_lang=detected_lang)
    answer   = result["answer"]

    # 4. Output guardrails
    passed, message = output_guard.run(answer, context, reranked)
    if not passed:
        return SearchResponse(
            query=req.query, answer=message,
            sources=[], followups=[], blocked=True,
        )

    # 4. Build source list for AEM
    sources = [
        Source(
            filename=c["filename"],
            clean_name=c.get("clean_name", c["filename"]),
            page=c["page"],
            pdf_url=c.get("pdf_url", ""),
            page_url=c.get("page_url", ""),
            resource_type=c.get("resource_type", ""),
            preview=c["text"].split("\n\n",1)[-1][:200].strip(),
            relevance_score=round(c.get("rerank_score", 0.0), 4),
        )
        for c in reranked
    ]

    return SearchResponse(
        query=req.query,
        answer=answer,
        sources=sources,
        followups=result.get("followups", []),
        cached=result.get("cache_hit", False),
    )


@router.get("/health")
async def health():
    return {"status": "ok", "environment": settings.ENVIRONMENT}


@router.get("/cache/stats")
async def cache_stats(_: str = Depends(verify_api_key)):
    from cache.factory import cache
    c = cache()
    if hasattr(c, "stats"):
        return c.stats()
    return {"message": "Stats not available for this backend"}


@router.delete("/cache")
async def clear_cache(_: str = Depends(verify_api_key)):
    from cache.factory import cache
    cache().clear()
    return {"message": "Cache cleared"}


# ── Summarise endpoint ────────────────────────────────────────────

class SummariseRequest(BaseModel):
    filename: str = Field(..., min_length=1, max_length=500)

class SummariseResponse(BaseModel):
    filename:       str
    summary:        str = ""
    key_topics:     list[str] = []
    suggested_name: str = ""
    suggested_type: str = ""
    cached:         bool = False
    error:          str = ""

@router.post("/summarise", response_model=SummariseResponse)
async def summarise(req: SummariseRequest, _: str = Depends(verify_api_key)):
    from pipeline.generator import summarise_document
    result = summarise_document(req.filename)
    if "error" in result:
        return SummariseResponse(filename=req.filename, error=result["error"])
    return SummariseResponse(**result)


@router.get("/registry")
async def list_registry(_: str = Depends(verify_api_key)):
    """List all documents in the DynamoDB registry."""
    from pipeline.registry import list_documents
    docs = list_documents()
    return {
        "total": len(docs),
        "documents": docs,
    }
