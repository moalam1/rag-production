"""
search_app.py — SEARCH-scoped FastAPI app for the Lambda deployment.

The full app (main.py) includes the ingestion router, which transitively imports
LlamaIndex (~866 modules, hundreds of MB) — that belongs to the FARGATE ingestion
container, NOT the search Lambda. This app mounts only the search/read-side
routers, so the search Lambda carries ZERO LlamaIndex. Same routers, middleware,
and limiter as main.py, minus ingest.

main.py stays the full app (EC2/local dev + the eval harness). lambda_handler.py
wraps THIS app for the search Lambda.
"""
import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

from config import settings
from limiter import limiter

# Search / read-side routers ONLY — NO ingest (that's Fargate, pulls LlamaIndex)
from api.routes.search    import router as search_router
from api.routes.analytics import router as analytics_router
from api.routes.admin     import router as admin_router
from api.routes.system    import router as system_router
from api.routes.visitor   import router as visitor_router
from api.feedback         import router as feedback_router

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

app = FastAPI(
    title="Document Search API (search-scoped)",
    description="RAG search — Lambda deployment (no ingestion stack)",
    version="1.0.0",
    docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(search_router)
app.include_router(analytics_router)
app.include_router(admin_router)
app.include_router(system_router)
app.include_router(visitor_router)
app.include_router(feedback_router, prefix="/api/v1")


@app.on_event("startup")
async def startup():
    logging.getLogger(__name__).info(
        "Starting Search API (search-scoped, no ingest) [%s]", settings.ENVIRONMENT
    )
