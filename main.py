"""
main.py — FastAPI application entrypoint for production.
Run: uvicorn main:app --host 0.0.0.0 --port 8000
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from config import settings
from api.search import router as search_router
from api.ingest import router as ingest_router
from api.feedback import router as feedback_router
from api.routes.analytics import router as analytics_router
from api.routes.admin import router as admin_router

# ── Logging ───────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ── Rate limiter ──────────────────────────────────────────────────
from limiter import limiter

# ── App ───────────────────────────────────────────────────────────
app = FastAPI(
    title="Document Search API",
    description="RAG-powered document search for AEM integration",
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

import os
if os.path.exists("static"):
    app.mount("/dashboard", StaticFiles(directory="static", html=True), name="static")
app.include_router(search_router)
app.include_router(analytics_router)
app.include_router(admin_router)
app.include_router(ingest_router,  prefix="/api/v1")
app.include_router(feedback_router, prefix="/api/v1")


@app.on_event("startup")
async def startup():
    logging.getLogger(__name__).info(
        "Starting Document Search API [%s]", settings.ENVIRONMENT
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
