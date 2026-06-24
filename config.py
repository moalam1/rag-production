"""
config.py — Central configuration. All env vars and constants live here.
Never import os.getenv() anywhere else.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # ── OpenAI ────────────────────────────────────────────────────
    OPENAI_API_KEY:    str = os.getenv("OPENAI_API_KEY", "")
    EMBED_MODEL:       str = "text-embedding-3-small"
    EMBED_DIMS:        int = 1024
    GENERATION_MODEL:  str = "gpt-4o"
    GUARDRAIL_MODEL:   str = "gpt-4o-mini"   # cheap model for safety checks
    AWS_REGION:                str = "us-east-1"
    BEDROCK_GUARDRAIL_ID:      str = ""
    BEDROCK_GUARDRAIL_VERSION: str = "DRAFT"

    # ── Pinecone ──────────────────────────────────────────────────
    PINECONE_API_KEY:  str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX:    str = os.getenv("PINECONE_INDEX", "rag-poc")
    PINECONE_SUMMARY_INDEX: str = os.getenv("PINECONE_SUMMARY_INDEX", "rag-summary")
    TOP_K_RETRIEVE:    int = 20
    TOP_K_RERANK:      int = 5

    # ── Cohere ────────────────────────────────────────────────────
    COHERE_API_KEY:    str = os.getenv("COHERE_API_KEY", "")
    RERANK_MODEL:      str = "rerank-english-v3.0"

    # ── LlamaParse ────────────────────────────────────────────────
    LLAMA_CLOUD_API_KEY: str = os.getenv("LLAMA_CLOUD_API_KEY", "")

    # ── HuggingFace ───────────────────────────────────────────────
    HF_TOKEN:          str = os.getenv("HF_TOKEN", "")
    HF_REPO_ID:        str = os.getenv("HF_REPO_ID", "perwaizalam/rag-poc-demo")
    HF_REPO_TYPE:      str = "space"
    PDF_DIR:           str = "pdfs"
    LOCAL_PDF_DIR:     str = "/app/pdfs"

    # ── Guardrails ────────────────────────────────────────────────
    DOC_TOPIC:         str = os.getenv("DOC_TOPIC", "enterprise technology and data centers")
    MAX_QUERY_LENGTH:  int = 1000
    MIN_QUERY_LENGTH:  int = 3

    # ── Cache ─────────────────────────────────────────────────────
    CACHE_BACKEND:     str = os.getenv("CACHE_BACKEND", "memory")   # memory | redis | dynamodb
    REDIS_URL:         str = os.getenv("REDIS_URL", "redis://localhost:6379")
    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "21600"))  # 6 hour default
    CACHE_MAX_SIZE:    int = int(os.getenv("CACHE_MAX_SIZE", "1000"))       # for in-memory
    # ── BM25 artifact (S3 — serverless, replaces Redis) ───────────
    BM25_S3_BUCKET:    str = os.getenv("BM25_S3_BUCKET", "rag-artifacts")

    # ── API ───────────────────────────────────────────────────────
    API_KEY:           str = os.getenv("API_KEY", "")                # for AEM auth
    RATE_LIMIT:        str = os.getenv("RATE_LIMIT", "30/minute")
    CORS_ORIGINS:      list = os.getenv("CORS_ORIGINS", "*").split(",")

    # ── Logging ───────────────────────────────────────────────────
    LOG_LEVEL:         str = os.getenv("LOG_LEVEL", "INFO")
    ENVIRONMENT:       str = os.getenv("ENVIRONMENT", "development")


settings = Settings()
