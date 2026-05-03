# Document Search API — Production

## Folder Structure

```
rag_production/
├── config.py               # All settings — one place, driven by env vars
├── main.py                 # FastAPI app entrypoint
├── Dockerfile
├── requirements.txt
│
├── cache/
│   ├── base.py             # Abstract cache interface
│   ├── memory.py           # In-process LRU cache (default)
│   ├── redis_cache.py      # Redis cache (production)
│   └── factory.py          # Returns right backend from CACHE_BACKEND env var
│
├── pipeline/
│   ├── embedder.py         # OpenAI embeddings (cached)
│   ├── retriever.py        # Pinecone retrieval (cached)
│   ├── reranker.py         # Cohere reranking (cached)
│   ├── generator.py        # GPT-4o generation (cached)
│   └── ingester.py         # PDF → parse → chunk → upsert
│
├── guardrails/
│   ├── input.py            # Length, injection, relevance checks
│   └── output.py           # PII, citations, grounding checks
│
└── api/
    └── search.py           # FastAPI router — POST /api/v1/search
```

---

## Cache Options

| Backend | Config | Best For | Shared Across Pods |
|---------|--------|----------|--------------------|
| Memory  | `CACHE_BACKEND=memory` | Local dev, single instance, HF Spaces | ❌ |
| Redis   | `CACHE_BACKEND=redis`  | Production, multi-pod, persistent | ✅ |

### What gets cached

| Layer | Cache Key | TTL | Savings |
|-------|-----------|-----|---------|
| Embed | hash(query text) | 1h | ~$0.0001/call |
| Retrieve | hash(vector[:8]) | 10min | Pinecone latency |
| Rerank | hash(query + chunk ids) | 1h | Cohere API cost |
| Answer | hash(query + context[:500]) | 1h | ~$0.01-0.05/call |

### Redis setup (production)
```bash
# Docker
docker run -d -p 6379:6379 redis:7-alpine

# Env vars
CACHE_BACKEND=redis
REDIS_URL=redis://your-redis-host:6379
CACHE_TTL_SECONDS=3600
```

---

## Running Locally

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your keys
uvicorn main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

---

## Docker

```bash
docker build -t doc-search-api .
docker run -p 8000:8000 --env-file .env doc-search-api
```

---

## AEM Integration

AEM calls `POST /api/v1/search` with an API key header:

```json
// Request
POST /api/v1/search
X-API-Key: your-secret-key
Content-Type: application/json

{
  "query": "What does Equinix do?",
  "top_k": 5
}

// Response
{
  "query": "What does Equinix do?",
  "answer": "Equinix is a global digital infrastructure company [1] ...",
  "sources": [
    {
      "filename": "equinix_overview.pdf",
      "clean_name": "Equinix Overview",
      "page": "4",
      "pdf_url": "https://.../equinix_overview.pdf#page=4",
      "preview": "Equinix connects the world's leading businesses...",
      "relevance_score": 0.9821
    }
  ],
  "followups": ["What markets does Equinix operate in?"],
  "blocked": false
}
```

---

## Environment Variables

```bash
# Required
OPENAI_API_KEY=
PINECONE_API_KEY=
PINECONE_INDEX=rag-poc
COHERE_API_KEY=
LLAMA_CLOUD_API_KEY=
HF_TOKEN=
HF_REPO_ID=perwaizalam/rag-poc-demo

# Cache
CACHE_BACKEND=memory          # memory | redis
REDIS_URL=redis://localhost:6379
CACHE_TTL_SECONDS=3600

# API security
API_KEY=your-secret-key       # AEM sends this in X-API-Key header
RATE_LIMIT=30/minute
CORS_ORIGINS=https://your-aem-domain.com

# Guardrails
DOC_TOPIC=enterprise technology and data centers

# App
ENVIRONMENT=production        # hides /docs in production
LOG_LEVEL=INFO
```
