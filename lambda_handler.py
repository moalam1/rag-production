"""
lambda_handler.py — AWS Lambda entry point for the SEARCH path.

Wraps the SEARCH-scoped FastAPI `app` (search_app.py — no ingestion/LlamaIndex) with Mangum so API Gateway can invoke
it as a Lambda. The SAME app runs on uvicorn (EC2/local) and Lambda (here) — no
app changes, full dev/prod parity.

API Gateway → Lambda(this handler) → Mangum → FastAPI(app) → response → Mangum →
API Gateway. Mangum translates the API Gateway event ↔ ASGI both directions.

lifespan="auto": runs FastAPI startup/shutdown events if present, tolerates their
absence. The app's startup hook only logs, so cold-start cost here is ~nil.

The BM25 index hydrates from S3 at module scope when pipeline.retriever is first
imported (on the first search) — once per cold container, reused across warm
invocations (the 'global container scope' pattern). See pipeline/retriever.py.
"""
from mangum import Mangum
from search_app import app

handler = Mangum(app, lifespan="auto")
