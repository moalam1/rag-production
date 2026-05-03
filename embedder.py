"""
pipeline/embedder.py — Embedding with cache.

Cache key: hash of the query string.
Cache hit:  returns stored vector, skips OpenAI call entirely.
Cache miss: calls OpenAI, stores result, returns vector.
"""
import logging
from openai import OpenAI

from config import settings
from cache.factory import cache
from cache.memory import MemoryCache

log = logging.getLogger(__name__)

_client = OpenAI(api_key=settings.OPENAI_API_KEY)


def embed_text(text: str) -> list[float]:
    """
    Embed a single string. Results are cached so repeated queries
    (e.g. 'what does Equinix do') never hit OpenAI twice.
    """
    c   = cache()
    key = MemoryCache.make_key("embed", text.strip().lower())

    cached = c.get(key)
    if cached is not None:
        log.debug("embed cache HIT for: %s...", text[:40])
        return cached

    log.debug("embed cache MISS — calling OpenAI")
    response = _client.embeddings.create(
        input=text,
        model=settings.EMBED_MODEL,
        dimensions=settings.EMBED_DIMS,
    )
    vector = response.data[0].embedding
    c.set(key, vector, ttl=settings.CACHE_TTL_SECONDS)
    return vector
