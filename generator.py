"""
pipeline/generator.py — GPT-4o answer generation with cache.

Cache key: hash of (query + context).
This is the most expensive cache — saves ~$0.01-0.05 per repeated query.
"""
import json
import logging
from openai import OpenAI

from config import settings
from cache.factory import cache
from cache.memory import MemoryCache

log = logging.getLogger(__name__)

_client = OpenAI(api_key=settings.OPENAI_API_KEY)

SYSTEM_PROMPT = """You are a research assistant. Answer the query using ONLY the provided context chunks.

Rules:
- Write a clear, flowing answer of 2-4 sentences.
- Cite sources inline using [1], [2], etc. matching the chunk numbers provided.
- Be factual and concise.
- Do NOT make up information not in the context.
- If nothing relevant, say: "I couldn't find relevant information in the documents."

Return ONLY a JSON object with exactly these two fields:
{
  "answer": "Your answer here with inline [1] citations [2].",
  "followups": ["Follow-up question 1?", "Follow-up question 2?", "Follow-up question 3?"]
}
No markdown, no explanation outside the JSON.
"""


def generate_answer(query: str, context: str) -> dict:
    """
    Generate a cited answer using GPT-4o.
    Identical (query + context) pairs return cached answers instantly.
    """
    c   = cache()
    key = MemoryCache.make_key("answer", {"q": query, "ctx": context[:500]})

    cached = c.get(key)
    if cached is not None:
        log.debug("answer cache HIT")
        return cached

    try:
        response = _client.chat.completions.create(
            model=settings.GENERATION_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Query: {query}\n\nContext:\n{context}"}
            ],
            temperature=0.2,
            max_tokens=600,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if model wraps in ```json
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        parsed = json.loads(raw)
        result = {
            "answer":    parsed.get("answer", ""),
            "followups": parsed.get("followups", []),
        }
    except Exception as e:
        log.error("generate_answer error: %s", e)
        result = {"answer": "Error generating answer.", "followups": []}

    c.set(key, result)
    return result
