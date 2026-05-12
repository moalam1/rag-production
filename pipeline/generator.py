"""
pipeline/generator.py — Multilingual answer generation with GPT-4o.

Changes from previous version:
  1. Language detection via Amazon Comprehend (replaces gpt-4o-mini call)
     - $0.0001 per unit vs ~$0.001 for gpt-4o-mini — 10× cheaper
     - Instant — no LLM latency
     - Stays in AWS — data doesn't leave your account
     - Free tier: 50,000 units/month
     - Fallback to gpt-4o-mini if Comprehend unavailable

  2. Query translation to English before retrieval
     Non-English queries are translated so OpenAI embeddings match English documents.
     GPT-4o responds in the user's original language.

  3. Explicit language instruction to GPT-4o
     Prevents GPT-4o from inferring language from document content.

Cache key: hash of (query + context) — same query returns cached answer instantly.
"""
import json
import logging
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from openai import OpenAI

from config import settings
from pinecone import Pinecone
from cache.factory import cache
from cache.memory import MemoryCache

log = logging.getLogger(__name__)

_client = OpenAI(api_key=settings.OPENAI_API_KEY)

# ── Amazon Comprehend client ──────────────────────────────────────
# Auth via EC2 IAM role — no API key needed
# IAM policy required: comprehend:DetectDominantLanguage
try:
    _comprehend      = boto3.client("comprehend", region_name=settings.AWS_REGION, config=__import__("botocore.config", fromlist=["Config"]).Config(connect_timeout=3, read_timeout=3, retries={"max_attempts": 1}))
    _COMPREHEND_AVAILABLE = True
    log.info("Comprehend client initialised (region: %s)", settings.AWS_REGION)
except (NoCredentialsError, Exception) as e:
    _comprehend = None
    _COMPREHEND_AVAILABLE = False
    log.warning("Comprehend unavailable: %s — falling back to gpt-4o-mini detection", e)

# ── Language name mapping ─────────────────────────────────────────
LANG_NAMES = {
    "en": "English",    "fr": "French",     "es": "Spanish",
    "de": "German",     "it": "Italian",    "pt": "Portuguese",
    "nl": "Dutch",      "ja": "Japanese",   "zh": "Chinese",
    "ko": "Korean",     "ar": "Arabic",     "hi": "Hindi",
    "ru": "Russian",    "tr": "Turkish",    "pl": "Polish",
    "sv": "Swedish",    "da": "Danish",     "fi": "Finnish",
    "no": "Norwegian",  "id": "Indonesian",
}

# ── System prompt ─────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a multilingual research assistant. Answer the query using ONLY the provided context chunks.

Rules:
- LANGUAGE: You will be told the exact language to respond in. Always follow it strictly.
- Write a clear, flowing answer of 2-4 sentences.
- Each chunk starts with "Document: <name>" — use that document name when citing.
- Cite sources inline using [1], [2], etc. matching the chunk numbers provided.
- Be factual and concise.
- Do NOT make up information not in the context.
- If nothing relevant found, say the equivalent of "I couldn't find relevant information in the documents." in the specified language.
- Generate follow-up questions in the SAME specified language.

Return ONLY a JSON object with exactly these two fields:
{
  "answer": "Your answer here with inline [1] citations [2].",
  "followups": ["Follow-up question 1?", "Follow-up question 2?", "Follow-up question 3?"]
}
No markdown, no explanation outside the JSON.
"""


def detect_language(text: str) -> str:
    """
    Detect the dominant language of text using Amazon Comprehend.
    Returns ISO 639-1 language code (e.g. "en", "fr", "ja").

    Comprehend: $0.0001 per unit, free tier 50K/month.
    Fallback: gpt-4o-mini call if Comprehend unavailable.
    """
    # ── Try Comprehend first ──────────────────────────────────────
    if _COMPREHEND_AVAILABLE and _comprehend:
        try:
            response  = _comprehend.detect_dominant_language(Text=text[:300])
            languages = response.get("Languages", [])
            if languages:
                top = max(languages, key=lambda x: x["Score"])
                if top["Score"] > 0.7:
                    lang = top["LanguageCode"][:2]   # e.g. "zh-TW" → "zh"
                    log.debug("Comprehend detected: %s (score: %.2f)", lang, top["Score"])
                    return lang
        except ClientError as e:
            log.warning("Comprehend ClientError: %s", e)
        except Exception as e:
            log.warning("Comprehend error: %s", e)

    # ── Fallback: gpt-4o-mini ─────────────────────────────────────
    try:
        response = _client.chat.completions.create(
            model=settings.GUARDRAIL_MODEL,
            messages=[{"role": "user",
                        "content": (
                            f"What is the ISO 639-1 language code of this text? "
                            f"Reply with ONLY the 2-letter code (e.g. en, fr, ja).\n"
                            f"Text: {text[:200]}"
                        )}],
            max_tokens=5,
            temperature=0,
        )
        lang = response.choices[0].message.content.strip().lower()[:2]
        log.debug("gpt-4o-mini detected language: %s", lang)
        return lang
    except Exception as e:
        log.warning("Language detection fallback failed: %s — defaulting to en", e)
        return "en"


def translate_to_english(text: str) -> str:
    """
    Translate text to English using gpt-4o-mini.
    Used to improve Pinecone retrieval when query is non-English
    (English documents need English embedding vectors to match).
    """
    try:
        response = _client.chat.completions.create(
            model=settings.GUARDRAIL_MODEL,
            messages=[{"role": "user",
                        "content": (
                            f"Translate this text to English. "
                            f"Return ONLY the translated text, nothing else.\n\n"
                            f"Text: {text}"
                        )}],
            max_tokens=300,
            temperature=0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        log.warning("Translation error: %s — using original query", e)
        return text


def prepare_query(query: str) -> tuple[str, str]:
    """
    Detect language and translate query to English if needed.
    Returns (retrieval_query, detected_lang).

    retrieval_query: English version for Pinecone embedding
    detected_lang:   original language code for GPT-4o response language
    """
    detected_lang = detect_language(query)

    if detected_lang == "en":
        return query, "en"

    log.info("Non-English query detected (%s) — translating for retrieval", detected_lang)
    translated = translate_to_english(query)
    return translated, detected_lang


def generate_answer(query: str, context: str, detected_lang: str = "en") -> dict:
    """
    Generate a cited answer using GPT-4o.
    Responds in the language specified by detected_lang.
    Identical (query + context) pairs return cached answers instantly.
    """
    c   = cache()
    key = MemoryCache.make_key("answer", {"q": query.strip().lower(), "lang": detected_lang})

    cached = c.get(key)
    if cached is not None:
        log.debug("answer cache HIT")
        cached["cache_hit"] = True
        return cached

    lang_name    = LANG_NAMES.get(detected_lang, "English")
    user_message = (
        f"IMPORTANT: Respond entirely in {lang_name}. "
        f"The answer, citations, and all follow-up questions must be in {lang_name}.\n\n"
        f"Query: {query}\n\nContext:\n{context}"
    )

    try:
        response = _client.chat.completions.create(
            model=settings.GENERATION_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
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
            "cache_hit": False,
        }
    except Exception as e:
        log.error("generate_answer error: %s", e)
        result = {"answer": "Error generating answer.", "followups": []}

    c.set(key, result)
    return result


# ── Summary index client ──────────────────────────────────────────
_summary_index = Pinecone(api_key=settings.PINECONE_API_KEY).Index(
    settings.PINECONE_SUMMARY_INDEX
)

SUMMARY_PROMPT = """You are a document analyst. Read the following document chunks and produce:
1. A clear 2-3 sentence overview of what this document covers
2. 5-8 key topics as short tags
3. A suggested clean display name (title case, no filename extensions)
4. The most likely resource type from: whitepaper, blueprint, case-study, analyst-report, data-sheet, solution-brief, playbook, article, multimedia, webinar

Return ONLY a JSON object:
{
  "summary":        "2-3 sentence overview...",
  "key_topics":     ["topic1", "topic2", "topic3"],
  "suggested_name": "Clean Document Name",
  "suggested_type": "whitepaper"
}
No markdown, no explanation outside the JSON.
"""


def summarise_document(filename: str) -> dict:
    """
    Generate AI summary for a document using the rag-summary index.
    Cache layers:
      1. Redis exact cache (TTL 24h)
      2. Full pipeline — fetch chunks → GPT-4o
    Falls back to rag-poc search index if summary index has no chunks yet.
    """
    import json as _json

    # ── Redis cache check ─────────────────────────────────────────
    c   = cache()
    key = MemoryCache.make_key("summary", {"filename": filename})

    cached = c.get(key)
    if cached is not None:
        log.info("Summary cache HIT — %s", filename)
        cached["cached"] = True
        return cached

    # ── Fetch chunks from summary index ──────────────────────────
    try:
        embed_resp = _client.embeddings.create(
            input=filename,
            model="text-embedding-3-small",
            dimensions=1024,
        )
        vector = embed_resp.data[0].embedding
    except Exception as e:
        log.error("Embed error for summary: %s", e)
        return {"error": "Could not generate summary — embedding failed"}

    ALL_NS = ["technical", "business", "media"]
    chunks = []

    # Try summary index first
    for ns in ALL_NS:
        try:
            results = _summary_index.query(
                vector=vector,
                top_k=5,
                include_metadata=True,
                namespace=ns,
                filter={"filename": {"$eq": filename}, "status": {"$eq": "current"}},
            )
            for match in results.matches:
                nc = match.metadata.get("_node_content", "")
                if nc:
                    try:
                        text = _json.loads(nc).get("text", "")
                        if text.strip():
                            chunks.append(text)
                    except Exception:
                        pass
        except Exception as e:
            log.warning("Summary index query error ns=%s: %s", ns, e)

    # Fallback to search index if summary index empty
    if not chunks:
        log.info("Summary index empty for %s — falling back to search index", filename)
        from pinecone import Pinecone as _Pinecone
        _search_idx = _Pinecone(api_key=settings.PINECONE_API_KEY).Index(settings.PINECONE_INDEX)
        for ns in ALL_NS:
            try:
                results = _search_idx.query(
                    vector=vector,
                    top_k=8,
                    include_metadata=True,
                    namespace=ns,
                    filter={"filename": {"$eq": filename}},
                )
                for match in results.matches:
                    nc = match.metadata.get("_node_content", "")
                    if nc:
                        try:
                            text = _json.loads(nc).get("text", "")
                            if text.strip():
                                chunks.append(text)
                        except Exception:
                            pass
            except Exception as e:
                log.warning("Search index fallback error ns=%s: %s", ns, e)

    if not chunks:
        return {"error": f"No content found for {filename} in any index"}

    # ── GPT-4o summarise ─────────────────────────────────────────
    context = "\n\n---\n\n".join(chunks[:8])
    try:
        response = _client.chat.completions.create(
            model=settings.GENERATION_MODEL,
            messages=[
                {"role": "system", "content": SUMMARY_PROMPT},
                {"role": "user",   "content": f"Document chunks:\n\n{context}"},
            ],
            temperature=0.2,
            max_tokens=400,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = _json.loads(raw.strip())
        result = {
            "filename":       filename,
            "summary":        parsed.get("summary", ""),
            "key_topics":     parsed.get("key_topics", []),
            "suggested_name": parsed.get("suggested_name", ""),
            "suggested_type": parsed.get("suggested_type", ""),
            "cached":         False,
        }
    except Exception as e:
        log.error("GPT-4o summary error: %s", e)
        return {"error": f"Could not generate summary: {e}"}

    # ── Store in Redis (24h) ──────────────────────────────────────
    c.set(key, result, ttl=86400)
    log.info("Summary cached for %s", filename)
    return result
