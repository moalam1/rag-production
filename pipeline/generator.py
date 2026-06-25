from pipeline.prompt_registry import get_prompt, get_prompt_version
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
from langsmith import traceable

log = logging.getLogger(__name__)

# Spell correction for English queries
try:
    from spellchecker import SpellChecker
    _spell = SpellChecker()
    _SPELL_AVAILABLE = True
except ImportError:
    _SPELL_AVAILABLE = False
    log.warning("pyspellchecker not installed — spell correction disabled")

# Terms that must never be corrected — Equinix product names, codes, acronyms
_PROTECTED = {
    "equinix","fabric","xscale","ibx","amer","emea","apac","ioa",
    "megaport","zayo","fastly","coresite","verizon","lumen","cogent",
    "pinecone","llamaparse","langsmith","openai","cohere","bedrock",
    "10gbps","100gbps","400gbps","1gbps","gbps","mbps","tbps",
    "sv5","sv1","dc2","ny2","la1","ld4","fr5","sg1","ty2","os1","hk1",
    "colocation","colo","interconnection","mpls","bgp","sdwan","sdn",
    "multicloud","iaas","paas","saas","vpc","cdn","dns","api","url",
    "whitepaper","datasheet","infopaper","webinar",
}

def _correct_spelling(text: str) -> str:
    """
    Correct obvious spelling mistakes in English queries.
    Skips: short words (<=3 chars), PROTECTED domain terms,
           capitalised words (likely proper nouns), numbers.
    """
    if not _SPELL_AVAILABLE:
        return text
    words = text.split()
    corrected = []
    changed = []
    for word in words:
        # Strip punctuation for checking but preserve it
        clean = ''.join(c for c in word if c.isalpha() or c == '-')
        lower = clean.lower()
        # Skip: short, protected, capitalised (proper noun), numeric, already correct
        if (len(clean) <= 3
                or lower in _PROTECTED
                or (clean[0].isupper() and len(words) > 1)
                or any(c.isdigit() for c in clean)
                or not _spell.unknown([lower])):
            corrected.append(word)
            continue
        suggestion = _spell.correction(lower)
        if suggestion and suggestion != lower:
            # Preserve original capitalisation pattern
            if clean[0].isupper():
                suggestion = suggestion.capitalize()
            corrected.append(word.replace(clean, suggestion))
            changed.append(f"{clean}→{suggestion}")
        else:
            corrected.append(word)
    if changed:
        log.info("Spell correction: %s", ", ".join(changed))
    return " ".join(corrected)
from langsmith.wrappers import wrap_openai
from pinecone import Pinecone
from cache.factory import cache
from cache.memory import MemoryCache

log = logging.getLogger(__name__)

_client = wrap_openai(OpenAI(api_key=settings.OPENAI_API_KEY))

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

# ══════════════════════════════════════════════════════════════
# PROMPT CHANGE DEPLOYMENT CHECKLIST
# When SYSTEM_PROMPT in generator.py changes:
#   1. Increment PROMPT_VERSION here          (semantic_cache.py)
#   2. Increment pv= in generator.py          (MemoryCache key)
#   3. Increment CACHE_VERSION here           (full cache invalidation)
#   4. sudo systemctl restart rag-api         (clears in-process cache)
#   5. Run cache clear script MODE="semantic" (purge Pinecone entries)
# ══════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are an expert technical advisor for Equinix — the world's largest digital infrastructure company.
You help enterprise IT leaders, network architects and procurement decision-makers find precise answers from Equinix's resource library.

LANGUAGE: Respond entirely in the language specified in the user message.
All content — answer, citations and follow-up questions — must be in that language.

ANSWER RULES:
- Use ONLY the provided context chunks. Never invent facts.
- Write 3-5 sentences in a confident, expert tone as if briefing a CTO or enterprise architect.
- Lead with the direct answer. Never open with "Based on the documents..." or "According to...".
- Be specific: include numbers, product names, port speeds, SLAs, and limitations when present in context.
- Pricing queries with no pricing data in context: say "Pricing for [product] is not in the resource library. Visit equinix.com/contact or speak with a Solutions Architect for a custom quote."
- Vague queries (e.g. "what equinix does"): answer at a high level and use a clarifying follow-up question.
- Nothing relevant found: say "I could not find specific information on this in Equinix's resource library. Try rephrasing or contact our team directly."

CITATION RULES:
- Each chunk starts with "Document: <name>" — use that name when citing.
- Cite inline using [1], [2] matching chunk numbers.
- Only cite chunks you actually used. One citation is enough if one chunk answers the question.

FOLLOW-UP RULES:
- Generate exactly 3 follow-up questions a serious enterprise buyer would ask next.
- Make them specific to the products and use case — not generic.
- Progress the buyer journey: specs query → follow-ups probe deployment, pricing, comparison.
- Never generate "What else can I help you with?" or "Would you like more details?"

Return ONLY valid JSON — no markdown, no backticks, no explanation outside:
{
  "answer": "Your expert answer with inline [1] citations.",
  "followups": ["Specific follow-up 1?", "Specific follow-up 2?", "Specific follow-up 3?"]
}
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
        corrected = _correct_spelling(query)
        return corrected, "en"

    log.info("Non-English query detected (%s) — translating for retrieval", detected_lang)
    translated = translate_to_english(query)
    return translated, detected_lang


@traceable(name="generate-answer", run_type="llm")
def generate_answer(query: str, context: str, detected_lang: str = "en", visitor_profile: str = "") -> dict:
    """
    Generate a cited answer using GPT-4o.
    Responds in the language specified by detected_lang.
    Identical (query + context) pairs return cached answers instantly.
    """
    c   = cache()
    key = MemoryCache.make_key("answer", {"q": query.strip().lower(), "lang": detected_lang, "pv": get_prompt_version("generation", 2)})

    cached = c.get(key)
    if cached is not None:
        log.debug("answer cache HIT")
        cached["cache_hit"] = True
        return cached

    lang_name    = LANG_NAMES.get(detected_lang, "English")
    profile_block = f"\n\n[VISITOR MEMORY PROFILE]\n{visitor_profile.strip()}" if visitor_profile and visitor_profile.strip() else ""
    user_message = (
        f"IMPORTANT: Respond entirely in {lang_name}. "
        f"The answer, citations, and all follow-up questions must be in {lang_name}.\n\n"
        f"Query: {query}\n\nContext:\n{context}"
        f"{profile_block}"
    )

    try:
        response = _client.chat.completions.create(
            model=settings.GENERATION_MODEL,
            messages=[
                {"role": "system", "content": get_prompt("generation", SYSTEM_PROMPT)},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.1,
            max_tokens=900,
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


@traceable(name="summarise-document", run_type="llm")
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
