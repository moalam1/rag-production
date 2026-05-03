"""
guardrails/input.py — Input safety checks run before the RAG pipeline.
All checks return (passed: bool, message: str).
"""
import re
import logging
from openai import OpenAI

from config import settings

log = logging.getLogger(__name__)

_client = OpenAI(api_key=settings.OPENAI_API_KEY)

INJECTION_PATTERNS = [
    r"ignore (previous|all|above) instructions",
    r"you are now",
    r"act as (if you are|a|an)",
    r"jailbreak",
    r"system prompt",
    r"forget everything",
    r"disregard (all|previous|your)",
    r"new persona",
    r"pretend (you are|to be)",
]


def check_length(query: str) -> tuple[bool, str]:
    if len(query.strip()) < settings.MIN_QUERY_LENGTH:
        return False, "Query too short. Please ask a complete question."
    if len(query) > settings.MAX_QUERY_LENGTH:
        return False, f"Query too long. Please keep it under {settings.MAX_QUERY_LENGTH} characters."
    return True, ""


def check_injection(query: str) -> tuple[bool, str]:
    q = query.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, q):
            log.warning("Injection attempt detected: %s", query[:80])
            return False, "Invalid query detected. Please ask a genuine question."
    return True, ""


def check_relevance(query: str) -> tuple[bool, str]:
    """LLM-based topic relevance check using a cheap model."""
    try:
        response = _client.chat.completions.create(
            model=settings.GUARDRAIL_MODEL,
            messages=[{
                "role": "user",
                "content": (
                    f"Is this query relevant to {settings.DOC_TOPIC}?\n"
                    f"Query: \"{query}\"\n"
                    f"Answer with only YES or NO."
                )
            }],
            max_tokens=5,
            temperature=0,
        )
        answer = response.choices[0].message.content.strip().upper()
        if "NO" in answer:
            return False, (
                f"This question is outside the scope of the document library. "
                f"Try asking about {settings.DOC_TOPIC}."
            )
    except Exception as e:
        # If guardrail fails, fail open (allow through) to avoid blocking legitimate users
        log.warning("Relevance check failed (allowing through): %s", e)
    return True, ""


def run(query: str) -> tuple[bool, str]:
    """Run all input guardrails in order. Returns on first failure."""
    for check in [check_length, check_injection, check_relevance]:
        passed, message = check(query)
        if not passed:
            return False, message
    return True, ""
