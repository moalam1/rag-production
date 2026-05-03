"""
guardrails/output.py — Output safety checks run after generation.
All checks return (passed: bool, message: str).
"""
import re
import logging
from openai import OpenAI

from config import settings

log = logging.getLogger(__name__)

_client = OpenAI(api_key=settings.OPENAI_API_KEY)

PII_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",                                      # SSN
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",      # email
    r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",                   # credit card
    r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",                          # phone
]


def check_pii(answer: str) -> tuple[bool, str]:
    for pattern in PII_PATTERNS:
        if re.search(pattern, answer):
            log.warning("PII detected in answer — blocking response")
            return False, "Response withheld — contains sensitive information."
    return True, ""


def check_citations(answer: str, sources: list) -> tuple[bool, str]:
    if not sources:
        return False, "No source documents found to support an answer."
    has_citation = any(f"[{i}]" in answer for i in range(1, len(sources) + 1))
    if not has_citation:
        return False, "Could not generate a cited answer from the available documents."
    return True, ""


def check_grounding(answer: str, context: str) -> tuple[bool, str]:
    """LLM-based hallucination check."""
    try:
        response = _client.chat.completions.create(
            model=settings.GUARDRAIL_MODEL,
            messages=[{
                "role": "user",
                "content": (
                    f"Does this answer contain ONLY information present in the context?\n"
                    f"Context: {context[:2000]}\n"
                    f"Answer: {answer}\n"
                    f"Reply with only YES or NO."
                )
            }],
            max_tokens=5,
            temperature=0,
        )
        result = response.choices[0].message.content.strip().upper()
        if "NO" in result:
            log.warning("Grounding check failed — answer may contain hallucinations")
            return False, "Answer could not be fully verified against source documents."
    except Exception as e:
        log.warning("Grounding check failed (allowing through): %s", e)
    return True, ""


def run(answer: str, context: str, sources: list) -> tuple[bool, str]:
    """Run all output guardrails in order. Returns on first failure."""
    for check, args in [
        (check_pii,        (answer,)),
        (check_citations,  (answer, sources)),
        (check_grounding,  (answer, context)),
    ]:
        passed, message = check(*args)
        if not passed:
            return False, message
    return True, ""
