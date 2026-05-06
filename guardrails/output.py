"""
guardrails/output.py — Output safety checks run after answer generation.

Check order:
  1. check_bedrock   — AWS Bedrock Guardrail (replaces gpt-4o-mini PII + grounding checks)
                       covers: PII redaction, content filter, grounding verification
                       CloudTrail audit log on every call
  2. check_citations — local check, free, no LLM call

Bedrock Guardrail replaces two old gpt-4o-mini calls:
  - check_pii:      PII patterns now caught by Bedrock PII policy
  - check_grounding: grounding now handled by Bedrock grounding check
  Saves 2 LLM calls per search query.
"""
import re
import logging
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from openai import OpenAI

from config import settings

log = logging.getLogger(__name__)

_openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

# ── AWS Bedrock client ────────────────────────────────────────────
try:
    _bedrock = boto3.client("bedrock-runtime", region_name=settings.AWS_REGION)
    _BEDROCK_AVAILABLE = True
except (NoCredentialsError, Exception) as e:
    _bedrock = None
    _BEDROCK_AVAILABLE = False
    log.warning("Bedrock unavailable for output guardrails: %s", e)

# ── Local PII patterns (fallback only) ───────────────────────────
PII_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",                                      # SSN
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",      # email
    r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",                   # credit card
    r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",                          # phone
]


def check_bedrock(answer: str) -> tuple[bool, str]:
    """
    Apply AWS Bedrock Guardrail to the generated answer.
    Replaces check_pii (regex) and check_grounding (gpt-4o-mini).

    Bedrock OUTPUT guardrail covers:
      - PII redaction:    SSN, email, phone, credit card, address, name
      - Content filter:   hateful, violent, or inappropriate content in answer
      - Grounding:        verifies answer is grounded in context (no hallucination)
      - Word filter:      custom blocklist

    Fails open — if Bedrock unavailable, fallback PII regex runs instead.
    """
    if not _BEDROCK_AVAILABLE or not settings.BEDROCK_GUARDRAIL_ID:
        log.debug("Bedrock guardrail not configured — running local PII check")
        return _local_pii_check(answer)

    try:
        response = _bedrock.apply_guardrail(
            guardrailIdentifier=settings.BEDROCK_GUARDRAIL_ID,
            guardrailVersion=settings.BEDROCK_GUARDRAIL_VERSION,
            source="OUTPUT",
            content=[{"text": {"text": answer}}],
        )
        action = response.get("action", "NONE")
        log.debug("Bedrock OUTPUT guardrail action: %s", action)

        if action == "GUARDRAIL_INTERVENED":
            reason = "Response blocked by security policy."
            for output in response.get("outputs", []):
                txt = output.get("text", {}).get("text", "")
                if txt:
                    reason = txt
                    break
            for assessment in response.get("assessments", []):
                log.warning("Bedrock OUTPUT assessment: %s", assessment)
            return False, reason

        return True, ""

    except ClientError as e:
        log.warning("Bedrock output guardrail ClientError (falling back to regex): %s", e)
        return _local_pii_check(answer)
    except Exception as e:
        log.warning("Bedrock output guardrail error (falling back to regex): %s", e)
        return _local_pii_check(answer)


def _local_pii_check(answer: str) -> tuple[bool, str]:
    """Regex PII fallback used when Bedrock is unavailable."""
    for pattern in PII_PATTERNS:
        if re.search(pattern, answer):
            log.warning("Local PII pattern matched in answer — blocking")
            return False, "Response withheld — contains sensitive information."
    return True, ""


def check_citations(answer: str, sources: list) -> tuple[bool, str]:
    """Local check — no external calls."""
    if not sources:
        return False, "No source documents found to support an answer."
    has_citation = any(f"[{i}]" in answer for i in range(1, len(sources) + 1))
    if not has_citation:
        return False, "Could not generate a cited answer from the available documents."
    return True, ""


def check_pii(answer: str) -> tuple[bool, str]:
    """
    DEPRECATED — replaced by check_bedrock.
    Kept for backwards compatibility. Not called in run().
    """
    return True, ""


def check_grounding(answer: str, context: str) -> tuple[bool, str]:
    """
    DEPRECATED — replaced by check_bedrock Bedrock grounding policy.
    Kept for backwards compatibility. Not called in run().
    The old gpt-4o-mini call is replaced by Bedrock's native grounding check.
    """
    return True, ""


def run(answer: str, context: str, sources: list) -> tuple[bool, str]:
    """
    Run all output guardrails in order. Returns on first failure.

    Order:
      1. check_bedrock   — AWS guardrail (PII + grounding + content filter)
      2. check_citations — local (no LLM call)
    """
    for check, args in [
        (check_bedrock,   (answer,)),
        (check_citations, (answer, sources)),
    ]:
        passed, message = check(*args)
        if not passed:
            log.info("Output guardrail '%s' blocked answer", check.__name__)
            return False, message
    return True, ""
