"""
guardrails/input.py — Input safety checks run before the RAG pipeline.

Check order:
  1. check_length     — local regex, free, instant
  2. check_injection  — local regex, free, instant
  3. check_bedrock    — AWS Bedrock Guardrail (replaces gpt-4o-mini relevance check)
                        covers: topic denial, prompt injection, PII in query, content filter
                        CloudTrail audit log on every call

Bedrock Guardrail replaces the old gpt-4o-mini check_relevance:
  - No data sent to OpenAI for safety checks
  - $0.75 per 1000 units vs ~$0.001 per gpt-4o-mini call
  - Instant — no LLM generation latency
  - AWS CloudTrail logs every guardrail decision
  - Fails open on error — never blocks users on AWS outage
"""
import re
import logging
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from openai import OpenAI

from config import settings

log = logging.getLogger(__name__)

# ── OpenAI client (fallback only) ────────────────────────────────
_openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

# ── AWS Bedrock client ────────────────────────────────────────────
# Auth via EC2 IAM role — no API key needed
# IAM policy required: bedrock:ApplyGuardrail
try:
    _bedrock = boto3.client("bedrock-runtime", region_name=settings.AWS_REGION)
    _BEDROCK_AVAILABLE = True
    log.info("Bedrock runtime client initialised (region: %s)", settings.AWS_REGION)
except (NoCredentialsError, Exception) as e:
    _bedrock = None
    _BEDROCK_AVAILABLE = False
    log.warning("Bedrock unavailable: %s — falling back to open policy", e)

# ── Injection patterns (local regex — always runs) ────────────────
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
    """Local check — no external calls."""
    if len(query.strip()) < settings.MIN_QUERY_LENGTH:
        return False, "Query too short. Please ask a complete question."
    if len(query) > settings.MAX_QUERY_LENGTH:
        return False, f"Query too long. Please keep it under {settings.MAX_QUERY_LENGTH} characters."
    return True, ""


def check_injection(query: str) -> tuple[bool, str]:
    """Local regex check — no external calls."""
    q = query.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, q):
            log.warning("Injection attempt detected: %s", query[:80])
            return False, "Invalid query detected. Please ask a genuine question."
    return True, ""


def check_bedrock(query: str) -> tuple[bool, str]:
    """
    Apply AWS Bedrock Guardrail to the input query.
    Replaces the old gpt-4o-mini check_relevance function.

    Configure the guardrail in AWS Console → Bedrock → Guardrails with:
      - Topic denial:    block off-topic queries (non enterprise-tech)
      - Prompt attack:   injection detection (complements local regex above)
      - Content filters: hate, violence, insults
      - PII:             detect sensitive data in queries
      - Word filter:     custom blocklist

    Set BEDROCK_GUARDRAIL_ID in .env or Secrets Manager to enable.
    Fails open — if Bedrock unavailable, query is allowed through.
    """
    if not _BEDROCK_AVAILABLE or not settings.BEDROCK_GUARDRAIL_ID:
        log.debug("Bedrock guardrail not configured — skipping input check")
        return True, ""

    try:
        response = _bedrock.apply_guardrail(
            guardrailIdentifier=settings.BEDROCK_GUARDRAIL_ID,
            guardrailVersion=settings.BEDROCK_GUARDRAIL_VERSION,
            source="INPUT",
            content=[{"text": {"text": query}}],
        )
        action = response.get("action", "NONE")
        log.debug("Bedrock INPUT guardrail action: %s", action)

        if action == "GUARDRAIL_INTERVENED":
            reason = "Query blocked by security policy."
            for output in response.get("outputs", []):
                txt = output.get("text", {}).get("text", "")
                if txt:
                    reason = txt
                    break
            # Log assessment details — captured in CloudTrail
            for assessment in response.get("assessments", []):
                log.warning("Bedrock guardrail assessment: %s", assessment)
            return False, reason

        return True, ""

    except ClientError as e:
        log.warning("Bedrock guardrail ClientError (allowing through): %s", e)
        return True, ""
    except Exception as e:
        log.warning("Bedrock guardrail error (allowing through): %s", e)
        return True, ""


def check_relevance(query: str) -> tuple[bool, str]:
    """
    DEPRECATED — replaced by check_bedrock.
    Kept for backwards compatibility. Not called in run().
    """
    return True, ""


def run(query: str) -> tuple[bool, str]:
    """
    Run all input guardrails in order. Returns on first failure.

    Order:
      1. check_length   — cheapest, catches obviously bad inputs first
      2. check_injection — cheap regex before any AWS call
      3. check_bedrock  — AWS guardrail, covers everything else
    """
    for check in [check_length, check_injection, check_bedrock]:
        passed, message = check(query)
        if not passed:
            log.info("Input guardrail '%s' blocked query", check.__name__)
            return False, message
    return True, ""
