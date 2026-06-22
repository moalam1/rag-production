"""
api/routes/visitor.py — Visitor identity, suggestions, and history endpoints.

Extracted from api/search.py (Tier 1d). Three endpoints: identity capture
(POST /visitor/identify), personalised query suggestions, and query history.
The two suggestion helpers (_default_suggestions, _generate_suggestions) are a
self-contained pair and travel with the suggestions endpoint. Auth + log from
api.deps; IdentifyRequest from api.models. Endpoint bodies are verbatim moves.
"""
from fastapi import APIRouter, Depends, HTTPException

from api.deps import verify_api_key, log
from api.models import IdentifyRequest

router = APIRouter(prefix="/api/v1", tags=["visitor"])


@router.post("/visitor/identify")
async def identify_visitor(req: IdentifyRequest, _: str = Depends(verify_api_key)):
    try:
        import boto3 as _b3
        from datetime import datetime, timezone

        _ddb   = _b3.resource("dynamodb", region_name="us-east-1")
        _table = _ddb.Table("rag-episodic")

        _table.put_item(Item={
            "visitor_id":       req.visitor_id,
            "timestamp":        datetime.now(timezone.utc).isoformat(),
            "intent":           "identity_capture",
            "query":            f"IDENTITY — {req.email}",
            "email":            req.email,
            "name":             req.name,
            "source":           req.source,
            "products":         req.products,
            "lead_quality_tag": "SOLID_LEAD_COMMERCIAL",
            "company":          req.company or "",
            "country":          req.country or "",
        })

        log.info("Identity captured: visitor=%s email=%s", req.visitor_id, req.email)
        return {"status": "ok", "visitor_id": req.visitor_id, "email": req.email}

    except Exception as e:
        log.error("identify_visitor error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/visitor/{visitor_id}/suggestions")
async def visitor_suggestions(visitor_id: str, _: str = Depends(verify_api_key)):
    """Return personalised query suggestions based on visitor history."""
    try:
        from pipeline.episodic_memory import get_visitor_context
        context = get_visitor_context(visitor_id)
        suggestions = _generate_suggestions(context)
        return {
            "suggestions":   suggestions,
            "personalised":  len(context.get("interests", [])) > 0,
            "stage":         context.get("stage", "awareness"),
            "query_count":   context.get("query_count", 0),
        }
    except Exception as e:
        log.warning(f"Suggestions error: {e}")
        return {"suggestions": _default_suggestions(), "personalised": False}


def _default_suggestions() -> list[dict]:
    return [
        {"icon": "💡", "text": "What is Equinix Fabric and how does it work?",       "intent": "learn_concept"},
        {"icon": "📊", "text": "What port speeds and SLAs does Equinix Fabric support?", "intent": "evaluate_specs"},
        {"icon": "⚖️",  "text": "Equinix Fabric vs Network Edge for SD-WAN deployment", "intent": "compare"},
        {"icon": "📄", "text": "Show me a hybrid multicloud networking blueprint",    "intent": "find_resource"},
    ]


def _generate_suggestions(context: dict) -> list[dict]:
    intents     = context.get("intents", [])
    products    = context.get("interests", [])
    stage       = context.get("stage", "awareness")
    last_intent = context.get("last_intent", "")
    last_query  = context.get("last_query", "")

    # New visitor — return defaults
    if not intents or not products:
        return _default_suggestions()

    suggestions = []
    used_intents = set()
    p0 = products[0] if products else "Equinix Fabric"
    # Shorten long product names for card display
    p0_short = p0.replace("Equinix Fabric Cloud Router", "Fabric Cloud Router")

    # ── Rule 1: Continue the thread ───────────────────────────────────────────
    if last_intent == "learn_concept" and "evaluate_specs" not in intents:
        suggestions.append({
            "icon": "📊",
            "text": f"What are the technical specifications for {p0_short}?",
            "intent": "evaluate_specs",
            "rule": "continue"
        })
        used_intents.add("evaluate_specs")

    elif last_intent == "evaluate_specs" and "compare" not in intents:
        suggestions.append({
            "icon": "⚖️",
            "text": f"How does {p0_short} compare to Network Edge?",
            "intent": "compare",
            "rule": "continue"
        })
        used_intents.add("compare")

    elif last_intent == "compare" and "find_resource" not in intents:
        suggestions.append({
            "icon": "📄",
            "text": f"Show me a {p0_short} deployment blueprint",
            "intent": "find_resource",
            "rule": "continue"
        })
        used_intents.add("find_resource")

    elif last_intent == "find_resource" and stage in ("evaluation", "intent"):
        suggestions.append({
            "icon": "🎯",
            "text": f"What does an enterprise {p0_short} rollout look like?",
            "intent": "find_resource",
            "rule": "commercial"
        })
        used_intents.add("find_resource")

    # ── Rule 2: Fill the product gap ──────────────────────────────────────────
    ALL_CORE = ["Equinix Fabric", "Network Edge", "Equinix Metal",
                "Equinix Fabric Cloud Router", "Internet Access"]
    unseen = [p for p in ALL_CORE if p not in products]
    if unseen and len(suggestions) < 3 and "compare" not in used_intents:
        p1_short = unseen[0].replace("Equinix Fabric Cloud Router", "Fabric Cloud Router")
        suggestions.append({
            "icon": "⚖️",
            "text": f"{p0_short} vs {p1_short} — what's the difference?",
            "intent": "compare",
            "rule": "gap"
        })
        used_intents.add("compare")

    # ── Rule 3: Stage advancement ─────────────────────────────────────────────
    if stage == "consideration" and len(suggestions) < 3:
        if "evaluate_specs" not in used_intents:
            suggestions.append({
                "icon": "📊",
                "text": f"What SLAs and performance guarantees does {p0_short} offer?",
                "intent": "evaluate_specs",
                "rule": "stage"
            })
            used_intents.add("evaluate_specs")

    elif stage in ("evaluation", "intent") and len(suggestions) < 3:
        if "find_resource" not in used_intents:
            suggestions.append({
                "icon": "🏗️",
                "text": f"Show me a {p0_short} enterprise deployment case study",
                "intent": "find_resource",
                "rule": "stage"
            })
            used_intents.add("find_resource")

    # ── Rule 4: Always one generic broadening card ────────────────────────────
    # Pick a topic the visitor hasn't explored
    broadening = [
        {"icon": "🏢", "text": "What is colocation and why use Equinix IBX?",     "intent": "learn_concept"},
        {"icon": "☁️",  "text": "How does Equinix Fabric connect to AWS and Azure?", "intent": "learn_concept"},
        {"icon": "🔀", "text": "What are Equinix Fabric Cloud Router capabilities?", "intent": "evaluate_specs"},
        {"icon": "🏦", "text": "Show me a financial services case study for Equinix", "intent": "find_resource"},
        {"icon": "💡", "text": "What is Equinix Metal bare-metal infrastructure?",  "intent": "learn_concept"},
    ]
    # Pick one not already covered by visitor history
    for b in broadening:
        text_lower = b["text"].lower()
        already_seen = any(
            p.lower().replace("equinix ", "") in text_lower
            for p in products
        )
        if not already_seen or len(suggestions) >= 3:
            suggestions.append({**b, "rule": "broadening"})
            break

    # Pad to 4 with defaults if needed
    defaults = _default_suggestions()
    for d in defaults:
        if len(suggestions) >= 4:
            break
        if d["intent"] not in used_intents:
            suggestions.append(d)

    return suggestions[:4]


@router.get("/visitor/{visitor_id}/history")
async def visitor_history(visitor_id: str, _: str = Depends(verify_api_key)):
    """Return visitor query history from DynamoDB episodic memory."""
    import sys
    _p = "/home/ssm-user/rag-production/pipeline"
    if _p not in sys.path: sys.path.insert(0, _p)
    try:
        from pipeline.episodic_memory import get_visitor_stats
        stats = get_visitor_stats(visitor_id)
        return stats
    except Exception as e:
        import logging
        logging.getLogger("api.search").warning(f"Visitor history endpoint error: {e}")
        return {"queries": [], "total": 0}
