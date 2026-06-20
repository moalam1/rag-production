"""
api/routes/admin.py — Admin console endpoints.

Extracted from api/search.py (Tier 1c). The /admin/* cluster: config editor
(GET/PUT) and prompt registry (GET/PUT, single + multi). Auth + config-cache
busting come from api.deps. The two module constants this cluster owns
(_ADMIN_CONFIG_KEYS, _PROMPT_META) travel with it. Endpoint bodies are verbatim.

Note: the PUT endpoints write to the rag-config DynamoDB table and call
invalidate_config() so the in-process cache reloads on the next request.
"""
from fastapi import APIRouter, Depends, HTTPException

from api.deps import verify_api_key, invalidate_config, log

router = APIRouter(prefix="/api/v1", tags=["admin"])

_ADMIN_CONFIG_KEYS = ["workload_signals","product_signals","commercial_keywords","workload_badge_styles","equinix_products","equinix_use_cases","competitor_signals"]


@router.get("/admin/config")
async def admin_get_config(_: str = Depends(verify_api_key)):
    """All editable rag-config keys for the admin console."""
    import boto3 as _b3
    _t = _b3.resource("dynamodb", region_name="us-east-1").Table("rag-config")
    out = {}
    for k in _ADMIN_CONFIG_KEYS:
        resp = _t.get_item(Key={"config_key": k})
        if "Item" in resp:
            out[k] = resp["Item"].get("data", {})
    return {"config": out}


@router.put("/admin/config/{config_key}")
async def admin_put_config(config_key: str, body: dict,
                           _: str = Depends(verify_api_key)):
    """Update one rag-config key from the admin console. 5-min TTL applies."""
    if config_key not in _ADMIN_CONFIG_KEYS:
        raise HTTPException(400, f"Unknown config key: {config_key}")
    data = body.get("data")
    if data is None:
        raise HTTPException(400, "Missing 'data' in body")
    import boto3 as _b3, datetime as _dt
    _t = _b3.resource("dynamodb", region_name="us-east-1").Table("rag-config")
    _t.update_item(
        Key={"config_key": config_key},
        UpdateExpression="SET #d = :d, updated_at = :u",
        ExpressionAttributeNames={"#d": "data"},
        ExpressionAttributeValues={
            ":d": data,
            ":u": _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
        },
    )
    # Bust the in-process config cache so it reloads next request
    invalidate_config()
    log.info("admin: rag-config %s updated (%s items)", config_key,
             len(data) if isinstance(data, (list, dict)) else 1)
    return {"ok": True, "config_key": config_key}


@router.get("/admin/prompt")
async def admin_get_prompt(_: str = Depends(verify_api_key)):
    """Current system prompt + version. Reads rag-config override or code default."""
    import boto3 as _b3
    _t = _b3.resource("dynamodb", region_name="us-east-1").Table("rag-config")
    resp = _t.get_item(Key={"config_key": "system_prompt"})
    from pipeline.generator import SYSTEM_PROMPT as _code_prompt
    from pipeline.prompt_registry import get_prompt_version
    _pv = get_prompt_version('generation', 2)
    if "Item" in resp:
        item = resp["Item"]
        return {"prompt": item.get("data", _code_prompt),
                "prompt_version": int(item.get("prompt_version", _pv)),
                "source": "rag-config"}
    return {"prompt": _code_prompt, "prompt_version": _pv, "source": "code"}


@router.put("/admin/prompt")
async def admin_put_prompt(body: dict, _: str = Depends(verify_api_key)):
    """Save prompt to rag-config, bump version, clear answer caches."""
    prompt = (body.get("prompt") or "").strip()
    if len(prompt) < 50:
        raise HTTPException(400, "Prompt too short — refusing to save")
    import boto3 as _b3, datetime as _dt
    _t = _b3.resource("dynamodb", region_name="us-east-1").Table("rag-config")
    resp = _t.get_item(Key={"config_key": "system_prompt"})
    from pipeline.prompt_registry import get_prompt_version
    _pv_code = get_prompt_version('generation', 2)
    current_pv = int(resp["Item"].get("prompt_version", _pv_code)) if "Item" in resp else _pv_code
    new_pv = current_pv + 1
    _t.put_item(Item={
        "config_key": "system_prompt",
        "data": prompt,
        "prompt_version": new_pv,
        "updated_at": _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
    })
    log.warning("admin: system prompt updated → pv=%s. NOTE: generator must read "
                "rag-config prompt for this to take effect (see deploy notes).", new_pv)
    return {"ok": True, "prompt_version": new_pv,
            "note": "Restart rag-api + clear semantic cache to fully apply"}


_PROMPT_META = {
    "generation": {"model": "gpt-4o",      "label": "Answer generation",
                   "note": "Saving bumps version → semantic + memory caches auto-invalidate"},
    "intent":     {"model": "gpt-4o-mini", "label": "Intent detection",
                   "note": "Keep {products} and {use_cases} placeholders intact"},
    "profiles":   {"model": "gpt-4o-mini", "label": "Nightly buyer briefs",
                   "note": "Applies on next consolidation run"},
}


@router.get("/admin/prompts")
async def admin_list_prompts(_: str = Depends(verify_api_key)):
    import boto3 as _b3
    t = _b3.resource("dynamodb", region_name="us-east-1").Table("rag-config")
    out = []
    for pid, meta in _PROMPT_META.items():
        r = t.get_item(Key={"config_key": f"prompt#{pid}"})
        item = r.get("Item", {})
        out.append({"id": pid, **meta,
                    "version": int(item.get("prompt_version", 0)),
                    "chars": len(item.get("data", "")),
                    "source": "registry" if item else "code-fallback",
                    "updated_at": item.get("updated_at", "")})
    return {"prompts": out}


@router.get("/admin/prompts/{pid}")
async def admin_get_prompt_v2(pid: str, _: str = Depends(verify_api_key)):
    if pid not in _PROMPT_META:
        raise HTTPException(404, f"Unknown prompt id: {pid}")
    import boto3 as _b3
    t = _b3.resource("dynamodb", region_name="us-east-1").Table("rag-config")
    r = t.get_item(Key={"config_key": f"prompt#{pid}"})
    if "Item" in r:
        return {"id": pid, "prompt": r["Item"].get("data", ""),
                "version": int(r["Item"].get("prompt_version", 1)), "source": "registry"}
    return {"id": pid, "prompt": "", "version": 0, "source": "code-fallback",
            "note": "Not yet in registry — save once to take control"}


@router.put("/admin/prompts/{pid}")
async def admin_put_prompt_v2(pid: str, body: dict, _: str = Depends(verify_api_key)):
    if pid not in _PROMPT_META:
        raise HTTPException(404, f"Unknown prompt id: {pid}")
    prompt = (body.get("prompt") or "").strip()
    if len(prompt) < 50:
        raise HTTPException(400, "Prompt too short — refusing to save")
    if pid == "intent" and ("{products}" not in prompt or "{use_cases}" not in prompt):
        raise HTTPException(400, "Intent prompt must keep {products} and {use_cases} placeholders")
    import boto3 as _b3, datetime as _dt
    t = _b3.resource("dynamodb", region_name="us-east-1").Table("rag-config")
    r = t.get_item(Key={"config_key": f"prompt#{pid}"})
    new_v = (int(r["Item"].get("prompt_version", 0)) if "Item" in r else 0) + 1
    t.put_item(Item={"config_key": f"prompt#{pid}", "data": prompt,
                     "prompt_version": new_v,
                     "updated_at": _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
                     "description": _PROMPT_META[pid]["label"]})
    from pipeline.prompt_registry import bust as _bust; _bust()
    invalidate_config()
    log.warning("admin: prompt#%s saved → v%s (%s chars)", pid, new_v, len(prompt))
    return {"ok": True, "id": pid, "version": new_v, "note": _PROMPT_META[pid]["note"]}
    
