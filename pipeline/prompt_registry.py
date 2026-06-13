"""Prompt registry — prompts live in rag-config (prompt#<id>), 5-min TTL cache.
Consumers fall back to their hardcoded literals if the registry is empty/unreachable."""
import time, threading, logging

log = logging.getLogger("prompt_registry")
_cache: dict = {}
_loaded_at = 0.0
_lock = threading.Lock()
TTL = 300
PROMPT_IDS = ("generation", "intent", "profiles")

def _load():
    global _cache, _loaded_at
    now = time.time()
    if _cache and (now - _loaded_at) < TTL:
        return _cache
    with _lock:
        if _cache and (now - _loaded_at) < TTL:
            return _cache
        try:
            import boto3
            t = boto3.resource("dynamodb", region_name="us-east-1").Table("rag-config")
            fresh = {}
            for pid in PROMPT_IDS:
                r = t.get_item(Key={"config_key": f"prompt#{pid}"})
                if "Item" in r:
                    fresh[pid] = {
                        "prompt":  r["Item"].get("data", ""),
                        "version": int(r["Item"].get("prompt_version", 1)),
                    }
            _cache, _loaded_at = fresh, now
            log.info("prompt registry loaded: %s", {k: v["version"] for k, v in fresh.items()})
        except Exception as e:
            log.warning("prompt registry load failed — code fallbacks in use: %s", e)
    return _cache

def get_prompt(pid: str, fallback: str = "") -> str:
    p = _load().get(pid, {}).get("prompt", "")
    return p if p and p.strip() else fallback

def get_prompt_version(pid: str, fallback: int = 1) -> int:
    return _load().get(pid, {}).get("version", fallback)

def bust():
    global _loaded_at
    _loaded_at = 0
