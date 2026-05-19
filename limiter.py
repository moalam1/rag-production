from slowapi import Limiter
from starlette.requests import Request

def client_identifier(request: Request) -> str:
    api_key = request.headers.get("X-API-Key", "").strip()
    return f"key:{api_key}" if api_key else request.client.host

limiter = Limiter(key_func=client_identifier, default_limits=["20/minute", "100/day"])
