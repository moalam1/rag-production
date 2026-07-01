"""
secrets_loader.py — Load rag/<env>/* secrets from AWS Secrets Manager into
os.environ at Lambda cold start, BEFORE config.py reads them via os.getenv().

Why: the search Lambda has no .env file (unlike EC2/local). config.py reads all
secrets from environment variables, so without this they are empty and clients
(Pinecone, OpenAI, Cohere) fail to initialise.

Safe by design:
- Only runs inside Lambda (guarded on AWS_LAMBDA_FUNCTION_NAME), so it is a
  complete no-op on EC2/local where .env already provides everything.
- Never overrides a variable that is already set.
- Per-secret try/except: a missing/unreadable secret logs and continues rather
  than crashing the whole cold start.
"""
import os
import boto3

# secret name in Secrets Manager  ->  env var name config.py expects
_SECRET_MAP = {
    "openai-api-key":       "OPENAI_API_KEY",
    "pinecone-api-key":     "PINECONE_API_KEY",
    "bedrock-guardrail-id": "BEDROCK_GUARDRAIL_ID",
}


def load_secrets(env: str = None, region: str = None) -> None:
    """Fetch rag/<env>/* into os.environ. No-op outside Lambda."""
    # Only run inside Lambda — on EC2/local the .env file provides everything.
    if not os.getenv("AWS_LAMBDA_FUNCTION_NAME"):
        return

    env    = env    or os.getenv("ENVIRONMENT", "uat")
    region = region or os.getenv("AWS_REGION", "us-east-1")

    client = boto3.client("secretsmanager", region_name=region)

    for secret_suffix, env_var in _SECRET_MAP.items():
        if os.getenv(env_var):           # already set — never override
            continue
        secret_id = f"rag/{env}/{secret_suffix}"
        try:
            resp = client.get_secret_value(SecretId=secret_id)
            value = resp.get("SecretString", "")
            if value:
                os.environ[env_var] = value
        except Exception as e:
            print(f"[secrets_loader] skipped {secret_id}: {e}")
