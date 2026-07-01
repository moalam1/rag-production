###############################################################################
# modules/foundation/main.tf
#
# DynamoDB tables, S3 buckets, Secrets Manager entries for one environment.
# Table/key shapes mirror the VALIDATED prototype (rag-config, rag-document-
# registry, rag-cache w/ TTL, rag-episodic, rag-feedback; rag-artifacts S3 with
# bm25/ + lambda/ prefixes).
#
# All resources are prefixed rag-{env}-* for UAT/prod isolation in the
# standalone account.
###############################################################################

data "aws_caller_identity" "current" {}

locals {
  prefix     = "rag-${var.env}"
  account_id = data.aws_caller_identity.current.account_id

  artifacts_bucket = var.artifacts_bucket_name != "" ? var.artifacts_bucket_name : "${local.prefix}-artifacts-${local.account_id}"

  common_tags = merge({
    Project = "rag"
    Env     = var.env
    Module  = "foundation"
  }, var.tags)
}

###############################################################################
# DYNAMODB TABLES
#
# All PAY_PER_REQUEST (on-demand) — matches the prototype, no capacity planning,
# scales to zero cost when idle. Key schemas mirror the prototype's access
# patterns (single-key get/put/query; registry needs Scan for dedup).
###############################################################################

# --- rag-{env}-cache : L1 answer cache (Lambda backend). NEEDS TTL. ---
resource "aws_dynamodb_table" "cache" {
  name         = "${local.prefix}-cache"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "cache_key"

  attribute {
    name = "cache_key"
    type = "S"
  }

  # The TTL field the DynamoDBCache class writes ('expires_at'). Without this,
  # cached answers live forever → stale-forever bug. cache_ttl_enabled MUST be true.
  ttl {
    attribute_name = "expires_at"
    enabled        = var.cache_ttl_enabled
  }

  tags = merge(local.common_tags, { Table = "cache", Layer = "L1" })
}

# --- rag-{env}-config : app config + prompt registry (prompt#* items) ---
resource "aws_dynamodb_table" "config" {
  name         = "${local.prefix}-config"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "config_key"

  attribute {
    name = "config_key"
    type = "S"
  }

  point_in_time_recovery {
    enabled = var.point_in_time_recovery
  }

  tags = merge(local.common_tags, { Table = "config" })
}

# --- rag-{env}-document-registry : ingestion dedup. NEEDS Scan (task role). ---
resource "aws_dynamodb_table" "document_registry" {
  name         = "${local.prefix}-document-registry"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "page_url"

  attribute {
    name = "page_url"
    type = "S"
  }

  point_in_time_recovery {
    enabled = var.point_in_time_recovery
  }

  tags = merge(local.common_tags, { Table = "document-registry" })
}

# --- rag-{env}-episodic : session/visitor memory (Phase 2, flag-gated) ---
resource "aws_dynamodb_table" "episodic" {
  name         = "${local.prefix}-episodic"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "visitor_id"
  range_key    = "timestamp"

  attribute {
    name = "visitor_id"
    type = "S"
  }
  attribute {
    name = "timestamp"
    type = "S"
  }

  # Episodic memory is privacy-sensitive (Phase 2 / GDPR) — TTL to auto-expire.
  ttl {
    attribute_name = "expires_at"
    enabled        = true
  }

  tags = merge(local.common_tags, { Table = "episodic", Phase = "2" })
}

# --- rag-{env}-feedback : thumbs up/down + query logging (Phase 1) ---
resource "aws_dynamodb_table" "feedback" {
  name         = "${local.prefix}-feedback"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "feedback_id"

  attribute {
    name = "feedback_id"
    type = "S"
  }

  tags = merge(local.common_tags, { Table = "feedback" })
}

###############################################################################
# S3 — artifacts bucket (lambda zip + bm25 index)
###############################################################################

resource "aws_s3_bucket" "artifacts" {
  bucket = local.artifacts_bucket
  tags   = merge(local.common_tags, { Bucket = "artifacts" })
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration {
    # Versioning lets redeploy_lambda.sh keep search-lambda-PREV.zip rollbacks.
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Lifecycle: expire old noncurrent lambda zips so versioning doesn't accumulate.
resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    id     = "expire-old-lambda-zips"
    status = "Enabled"
    filter {
      prefix = "lambda/"
    }
    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }
}

###############################################################################
# SECRETS MANAGER — rag/{env}/{vendor-key}
#
# Creates the secret CONTAINERS. The secret VALUES are set out-of-band (CLI or
# console) — Terraform should NOT hold plaintext secrets in state/code.
# Convention (decided): rag/{env}/openai-api-key, /pinecone-api-key,
# /bedrock-guardrail-id, /app-api-key.
###############################################################################

locals {
  secret_names = [
    "openai-api-key",
    "pinecone-api-key",
    "cohere-api-key",
    "bedrock-guardrail-id",
    "app-api-key", # the X-API-Key for AEM/admin auth
  ]
}

resource "aws_secretsmanager_secret" "app" {
  for_each = toset(local.secret_names)

  name        = "rag/${var.env}/${each.value}"
  description = "RAG ${var.env} — ${each.value}. Value set out-of-band, NOT via Terraform."
  tags        = merge(local.common_tags, { Secret = each.value })

  # Don't accidentally delete secrets on destroy in prod — recovery window.
  recovery_window_in_days = var.env == "prod" ? 30 : 7
}

# NOTE: deliberately NO aws_secretsmanager_secret_version here. Setting the
# value in Terraform would put the plaintext in state. Set it manually:
#   aws secretsmanager put-secret-value \
#     --secret-id rag/uat/openai-api-key \
#     --secret-string 'sk-...' --region us-east-1
