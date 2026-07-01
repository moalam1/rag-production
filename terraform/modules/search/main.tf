###############################################################################
# modules/search/main.tf
#
# Search API. Sections:
#   1. Placeholder zip (chicken-and-egg: Lambda needs a zip to exist in S3)
#   2. Lambda + its IAM role (DynamoDB, S3, Secrets, Bedrock)
#   3. Provisioned concurrency (alias)
#   4. API Gateway REST + Lambda integration + CORS
#   5. Usage plan + API key (the REAL rate limiter)
#   6. AWS WAF (port RAG-WAF-RULE) on the stage
###############################################################################

data "aws_caller_identity" "current" {}

locals {
  prefix     = "rag-${var.env}"
  account_id = data.aws_caller_identity.current.account_id
  common_tags = merge({
    Project = "rag"
    Env     = var.env
    Module  = "search"
  }, var.tags)
}

###############################################################################
# 1. PLACEHOLDER ZIP
# Lambda's S3 object must exist before the function can be created. We upload a
# tiny placeholder so `terraform apply` succeeds; redeploy_lambda.sh then
# overwrites this key with the real code. We IGNORE future changes to the
# object so Terraform doesn't fight the deploy script.
###############################################################################

data "archive_file" "placeholder" {
  type        = "zip"
  output_path = "${path.module}/placeholder.zip"
  source {
    content  = "def handler(event, context):\n    return {'statusCode': 200, 'body': 'placeholder - deploy real code via redeploy_lambda.sh'}\n"
    filename = "lambda_handler.py"
  }
}

resource "aws_s3_object" "lambda_zip" {
  bucket = var.artifacts_bucket
  key    = var.lambda_s3_key
  source = data.archive_file.placeholder.output_path
  etag   = data.archive_file.placeholder.output_md5

  lifecycle {
    # The redeploy script overwrites this object with real code — don't revert it.
    ignore_changes = [etag, source, source_hash]
  }
}

###############################################################################
# 2. LAMBDA + IAM ROLE
###############################################################################

# --- execution role trust ---
data "aws_iam_policy_document" "lambda_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "lambda" {
  name               = "${local.prefix}-search-exec-role"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
  tags               = local.common_tags
}

# --- basic Lambda logging (managed policy) ---
resource "aws_iam_role_policy_attachment" "lambda_logs" {
  role       = aws_iam_role.lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# --- the scoped runtime permissions ---
data "aws_iam_policy_document" "lambda_permissions" {

  # DynamoDB: cache (read/write), config (read prompts), feedback (write).
  statement {
    sid    = "DynamoDB"
    effect = "Allow"
    actions = [
      "dynamodb:GetItem",
      "dynamodb:PutItem",
      "dynamodb:UpdateItem",
      "dynamodb:Query",
      "dynamodb:BatchGetItem",
      "dynamodb:BatchWriteItem",
    ]
    resources = [
      var.table_arns["cache"],
      var.table_arns["config"],
      var.table_arns["feedback"],
      var.table_arns["episodic"],
      "${var.table_arns["config"]}/index/*",
      "${var.table_arns["episodic"]}/index/*",
    ]
  }

  # S3: read the bm25 index + the lambda zip from artifacts.
  statement {
    sid    = "S3Read"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:ListBucket",
    ]
    resources = [
      var.artifacts_bucket_arn,
      "${var.artifacts_bucket_arn}/*",
    ]
  }

  # Secrets: this env's vendor keys.
  statement {
    sid       = "Secrets"
    effect    = "Allow"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [var.secret_arn_prefix]
  }

  # Bedrock: rerank model invocation + guardrail application.
  statement {
    sid    = "Bedrock"
    effect = "Allow"
    actions = [
      "bedrock:InvokeModel",      # Cohere Rerank (the Phase-D target for rerank)
      "bedrock:ApplyGuardrail",   # Equinix-POC guardrail
    ]
    resources = ["*"] # model + guardrail ARNs vary; scope later if desired
  }
}

resource "aws_iam_role_policy" "lambda" {
  name   = "${local.prefix}-search-policy"
  role   = aws_iam_role.lambda.id
  policy = data.aws_iam_policy_document.lambda_permissions.json
}

# --- log group (explicit, so retention is managed) ---
resource "aws_cloudwatch_log_group" "lambda" {
  name              = "/aws/lambda/${local.prefix}-search"
  retention_in_days = var.log_retention_days
  tags              = local.common_tags
}

# --- the function ---
resource "aws_lambda_function" "search" {
  function_name = "${local.prefix}-search"
  role          = aws_iam_role.lambda.arn
  handler       = var.lambda_handler
  runtime       = var.lambda_runtime
  memory_size   = var.lambda_memory
  timeout       = var.lambda_timeout

  s3_bucket = var.artifacts_bucket
  s3_key    = var.lambda_s3_key

  environment {
    variables = {
      ENVIRONMENT    = var.env
      AWS_REGION_APP = var.region
      CACHE_BACKEND  = "dynamodb"
      BM25_S3_BUCKET = var.artifacts_bucket
      PINECONE_INDEX = "rag-poc"
      CORS_ORIGINS   = join(",", var.cors_allowed_origins)
    }
  }

  depends_on = [
    aws_s3_object.lambda_zip,
    aws_cloudwatch_log_group.lambda,
  ]

  lifecycle {
    # The redeploy script updates the code; don't let TF revert it on the next plan.
    ignore_changes = [s3_key, source_code_hash]
  }

  tags = local.common_tags
}

# --- alias + provisioned concurrency (warm instances; kills cold start) ---
resource "aws_lambda_alias" "live" {
  name             = "live"
  function_name    = aws_lambda_function.search.function_name
  function_version = aws_lambda_function.search.version
}

resource "aws_lambda_provisioned_concurrency_config" "live" {
  count                             = var.provisioned_concurrency > 0 ? 1 : 0
  function_name                     = aws_lambda_function.search.function_name
  qualifier                         = aws_lambda_alias.live.name
  provisioned_concurrent_executions = var.provisioned_concurrency
}
