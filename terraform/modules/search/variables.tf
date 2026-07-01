###############################################################################
# modules/search/variables.tf
#
# The synchronous search API: Lambda (code from S3) + provisioned concurrency,
# API Gateway (REST) + usage plans (the REAL rate-limiter), AWS WAF, Bedrock
# permissions, scoped CORS. Encodes the audit findings:
#   - rate limiting → API Gateway usage plans (not the in-memory app limiter)
#   - WAF → on API Gateway (port RAG-WAF-RULE, drop PHP, add rate-based)
#   - CORS → scoped origins (not wildcard)
###############################################################################

variable "env" {
  type = string
  validation {
    condition     = contains(["dev","uat", "prod"], var.env)
    error_message = "env must be 'uat' or 'prod'."
  }
}

variable "region" {
  type    = string
  default = "us-east-1"
}

# --- Foundation wiring ---
variable "table_arns" {
  description = "DynamoDB table ARNs from foundation."
  type        = map(string)
}

variable "artifacts_bucket" {
  description = "Artifacts bucket NAME (holds the lambda zip + bm25 index)."
  type        = string
}

variable "artifacts_bucket_arn" {
  description = "Artifacts bucket ARN (for IAM scoping)."
  type        = string
}

variable "secret_arn_prefix" {
  description = "Secrets ARN prefix (rag/{env}/*)."
  type        = string
}

# --- Lambda code (from S3) ---
variable "lambda_s3_key" {
  description = "S3 key of the search lambda zip in the artifacts bucket."
  type        = string
  default     = "lambda/search-lambda.zip"
}

variable "lambda_handler" {
  description = "Lambda handler. The prototype requires lambda_handler.handler."
  type        = string
  default     = "lambda_handler.handler"
}

variable "lambda_runtime" {
  type    = string
  default = "python3.11"
}

variable "lambda_memory" {
  description = "Lambda memory MB (CPU scales with it). Prototype validated 1024."
  type        = number
  default     = 1024
}

variable "lambda_timeout" {
  description = "Lambda timeout seconds. Search should answer well under this; generous for safety."
  type        = number
  default     = 30
}

variable "provisioned_concurrency" {
  description = "Number of warm Lambda instances (kills the 4.9s cold start on live traffic). 0 = off."
  type        = number
  default     = 0 # UAT default off (cost); turn on for prod / load tests
}

# --- API Gateway usage plan (THE rate limiter — audit finding) ---
variable "rate_limit_per_second" {
  description = "API Gateway usage plan steady-state rate (req/sec)."
  type        = number
  default     = 50
}

variable "rate_limit_burst" {
  description = "API Gateway usage plan burst capacity."
  type        = number
  default     = 100
}

variable "quota_per_day" {
  description = "API Gateway usage plan daily request quota."
  type        = number
  default     = 50000
}

# --- WAF (audit finding: port RAG-WAF-RULE to the Gateway) ---
variable "waf_enabled" {
  description = "Attach an AWS WAF web ACL to the API Gateway stage."
  type        = bool
  default     = true
}

variable "waf_rate_limit" {
  description = "WAF rate-based rule: max requests per 5 min per IP before blocking."
  type        = number
  default     = 2000
}

# --- CORS (audit finding: scope, don't wildcard) ---
variable "cors_allowed_origins" {
  description = "Allowed CORS origins. PROD must be the real equinix.com + AEM origins, NOT '*'."
  type        = list(string)
  default     = ["*"] # UAT convenience; OVERRIDE in prod tfvars
}

# --- Bedrock ---
variable "bedrock_guardrail_arn" {
  description = "Optional Bedrock guardrail ARN to scope ApplyGuardrail. Empty = allow account guardrails."
  type        = string
  default     = ""
}

variable "log_retention_days" {
  type    = number
  default = 30
}

variable "tags" {
  type    = map(string)
  default = {}
}
