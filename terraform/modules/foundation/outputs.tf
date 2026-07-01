###############################################################################
# modules/foundation/outputs.tf
#
# These feed the ingestion + search modules (table names/ARNs for IAM scoping,
# bucket name, secret ARNs for the execution roles).
###############################################################################

# --- DynamoDB ---
output "table_names" {
  description = "Map of logical name → DynamoDB table name."
  value = {
    cache             = aws_dynamodb_table.cache.name
    config            = aws_dynamodb_table.config.name
    document_registry = aws_dynamodb_table.document_registry.name
    episodic          = aws_dynamodb_table.episodic.name
    feedback          = aws_dynamodb_table.feedback.name
  }
}

output "table_arns" {
  description = "Map of logical name → DynamoDB table ARN (for IAM policy scoping)."
  value = {
    cache             = aws_dynamodb_table.cache.arn
    config            = aws_dynamodb_table.config.arn
    document_registry = aws_dynamodb_table.document_registry.arn
    episodic          = aws_dynamodb_table.episodic.arn
    feedback          = aws_dynamodb_table.feedback.arn
  }
}

# --- S3 ---
output "artifacts_bucket" {
  description = "Artifacts bucket name (lambda zip + bm25 index)."
  value       = aws_s3_bucket.artifacts.id
}

output "artifacts_bucket_arn" {
  description = "Artifacts bucket ARN (for IAM scoping)."
  value       = aws_s3_bucket.artifacts.arn
}

# --- Secrets ---
output "secret_arns" {
  description = "Map of secret short-name → ARN (for execution-role GetSecretValue)."
  value       = { for k, s in aws_secretsmanager_secret.app : k => s.arn }
}

output "secret_arn_prefix" {
  description = "ARN prefix for scoping secretsmanager:GetSecretValue to rag/{env}/*."
  value       = "arn:aws:secretsmanager:${var.region}:${local.account_id}:secret:rag/${var.env}/*"
}

output "prefix" {
  description = "The rag-{env} resource prefix."
  value       = local.prefix
}
