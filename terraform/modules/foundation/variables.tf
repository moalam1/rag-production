###############################################################################
# modules/foundation/variables.tf
#
# The data layer: DynamoDB, S3, Secrets Manager. Lowest-risk module — no
# compute, no networking. Everything is name-prefixed per environment so UAT
# and prod coexist safely in the standalone account.
###############################################################################

variable "env" {
  description = "Environment name (uat | prod). Drives the rag-{env}-* prefix."
  type        = string
  validation {
    condition     = contains(["uat", "prod"], var.env)
    error_message = "env must be 'uat' or 'prod'."
  }
}

variable "region" {
  description = "Workload region (us-east-1)."
  type        = string
  default     = "us-east-1"
}

variable "artifacts_bucket_name" {
  description = <<-EOT
    S3 bucket for build artifacts (lambda zip + bm25 index). Must be globally
    unique → defaults to rag-{env}-artifacts-{account_id}. Override if needed.
  EOT
  type        = string
  default     = ""
}

variable "cache_ttl_enabled" {
  description = "Enable DynamoDB native TTL on the answer cache table. MUST be true (else cached answers never expire)."
  type        = bool
  default     = true
}

variable "point_in_time_recovery" {
  description = "Enable PITR (continuous backups) on registry/config tables. Recommended true for prod."
  type        = bool
  default     = false
}

variable "tags" {
  description = "Extra tags merged onto every resource."
  type        = map(string)
  default     = {}
}
