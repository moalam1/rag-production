###############################################################################
# bootstrap/variables.tf
###############################################################################

variable "region" {
  description = "Workload region. Locked to us-east-1 (Bedrock Cohere Rerank + Guardrails availability)."
  type        = string
  default     = "us-east-1"
}

variable "state_bucket_name" {
  description = "S3 bucket for Terraform remote state. Must be globally unique."
  type        = string
  default     = "rag-terraform-state"
  # NOTE: if 'rag-terraform-state' is taken globally, suffix the account id,
  # e.g. rag-terraform-state-019313283759
}

variable "lock_table_name" {
  description = "DynamoDB table for Terraform state locking."
  type        = string
  default     = "rag-terraform-locks"
}

variable "deploy_role_name" {
  description = "Name of the IAM role Terraform assumes to provision the stack."
  type        = string
  default     = "rag-terraform-deploy"
}

variable "deploy_role_trusted_principals" {
  description = <<-EOT
    ARNs allowed to assume the deploy role. Start with your admin / SSO
    permission-set role ARN(s). Add the CI role ARN here later.
    Find your SSO role ARN with:
      aws sts get-caller-identity   (the 'Arn' field, strip the session part)
    For Identity Center, it looks like:
      arn:aws:iam::ACCOUNT:role/aws-reserved/sso.amazonaws.com/REGION/AWSReservedSSO_AdministratorAccess_xxxx
  EOT
  type        = list(string)
  # MUST be set in terraform.tfvars — no safe default (don't want to trust everyone).
}

variable "require_mfa_to_assume" {
  description = "Require MFA to assume the deploy role. Recommended true once SSO MFA is enforced."
  type        = bool
  default     = false
}
