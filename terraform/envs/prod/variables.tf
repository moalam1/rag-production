###############################################################################
# envs/prod/variables.tf
###############################################################################

variable "region" {
  description = "Workload region."
  type        = string
  default     = "us-east-1"
}

variable "deploy_role_arn" {
  description = "Deploy role ARN from bootstrap output. Set in terraform.tfvars."
  type        = string
}
