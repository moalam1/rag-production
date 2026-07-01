###############################################################################
# bootstrap/outputs.tf
#
# These outputs feed the rest of the Terraform:
#   - state_bucket / lock_table  → the backend "s3" block in each env
#   - deploy_role_arn            → the provider assume_role in each env
###############################################################################

output "state_bucket" {
  description = "S3 bucket name for remote state. Put this in each env's backend block."
  value       = aws_s3_bucket.tf_state.id
}

output "lock_table" {
  description = "DynamoDB lock table name. Put this in each env's backend block."
  value       = aws_dynamodb_table.tf_lock.name
}

output "deploy_role_arn" {
  description = "Deploy role ARN. Put this in each env provider's assume_role.role_arn."
  value       = aws_iam_role.deploy.arn
}

output "region" {
  value = var.region
}

###############################################################################
# NOTE on bootstrap state:
# Bootstrap uses LOCAL state (terraform.tfstate in this dir). It's small and
# rarely changes. Options:
#   (a) Keep it local + commit to a SECURE private repo location, OR
#   (b) After first apply, add a backend block pointing at the bucket it just
#       created + `terraform init -migrate-state` to move it remote.
# (a) is simplest; (b) is cleaner. Either is fine. Do NOT commit it to a
# public repo (it can contain the role policy json — low-sensitivity, but
# treat as private).
###############################################################################
