###############################################################################
# envs/prod/backend.tf
#
# Remote state in the bucket bootstrap created. The deploy role (provider
# assume_role) needs the StateBackend permissions bootstrap granted it.
#
# Fill in the bucket name from `terraform output state_bucket` in bootstrap.
# If you suffixed the account id in bootstrap, match it here.
###############################################################################

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.4"
    }
  }

  backend "s3" {
    bucket         = "rag-terraform-state-019313283759" # ← match bootstrap output state_bucket
    key            = "prod/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "rag-terraform-locks" # ← match bootstrap output lock_table
    encrypt        = true
  }
}
