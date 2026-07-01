###############################################################################
# bootstrap/main.tf
#
# RUN THIS FIRST, ONCE, as an ADMIN principal (not the deploy role — this
# CREATES the deploy role). Uses LOCAL state on purpose: it provisions the
# S3 bucket + DynamoDB lock table that every OTHER stack will use as its
# remote backend (chicken-and-egg — the backend can't manage itself).
#
# After `terraform apply` here:
#   1. commit the generated state? NO — see note in outputs.tf. Keep it, but
#      bootstrap rarely changes. Some teams import it to remote later.
#   2. the deploy role ARN (output) goes into every env's provider assume_role.
#
# Idempotent: safe to re-run. Creates nothing app-specific.
###############################################################################

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  # NOTE: no backend block — bootstrap uses local state by design.
}

provider "aws" {
  region = var.region
  # Run as your admin. No assume_role here — this is the one stack that
  # creates the role others assume.
  default_tags {
    tags = {
      Project   = "rag"
      ManagedBy = "terraform-bootstrap"
    }
  }
}

data "aws_caller_identity" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
}

###############################################################################
# 1. REMOTE STATE BACKEND — S3 bucket (state) + DynamoDB table (lock)
###############################################################################

resource "aws_s3_bucket" "tf_state" {
  bucket = var.state_bucket_name
}

# Versioning: keep state history so a bad apply can be rolled back.
resource "aws_s3_bucket_versioning" "tf_state" {
  bucket = aws_s3_bucket.tf_state.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Encrypt state at rest (state can contain secrets in plaintext).
resource "aws_s3_bucket_server_side_encryption_configuration" "tf_state" {
  bucket = aws_s3_bucket.tf_state.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

# Block ALL public access to the state bucket.
resource "aws_s3_bucket_public_access_block" "tf_state" {
  bucket                  = aws_s3_bucket.tf_state.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# DynamoDB table for state locking (prevents concurrent applies corrupting state).
resource "aws_dynamodb_table" "tf_lock" {
  name         = var.lock_table_name
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "LockID"

  attribute {
    name = "LockID"
    type = "S"
  }
}

###############################################################################
# 2. DEPLOY ROLE — what Terraform (and later CI) assumes to provision the stack
###############################################################################

# Trust policy: who is ALLOWED to assume the deploy role.
# Start with: your admin/SSO principals in this account. Add CI's role ARN later.
data "aws_iam_policy_document" "deploy_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "AWS"
      identifiers = var.deploy_role_trusted_principals
    }
    # Optional hardening: require MFA to assume the deploy role.
    dynamic "condition" {
      for_each = var.require_mfa_to_assume ? [1] : []
      content {
        test     = "Bool"
        variable = "aws:MultiFactorAuthPresent"
        values   = ["true"]
      }
    }
  }
}

resource "aws_iam_role" "deploy" {
  name                 = var.deploy_role_name
  assume_role_policy   = data.aws_iam_policy_document.deploy_assume.json
  max_session_duration = 3600
  description          = "Terraform deploy role for the RAG platform. Assumed by admins/CI to provision infra."
}

# Permissions the deploy role needs to build the RAG stack.
# Scoped to the services Terraform manages. NOT AdministratorAccess — but
# broad within those services (it creates IAM roles, so it's privileged).
data "aws_iam_policy_document" "deploy_permissions" {

  # Core service provisioning
  statement {
    sid    = "ServiceProvisioning"
    effect = "Allow"
    actions = [
      "lambda:*",
      "ecs:*",
      "ecr:*",
      "dynamodb:*",
      "s3:*",
      "apigateway:*",
      "events:*",          # EventBridge
      "logs:*",            # CloudWatch Logs
      "cloudwatch:*",
      "secretsmanager:*",
      "wafv2:*",
      "elasticloadbalancing:*",
      "ec2:Describe*",     # read VPC/subnet/SG for networking config
    ]
    resources = ["*"]
  }

  # Bedrock (guardrails + rerank model invocation provisioning)
  statement {
    sid    = "Bedrock"
    effect = "Allow"
    actions = [
      "bedrock:*",
    ]
    resources = ["*"]
  }

  # IAM — the deploy role creates the app roles (lambda exec, ingest task/exec).
  # Scoped to rag-* role names so the deploy role can't touch unrelated IAM.
  statement {
    sid    = "IamRoleManagement"
    effect = "Allow"
    actions = [
      "iam:CreateRole",
      "iam:DeleteRole",
      "iam:GetRole",
      "iam:PassRole",
      "iam:TagRole",
      "iam:UntagRole",
      "iam:AttachRolePolicy",
      "iam:DetachRolePolicy",
      "iam:PutRolePolicy",
      "iam:DeleteRolePolicy",
      "iam:GetRolePolicy",
      "iam:ListRolePolicies",
      "iam:ListAttachedRolePolicies",
      "iam:CreatePolicy",
      "iam:DeletePolicy",
      "iam:GetPolicy",
      "iam:CreatePolicyVersion",
      "iam:DeletePolicyVersion",
      "iam:ListPolicyVersions",
      "iam:CreateServiceLinkedRole",
    ]
    resources = [
      "arn:aws:iam::${local.account_id}:role/rag-*",
      "arn:aws:iam::${local.account_id}:policy/rag-*",
      # service-linked roles (ECS, etc.)
      "arn:aws:iam::${local.account_id}:role/aws-service-role/*",
    ]
  }

  # Remote state access (so the deploy role can read/write its own state).
  statement {
    sid    = "StateBackend"
    effect = "Allow"
    actions = [
      "s3:ListBucket",
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
    ]
    resources = [
      aws_s3_bucket.tf_state.arn,
      "${aws_s3_bucket.tf_state.arn}/*",
    ]
  }
  statement {
    sid       = "StateLock"
    effect    = "Allow"
    actions   = ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:DeleteItem"]
    resources = [aws_dynamodb_table.tf_lock.arn]
  }
}

resource "aws_iam_role_policy" "deploy" {
  name   = "${var.deploy_role_name}-policy"
  role   = aws_iam_role.deploy.id
  policy = data.aws_iam_policy_document.deploy_permissions.json
}
