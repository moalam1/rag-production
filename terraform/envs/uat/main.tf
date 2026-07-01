###############################################################################
# envs/uat/main.tf
#
# Composes the modules for the UAT environment. Provider assumes the deploy
# role created by bootstrap. Build order: foundation → ingestion → search
# (ingestion + search modules added as they're built).
###############################################################################

provider "aws" {
  region = var.region

  # Run AS the deploy role, not personal admin.
  assume_role {
    role_arn = var.deploy_role_arn
  }

  default_tags {
    tags = {
      Project   = "rag"
      Env       = "uat"
      ManagedBy = "terraform"
    }
  }
}

# --- Foundation: data layer (DynamoDB, S3, Secrets) ---
module "foundation" {
  source = "../../modules/foundation"

  env    = "uat"
  region = var.region

  # UAT: lighter durability settings (save cost; prod will enable these).
  point_in_time_recovery = false
  cache_ttl_enabled      = true # never disable — stale-forever bug
}

# --- Ingestion: async batch loading (ECR, ECS Fargate, IAM, EventBridge) ---
module "ingestion" {
  source = "../../modules/ingestion"

  env    = "uat"
  region = var.region

  # Foundation wiring
  table_arns           = module.foundation.table_arns
  artifacts_bucket_arn = module.foundation.artifacts_bucket_arn
  secret_arn_prefix    = module.foundation.secret_arn_prefix

  # Networking — corporate default VPC public subnets (matches prototype:
  # public subnet + public IP, no NAT). 3 AZs for placement resilience.
  subnet_ids = [
    "subnet-0ac8ce76c20ac23eb", # us-east-1a
    "subnet-0bfa25a81270e3e86", # us-east-1b
    "subnet-0d384bd9cda5e1c20", # us-east-1c
  ]
  security_group_ids = ["sg-04069ae01c6942758"] # default SG (outbound allowed)

  # Schedule OFF in UAT — run ingestion manually until validated.
  schedule_enabled = false

  content_host   = "www.equinix.com"
  pinecone_index = "rag-poc"
}

# --- Search module: added in build order 3 ---
# module "search" {
#   source            = "../../modules/search"
#   env               = "uat"
#   region            = var.region
#   table_arns        = module.foundation.table_arns
#   artifacts_bucket  = module.foundation.artifacts_bucket
#   secret_arn_prefix = module.foundation.secret_arn_prefix
#   ...
# }
module "search" {
  source = "../../modules/search"

  env    = "uat"
  region = var.region

  # Foundation wiring
  table_arns           = module.foundation.table_arns
  artifacts_bucket     = module.foundation.artifacts_bucket
  artifacts_bucket_arn = module.foundation.artifacts_bucket_arn
  secret_arn_prefix    = module.foundation.secret_arn_prefix

  # UAT settings (cost-conscious; prod will override)
  provisioned_concurrency = 0       # off in UAT (turn on for load tests / prod)
  waf_enabled             = true     # WAF on from the start
  cors_allowed_origins    = ["*"]    # UAT only — prod MUST scope to equinix.com
}
