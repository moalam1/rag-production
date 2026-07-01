###############################################################################
# modules/ingestion/variables.tf
#
# The async batch-ingestion engine: ECR (image), ECS Fargate (compute), the two
# IAM roles, CloudWatch logs, and an EventBridge schedule. Compute shapes mirror
# the VALIDATED prototype (761MB image, public-subnet + public-IP networking,
# dynamodb:Scan on the task role for registry dedup).
###############################################################################

variable "env" {
  description = "Environment (dev | uat | prod)."
  type        = string
  validation {
    condition     = contains(["dev","uat", "prod"], var.env)
    error_message = "env must be 'uat' or 'prod'."
  }
}

variable "region" {
  description = "Workload region."
  type        = string
  default     = "us-east-1"
}

# --- Foundation wiring (passed from the env composition) ---
variable "table_arns" {
  description = "DynamoDB table ARNs from the foundation module (for task-role IAM scoping)."
  type        = map(string)
}

variable "artifacts_bucket_arn" {
  description = "Artifacts S3 bucket ARN from foundation (task writes the BM25 index here)."
  type        = string
}

variable "secret_arn_prefix" {
  description = "Secrets ARN prefix (rag/{env}/*) from foundation, for GetSecretValue scoping."
  type        = string
}

# --- Networking (the prototype: public subnet + public IP, NO NAT) ---
variable "subnet_ids" {
  description = <<-EOT
    Subnet IDs for the Fargate task. Prototype used a PUBLIC subnet with a
    public IP (assign_public_ip = true) so the crawler reaches the internet
    without a NAT gateway. Provide public subnet IDs in the corporate VPC.
  EOT
  type        = list(string)
}

variable "security_group_ids" {
  description = "Security group IDs for the Fargate task (egress to internet for crawling + AWS APIs)."
  type        = list(string)
  default     = []
}

# --- Task sizing (Fargate) ---
variable "task_cpu" {
  description = "Fargate task CPU units. 1024 = 1 vCPU. Ingestion is I/O-bound (parse/embed waits)."
  type        = string
  default     = "1024"
}

variable "task_memory" {
  description = "Fargate task memory (MB). 761MB image + parsing headroom."
  type        = string
  default     = "4096"
}

# --- Schedule ---
variable "schedule_expression" {
  description = "EventBridge schedule for nightly ingestion. Empty string disables the schedule."
  type        = string
  default     = "cron(0 6 * * ? *)" # 06:00 UTC daily
}

variable "schedule_enabled" {
  description = "Whether the nightly EventBridge schedule is enabled."
  type        = bool
  default     = false # default OFF in UAT; turn on deliberately
}

# --- Container env (non-secret config passed to the task) ---
variable "content_host" {
  description = "CONTENT_HOST for the crawler. Public sitemap host (prod.equinix.com only resolves in-network)."
  type        = string
  default     = "www.equinix.com"
}

variable "pinecone_index" {
  description = "Pinecone index name (non-secret)."
  type        = string
  default     = "rag-poc"
}

variable "log_retention_days" {
  description = "CloudWatch log retention for the ingestion task."
  type        = number
  default     = 30
}

variable "tags" {
  description = "Extra tags."
  type        = map(string)
  default     = {}
}
