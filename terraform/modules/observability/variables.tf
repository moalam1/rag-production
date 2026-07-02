###############################################################################
# modules/observability/variables.tf
###############################################################################

variable "env" {
  type        = string
  description = "Environment name (dev/uat/prod)."
  validation {
    condition     = contains(["dev", "uat", "prod"], var.env)
    error_message = "env must be 'dev', 'uat' or 'prod'."
  }
}

variable "region" {
  type    = string
  default = "us-east-1"
}

variable "alarm_emails" {
  type        = list(string)
  description = "Emails subscribed to the alarm SNS topic. Each must confirm the AWS email."
  default     = []
}

variable "api_latency_p99_ms" {
  type        = number
  description = "p99 API latency alarm threshold in milliseconds."
  default     = 10000
}
