###############################################################################
# modules/observability/main.tf
#
# Production health monitoring: SNS topic + email, 5 alarms, dashboard.
# Complementary to LangSmith (which does eval/response quality). This is ops:
# "is the service up, fast, not erroring — and page me if it breaks."
#
# Written to MATCH the console-created uat resources exactly, so existing envs
# can `terraform import` them with no drift. New envs get them created.
#
# The 5 alarms: Lambda errors, Lambda throttles, API 5xx, API p99 latency,
# and (via EventBridge) ingestion Fargate task failure.
###############################################################################

locals {
  prefix = "rag-${var.env}"
}

data "aws_caller_identity" "current" {}

# ── SNS topic + email subscription ────────────────────────────────────────
resource "aws_sns_topic" "alarms" {
  name = "${local.prefix}-alarms"
}

# Allow CloudWatch alarms + EventBridge to publish (merged with default owner).
resource "aws_sns_topic_policy" "alarms" {
  arn = aws_sns_topic.alarms.arn
  policy = jsonencode({
    Version = "2008-10-17"
    Id      = "__default_policy_ID"
    Statement = [
      {
        Sid       = "__default_statement_ID"
        Effect    = "Allow"
        Principal = { AWS = "*" }
        Action = [
          "SNS:GetTopicAttributes", "SNS:SetTopicAttributes", "SNS:AddPermission",
          "SNS:RemovePermission", "SNS:DeleteTopic", "SNS:Subscribe",
          "SNS:ListSubscriptionsByTopic", "SNS:Publish"
        ]
        Resource  = aws_sns_topic.alarms.arn
        Condition = { StringEquals = { "AWS:SourceOwner" = data.aws_caller_identity.current.account_id } }
      },
      {
        Sid       = "AllowEventBridgePublish"
        Effect    = "Allow"
        Principal = { Service = "events.amazonaws.com" }
        Action    = "sns:Publish"
        Resource  = aws_sns_topic.alarms.arn
      }
    ]
  })
}

# One subscription per alert email. Each recipient must confirm via the email
# AWS sends. Manage the team list here (or point at a distribution list).
resource "aws_sns_topic_subscription" "email" {
  for_each  = toset(var.alarm_emails)
  topic_arn = aws_sns_topic.alarms.arn
  protocol  = "email"
  endpoint  = each.value
}

# ── Alarm 1: Lambda errors ────────────────────────────────────────────────
resource "aws_cloudwatch_metric_alarm" "lambda_errors" {
  alarm_name          = "${local.prefix}-search-lambda-errors"
  alarm_description   = "Search Lambda throwing errors"
  namespace           = "AWS/Lambda"
  metric_name         = "Errors"
  dimensions          = { FunctionName = "${local.prefix}-search" }
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alarms.arn]
  ok_actions          = [aws_sns_topic.alarms.arn]
}

# ── Alarm 2: Lambda throttles ─────────────────────────────────────────────
resource "aws_cloudwatch_metric_alarm" "lambda_throttles" {
  alarm_name          = "${local.prefix}-search-lambda-throttles"
  alarm_description   = "Search Lambda being throttled (concurrency)"
  namespace           = "AWS/Lambda"
  metric_name         = "Throttles"
  dimensions          = { FunctionName = "${local.prefix}-search" }
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alarms.arn]
  ok_actions          = [aws_sns_topic.alarms.arn]
}

# ── Alarm 3: API Gateway 5xx ──────────────────────────────────────────────
# NOTE: requires detailed CloudWatch metrics enabled on the API stage, else no
# data (see the foundation/search stage config). Dimension keys on API NAME.
resource "aws_cloudwatch_metric_alarm" "api_5xx" {
  alarm_name          = "${local.prefix}-api-5xx"
  alarm_description   = "API Gateway returning 5xx to users"
  namespace           = "AWS/ApiGateway"
  metric_name         = "5XXError"
  dimensions          = { ApiName = "${local.prefix}-search-api" }
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alarms.arn]
  ok_actions          = [aws_sns_topic.alarms.arn]
}

# ── Alarm 4: API p99 latency ──────────────────────────────────────────────
resource "aws_cloudwatch_metric_alarm" "api_latency_p99" {
  alarm_name          = "${local.prefix}-api-latency-p99"
  alarm_description   = "API p99 latency degraded (>${var.api_latency_p99_ms}ms)"
  namespace           = "AWS/ApiGateway"
  metric_name         = "Latency"
  dimensions          = { ApiName = "${local.prefix}-search-api" }
  extended_statistic  = "p99"
  period              = 300
  evaluation_periods  = 2
  threshold           = var.api_latency_p99_ms
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alarms.arn]
  ok_actions          = [aws_sns_topic.alarms.arn]
}

# ── Alarm 5: Fargate ingestion task failure (via EventBridge) ─────────────
resource "aws_cloudwatch_event_rule" "ingest_task_failed" {
  name        = "${local.prefix}-ingest-task-failed"
  description = "Ingestion Fargate task stopped with non-zero exit"
  event_pattern = jsonencode({
    source      = ["aws.ecs"]
    detail-type = ["ECS Task State Change"]
    detail = {
      clusterArn = [{ suffix = "${local.prefix}-cluster" }]
      lastStatus = ["STOPPED"]
      containers = { exitCode = [{ "anything-but" = [0] }] }
    }
  })
}

resource "aws_cloudwatch_event_target" "ingest_task_failed_sns" {
  rule      = aws_cloudwatch_event_rule.ingest_task_failed.name
  target_id = "1"
  arn       = aws_sns_topic.alarms.arn
}

# ── Dashboard ─────────────────────────────────────────────────────────────
resource "aws_cloudwatch_dashboard" "health" {
  dashboard_name = "${local.prefix}-health"
  dashboard_body = jsonencode({
    widgets = [
      {
        type = "metric", x = 0, y = 0, width = 12, height = 6
        properties = {
          title  = "Search Lambda — Invocations / Errors / Throttles"
          region = var.region, stat = "Sum", period = 300
          metrics = [
            ["AWS/Lambda", "Invocations", "FunctionName", "${local.prefix}-search"],
            ["AWS/Lambda", "Errors", "FunctionName", "${local.prefix}-search"],
            ["AWS/Lambda", "Throttles", "FunctionName", "${local.prefix}-search"]
          ]
        }
      },
      {
        type = "metric", x = 12, y = 0, width = 12, height = 6
        properties = {
          title  = "Search Lambda — Duration (p50 / p99)"
          region = var.region, period = 300
          metrics = [
            ["AWS/Lambda", "Duration", "FunctionName", "${local.prefix}-search", { stat = "p50" }],
            ["AWS/Lambda", "Duration", "FunctionName", "${local.prefix}-search", { stat = "p99" }]
          ]
        }
      },
      {
        type = "metric", x = 0, y = 6, width = 12, height = 6
        properties = {
          title  = "API Gateway — Requests / 4xx / 5xx"
          region = var.region, stat = "Sum", period = 300
          metrics = [
            ["AWS/ApiGateway", "Count", "ApiName", "${local.prefix}-search-api"],
            ["AWS/ApiGateway", "4XXError", "ApiName", "${local.prefix}-search-api"],
            ["AWS/ApiGateway", "5XXError", "ApiName", "${local.prefix}-search-api"]
          ]
        }
      },
      {
        type = "metric", x = 12, y = 6, width = 12, height = 6
        properties = {
          title  = "API Gateway — Latency (p50 / p99)"
          region = var.region, period = 300
          metrics = [
            ["AWS/ApiGateway", "Latency", "ApiName", "${local.prefix}-search-api", { stat = "p50" }],
            ["AWS/ApiGateway", "Latency", "ApiName", "${local.prefix}-search-api", { stat = "p99" }]
          ]
        }
      }
    ]
  })
}
