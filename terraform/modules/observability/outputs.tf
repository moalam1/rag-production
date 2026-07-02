###############################################################################
# modules/observability/outputs.tf
###############################################################################

output "sns_topic_arn" {
  description = "Alarm SNS topic ARN."
  value       = aws_sns_topic.alarms.arn
}

output "dashboard_name" {
  value = aws_cloudwatch_dashboard.health.dashboard_name
}

output "alarm_names" {
  value = [
    aws_cloudwatch_metric_alarm.lambda_errors.alarm_name,
    aws_cloudwatch_metric_alarm.lambda_throttles.alarm_name,
    aws_cloudwatch_metric_alarm.api_5xx.alarm_name,
    aws_cloudwatch_metric_alarm.api_latency_p99.alarm_name,
    aws_cloudwatch_event_rule.ingest_task_failed.name,
  ]
}
