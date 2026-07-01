###############################################################################
# modules/ingestion/outputs.tf
###############################################################################

output "ecr_repository_url" {
  description = "ECR repo URL — push the ingestion image here (redeploy_ingest.sh)."
  value       = aws_ecr_repository.ingest.repository_url
}

output "ecr_repository_name" {
  value = aws_ecr_repository.ingest.name
}

output "cluster_name" {
  description = "ECS cluster name (for manual ecs:RunTask via the admin endpoint)."
  value       = aws_ecs_cluster.main.name
}

output "cluster_arn" {
  value = aws_ecs_cluster.main.arn
}

output "task_definition_family" {
  description = "Task definition family (for ecs:RunTask)."
  value       = aws_ecs_task_definition.ingest.family
}

output "task_definition_arn" {
  value = aws_ecs_task_definition.ingest.arn
}

output "execution_role_arn" {
  value = aws_iam_role.execution.arn
}

output "task_role_arn" {
  description = "Task role ARN — what the container's code runs as (has dynamodb:Scan)."
  value       = aws_iam_role.task.arn
}

output "log_group" {
  description = "CloudWatch log group for ingestion runs."
  value       = aws_cloudwatch_log_group.ingest.name
}
