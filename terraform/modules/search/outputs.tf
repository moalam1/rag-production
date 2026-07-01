###############################################################################
# modules/search/outputs.tf
###############################################################################

output "lambda_function_name" {
  value = aws_lambda_function.search.function_name
}

output "lambda_function_arn" {
  value = aws_lambda_function.search.arn
}

output "api_invoke_url" {
  description = "Base URL for the search API (append paths)."
  value       = aws_api_gateway_stage.search.invoke_url
}

output "api_id" {
  value = aws_api_gateway_rest_api.search.id
}

output "api_key_id" {
  description = "API key ID. Retrieve the VALUE with: aws apigateway get-api-key --api-key <id> --include-value"
  value       = aws_api_gateway_api_key.client.id
}

output "usage_plan_id" {
  value = aws_api_gateway_usage_plan.search.id
}

output "waf_web_acl_arn" {
  value = var.waf_enabled ? aws_wafv2_web_acl.search[0].arn : null
}

output "lambda_s3_target" {
  description = "Where to push the real lambda zip (redeploy_lambda.sh)."
  value       = "s3://${var.artifacts_bucket}/${var.lambda_s3_key}"
}
