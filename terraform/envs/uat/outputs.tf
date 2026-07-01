###############################################################################
# envs/uat/outputs.tf
###############################################################################

output "foundation" {
  description = "Foundation layer outputs (tables, bucket, secrets)."
  value = {
    table_names      = module.foundation.table_names
    artifacts_bucket = module.foundation.artifacts_bucket
    secret_arns      = module.foundation.secret_arns
    prefix           = module.foundation.prefix
  }
}

output "ingestion" {
  description = "Ingestion layer outputs (ECR, cluster, task def, roles)."
  value = {
    ecr_repository_url     = module.ingestion.ecr_repository_url
    cluster_name           = module.ingestion.cluster_name
    task_definition_family = module.ingestion.task_definition_family
    task_role_arn          = module.ingestion.task_role_arn
    log_group              = module.ingestion.log_group
  }
}

output "search" {
  description = "Search layer outputs (Lambda, API URL, key, WAF)."
  value = {
    lambda_function_name = module.search.lambda_function_name
    api_invoke_url       = module.search.api_invoke_url
    api_key_id           = module.search.api_key_id
    usage_plan_id        = module.search.usage_plan_id
    lambda_s3_target     = module.search.lambda_s3_target
    waf_web_acl_arn      = module.search.waf_web_acl_arn
  }
}
