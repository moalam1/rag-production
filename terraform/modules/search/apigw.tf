###############################################################################
# modules/search/apigw.tf
#
# API Gateway (REST) → Lambda, with:
#   - usage plan + API key  = THE rate limiter (audit finding: replaces the
#     in-memory app limiter that breaks across Lambda instances)
#   - AWS WAF on the stage  = port of the prototype RAG-WAF-RULE
###############################################################################

###############################################################################
# REST API + proxy integration
###############################################################################

resource "aws_api_gateway_rest_api" "search" {
  name        = "${local.prefix}-search-api"
  description = "RAG ${var.env} search API"

  endpoint_configuration {
    types = ["REGIONAL"]
  }

  tags = local.common_tags
}

# Proxy resource: /{proxy+} catches all paths, forwards to the FastAPI app.
resource "aws_api_gateway_resource" "proxy" {
  rest_api_id = aws_api_gateway_rest_api.search.id
  parent_id   = aws_api_gateway_rest_api.search.root_resource_id
  path_part   = "{proxy+}"
}

# ANY method on the proxy — API key required (the usage plan enforces limits).
resource "aws_api_gateway_method" "proxy_any" {
  rest_api_id      = aws_api_gateway_rest_api.search.id
  resource_id      = aws_api_gateway_resource.proxy.id
  http_method      = "ANY"
  authorization    = "NONE"
  api_key_required = true
}

resource "aws_api_gateway_integration" "proxy" {
  rest_api_id             = aws_api_gateway_rest_api.search.id
  resource_id             = aws_api_gateway_resource.proxy.id
  http_method             = aws_api_gateway_method.proxy_any.http_method
  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.search.invoke_arn
}

# Also handle the root path "/" (not just /{proxy+}).
resource "aws_api_gateway_method" "root_any" {
  rest_api_id      = aws_api_gateway_rest_api.search.id
  resource_id      = aws_api_gateway_rest_api.search.root_resource_id
  http_method      = "ANY"
  authorization    = "NONE"
  api_key_required = true
}

resource "aws_api_gateway_integration" "root" {
  rest_api_id             = aws_api_gateway_rest_api.search.id
  resource_id             = aws_api_gateway_rest_api.search.root_resource_id
  http_method             = aws_api_gateway_method.root_any.http_method
  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.search.invoke_arn
}

# Let API Gateway invoke the Lambda.
resource "aws_lambda_permission" "apigw" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.search.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_api_gateway_rest_api.search.execution_arn}/*/*"
}

###############################################################################
# Deployment + stage
###############################################################################

resource "aws_api_gateway_deployment" "search" {
  rest_api_id = aws_api_gateway_rest_api.search.id

  # Redeploy when the API shape changes.
  triggers = {
    redeployment = sha1(jsonencode([
      aws_api_gateway_resource.proxy.id,
      aws_api_gateway_method.proxy_any.id,
      aws_api_gateway_integration.proxy.id,
      aws_api_gateway_method.root_any.id,
      aws_api_gateway_integration.root.id,
    ]))
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_api_gateway_stage" "search" {
  rest_api_id   = aws_api_gateway_rest_api.search.id
  deployment_id = aws_api_gateway_deployment.search.id
  stage_name    = var.env

  tags = local.common_tags
}

###############################################################################
# USAGE PLAN + API KEY  (the REAL rate limiter — audit finding)
#
# This enforces per-key throttling at the EDGE, before Lambda runs — so it
# works correctly across any number of Lambda instances, unlike the in-memory
# slowapi limiter (which counts per-process and multiplies by instance count).
###############################################################################

resource "aws_api_gateway_api_key" "client" {
  name = "${local.prefix}-search-key"
  tags = local.common_tags
}

resource "aws_api_gateway_usage_plan" "search" {
  name = "${local.prefix}-search-usage-plan"

  api_stages {
    api_id = aws_api_gateway_rest_api.search.id
    stage  = aws_api_gateway_stage.search.stage_name
  }

  throttle_settings {
    rate_limit  = var.rate_limit_per_second
    burst_limit = var.rate_limit_burst
  }

  quota_settings {
    limit  = var.quota_per_day
    period = "DAY"
  }

  tags = local.common_tags
}

resource "aws_api_gateway_usage_plan_key" "client" {
  key_id        = aws_api_gateway_api_key.client.id
  key_type      = "API_KEY"
  usage_plan_id = aws_api_gateway_usage_plan.search.id
}

###############################################################################
# AWS WAF  (port of the prototype RAG-WAF-RULE — audit finding)
#
# Prototype had: CommonRuleSet, KnownBadInputs, PHPRuleSet (dropped — Python
# app), AmazonIpReputationList. We ADD a rate-based rule (the prototype lacked
# one). Attached to the API Gateway stage.
###############################################################################

resource "aws_wafv2_web_acl" "search" {
  count = var.waf_enabled ? 1 : 0

  name  = "${local.prefix}-search-waf"
  scope = "REGIONAL" # API Gateway = regional (not CLOUDFRONT)

  default_action {
    allow {}
  }

  # 1. AWS Common Rule Set
  rule {
    name     = "CommonRuleSet"
    priority = 1
    override_action {
      none {}
    }
    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesCommonRuleSet"
        vendor_name = "AWS"
      }
    }
    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "${local.prefix}-common"
      sampled_requests_enabled   = true
    }
  }

  # 2. Known Bad Inputs
  rule {
    name     = "KnownBadInputs"
    priority = 2
    override_action {
      none {}
    }
    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesKnownBadInputsRuleSet"
        vendor_name = "AWS"
      }
    }
    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "${local.prefix}-badinputs"
      sampled_requests_enabled   = true
    }
  }

  # 3. Amazon IP Reputation
  rule {
    name     = "IpReputation"
    priority = 3
    override_action {
      none {}
    }
    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesAmazonIpReputationList"
        vendor_name = "AWS"
      }
    }
    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "${local.prefix}-iprep"
      sampled_requests_enabled   = true
    }
  }

  # 4. Rate-based rule (NEW — prototype lacked one). Per-IP volumetric guard.
  rule {
    name     = "RateLimit"
    priority = 4
    action {
      block {}
    }
    statement {
      rate_based_statement {
        limit              = var.waf_rate_limit
        aggregate_key_type = "IP"
      }
    }
    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "${local.prefix}-ratelimit"
      sampled_requests_enabled   = true
    }
  }

  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "${local.prefix}-search-waf"
    sampled_requests_enabled   = true
  }

  tags = local.common_tags
}

resource "aws_wafv2_web_acl_association" "search" {
  count        = var.waf_enabled ? 1 : 0
  resource_arn = aws_api_gateway_stage.search.arn
  web_acl_arn  = aws_wafv2_web_acl.search[0].arn
}
