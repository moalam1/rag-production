###############################################################################
# modules/ingestion/main.tf
#
# Fargate batch ingestion. Two IAM roles (the important part):
#   - execution role: lets ECS PULL the image + write logs (AWS-facing)
#   - task role:      what the CONTAINER's code can do (DynamoDB, S3, Secrets)
#                     — carries the dynamodb:Scan the prototype proved necessary.
###############################################################################

data "aws_caller_identity" "current" {}

locals {
  prefix     = "rag-${var.env}"
  account_id = data.aws_caller_identity.current.account_id

  common_tags = merge({
    Project = "rag"
    Env     = var.env
    Module  = "ingestion"
  }, var.tags)
}

###############################################################################
# ECR — container registry for the ingestion image
###############################################################################

resource "aws_ecr_repository" "ingest" {
  name                 = "${local.prefix}-ingest"
  image_tag_mutability = "MUTABLE" # redeploy_ingest.sh pushes :latest + :git-sha

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = merge(local.common_tags, { Repo = "ingest" })
}

# Lifecycle: keep the last 10 images, expire older untagged ones to control cost.
resource "aws_ecr_lifecycle_policy" "ingest" {
  repository = aws_ecr_repository.ingest.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 10 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}

###############################################################################
# CloudWatch Logs — task log group
###############################################################################

resource "aws_cloudwatch_log_group" "ingest" {
  name              = "/ecs/${local.prefix}-ingest-task"
  retention_in_days = var.log_retention_days
  tags              = local.common_tags
}

###############################################################################
# ECS cluster
###############################################################################

resource "aws_ecs_cluster" "main" {
  name = "${local.prefix}-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = local.common_tags
}

###############################################################################
# IAM ROLE #1 — EXECUTION ROLE
# What ECS/Fargate itself needs: pull the image from ECR, write logs.
# This is AWS infrastructure acting on your behalf BEFORE your code runs.
###############################################################################

data "aws_iam_policy_document" "execution_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "execution" {
  name               = "${local.prefix}-ingest-exec-role"
  assume_role_policy = data.aws_iam_policy_document.execution_assume.json
  tags               = local.common_tags
}

# AWS-managed policy covers ECR pull + CloudWatch logs for Fargate execution.
resource "aws_iam_role_policy_attachment" "execution_managed" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# The execution role also needs to read the secrets it injects into the
# container env at startup (Secrets Manager → container environment).
data "aws_iam_policy_document" "execution_secrets" {
  statement {
    sid       = "ReadSecretsForInjection"
    effect    = "Allow"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [var.secret_arn_prefix]
  }
}

resource "aws_iam_role_policy" "execution_secrets" {
  name   = "${local.prefix}-ingest-exec-secrets"
  role   = aws_iam_role.execution.id
  policy = data.aws_iam_policy_document.execution_secrets.json
}

###############################################################################
# IAM ROLE #2 — TASK ROLE  (the important one)
# What YOUR CODE inside the container is allowed to do.
# Least-privilege, scoped to THIS env's resources. Carries dynamodb:Scan on
# the document-registry (the dedup step — a live prototype run proved this).
###############################################################################

data "aws_iam_policy_document" "task_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "task" {
  name               = "${local.prefix}-ingest-task-role"
  assume_role_policy = data.aws_iam_policy_document.task_assume.json
  tags               = local.common_tags
}

data "aws_iam_policy_document" "task_permissions" {

  # DynamoDB: registry needs Scan (dedup) + read/write. Other tables read/write.
  statement {
    sid    = "DynamoDBRegistry"
    effect = "Allow"
    actions = [
      "dynamodb:Scan",      # ← the dedup permission the prototype exposed
      "dynamodb:GetItem",
      "dynamodb:PutItem",
      "dynamodb:UpdateItem",
      "dynamodb:DeleteItem",
      "dynamodb:BatchGetItem",
      "dynamodb:BatchWriteItem",
      "dynamodb:Query",
    ]
    resources = [
      var.table_arns["document_registry"],
      var.table_arns["config"],
      "${var.table_arns["document_registry"]}/index/*",
      "${var.table_arns["config"]}/index/*",
    ]
  }

  # S3: write the BM25 index + read/write artifacts.
  statement {
    sid    = "S3Artifacts"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:ListBucket",
    ]
    resources = [
      var.artifacts_bucket_arn,
      "${var.artifacts_bucket_arn}/*",
    ]
  }

  # Secrets: read this env's vendor keys (OpenAI, Pinecone, etc.).
  statement {
    sid       = "Secrets"
    effect    = "Allow"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [var.secret_arn_prefix]
  }
}

resource "aws_iam_role_policy" "task" {
  name   = "${local.prefix}-ingest-task-policy"
  role   = aws_iam_role.task.id
  policy = data.aws_iam_policy_document.task_permissions.json
}

###############################################################################
# ECS TASK DEFINITION — the Fargate ingestion task
###############################################################################

resource "aws_ecs_task_definition" "ingest" {
  family                   = "${local.prefix}-ingest-task"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([
    {
      name      = "ingest"
      image     = "${aws_ecr_repository.ingest.repository_url}:latest"
      essential = true

      environment = [
        { name = "ENVIRONMENT", value = var.env },
        { name = "AWS_REGION", value = var.region },
        { name = "CONTENT_HOST", value = var.content_host },
        { name = "PINECONE_INDEX", value = var.pinecone_index },
        { name = "CACHE_BACKEND", value = "dynamodb" },
        { name = "BM25_S3_BUCKET", value = replace(var.artifacts_bucket_arn, "arn:aws:s3:::", "") },
        { name = "DYNAMODB_REGISTRY_TABLE", value = var.table_arns["document_registry"] != null ? "${local.prefix}-document-registry" : "" },
        { name = "DYNAMODB_CONFIG_TABLE", value = "${local.prefix}-config" },
      ]

      # Secrets injected from Secrets Manager → container env (by the EXECUTION role).
      secrets = [
        { name = "OPENAI_API_KEY", valueFrom = "arn:aws:secretsmanager:${var.region}:${local.account_id}:secret:rag/${var.env}/openai-api-key" },
        { name = "PINECONE_API_KEY", valueFrom = "arn:aws:secretsmanager:${var.region}:${local.account_id}:secret:rag/${var.env}/pinecone-api-key" },
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ingest.name
          "awslogs-region"        = var.region
          "awslogs-stream-prefix" = "ingest"
        }
      }
    }
  ])

  tags = local.common_tags
}

###############################################################################
# EVENTBRIDGE — nightly scheduled ingestion (optional; default OFF in UAT)
###############################################################################

# Role EventBridge assumes to run the ECS task.
data "aws_iam_policy_document" "events_assume" {
  count = var.schedule_expression != "" ? 1 : 0
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["events.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "events" {
  count              = var.schedule_expression != "" ? 1 : 0
  name               = "${local.prefix}-ingest-events-role"
  assume_role_policy = data.aws_iam_policy_document.events_assume[0].json
  tags               = local.common_tags
}

data "aws_iam_policy_document" "events_run_task" {
  count = var.schedule_expression != "" ? 1 : 0

  statement {
    sid       = "RunIngestTask"
    effect    = "Allow"
    actions   = ["ecs:RunTask"]
    resources = ["${aws_ecs_task_definition.ingest.arn_without_revision}:*"]
    condition {
      test     = "ArnLike"
      variable = "ecs:cluster"
      values   = [aws_ecs_cluster.main.arn]
    }
  }

  # PassRole so EventBridge can hand the task its execution + task roles.
  statement {
    sid       = "PassTaskRoles"
    effect    = "Allow"
    actions   = ["iam:PassRole"]
    resources = [aws_iam_role.execution.arn, aws_iam_role.task.arn]
  }
}

resource "aws_iam_role_policy" "events_run_task" {
  count  = var.schedule_expression != "" ? 1 : 0
  name   = "${local.prefix}-ingest-events-policy"
  role   = aws_iam_role.events[0].id
  policy = data.aws_iam_policy_document.events_run_task[0].json
}

resource "aws_cloudwatch_event_rule" "nightly" {
  count               = var.schedule_expression != "" ? 1 : 0
  name                = "${local.prefix}-ingest-nightly"
  description         = "Nightly RAG ingestion run"
  schedule_expression = var.schedule_expression
  state               = var.schedule_enabled ? "ENABLED" : "DISABLED"
  tags                = local.common_tags
}

resource "aws_cloudwatch_event_target" "nightly" {
  count     = var.schedule_expression != "" ? 1 : 0
  rule      = aws_cloudwatch_event_rule.nightly[0].name
  arn       = aws_ecs_cluster.main.arn
  role_arn  = aws_iam_role.events[0].arn

  ecs_target {
    task_definition_arn = aws_ecs_task_definition.ingest.arn
    task_count          = 1
    launch_type         = "FARGATE"

    network_configuration {
      subnets          = var.subnet_ids
      security_groups  = var.security_group_ids
      assign_public_ip = true # prototype: public IP for internet access, no NAT
    }
  }
}
