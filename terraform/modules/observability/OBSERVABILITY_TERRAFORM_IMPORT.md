# Observability Module — Wire-in + Import Guide

Adopt the console-created uat observability resources into Terraform with NO
recreation. dev/prod (if not yet created) get created by apply.

---

## 1. Copy the module into the repo
```
terraform/modules/observability/{main.tf,variables.tf,outputs.tf}
```

## 2. Add the module call to each env's main.tf

**uat** (`terraform/envs/uat/main.tf`):
```hcl
module "observability" {
  source             = "../../modules/observability"
  env                = "uat"
  region             = var.region
  alarm_emails       = ["moalam@equinix.com"]   # add teammates here
  api_latency_p99_ms = 10000
}
```
**dev**: same block, `env = "dev"`. **prod** later: same, `env = "prod"`.

## 3. uat ONLY — import the existing resources (they were console-created)

Run each from `terraform/envs/uat`. Single-quote the addresses (zsh globs `[`/`"`).

```bash
cd ~/Arac-POC/terraform/envs/uat

# SNS topic (ARN as the import id)
terraform import 'module.observability.aws_sns_topic_policy.alarms' arn:aws:sns:us-east-1:019313283759:rag-uat-alarms

terraform import 'module.observability.aws_sns_topic_subscription.email["moalam@equinix.com"]' <SUBSCRIPTION_ARN>

terraform import 'module.observability.aws_cloudwatch_metric_alarm.lambda_errors' rag-uat-search-lambda-errors
terraform import 'module.observability.aws_cloudwatch_metric_alarm.lambda_throttles' rag-uat-search-lambda-throttles
terraform import 'module.observability.aws_cloudwatch_metric_alarm.api_5xx' rag-uat-api-5xx
terraform import 'module.observability.aws_cloudwatch_metric_alarm.api_latency_p99' rag-uat-api-latency-p99

terraform import 'module.observability.aws_cloudwatch_event_rule.ingest_task_failed' rag-uat-ingest-task-failed
terraform import 'module.observability.aws_cloudwatch_event_target.ingest_task_failed_sns' rag-uat-ingest-task-failed/1

terraform import 'module.observability.aws_cloudwatch_dashboard.health' rag-uat-health

# SNS topic policy (same ARN)
terraform import 'module.observability.aws_sns_topic_policy.alarms' \
  arn:aws:sns:us-east-1:019313283759:rag-uat-alarms

# Email subscription — for_each keyed by email. Import id is the subscription ARN.
#   get it: aws sns list-subscriptions-by-topic --topic-arn <topic> --query "Subscriptions[].SubscriptionArn"
terraform import 'module.observability.aws_sns_topic_subscription.email["moalam@equinix.com"]' \
  <SUBSCRIPTION_ARN>

# 4 metric alarms (import id = alarm name)
terraform import 'module.observability.aws_cloudwatch_metric_alarm.lambda_errors'    rag-uat-search-lambda-errors
terraform import 'module.observability.aws_cloudwatch_metric_alarm.lambda_throttles' rag-uat-search-lambda-throttles
terraform import 'module.observability.aws_cloudwatch_metric_alarm.api_5xx'          rag-uat-api-5xx
terraform import 'module.observability.aws_cloudwatch_metric_alarm.api_latency_p99'  rag-uat-api-latency-p99

# EventBridge rule (import id = rule name) + its SNS target (rule/target-id)
terraform import 'module.observability.aws_cloudwatch_event_rule.ingest_task_failed'        rag-uat-ingest-task-failed
terraform import 'module.observability.aws_cloudwatch_event_target.ingest_task_failed_sns'  rag-uat-ingest-task-failed/1

# Dashboard (import id = dashboard name)
terraform import 'module.observability.aws_cloudwatch_dashboard.health' rag-uat-health
```

## 4. Plan — the goal is NEAR-ZERO changes
```bash
terraform plan
```
Expect: mostly "no changes." Small in-place diffs are OK and expected if the
console defaults differ slightly from the module (e.g. an alarm description
string, `ok_actions` that the console omitted). **Confirm 0 to destroy.**

If a diff would DESTROY/recreate a resource, STOP — the module doesn't match
the imported resource; adjust the module to match, then re-plan. (Common cause:
a dimension or name mismatch. The whole point is adopt-not-recreate.)

Apply once the plan is clean:
```bash
terraform apply
```

## 5. dev / prod — no import, just apply
dev's alarms/dashboard were NOT console-created, so add the module block and:
```bash
cd ~/Arac-POC/terraform/envs/dev
terraform apply   # creates rag-dev-alarms, alarms, dashboard fresh
```
(Confirm + subscribe the dev alarm email after.)

---

## GOTCHAS
- **`ok_actions`**: the module sets `ok_actions` (notify on recovery too). The
  console alarms may NOT have this → plan shows an in-place add. Harmless; apply it.
- **Subscription import** needs the real subscription ARN (the confirmed one, not
  `PendingConfirmation`). Unconfirmed subs can't be imported.
- **API stage metrics** must stay enabled (separate from this module — it's a
  stage setting) or alarms 3/4 have no data.
- **Prod** should use a lower `api_latency_p99_ms` and probably a stricter error
  threshold; tune per env via the variable.
