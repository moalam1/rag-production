# RAG Platform — Terraform

Infrastructure-as-code for the Equinix RAG platform. Region: **us-east-1**
(locked by Bedrock Cohere Rerank + Guardrails availability). Account model:
**standalone** (UAT + prod separated by name-prefix/workspace within one account).

## Layout
```
bootstrap/          run FIRST, once, as admin. Creates state backend + deploy role.
modules/
  foundation/       DynamoDB, S3, Secrets Manager   (build order 1)
  ingestion/        ECR, ECS, IAM roles, EventBridge (build order 2)
  search/           Lambda, API Gateway, WAF, Bedrock perms (build order 3)
envs/
  uat/              composes the modules for UAT
  prod/             composes the modules for prod
```

## Order of operations

### Step 0 — Bootstrap (once, as ADMIN)
```bash
cd bootstrap
cp terraform.tfvars.example terraform.tfvars
# edit terraform.tfvars → set deploy_role_trusted_principals to your admin ARN
terraform init
terraform apply
# note the 3 outputs: state_bucket, lock_table, deploy_role_arn
```

### Step 1+ — Everything else (as the DEPLOY ROLE)
Each env's `backend.tf` points at the state bucket from bootstrap, and its
provider assumes `deploy_role_arn`. Then:
```bash
cd envs/uat
terraform init
terraform plan
terraform apply        # build UAT first, prove it
# then envs/prod once UAT is validated
```

## Principles
- **Deploy role, not personal admin** — Terraform runs AS rag-terraform-deploy
  (scoped blast radius, reusable by CI, auditable). Admin only bootstraps it.
- **UAT first → prove → prod.** Same modules, different env composition + prefix.
- **Name-prefix + tags for isolation** — rag-{uat,prod}-* everywhere, since
  standalone means UAT/prod share an account boundary.
- **Validated shapes only** — the IAM policies, networking, and env vars here
  come from the prototype that ran in AWS, not from guesses.

## Status
- [x] bootstrap (state backend + deploy role)
- [x] modules/foundation
- [ ] modules/ingestion
- [ ] modules/search
- [ ] envs/uat
- [ ] envs/prod
