#!/usr/bin/env bash
# Rebuild + push the ingestion Fargate image. Run after merging an ingestion change to main.
# The next Fargate task picks up :latest automatically (no task-def change needed).
set -euo pipefail
ACCOUNT=141927126501
REGION=us-east-1
ECR=$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/rag-ingest

echo ">> [1/4] building image..."
docker build -f Dockerfile.ingest -t rag-ingest:local .

SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "manual")
echo ">> [2/4] tagging (latest + $SHA for rollback)..."
docker tag rag-ingest:local "$ECR:latest"
docker tag rag-ingest:local "$ECR:$SHA"

echo ">> [3/4] login to ECR..."
aws ecr get-login-password --region $REGION \
  | docker login --username AWS --password-stdin "$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"

echo ">> [4/4] pushing..."
docker push "$ECR:latest"
docker push "$ECR:$SHA"

echo ""
echo ">> DONE. Pushed :latest and :$SHA"
echo ">> Test: ECS console -> rag-cluster -> Run task -> rag-ingest-task"
echo ">>       (public subnet + Public IP ON), env INGEST_SECTION=customer-success INGEST_LIMIT=2 REBUILD_BM25=false"
echo ">> Rollback: re-push a prior SHA as :latest (see MAINTENANCE.md section 2)"
