#!/usr/bin/env bash
# Rebuild + redeploy the search Lambda. Run after merging a search-path change to main.
# Builds the LEAN package (search subset + glibc pins), pushes to S3, updates the function.
set -euo pipefail
FN="${LAMBDA_FN:-rag-search}"          # override: LAMBDA_FN=othername ./scripts/redeploy_lambda.sh
BUCKET=rag-artifacts-s3
KEY=lambda/search-lambda.zip
REGION=us-east-1

echo ">> [1/5] building lean package (glibc-pinned wheels)..."
rm -rf /tmp/lam && mkdir -p /tmp/lam/pkg
python3.11 -m pip install -r requirements-search.txt --target /tmp/lam/pkg \
  --platform manylinux2014_x86_64 --python-version 3.11 --only-binary=:all: --upgrade -q

echo ">> [2/5] copying app code (search subset, excludes api/ingest.py)..."
cp config.py config_dynamic.py search_app.py lambda_handler.py limiter.py /tmp/lam/pkg/
cp -r pipeline cache guardrails /tmp/lam/pkg/
mkdir -p /tmp/lam/pkg/api/routes
cp api/*.py /tmp/lam/pkg/api/ 2>/dev/null || true
cp -r api/routes /tmp/lam/pkg/api/
rm -f /tmp/lam/pkg/api/ingest.py
find /tmp/lam/pkg -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

echo ">> [3/5] zipping + keeping rollback copy (search-lambda-PREV.zip)..."
( cd /tmp/lam/pkg && zip -r9 -q /tmp/lam/search-lambda.zip . )
aws s3 cp "s3://$BUCKET/$KEY" "s3://$BUCKET/lambda/search-lambda-PREV.zip" --region $REGION 2>/dev/null \
  || echo "   (no prior version to back up - first deploy)"

echo ">> [4/5] uploading to S3..."
aws s3 cp /tmp/lam/search-lambda.zip "s3://$BUCKET/$KEY" --region $REGION

echo ">> [5/5] updating function code ($FN)..."
aws lambda update-function-code --function-name "$FN" \
  --s3-bucket "$BUCKET" --s3-key "$KEY" --region $REGION >/dev/null 2>&1 \
  && echo ">> DONE. $FN updated." \
  || echo ">> NOTE: function '$FN' not found/updatable yet (deploy the keeper Lambda first, or set LAMBDA_FN). Package IS in S3."

echo ">> Verify: curl -s -X POST <FUNCTION_URL>/api/v1/search -H \"X-API-Key: \$API_KEY\" -H 'Content-Type: application/json' -d '{\"query\":\"what is equinix fabric\"}'"
echo ">> Rollback: copy search-lambda-PREV.zip back to $KEY and re-run update-function-code (see MAINTENANCE.md section 1)"
