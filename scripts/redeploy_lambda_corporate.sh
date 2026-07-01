#!/usr/bin/env bash
# CORPORATE deploy of the search Lambda → account 019313283759, rag-uat-*.
# Requires corporate creds exported as env vars BEFORE running.
# Builds the lean package identically to the personal script, but pushes ONLY
# to corporate-named resources, and HARD-ABORTS if not in the corporate account.
set -euo pipefail

# ---- corporate targets (all explicit; none overlap personal names) ----
FN=rag-uat-search
BUCKET=rag-uat-artifacts-019313283759
KEY=lambda/search-lambda.zip
REGION=us-east-1
EXPECTED_ACCOUNT=019313283759

# ---- SAFETY GUARD: refuse to run unless we're in the corporate account ----
ACCT=$(aws sts get-caller-identity --query Account --output text)
if [ "$ACCT" != "$EXPECTED_ACCOUNT" ]; then
  echo "ABORT: caller account is $ACCT, expected $EXPECTED_ACCOUNT (corporate)."
  echo "       Export corporate creds first. Refusing to deploy."
  exit 1
fi
echo ">> account confirmed: $ACCT (corporate). proceeding."

echo ">> [1/5] building lean package (glibc-pinned wheels)..."
rm -rf /tmp/lam-corp && mkdir -p /tmp/lam-corp/pkg
python3.11 -m pip install -r requirements-search.txt --target /tmp/lam-corp/pkg \
  --platform manylinux2014_x86_64 --python-version 3.11 --only-binary=:all: --upgrade -q

echo ">> [2/5] copying app code (search subset)..."
cp config.py config_dynamic.py search_app.py lambda_handler.py limiter.py secrets_loader.py /tmp/lam-corp/pkg/
cp -r pipeline cache guardrails /tmp/lam-corp/pkg/
mkdir -p /tmp/lam-corp/pkg/api/routes
cp api/*.py /tmp/lam-corp/pkg/api/ 2>/dev/null || true
cp -r api/routes /tmp/lam-corp/pkg/api/
rm -f /tmp/lam-corp/pkg/api/ingest.py
find /tmp/lam-corp/pkg -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

echo ">> [3/5] zipping + keeping rollback copy..."
( cd /tmp/lam-corp/pkg && zip -r9 -q /tmp/lam-corp/search-lambda.zip . )
aws s3 cp "s3://$BUCKET/$KEY" "s3://$BUCKET/lambda/search-lambda-PREV.zip" --region $REGION 2>/dev/null \
  || echo "   (no prior version to back up - first deploy)"

echo ">> [4/5] uploading to S3 ($BUCKET)..."
aws s3 cp /tmp/lam-corp/search-lambda.zip "s3://$BUCKET/$KEY" --region $REGION

echo ">> [5/5] updating function code ($FN)..."
aws lambda update-function-code --function-name "$FN" \
  --s3-bucket "$BUCKET" --s3-key "$KEY" --region $REGION >/dev/null \
  && echo ">> DONE. $FN updated in corporate." \
  || echo ">> NOTE: update failed — check the function exists + creds valid."
