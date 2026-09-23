#!/usr/bin/env bash
# Поднимает MLflow Tracking Server и Model Registry:
# метаданные пишутся в PostgreSQL, артефакты - в бакет S3.
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

set -a
source .env
set +a

export MLFLOW_S3_ENDPOINT_URL="https://storage.yandexcloud.net"

# в пароле встречаются спецсимволы, поэтому экранируем его перед подстановкой в URI
DB_PASSWORD_ENC=$(python3 -c "import os, urllib.parse; print(urllib.parse.quote_plus(os.environ['DB_DESTINATION_PASSWORD']))")
BACKEND_URI="postgresql://${DB_DESTINATION_USER}:${DB_PASSWORD_ENC}@${DB_DESTINATION_HOST}:${DB_DESTINATION_PORT}/${DB_DESTINATION_NAME}"

mlflow server \
    --backend-store-uri "$BACKEND_URI" \
    --registry-store-uri "$BACKEND_URI" \
    --default-artifact-root "s3://${S3_BUCKET_NAME}" \
    --host 0.0.0.0 \
    --port 5000
