#!/usr/bin/env bash
# run-integration-tests.sh — Run PANOSETI control integration tests in Docker
#
# Usage:
#   bash run-ci-tests/run-integration-tests.sh
#
# Note: integration tests live in control/tests/integration/ (Phase 4 — future)
# This script is a placeholder that currently reports no tests collected.

set -euo pipefail
cd "$(dirname "$0")/.."  # always run from control/

echo "==> Building test image..."
docker compose -f run-ci-tests/docker-compose.yml build integration-tests

echo "==> Starting services..."
docker compose -f run-ci-tests/docker-compose.yml up -d redis

echo "==> Running integration tests..."
docker compose -f run-ci-tests/docker-compose.yml run --rm \
  integration-tests \
  pytest tests/integration/ -v --tb=short --color=yes "$@"

EXIT_CODE=$?

echo "==> Tearing down..."
docker compose -f run-ci-tests/docker-compose.yml down

exit $EXIT_CODE
