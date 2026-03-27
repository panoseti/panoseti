#!/usr/bin/env bash
# run-unit-tests.sh — Run PANOSETI control unit tests in Docker
#
# Usage:
#   bash run-ci-tests/run-unit-tests.sh              # run all unit tests
#   bash run-ci-tests/run-unit-tests.sh -k test_pff  # filter by name
#   bash run-ci-tests/run-unit-tests.sh --co         # collect-only (list tests)
#
# Prerequisites: Docker + Docker Compose v2

set -euo pipefail
cd "$(dirname "$0")/.."  # always run from control/

echo "==> Building test image..."
docker compose -f run-ci-tests/docker-compose.yml build unit-tests

echo "==> Starting Redis..."
docker compose -f run-ci-tests/docker-compose.yml up -d redis

echo "==> Running unit tests..."
docker compose -f run-ci-tests/docker-compose.yml run --rm \
  unit-tests \
  pytest tests/unit/ -v --tb=short --color=yes "$@"

EXIT_CODE=$?

echo "==> Tearing down..."
docker compose -f run-ci-tests/docker-compose.yml down

exit $EXIT_CODE
