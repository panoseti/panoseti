#!/usr/bin/env bash
# run-integration-tests.sh — Build and run the PANOSETI integration test suite.
#
# Usage:
#   bash control/run-ci-tests/integration/run-integration-tests.sh [--no-build]
#
# Environment variables:
#   DAQNODE_DIRECT_HOST   — override direct daqnode IP (default: 192.168.0.10)
#   DAQNODE_GATEWAY_HOST  — override gateway IP (default: 10.0.1.1)
#   GRPC_PORT             — override gRPC port (default: 50051)
#   NO_BUILD              — skip Docker image builds (set to any non-empty value)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.integration.yml"

# ---------------------------------------------------------------------------
# Step 1: Build Docker images
# ---------------------------------------------------------------------------
if [ -z "${NO_BUILD:-}" ]; then
    echo "--- Building integration test Docker images ---"
    docker compose -f "$COMPOSE_FILE" build
fi

# ---------------------------------------------------------------------------
# Step 2: Run integration tests
# ---------------------------------------------------------------------------
echo "--- Starting integration test environment ---"
docker compose -f "$COMPOSE_FILE" up \
    --exit-code-from test-runner \
    --abort-on-container-exit \
    --attach test-runner
EXIT_CODE=$?

# ---------------------------------------------------------------------------
# Step 3: Tear down
# ---------------------------------------------------------------------------
echo "--- Tearing down integration environment ---"
docker compose -f "$COMPOSE_FILE" down --volumes --remove-orphans

if [ $EXIT_CODE -eq 0 ]; then
    echo "--- Integration tests PASSED ---"
else
    echo "--- Integration tests FAILED (exit code: $EXIT_CODE) ---"
    exit $EXIT_CODE
fi
