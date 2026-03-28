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
#   GRPC_SRC              — path to panoseti_grpc source (default: auto-detected sibling dir)
#   NO_BUILD              — skip Docker image builds (set to any non-empty value)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.integration.yml"
WHEELS_DIR="$SCRIPT_DIR/wheels"

# ---------------------------------------------------------------------------
# Step 1: Locate and build the panoseti_grpc wheel (enables offline Docker builds)
# ---------------------------------------------------------------------------
# Look for panoseti_grpc as a sibling of panoseti-software/
PANOSETI_DIR="$(realpath "$SCRIPT_DIR/../../../..")"  # panoseti/ parent
GRPC_SRC="${GRPC_SRC:-$PANOSETI_DIR/panoseti_grpc}"

mkdir -p "$WHEELS_DIR"

if [ -z "${NO_BUILD:-}" ]; then
    if [ -d "$GRPC_SRC" ]; then
        echo "--- Building panoseti_grpc wheel from $GRPC_SRC ---"
        pip wheel --no-deps --no-build-isolation -w "$WHEELS_DIR" "$GRPC_SRC" \
            || echo "Warning: wheel build failed — Docker will fall back to PyPI"
    else
        echo "Note: panoseti_grpc source not found at $GRPC_SRC"
        echo "      Docker images will install from PyPI (panoseti-grpc>=0.3.5)"
    fi
fi

# ---------------------------------------------------------------------------
# Step 2: Build Docker images
# ---------------------------------------------------------------------------
if [ -z "${NO_BUILD:-}" ]; then
    echo "--- Building integration test Docker images ---"
    docker compose -f "$COMPOSE_FILE" build
fi

# ---------------------------------------------------------------------------
# Step 3: Run integration tests
# ---------------------------------------------------------------------------
echo "--- Starting integration test environment ---"
docker compose -f "$COMPOSE_FILE" up \
    --exit-code-from test-runner \
    --abort-on-container-exit \
    --attach test-runner
EXIT_CODE=$?

# ---------------------------------------------------------------------------
# Step 4: Tear down
# ---------------------------------------------------------------------------
echo "--- Tearing down integration environment ---"
docker compose -f "$COMPOSE_FILE" down --volumes --remove-orphans

if [ $EXIT_CODE -eq 0 ]; then
    echo "--- Integration tests PASSED ---"
else
    echo "--- Integration tests FAILED (exit code: $EXIT_CODE) ---"
    exit $EXIT_CODE
fi
