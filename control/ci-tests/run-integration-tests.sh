#!/usr/bin/env bash
# run-integration-tests.sh — Run PANOSETI control integration tests in Docker
#
# Usage:
#   bash run-ci-tests/run-integration-tests.sh
#
# Note: integration tests live in control/tests/integration/ (Phase 4 — future)
# This script is a placeholder that currently reports no tests collected.

set -euo pipefail

TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$TEST_DIR/integration/docker-compose.integration.yml"

# ---------------------------------------------------------------------------
# THE FIX: Rigorous Lifecycle Management
# ---------------------------------------------------------------------------

# 1. TRAP: Guarantee teardown happens on SUCCESS, FAILURE, or CTRL+C
cleanup() {
    echo "--- Tearing down integration environment ---"
    # --volumes ensures mapped data volumes don't persist stale state between runs
    # --remove-orphans catches any stragglers
    docker compose -f "$COMPOSE_FILE" down --volumes --remove-orphans
    echo "--- Cleanup Complete ---"
}
# Attach the cleanup function to EXIT, SIGINT (Ctrl+C), and SIGTERM
trap cleanup EXIT INT TERM

# 2. PRE-CLEANUP: Kill any zombies currently holding the IP addresses
echo "--- Pre-emptively cleaning up any dangling containers ---"
docker compose -f "$COMPOSE_FILE" down --volumes --remove-orphans

# ---------------------------------------------------------------------------
# Step 1: Build Docker images
# ---------------------------------------------------------------------------
# Extract the --no-build flag if it exists, leave the rest of the arguments for Pytest
NO_BUILD=0
if [[ "${1:-}" == "--no-build" ]]; then
    NO_BUILD=1
    shift # Remove --no-build from the argument list
fi

if [ "$NO_BUILD" -eq 0 ]; then
    echo "--- Building integration test Docker images ---"
    docker compose -f "$COMPOSE_FILE" build
fi

# ---------------------------------------------------------------------------
# Step 2: Run integration tests
# ---------------------------------------------------------------------------
echo "--- Starting integration test environment & running tests ---"

# We use 'run' instead of 'up' for the test-runner. 
# This automatically boots dependencies (daqnode, gateway, redis, loki) in the background,
# runs the tests, and crucially allows us to pass Pytest arguments ("$@") through the CLI.
docker compose -f "$COMPOSE_FILE" run --rm \
    test-runner \
    pytest tests/integration/ -v --tb=short --color=yes "$@"

# No need for EXIT_CODE=$? or an explicit down command here.
# The 'trap' will automatically fire when the script finishes and handle teardown!
