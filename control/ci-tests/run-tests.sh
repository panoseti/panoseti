#!/usr/bin/env bash
# run-tests.sh — Unified test runner for PANOSETI control.
#
# Usage:
#   bash run-ci-tests/run-tests.sh unit [pytest args...]
#   bash run-ci-tests/run-tests.sh integration [pytest args...]

set -euo pipefail

SUITE="${1:-}"
if [[ "$SUITE" != "unit" && "$SUITE" != "integration" ]]; then
    echo "Error: You must specify 'unit' or 'integration'."
    echo "Usage: $0 <unit|integration> [pytest args...]"
    exit 1
fi
shift # remove 'unit' or 'integration' from arguments so we can pass "$@" to pytest

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. Setup Suite-Specific Variables
if [[ "$SUITE" == "unit" ]]; then
    COMPOSE_FILE="$SCRIPT_DIR/docker-compose-unit.yml"
    SERVICE_NAME="unit-tests"
    # Isolate state by prefixing the docker project
    export COMPOSE_PROJECT_NAME="panoseti_ci_unit" 
else
    COMPOSE_FILE="$SCRIPT_DIR/docker-compose.integration.yml"
    SERVICE_NAME="test-runner"
    # Isolate state by prefixing the docker project
    export COMPOSE_PROJECT_NAME="panoseti_ci_integration"
fi

# 2. Rigorous Teardown Management
cleanup() {

    echo "--- Tearing down $SUITE environment ---"
    docker compose -f "$COMPOSE_FILE" down --volumes --remove-orphans
    echo "--- $SUITE Cleanup Complete ---"
}
trap cleanup EXIT INT TERM

# Pre-emptive cleanup (scoped ONLY to this suite's project name!)
echo "--- Pre-emptively cleaning up any dangling $SUITE containers ---"
docker compose -f "$COMPOSE_FILE" down --volumes --remove-orphans

# 3. Build & Run
echo "--- Building $SUITE test Docker images ---"
docker compose -f "$COMPOSE_FILE" build "$SERVICE_NAME"

echo "--- Starting $SUITE test environment & running tests ---"
docker compose -f "$COMPOSE_FILE" run --rm \
    "$SERVICE_NAME" \
    pytest "tests/$SUITE/" -v --tb=short --color=yes "$@"