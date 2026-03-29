#!/usr/bin/env bash
# run.sh — Unified test runner for PANOSETI control.
#
# Usage:
#   bash ci-tests/run.sh unit [pytest args...]
#   bash ci-tests/run.sh integration [pytest args...]
#
# Environment variables:
#   ENABLE_TELEMETRY_TESTS  — set to 1 to run Loki/Redis telemetry tests
#                             (skips storeloki scale-down optimisation)

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
    COMPOSE_FILE="$SCRIPT_DIR/docker-compose.unit.yml"
    SERVICE_NAME="unit-test-runner"
    # Isolate state by prefixing the docker project
    export COMPOSE_PROJECT_NAME="ctl-unit"
else
    COMPOSE_FILE="$SCRIPT_DIR/docker-compose.integration.yml"
    SERVICE_NAME="integration-test-runner"
    # Isolate state by prefixing the docker project
    export COMPOSE_PROJECT_NAME="ctl-int"
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
DOCKER_BUILDKIT=1 docker compose -f "$COMPOSE_FILE" build

echo "--- Starting $SUITE test environment & running tests ---"
if [[ "$SUITE" == "unit" ]]; then
    # -n auto: run unit tests in parallel across all available CPUs (pytest-xdist)
    docker compose -f "$COMPOSE_FILE" run --rm \
        "$SERVICE_NAME" \
        pytest "ci-tests/$SUITE/" -v --tb=short --color=yes --timeout=60 "$@"
else
    # Integration: 60-second per-test timeout to catch hangs
    # When telemetry tests are disabled, scale storeloki to 0 to save startup time
    if [[ -z "${ENABLE_TELEMETRY_TESTS:-}" ]]; then
        #SCALE_FLAGS="--scale storeloki=0"
        SCALE_FLAGS=""
    else
        SCALE_FLAGS=""
    fi
    docker compose -f "$COMPOSE_FILE" run --rm $SCALE_FLAGS \
        "$SERVICE_NAME" \
        pytest "ci-tests/$SUITE/" -v --tb=short --color=yes --timeout=15 "$@"
fi
