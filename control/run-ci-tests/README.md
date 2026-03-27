# PANOSETI Control — Docker CI Test Runner

Convenience scripts to run the `control/` test suite in a reproducible Docker environment.

## Quick start

```bash
# From the control/ directory:
bash run-ci-tests/run-unit-tests.sh
```

Or, from anywhere in the repo:

```bash
cd panoseti-software/control
bash run-ci-tests/run-unit-tests.sh
```

## What gets started

| Service | Image | Purpose |
|---------|-------|---------|
| `redis` | `redis:7-alpine` | In-container Redis (for integration tests; unit tests use fakeredis) |
| `unit-tests` | Built from `Dockerfile.test` | Python 3.12 test runner |

## Running specific tests

```bash
# Run only pff tests
bash run-ci-tests/run-unit-tests.sh -k test_pff

# Run with verbose output and stop after first failure
bash run-ci-tests/run-unit-tests.sh -x -v

# Collect (list) all unit tests without running
bash run-ci-tests/run-unit-tests.sh --collect-only
```

## Local development (without Docker)

```bash
cd control
pip install -e ".[dev]"
pytest tests/unit/ -v --tb=short
```

## Test structure

```
control/tests/
├── unit/                     # No hardware required — run anywhere
│   ├── test_pydantic_models.py
│   ├── test_config_file.py
│   ├── test_global_validator.py
│   ├── test_pff.py
│   ├── test_util.py
│   ├── test_redis_utils.py
│   └── test_image_quantiles.py
└── integration/              # Phase 4 (future) — end-to-end session lifecycle
```

## Docker Compose profiles

The `docker-compose.yml` uses profiles to avoid starting unnecessary services:

```bash
# Unit tests only (starts Redis, unit-test runner)
docker compose -f run-ci-tests/docker-compose.yml --profile unit up

# Integration tests only
docker compose -f run-ci-tests/docker-compose.yml --profile integration up
```

## Requirements

- Docker Engine 24+
- Docker Compose v2 (`docker compose` — note: no hyphen)
- ~500 MB disk for the test image (Python 3.12 slim + deps)
