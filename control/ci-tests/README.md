# PANOSETI Control — Docker CI Test Runner

Convenience scripts to run the `control/` test suite in a reproducible Docker environment.

## Quick start

```bash
# From the control/ directory:
bash ci-tests/run.sh
```

Or, from anywhere in the repo:

```bash
cd panoseti-software/control
bash ci-tests/run.sh
```

## What gets started

| Service | Image | Purpose |
|---------|-------|---------|
| `redis` | `redis:7-alpine` | In-container Redis (for integration tests; unit tests use fakeredis) |
| `unit-tests` | Built from `Dockerfile.test` | Python 3.14 test runner |

## Running specific tests

```bash
# Run only pff tests
bash ci-tests/run.sh -k test_pff

# Run with verbose output and stop after first failure
bash ci-tests/run.sh -x -v

# Collect (list) all unit tests without running
bash ci-tests/run.sh --collect-only
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
docker compose -f ci-tests/docker-compose.unit.yml

# Integration tests only
docker compose -f ci-tests/docker-compose.integration.yml
```

## Requirements

- Docker Engine 24+
- Docker Compose v2 (`docker compose` — note: no hyphen)
- ~500 MB disk for the test image (Python 3.14 slim + deps)
