# PANOSETI Control — CI Test Suite

All tests live under `control/ci/` and run inside Docker via a single multi-stage `Dockerfile.ci` using `uv` for high-performance builds.

**Current status:** 524 unit tests passing · 109 integration tests passing · 98 chaos tests passing
 
---

## Quick Start

The PANOSETI QA runner (`pseti test`) manages isolated Docker environments for different test suites. Setup and teardown are automated per-command.

```bash
# 1. Run suites (automated setup/teardown)
pseti test sw unit        # Parallel unit tests (no Docker)
pseti test sw integration # E2E integration tests (Docker)
pseti test sw chaos       # Chaos/TDD scenarios (Docker)
pseti test lint           # Ruff & MyPy (concurrent)

# 2. Targeted debugging (bypass teardown to inspect containers)
pseti test sw integration --debug -k test_transfer_daemon_archives_run
# Now you can: docker exec -it pseti-integration-int-tester bash

# 3. Global Cleanup (if containers are left running by --debug)
pseti test sw cleanup
```

---

## 🛠️ Testing Principles

### State Isolation (Mandatory)
Integration tests must use `PSETI_STATE` and `PSETI_CONTROL` redirected to `tmp_path` to avoid collisions on shared ledgers and locks.
```python
def test_my_feature(tmp_path, monkeypatch):
    monkeypatch.setenv("PSETI_STATE", str(tmp_path))
    PanoPaths.ensure_state_dirs()
    # Now all state is isolated to this test run
```

### Async Mocking
When mocking `asyncio` logic, never patch the `asyncio` module directly. Instead, patch specific functions like `subprocess.run` or use `AsyncMock` for `asyncio.sleep`.

---

## 📋 Test Hierarchy

| Tier | Directory | Purpose |
|---|---|---|
| **Tier 1 (Unit)** | `ci/unit/` | Zero-dependency logic, parsing, and math. |
| **Tier 2 (Logic)** | `ci/integration/transfer/` | Distributed control logic using mocked gRPC but real filesystem state. |
| **Tier 3 (Fleet)** | `ci/integration/` | Full E2E with Docker containers, real gRPC servers, and shared volumes. |

### Chaos Suite (`ci/integration/scenarios/`)
These tests use `mock-quabo` to simulate hardware failures, network latency, and gRPC timeouts. They are the "TDD source of truth" for the control plane.

---

## Local Development (without Docker)

```bash
# Sync environment
uv sync --all-extras

# Run unit tests natively
uv run pytest ci/unit/

# Run specific integration tests natively (those tagged not skip_outside_ci)
uv run pytest ci/integration/transfer/test_transfer_chaos.py
```
