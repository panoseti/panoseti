# PANOSETI Control — CI Test Suite

All tests live under `control/ci/` and run inside Docker via a single multi-stage `Dockerfile.ci` using `uv` for high-performance builds.

**Current status:** 524 unit tests passing · integration suite: 65 tests passing

---

## Quick Start

The PANOSETI QA runner (`pseti test`) manages isolated Docker environments for different test suites. Setup and teardown are automated per-command.

```bash
# 1. Run suites (automated setup/teardown)
pseti test sw unit        # Parallel unit tests
pseti test sw integration # E2E integration tests
pseti test sw chaos       # Chaos/TDD-forcing scenarios
pseti test lint           # Ruff & MyPy (concurrent)

# 2. Targeted debugging (bypass teardown to inspect containers)
pseti test sw integration --debug -k TestDaqLifecycle
# Now you can: docker exec -it pseti-integration-int-tester bash

# 3. Global Cleanup (if containers are left running by --debug)
pseti test sw cleanup

# 4. Image management
pseti test sw build       # Rebuild images (uv cached)
```

---

## Test Runner Architecture (`qa.toml`)

The test runner is driven by a modular TOML configuration. Each suite defines its own environment, allowing for **concurrent execution** in isolated Docker projects.

### Defining a New Suite
To add a new test suite, add an entry to `control/ci/qa.toml`:

```toml
[suites.my_new_suite]
description = "My new feature tests"
type = "test"
requires_docker = true
compose_file = "ci/docker-compose.integration.yml"
service = "int-tester"
test_dir = "ci/my_feature/"
env = { DEBUG = "1" }
```

The runner automatically injects `COMPOSE_PROJECT_NAME=pseti-my_new_suite` to ensure network and container isolation.

---

## Unit Test Modules

Unit tests (`ci/unit/`) are hardware-agnostic and require no external services. All 524 pass natively with `uv run pytest ci/unit/`.

| File | Tests | Coverage |
|---|---|---|
| `test_run_state.py` | ~40 | RunStateManager, advisory lock, ledger TOML I/O |
| `test_run_state_extended.py` | 29 | Extended ledger statuses (17 total), new NodeReceipt fields |
| `test_transfer_queue.py` | 11 | TransferQueue enqueue/claim/complete/fail, idempotency |
| `test_stop_fast_path.py` | 4 | StopTransaction enqueues a job; ledger → RECORDING_ENDED in < 5 s |
| `test_transfer_daemon.py` | 19 | _process_job state machine, lock helpers, verify_manifest |
| `test_pydantic_config_models.py` | ~60 | Pydantic schema validation for all models |
| `test_config_file.py` | ~50 | Config loading, range expansion, cross-config validation |
| `test_pff.py` | ~40 | PFF file format parsing |
| `test_global_validator.py` | ~80 | Cross-config consistency rules |
| `test_file_xfer.py` | ~20 | SSH/rsync helpers |
| `test_transaction_*.py` | ~100 | StartTransaction and StopTransaction rollback ladders |
| `test_chaos_*.py` | ~70 | Isolated chaos unit cases |

---

## Integration Test Files

Integration tests (`ci/integration/`) require the Docker stack.

### Transfer Daemon Tests (`test_transfer_daemon_e2e.py`)

8 tests covering the decoupled transfer pipeline:

**In-process (run natively, no Docker):**
- `test_transfer_daemon_unit_integration` — enqueue job → `_process_job()` → `run_complete` written; uses mocked gRPC + rsync
- `test_transfer_queue_enqueue_then_process` — full `TransferQueue` lifecycle: `enqueue → claim → _process_job → complete`; verifies job lands in `completed/`
- `test_transfer_daemon_no_collect_integration` — `no_collect=True` skips rsync; `_process_job` still reaches `ARCHIVED`

**Docker CI only (skip outside Docker):**
- `test_transfer_daemon_archives_run` — full E2E with real hashpipe + daemon
- `test_transfer_daemon_resumes_after_crash` — kill daemon mid-rsync; restart completes the job
- `test_transfer_daemon_retry_on_transient_rsync_failure` — rsync fails twice, succeeds on third attempt
- `test_transfer_daemon_marks_failed_after_max_attempts` — exhausts MAX_ATTEMPTS; job moves to `failed/`
- `test_transfer_daemon_singleton_lock_in_container` — second daemon exits immediately; first keeps processing

### Other Key Integration Files

| File | What it tests |
|---|---|
| `test_daq_lifecycle.py` | Full start/stop cycle with real hashpipe |
| `test_concurrent_daq_operations.py` | Race conditions in StartDaq/StopDaq |
| `test_two_node_direct.py` | Two-DAQ-node topology |
| `test_data_collection.py` | rsync collection helpers |
| `test_loki_pipeline.py` | Log pipeline Redis→Loki |
| `test_gateway_topology.py` | VPN gateway socat bridge |
| `scenarios/` | 114 chaos/TDD scenarios for transaction integrity |

---

## Modern CI Architecture

We have migrated to **Python 3.14** and **uv**. Our containers are designed to be "inner-loop" friendly:

*   **Persistent Projects:** Each suite runs in its own Docker project (`pseti-unit`, `pseti-integration`, etc.).
*   **Live Mounting:** The local `control/` directory is volume-mounted into `/app`. Edits you make on your host are instantly available for the next test run.
*   **Venv Isolation:** Python dependencies live in `/opt/venv`, safely isolated from your local volume mounts.
*   **Blazing Fast Builds:** We use BuildKit cache mounts (`--mount=type=cache,target=/root/.cache/uv`) and `uv sync` layers to ensure dependencies are only re-evaluated when `pyproject.toml` or `uv.lock` changes.

### Integration Topology (Fleet Testing)

The integration suite simulates a Palomar-like VPN topology. We use two distinct topologies depending on whether we are testing E2E high-throughput data or distributed control logic.

#### 1. Loopback Data Path (E2E Integration)
Used in `real_data` tests to verify the high-throughput pipeline (`tcpreplay` -> `hashpipe`).
- **Isolation:** Each DAQ node runs its own **dedicated local `tcpreplay` instance** inside its container.
- **Path:** Science packets never leave the container; they flow through the local `lo` (loopback) interface to bypass MTU and MAC filtering overhead.

#### 2. Distributed Control Path (Chaos/Logic)
Used in chaos scenarios to verify `start.py`/`stop.py` transaction integrity and rollback logic.
- **Shared Service:** A single **`mock-quabo` service** simulates a full 4-quabo module (e.g. Module 200).
- **Path:** Command packets flow from `start.py` (Headnode) to `mock-quabo` (External), and telemetry flows back up. Science packets can be triggered to any node in the fleet via the external `eth0` network.

---

## Local Development (without Docker)

```bash
# Sync environment
uv sync --all-extras

# Run unit tests (all 524, no Docker required)
uv run pytest ci/unit/

# Run in-process integration tests natively
uv run pytest ci/integration/test_transfer_daemon_e2e.py -k "not skip_outside_ci"
```
