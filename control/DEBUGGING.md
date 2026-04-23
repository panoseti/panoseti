# Debugging Guide — PANOSETI Control Plane

This document captures hard-won debugging strategies for the control plane CI stack.
It is not a tutorial — it assumes familiarity with the architecture described in `CLAUDE.md`.

---

## 1. State Leak Identification

State leaks are the most common cause of test failures that reproduce intermittently or only in full-suite runs.

### Advisory lock stale on entry

The control plane uses a **Context Manager Architecture** for locking.
- **Lock Implementation**: Uses low-level `os.open` with `O_CREAT | os.O_EXCL` on `tmp/panoseti_control.lock`.
- **Race Condition Guard**: Atomic file creation prevents two processes from acquiring the same lock simultaneously.
- **Self-Healing**: If lock acquisition fails, the system reads the PID inside the lock file. If that process is not alive on the system, it automatically clears the stale lock and proceeds (SC-015/SC-021).

**Debugging orphaned locks**:
```bash
# Check if the lock file exists
ls -l tmp/panoseti_control.lock

# See which PID claims the lock
cat tmp/panoseti_control.lock

# Verify if that process is still running
ps -p $(cat tmp/panoseti_control.lock)
```

### Ledger left in ACTIVE state

`tmp/run_state.toml` persisting between tests causes `pseti start` to refuse with "A run is already in progress."
- **StartTransaction** and **StopTransaction** manage the status lifecycle.
- Inspect the ledger for status leaks:
```bash
cat tmp/run_state.toml | grep status
```

**Statuses that block a new start**: `STARTING`, `ACTIVE`, `STOPPING`, `RECORDING_ENDED`. Any of these means the prior run has not been fully torn down.

### Hashpipe process left running after a test

The DAQ Control server tracks only the PID it spawned. If a test fails mid-run and rollback is bypassed (e.g., violent container restart), kill it manually:
```bash
# On the daqnode container
pkill -KILL hashpipe
```

---

## 2. Transfer Queue Debugging

### Inspecting queue state

```bash
# What's waiting to be transferred?
ls -la tmp/transfer_queue/pending/

# What is the daemon actively processing?
ls -la tmp/transfer_queue/active/

# What failed?
ls -la tmp/transfer_queue/failed/
cat tmp/transfer_queue/failed/*.job.toml

# What completed successfully?
ls -la tmp/transfer_queue/completed/
```

Each `.job.toml` contains `run_name`, `head_data_dir`, `daq_nodes`, `created_at`, and `attempts`. A job in `failed/` has `attempts >= MAX_ATTEMPTS (3)`.

### Daemon singleton lock

The Transfer Daemon holds `tmp/panoseti_transfer.lock` using `fcntl.LOCK_EX | LOCK_NB`. The lock is kernel-managed and releases automatically when the daemon process exits.

```bash
# Check which PID holds the transfer lock
cat tmp/panoseti_transfer.lock

# Verify that process is alive
ps -p $(cat tmp/panoseti_transfer.lock)
```

If the daemon has crashed but the lock file exists (the kernel already released the flock), a fresh daemon start will succeed immediately — `flock` contention is only live-process contention, not file-existence contention.

### Manually re-enqueuing a failed job

If a job lands in `failed/` due to a transient error (network outage during rsync), move it back to `pending/` manually:

```bash
mv tmp/transfer_queue/failed/myrun.pffd.job.toml tmp/transfer_queue/pending/
```

Then restart the daemon, or wait for the next poll cycle if it's already running.

### run_complete marker missing

If the ledger shows `ARCHIVED` but the `run_complete` file is absent, the daemon may have crashed between writing the ledger and writing the marker. The marker write is the last step — re-enqueue the job and the daemon will write it idempotently (it checks for existence before overwriting).

---

## 3. Container Log Inspection

### Loki Log Pipeline Debugging

All logs from distributed nodes flow through a gRPC pipeline to the Headnode.
- **Pipeline**: `Node Logger` -> `gRPC` -> `Telemetry Service` -> `Redis (logs:ingress)` -> `storeLoki.py` -> `Grafana Loki`.

**Inspect Redis ingress queue**:
```bash
# Check queue length (should be near 0 if storeLoki is keeping up)
docker exec ctl-int-redis-1 redis-cli LLEN logs:ingress

# Inspect latest log entries (raw JSON)
docker exec ctl-int-redis-1 redis-cli LRANGE logs:ingress 0 5
```

**Robustness Features**:
- `storeLoki.py` handles non-UTF8 input and surrogate pairs by using `errors='replace'` during decoding.
- Large log payloads (100KB+) are handled via batch size optimization (reduced to 10 entries) to ensure pipeline responsiveness.

### DAQ Control server internal log

The server writes to `/var/log/panoseti/daq_control_server.log` inside the daqnode container.

---

## 4. Configuration Validation Debugging

### CI Validation Leniency
The control plane enforces strict pre-flight validation. In CI environments, we allow leniency for missing hardware/firmware if `head_node_container: true` is set in `daq_config.json`.
- If a test fails with "Pre-flight configuration validation failed," ensure the test's `DaqConfig` instance has `head_node_container=True`.

---

## 5. Test Infrastructure Gotchas

### Integration conftest `create_data_dirs` fixture

`ci/integration/conftest.py` has an `autouse=True` session-scoped fixture that tries to create `/data/head` and `/data/daq`. These paths only exist inside Docker CI. Outside Docker, the fixture now catches the `OSError` silently so in-process integration tests (like the transfer daemon tests) can run natively.

**Symptom**: Tests in `ci/integration/` fail at setup with `OSError: [Errno 30] Read-only file system: '/data'` when run outside Docker.

**Fix**: The fixture guards with `try/except OSError: pass`. If you see this error in a new test, check whether it was added inside `ci/integration/` but should be in `ci/unit/` instead — in-process tests with no Docker dependency belong in the unit folder.

### Patching daemon gRPC imports

`utils/transfer/daemon.py` imports `DaqControlClient` **inside** `_process_job()` (not at module level) to allow the module to load without `panoseti_grpc` installed. This means `patch("utils.transfer.daemon.DaqControlClient", ...)` will fail with `AttributeError`.

**Correct approach**: Inject fake modules via `sys.modules` before calling `_process_job()`:
```python
from types import ModuleType
from unittest.mock import MagicMock

stub = ModuleType("panoseti_grpc.daq_control.client")
stub.DaqControlClient = MagicMock(return_value=mock_client)
sys.modules["panoseti_grpc.daq_control.client"] = stub
# ... run _process_job ...
# restore sys.modules afterwards
```
See `ci/unit/test_transfer_daemon.py::_mock_grpc_modules` for the complete reusable context manager.

### `grpc_error_handler` on async generators

The `grpc_error_handler` decorator in `panoseti_grpc/util/error_handling.py` has special handling for async generator functions (`GetManifest` is server-streaming). It uses `inspect.isasyncgenfunction` to detect generators and wraps them in an `agen_wrapper` that yields items. If you add a new server-streaming RPC, verify the decorator handles it by checking `inspect.isasyncgenfunction(your_handler)`.

---

## 6. Advanced Insights

### asyncio TaskGroup and Concurrent RPCs
`pseti start` uses `asyncio.TaskGroup` for fail-fast concurrency.
- **Fail-Fast**: If one `StartDaq` RPC fails, all others are immediately cancelled.
- **Atomic Rollback**: The `StartTransaction.__aexit__` re-loads the ledger to identify which nodes received a receipt before cancellation and issues `StopDaq` to them in the correct priority order.

### Write Ahead Logging (WAL) Pattern
Always write the node receipt to `run_state.toml` **before** issuing the gRPC call. This ensures that if the process is killed during the RPC, the rollback ladder knows the node was "attempted" and can clean it up.

### TransferQueue idempotency
`TransferQueue.enqueue()` checks all four subdirs (`pending/`, `active/`, `completed/`, `failed/`) before writing. Double-enqueueing the same run name is a no-op. This is intentional — a crashed `pseti stop` that re-runs will not create a duplicate job.
