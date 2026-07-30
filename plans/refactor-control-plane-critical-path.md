# PANOSETI Control Plane Refactoring Plan - Phase 3: Transactional Refactor

**Objective**: Refactor `start.py`, `stop.py`, and create necessary state-tracking utilities in `control/utils/` to implement a strict transactional "Rollback Ladder" for startup and a "Best-Effort Shutdown" for teardown that satisfy the Chaos test suite scenarios.

## 1. Reliable Advisory Locking
- Implement `fcntl.flock` advisory locking using `control/tmp/panoseti_control.lock`.
- Ensure the `tmp/` directory is created automatically if it does not exist.
- Both `start.py` and `stop.py` must acquire this lock to prevent concurrent executions (SC-024) and start/stop race conditions.

## 2. Distributed State Ledger (`run_state.toml`)
- Replace the legacy `current_run` text file with a robust TOML ledger at `control/tmp/run_state.toml`.
- The ledger will record:
  - Intended `run_name`
  - `start_time` (ISO 8601)
  - Target configuration metadata
  - Status (e.g., `STARTING`, `ACTIVE`, `ABORTED`, `STOPPING`, `COMPLETED`)
  - Node-specific gRPC receipts (remote Hashpipe PIDs, remote data directories)

## 3. Transactional `start.py` (Rollback Ladder)
- **Concurrent gRPC Receipts**: `start.py` will invoke `StartDaq` concurrently across DAQ nodes, await and parse the responses. Successful receipts will be appended to the ledger.
- **Liveness Probes**: Pause for 2 seconds post-`StartDaq` and invoke `StatusDaq`. Explicitly verify that Hashpipe is `ALIVE` (SC-005).
- **Rollback Trigger**: If *any* node fails to return a valid receipt, times out (SC-001, SC-002, SC-069), or fails the liveness probe, the rollback ladder is triggered.
- **Rollback Execution**:
  - Wrap the entire sequence in a strict `try/except/finally`.
  - On failure: Stop any successfully started remote nodes using `StopDaq`, broadcast zeroed UDP params to the Quabos, mark the ledger as `ABORTED`, and preserve the partial run directory in `<head_node_data_dir>/_aborted/<run_name>/` with a failure context dump.

## 4. Transactional `stop.py` (Best-Effort Shutdown)
- **Ledger Integration**: `stop.py` will read the intended run from `run_state.toml`. If `--run` is passed, it must validate against the ledger to prevent orphaning the current run (SC-027). The ledger status will transition to `STOPPING` and finally `COMPLETED`.
- **Partial Failure Tolerance**: Wrap `StopDaq`, `CleanupData`, and hardware `stop_data_flow` calls in individual `try/except` blocks. If node A times out or fails, log the error and **continue** to node B (SC-006).
- **Data Collection Safety (`utils/collect.py`)**: Harden `collect_data` to retry transient `rsync` failures. If it fundamentally fails (e.g., node disk full), it must *not* write `collect_complete` and must *not* invoke `CleanupData` (SC-012, SC-029).
- **Force Cleanup Flag**: Add a `--force-cleanup` CLI argument. This passes `force=True` to the gRPC `CleanupData` request, authorizing the DAQ server to delete data even if it has an orphaned PID record (SC-010).
- **Interleave Hard-Kill**: `stop_interleave()` will send `SIGTERM`, wait for a 5-second `retry_limit`, and if the process survives, escalate to `SIGKILL`. It will then synchronously invoke `do_maroc_config()` to safely restore Quabo default registers (SC-034).
- **Signal Masking**: Trap `SIGINT` (Ctrl-C) during execution to ensure critical cleanup operations finish gracefully before exiting.

## Constraints & Guardrails
- **Scope**: Focus entirely on `start.py`, `stop.py`, `utils/collect.py`, and the new state-tracking utilities.
- **Typing**: Maintain 100% strict MyPy type annotations.
- **Dependencies**: Use Pydantic and standard library `tomllib` (Python 3.11+) for parsing state. TOML writing will be handled safely using standard string formatting or a dedicated writer if available.