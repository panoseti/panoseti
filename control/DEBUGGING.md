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

`tmp/run_state.toml` persisting between tests causes `start.py` to refuse with "A run is already in progress." 
- **StartTransaction** and **StopTransaction** manage the status lifecycle.
- Inspect the ledger for status leaks:
```bash
cat tmp/run_state.toml | grep status
```

### Hashpipe process left running after a test

The DAQ Control server tracks only the PID it spawned. If a test fails mid-run and rollback is bypassed (e.g., violent container restart), kill it manually:
```bash
# On the daqnode container
pkill -KILL hashpipe
```

---

## 2. Container Log Inspection

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

## 3. Configuration Validation Debugging

### CI Validation Leniency
The control plane enforces strict pre-flight validation. In CI environments, we allow leniency for missing hardware/firmware if `head_node_container: true` is set in `daq_config.json`.
- If a test fails with "Pre-flight configuration validation failed," ensure the test's `DaqConfigValidator` instance has `head_node_container=True`.

---

## 4. Advanced Insights

### asyncio TaskGroup and Concurrent RPCs
`start.py` uses `asyncio.TaskGroup` for fail-fast concurrency.
- **Fail-Fast**: If one `StartDaq` RPC fails, all others are immediately cancelled.
- **Atomic Rollback**: The `StartTransaction.__aexit__` re-loads the ledger to identify which nodes received a receipt before cancellation and issues `StopDaq` to them in the correct priority order.

### Write Ahead Logging (WAL) Pattern
Always write the node receipt to `run_state.toml` **before** issuing the gRPC call. This ensures that if the process is killed during the RPC, the rollback ladder knows the node was "attempted" and can clean it up.
