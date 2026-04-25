# Debugging Guide — PANOSETI Control Plane

This document captures hard-won debugging strategies and core principles for the control plane. It assumes familiarity with the architecture in `CLAUDE.md`.

---

## 🧠 Core Principles
1. **Atomic Receipt First (WAL)**: Always write the node receipt to the ledger *before* issuing a gRPC call. This ensures the rollback ladder knows which nodes to clean up if the process is killed mid-RPC.
2. **Path Totality**: Never use bare strings for paths (e.g. `open("data.json")`). Always use `PanoPaths` accessors. Construction of a path outside of `PanoPaths` is an architectural defect.
3. **Explicit Success Validation**: gRPC responses often return `success: bool`. Never assume a response without an exception is a success. Always check `resp.success` and log `resp.message`.
4. **State Isolation**: When debugging tests, always set `PSETI_STATE` to a unique temporary directory via `monkeypatch` to prevent pollution of the global `/app/state`.

---

## 1. State Management & Locks

We use a role-segregated hierarchy under `control/state/` (override via `PSETI_STATE`).

### Orphaned Advisory Locks
- **Location**: `state/locks/{control|transfer}.lock`
- **Mechanism**: Uses `os.O_EXCL` for atomic creation.
- **Self-Healing**: If acquisition fails, the system reads the PID in the lock file. If that process is dead, the system clears the lock automatically.
- **Manual Debugging**:
  ```bash
  cat state/locks/control.lock # See who claims it
  ps -p $(cat state/locks/control.lock) # Verify if alive
  ```

### Run Ledger Issues
- **Location**: `state/runs/ledger.toml`
- **Statuses that block a new start**: `STARTING`, `ACTIVE`, `STOPPING`, `RECORDING_ENDED`.
- **Resetting state**: `pseti state migrate` (idempotent) or `rm state/runs/ledger.toml`.

---

## 2. Transfer Queue Pipeline

### Inspecting Queue Stages
Jobs move between subdirectories in `state/transfer/queue/`:
- `pending/`: Enqueued by `pseti stop`.
- `active/`: Claimed by `transfer_daemon`.
- `completed/`: Success. Includes `run_complete` marker on head node.
- `failed/`: Exhausted `MAX_ATTEMPTS` (3).

### Daemon Troubleshooting
- **Heartbeat**: Written every 5s to `state/transfer/daemon.heartbeat`. Staleness >30s indicates a crashed daemon.
- **Mocking pitfall**: Patching the entire `asyncio` module breaks `TaskGroup` and `to_thread`. Patch only specific functions like `asyncio.sleep`.

---

## 3. Container Log & gRPC Pipeline

- **Redis Ingress**: `docker exec ctl-int-redis-1 redis-cli LLEN logs:ingress` (Should be near 0).
- **Server Internal Log**: `/var/log/panoseti/daq_control_server.log` inside the `daqnode` container.
- **Validation Failure**: If a gRPC call fails with "Validation Error," check `grpc/src/panoseti_grpc/daq_control/config.py`. The server enforces strict existence checks on `run_dir` and `module_id` paths.

---

## 4. Test Infrastructure Gotchas

### TaskGroup Error Suppression
`asyncio.TaskGroup` cancels all remaining tasks if one fails. In `_process_job`, we manually collect errors from the group to ensure one node's failure doesn't silently hide another's, and to prevent advancing to the `ARCHIVED` stage on partial success.

### Shared Volume Race Conditions
The `int-tester` and `daqnode` share `/data`. If a test creates a directory, the gRPC server sees it instantly. However, if hashpipe hasn't received packets, module directories may be missing, causing `GenerateManifest` to fail validation. **Fix**: Pre-create module directories in the test setup.
