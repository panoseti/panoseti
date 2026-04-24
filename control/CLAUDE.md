# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This file supplements the root-level `../CLAUDE.md`, which covers the full repo architecture, hardware topology, config system, and observing run lifecycle. Read that first for context.

---

## Test Counts & Runner

- **Python version**: `requires-python = ">=3.14"`
- **CI runner**: `pseti test <cmd>` (not `bash ci/run.sh`)
- **Unit tests**: 538 passing (12 modules) — `pseti test sw unit`
- **Integration tests**: 65 passing — `pseti test sw integration`
- **Chaos/scenario tests**: 114 tests (91 active, 23 stubs) — `pseti test sw chaos`
- **HW-in-the-loop tests**: HW-01 … HW-05 in `ci/hardware-software/` — `pseti test hw run`

---

## Verification & Quality Standards

### Linting and Type Safety
The project enforces strict linting via Ruff and type checking via MyPy. All new code must pass `pseti test lint`.

- **Pydantic Model Authority**: Instantiated models from `utils/pydantic_config_models.py` must be passed across call boundaries. Polymorphic functions must validate dictionaries into models at the entry point.
- **Attribute Access**: Always prefer model attribute access (`config.daq_nodes`) over dictionary indexing (`config['daq_nodes']`).
- **MyPy Strictness**: Avoid `type: ignore` whenever possible. If required, use it on a specific line with a comment explaining why. Ensure `unused-ignore` rules pass.

### Documentation (Google Style Docstrings)
All functions and methods must have high-quality docstrings. Preserving legacy comments (prefixed with `#`) by transforming them into formal docstrings is mandatory.

---

## Transaction Logic
The observatory uses a **Context Manager Architecture** to manage the lifecycle of an observing run.
- **StartTransaction**: Handles atomic locking and a multi-step rollback ladder. If any startup step fails, it guarantees all hardware and remote processes are restored to a safe state.
- **StopTransaction**: Implements a resilient teardown sequence for hardware (stop DAQs, kill daemons, stop quabos), then enqueues a transfer job and transitions the ledger to `RECORDING_ENDED`. Bulk I/O is decoupled from the lock.
- **Transfer Daemon** (`daemons/transfer_daemon.py`): Drains `tmp/transfer_queue/pending/` jobs and drives each through manifest → rsync → verify → selective cleanup → archive. Holds `tmp/panoseti_transfer.lock` (flock singleton).
- **Distributed Ledger**: State is persisted in `tmp/run_state.toml`. The `RunStateLedger.status` field has 17 possible values covering the full lifecycle from `STARTING` through `ARCHIVED`.

### Lock Hierarchy
| Lock file | Mechanism | Held by | Duration |
|---|---|---|---|
| `tmp/panoseti_control.lock` | `os.O_EXCL` + stale-PID healing | `pseti start` / `pseti stop` | Seconds (hardware I/O only) |
| `tmp/panoseti_transfer.lock` | `fcntl.LOCK_EX \| LOCK_NB` | Transfer Daemon | Job duration (minutes to hours) |

### RunStateLedger Status Vocabulary
`STARTING → ACTIVE → STOPPING → RECORDING_ENDED → MANIFEST_GENERATING → TRANSFERRING → VERIFYING → CLEANING → ARCHIVED`

Error exits: `ABORTED` (from start), `TRANSFER_FAILED`, `VERIFY_FAILED`, `STOPPED_WITH_ERRORS`.

The VERIFYING stage calls `utils/transfer/verify.py::verify_manifest()` on head-node manifest files.  Any digest mismatch → `VERIFY_FAILED`; cleanup is skipped.  The CLEANING stage passes `manifest_digest` to `CleanupData`; the server rejects deletion with `FAILED_PRECONDITION` if digests don't match.

### TransferQueue Layout
```
tmp/transfer_queue/
  pending/    {run_name}.job.toml   ← pseti stop writes here
  active/     {run_name}.job.toml   ← daemon moves here on claim()
  failed/     {run_name}.job.toml   ← daemon moves here after MAX_ATTEMPTS
  completed/  {run_name}.job.toml   ← daemon moves here on success
```
State transitions use `os.rename` (POSIX-atomic). Double-enqueue of the same run is idempotent.

Read [TRANSACTIONS.md](TRANSACTIONS.md) for detailed diagrams and rollback rules.

---

## Testing and Debugging
- **Unit Tests**: Add new cases to `src/control/ci/unit/` for every utility function. No hardware or network access is allowed.
- **Integration Tests**: Verify end-to-end flows in `src/control/ci/integration/`. Use `-k` to isolate failures.
- **Chaos Tests**: Verifies transaction integrity under failure conditions in `src/control/ci/integration/scenarios/`. Run via `pseti test chaos`.
- **Atomic Locking**: Locks are managed via `os.O_EXCL` file creation with stale PID detection. Orphaned locks from crashed runs are self-healing.
### Telemetry & Logging
- **Unified Logger**: Use `panoseti_grpc.telemetry.logger.get_logger(service_name, log_dir=...)`.
- **Path Resolution**: Use `control.utils.paths.PanoPaths.logs_dir()` for `log_dir`. `get_logger` accepts a `pathlib.Path`.
- **Four output paths**: console (Rich), `{service}.log` (plain text), `{service}.jsonl` (structured JSON for Grafana Alloy → Loki), and the legacy gRPC `Log` RPC (shadow period — all four run simultaneously during migration).
- **JSONL format**: one JSON object per line; fields: `timestamp`, `service`, `level`, `message`, `hostname`, `pid`, `thread`, plus any `extra` fields passed to the logger call (`git_commit`, `run_id`, etc.).
- **Alloy**: `alloy/config.alloy` ships `.jsonl` files to Loki. `alloy/docker-compose.yml` runs the Alloy agent.
- Avoid `logging.getLogger` and `print` for system events.

Read [DEBUGGING.md](DEBUGGING.md) for advanced troubleshooting techniques and [ci/README.md](ci/README.md) for test architecture details.

---

## Run tests

```bash
# Standard test suite
pseti test sw unit      # Parallel unit tests
pseti test sw integration # E2E with real hashpipe
pseti test sw chaos     # Chaos/TDD-forcing scenarios
pseti test lint         # ruff + mypy concurrently

# Targeted test runs
pseti test sw chaos -k SCN003 -vv    # Verbose scenario debugging
pseti test sw integration -k "real_data"

# Native (no Docker, unit tests only)
pseti test sw unit --native
```

The `chaos` command runs `src/control/ci/integration/scenarios/`.

---

## CI Architecture Notes
- **Persistent containers**: `pseti test up` starts containers that are reused across runs to minimize overhead.
- **Live mount**: `control/` is volume-mounted into containers; source edits are visible instantly.
- **Validation Leniency**: In CI, we bypass strict hardware checks if `daq_config.json` has `head_node_container: true`.
- **Networking**: `headnode_net` (10.0.1.0/24) hosts telemetry and Loki; `daqnode_net` (192.168.0.0/24) hosts the DAQ fleet.
- **Loki Pipeline**: Logs are queued in Redis (`logs:ingress`) and processed by `storeLoki.py` with non-blocking resilience.

Read [ci/README.md](ci/README.md) for the full network topology and test isolation strategy.
