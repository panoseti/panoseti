# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This file supplements the root-level `../CLAUDE.md`, which covers the full repo architecture, hardware topology, config system, and observing run lifecycle. Read that first for context.


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
- **Transfer Daemon** (`transfer/daemon.py`): Drains `state/transfer/queue/pending/` jobs and drives each through manifest → rsync → verify → selective cleanup → archive. Holds `state/locks/transfer.lock` (flock singleton).
- **Distributed Ledger**: State is persisted in `state/runs/ledger.toml`. The `RunStateLedger.status` field has 17 possible values covering the full lifecycle from `STARTING` through `ARCHIVED`.

### Lock Hierarchy
| Lock file | Mechanism | Held by | Duration |
|---|---|---|---|
| `state/locks/control.lock` | `os.O_EXCL` + stale-PID healing | `pseti start` / `pseti stop` | Seconds (hardware I/O only) |
| `state/locks/transfer.lock` | `fcntl.LOCK_EX \| LOCK_NB` | Transfer Daemon | Job duration (minutes to hours) |

### RunStateLedger Status Vocabulary
`STARTING → ACTIVE → STOPPING → RECORDING_ENDED → MANIFEST_GENERATING → TRANSFERRING → VERIFYING → CLEANING → ARCHIVED`

Error exits: `ABORTED` (from start), `TRANSFER_FAILED`, `VERIFY_FAILED`, `STOPPED_WITH_ERRORS`.

The VERIFYING stage calls `transfer/verify.py::verify_manifest()` on head-node manifest files.  Any digest mismatch → `VERIFY_FAILED`; cleanup is skipped.  The CLEANING stage passes `manifest_digest` to `CleanupData`; the server rejects deletion with `FAILED_PRECONDITION` if digests don't match.

### TransferQueue Layout
```
state/transfer/queue/
  pending/    {run_name}.job.toml   ← pseti stop writes here
  active/     {run_name}.job.toml   ← daemon moves here on claim()
  failed/     {run_name}.job.toml   ← daemon moves here after MAX_ATTEMPTS
  completed/  {run_name}.job.toml   ← daemon moves here on success
```
State transitions use `os.rename` (POSIX-atomic). Double-enqueue of the same run is idempotent.

Read [TRANSACTIONS.md](TRANSACTIONS.md) for detailed diagrams and rollback rules.

---

## Testing and Debugging
- **Tier 1 (Unit)**: `src/ci/tier1_unit/`. Zero-dependency logic and parsing.
- **Tier 2 (Logic)**: `src/ci/tier2_logic/`. Subsystem logic with mocked gRPC.
- **Tier 3 (Fleet)**: `src/ci/tier3_fleet/`. Multi-node E2E with testcontainers.
- **Tier 4 (Chaos)**: `src/ci/tier4_chaos/`. Fault injection and TDD-forcing failure scenarios.
- **Tier 5 (Integration)**: `src/ci/tier5_integration/`. Real Hashpipe binary and heavy telemetry (Loki/Redis).
- **Atomic Locking**: Locks are managed via `os.O_EXCL` file creation with stale PID detection. Orphaned locks from crashed runs are self-healing.
- **State Isolation**: ALL integration tests MUST isolate their state using `PSETI_STATE` redirected to a temporary directory.

### Telemetry & Logging
- **Unified Logger**: Use `panoseti_grpc.telemetry.logger.get_logger(service_name, log_dir=...)`.
- **Path Resolution**: Use `control.utils.paths.PanoPaths.logs_dir()` for `log_dir`. `get_logger` accepts a `pathlib.Path`.
- **Four output paths**: console (Rich), `{service}.log` (plain text), `{service}.jsonl` (structured JSON for Grafana Alloy → Loki), and the legacy gRPC `Log` RPC (shadow period — all four run simultaneously during migration).
- **JSONL format**: one JSON object per line; fields: `timestamp`, `service`, `level`, `message`, `hostname`, `pid`, `thread`, plus any `extra` fields passed to the logger call (`git_commit`, `run_id`, etc.).
- **Alloy**: `alloy/config.alloy` ships `.jsonl` files to Loki. `alloy/docker-compose.yml` runs the Alloy agent.
- Avoid `logging.getLogger` and `print` for system events.

---

## 📁 Critical Documentation Routing

| Document | Description |
|---|---|
| [CLI.md](CLI.md) | Main command-line interface `pseti` |
| [TRANSACTIONS.md](TRANSACTIONS.md) | Rollback ladder sequence, atomic locking, and run state transitions. |
| [DEBUGGING.md](DEBUGGING.md) | Core debugging principles, lock and Loki pipeline troubleshooting, state isolation. |
| [TEST.md](TEST.md) | Test suite architecture, Docker runner usage, and isolation mandates (`PSETI_STATE`). |
| [TEST-HW-SW.md](TEST-HW-SW.md) | UC Berkeley Hardware-Software system architecture and description |

---

## Run tests

```bash
# Standard test suite
pseti test sw unit         # Tier 1: Fast logic tests
pseti test sw logic        # Tier 2: State logic tests
pseti test sw fleet        # Tier 3: Dynamic node tests
pseti test sw chaos        # Tier 4: Fault injection
pseti test sw integration  # Tier 5: Heavy stack tests
pseti test lint            # ruff + mypy concurrently

# Targeted test runs
pseti test sw chaos -k SCN003 -vv    # Verbose scenario debugging
pseti test sw integration -k "real_data"
```

---

## CI Architecture Notes
- **Persistent containers**: `pseti test up` starts containers that are reused across runs to minimize overhead.
- **Live mount**: `control/` is volume-mounted into containers; source edits are visible instantly.
- **Validation Leniency**: In CI, we bypass strict hardware checks if `daq_config.json` has `head_node_container: true`.
- **Networking**: `headnode_net` (10.0.1.0/24) hosts telemetry and Loki; `daqnode_net` (192.168.0.0/24) hosts the DAQ fleet.
- **Loki Pipeline**: Logs are queued in Redis (`logs:ingress`) and processed by `storeLoki.py` with non-blocking resilience.

Read [ci/README.md](ci/README.md) for the full network topology and test isolation strategy.
