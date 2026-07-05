---
name: developing-control-code
description: Use when modifying or adding code under control/ — the observing-run lifecycle, start/stop transactions, the config and validation system, daemons, or the transfer queue. For Quabo UDP packet work use working-with-quabo-driver.
---

# Developing Control Code

## Overview

`control/` is the observatory's Python control plane. Entry point: `pseti` CLI. Core concepts: transaction/rollback ladder, config validation tiers, `PanoPaths` for file resolution, and a daemon model for long-running background processes.

## Transaction and ledger model

`StartTransaction` / `StopTransaction` context managers drive hardware through atomic steps with rollback on failure. State persists in `state/runs/ledger.toml` via `RunStateLedger` (17 status values):

```
STARTING → ACTIVE → STOPPING → RECORDING_ENDED
→ MANIFEST_GENERATING → TRANSFERRING → VERIFYING → CLEANING → ARCHIVED
Error exits: ABORTED, TRANSFER_FAILED, VERIFY_FAILED, STOPPED_WITH_ERRORS
```

Lock hierarchy (never invert):

| Lock | Mechanism | Holder | Duration |
|------|-----------|--------|----------|
| `state/locks/control.lock` | `os.O_EXCL` + stale-PID heal | `pseti start` / `stop` | Seconds |
| `state/locks/transfer.lock` | `fcntl.LOCK_EX\|LOCK_NB` | Transfer Daemon | Minutes–hours |

Full reference: `control/TRANSACTIONS.md`.

## Configuration system (6 files + 3 validation tiers)

| File | Role |
|------|------|
| `obs_config.json` | Domes → modules → quabos, timing mode, overvoltage |
| `daq_config.json` | DAQ node IPs and module assignments |
| `data_config.json` | Data products, integration time, interleave |
| `network_config.json` | VPN/gateway port forwarding |
| `daemons.json` | Which background daemons to enable |
| `firmware.json` | Quabo firmware binary mappings |

Validation tiers:
1. **Pydantic schema** (`utils/pydantic_config_models.py`) — types, ranges, constraints
2. **Cross-config rules** (`utils/global_validator.py`) — overvoltage consensus, port collisions
3. **Network reachability** (`utils/config_validator.py`) — parallel TCP checks

Always use Pydantic model instances across call boundaries; always prefer attribute access (`config.daq_nodes`) over dict indexing.

## Path resolution

`control/utils/paths.py` — `PanoPaths` resolves all critical dirs (`state/`, `logs/`, `runs/`, `transfer/`) from `PSETI_*` env vars, falling back to defaults relative to the repo root. Use `PanoPaths` everywhere; never hardcode paths.

## Data flow

```
Quabos (UDP 60001 science) → DAQ Nodes (Hashpipe) → PFF files
Quabos (UDP 60002 HK)      → Head Node (capture_hk.py) → Redis → InfluxDB → Grafana
```

Hashpipe is started/stopped via the `DaqControlClient` gRPC client (not SSH). See `developing-grpc-services` for the gRPC layer.

## Daemons

Started by `session_start.py` via `util.start_daemon()`; tracked by PID file; stopped by `util.stop_daemon()` (SIGTERM). Key ones: `capture_hk.py`, `capture_gps.py`, `capture_mount_ssh.py`, `storeInfluxDB.py`, `transfer_daemon.py`.

**Gotcha**: `start_daemon(prog)` accepts either a script path (str) or a full command list. For the list form, always use `[sys.executable, "-m", "control.transfer"]` — never the bare string `"python"`. A bare `"python"` resolves via the *subprocess's* `$PATH` at launch time, which may not be the interpreter running `pseti` itself (e.g. it resolved to a conda Python with no `control` package installed on one deployment, so the daemon subprocess died instantly with `ModuleNotFoundError` while `start_daemon()` printed "started ..." regardless — it doesn't wait to confirm the child stays up). Symptom: `pseti xfr stat` shows `NOT RUNNING` immediately after `pseti xfr start` reports success; check `state/logs/<daemon_name>/stderr.log` first.

## start.py / stop.py module split

`start.py` and `stop.py` are thin CLI + orchestrator entrypoints; the actual logic lives in sibling modules, re-exported at the top of each for backward compatibility (`__all__`, real top-level imports — not lazy):

- `start_transaction.py` / `stop_transaction.py` — `StartTransaction` / `StopTransaction` context managers (lock + rollback/teardown ladder).
- `start_preflight.py` — pure validation (`_check_daq_reachability`, `_check_quabo_reachability`, `_check_no_remote_hashpipe`, `_resolve_strict_mode`, etc.), independently testable, no CLI coupling.
- `hardware_ops.py` — `start_data_flow`, `make_run_dirs`.

When patching internals in tests, target the module where the code *actually lives* now, not `control.start`/`control.stop` — e.g. `patch("control.start_transaction.AsyncDaqControlClient")`, `patch("control.stop_transaction.TransferQueue")`. `control.start.StartTransaction` etc. still resolve (re-exported), but `patch()` needs the module that owns the attribute's *lookup*, which after the split is the sibling module, not `start.py`/`stop.py`.

**Cross-module globals don't survive a module split.** `hardware_ops.py`'s `start_data_flow`/`make_run_dirs` used to share a module-level `verbose` flag with `start.py` when they lived in the same file; splitting them broke that silently (`from x import verbose` captures a value at import time, not a live reference) until `verbose` was made an explicit function parameter. If you split a module further, grep for module-level mutable globals first.

## Hashpipe health: thread-count check

`panoseti_grpc.daq_control`'s `StartDaq`/`StatusDaq` poll `psutil.Process(pid).num_threads()` (`EXPECTED_HASHPIPE_THREADS = 4`: main + net_thread + compute_thread + output_thread) to distinguish "hashpipe alive" from "hashpipe alive but stuck mid-init" — the latter happens when a stale POSIX semaphore (`/dev/shm/sem.*hashpipe_status_N`, named by instance ID not PID, survives the old process's death) blocks `hashpipe_databuf_create()` forever before any worker thread spawns. `pseti stat` shows `[N/4 threads]` (healthy) or `[STUCK: N/4 threads]`. Cleanup is **opt-in**, not automatic: `pseti start --force-clean-semaphores` / `StartDaqRequest.force_clean_semaphores`. See `developing-grpc-services` for the proto/client side.

If you write a fake-hashpipe test double, it must actually spawn matching worker threads (see `testing-panoseti`) or this health check will always flag it unhealthy.

## Full references

- `control/CLAUDE.md` — transaction logic, test tiers, telemetry/logging
- `control/TRANSACTIONS.md` — rollback ladder diagrams, lock rules, transfer queue layout
- Root `CLAUDE.md` — hardware topology, config system, PFF format, timing
- `wiki_docs/Configuration-files.md`, `Observing-runs.md`, `Sessions-and-configuration.md`
- `wiki_docs/DAQ-system-overview.md`, `Nodes-and-modules.md`, `Control-system-implementation.md`
- `wiki_docs/Data-file-format.md`, `Data-file-names.md`, `Precise-Timing.md`
- For module/quabo IP math, Quabo UDP protocol, and MAROC config: use `working-with-quabo-driver`
