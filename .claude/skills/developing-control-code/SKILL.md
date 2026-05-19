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

## Full references

- `control/CLAUDE.md` — transaction logic, test tiers, telemetry/logging
- `control/TRANSACTIONS.md` — rollback ladder diagrams, lock rules, transfer queue layout
- Root `CLAUDE.md` — hardware topology, config system, PFF format, timing
- `wiki_docs/Configuration-files.md`, `Observing-runs.md`, `Sessions-and-configuration.md`
- `wiki_docs/DAQ-system-overview.md`, `Nodes-and-modules.md`, `Control-system-implementation.md`
- `wiki_docs/Data-file-format.md`, `Data-file-names.md`, `Precise-Timing.md`
- For module/quabo IP math, Quabo UDP protocol, and MAROC config: use `working-with-quabo-driver`
