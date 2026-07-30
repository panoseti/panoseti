---
name: using-pseti-cli
description: Use when running, scripting, or explaining the pseti command — the observing-run lifecycle, hardware/daemon configuration, status/ledger, or the transfer queue. Not for writing or debugging tests (use testing-panoseti).
---

# Using the pseti CLI

## Overview

`pseti` is the unified observatory control entry point, implemented as a lazy-loading Typer app (`control/src/control/pseti.py`). All sub-commands are registered in `lazy_mapping` and loaded on demand — never import them directly.

## Top-level commands

| Command | Purpose |
|---------|---------|
| `power` | WPS power control (on/off/status) |
| `uids` | Scan Quabo hardware UIDs |
| `cfg` | Hardware/daemon configuration (many subcommands) |
| `val` | Config validation (all/network/graph/debug) |
| `start` | Begin recording run |
| `stat` | Observatory health + ledger |
| `stop` | End recording run, enqueue transfer |
| `xfr` | Transfer queue management |
| `session-start` | Power on, calibrate, start daemons |
| `session-stop` | Power off, stop daemons |
| `show` | Visualize state (sci/pff subcommands) |
| `paths` | Top-level — show resolved paths and env overrides |
| `test` | QA suites (lint/sw/sw2/grpc/hw/pff/prune) |
| `grpc` | gRPC service operations |
| `admin` | Deploy/check the DAQ node stack over a Docker context (`deploy`, `status`; `--mode docker\|bare-metal`; node arg or `all`) |

## Canonical run lifecycle

```
pseti session-start     # power on, UIDs, calibrate, start daemons
pseti start             # configure quabos, start DAQ/Hashpipe
pseti stat              # monitor health and disk
pseti stop              # stop DAQ, enqueue transfer job
pseti session-stop      # power off, stop daemons
```

Transfer (rsync → verify → cleanup → archive) runs out-of-band via the Transfer Daemon. `pseti xfr` subcommands: `start`/`stop` (daemon lifecycle), `stat` (daemon health + queue summary — check this first), `queue [bucket]` (list jobs; default `pending`), `retry` (move a failed job back to pending), `tail` (daemon log), `verify` (manifest check on a completed run, no state changes).

If `pseti xfr start` reports success but `pseti xfr stat` immediately shows `NOT RUNNING`, check `state/logs/transfer_daemon/stderr.log` — the daemon subprocess may have died instantly (e.g. wrong Python interpreter resolved from `$PATH`). Also worth knowing: `pseti stop` accepts `--force-stop` (bypasses ledger-state validation to run the full teardown ladder anyway) — an older `--force-cleanup` flag was removed.

`pseti stat`'s DAQ node lines show `[N/4 threads]` (healthy) or `[STUCK: N/4 threads]` — a live Hashpipe PID alone doesn't mean it's actually running; see `developing-control-code` for why.

## Discovery

```bash
pseti -t          # full command tree
pseti <cmd> -h    # per-command help
```

There is no `pseti show commands`. Note `pseti paths` is top-level, not `pseti show paths`.

## Lazy-unwrap pattern

Single-command modules are auto-unwrapped: `pseti start --nsecs 60` works instead of `pseti start main --nsecs 60`. Multi-command modules expose subcommands (e.g. `pseti cfg hv-on`).

## Full reference

`control/CLI.md` — complete command and subcommand listing with flags.

For running or writing tests, use `testing-panoseti` instead of this skill.
