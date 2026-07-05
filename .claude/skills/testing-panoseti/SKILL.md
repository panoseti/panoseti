---
name: testing-panoseti
description: Use when running, writing, or debugging tests or CI for control/ or grpc/ code, when a test or CI run fails or hangs, or when test state leaks between runs.
---

# Testing PanoSETI

## Overview

Two test stacks: `pseti test sw2` (control/ v2, current) and `pseti test grpc` / `python tests/qa.py` (grpc/ submodule). Both share Docker-based CI runners but have separate isolation models.

## control/ test tiers (`pseti test sw2`)

| Tier | Command | Docker | Notes |
|------|---------|--------|-------|
| 1 | `pseti test sw2 unit` | No | Pure logic, Pydantic, driver |
| 2 | `pseti test sw2 logic` | No | State-machine, isolated workspace |
| 3 | `pseti test sw2 fleet` | testcontainers | Multi-node E2E |
| 4 | `pseti test sw2 chaos` | testcontainers | Fault injection |
| 5 | `pseti test sw2 integration` | static compose | Real Hashpipe + tcpreplay |

`pseti test sw v2 <tier>` is the legacy alias. `pseti test sw2 all` runs lint + all five tiers.

**Shared flags** (before the subcommand):
`--dev` (hot-mount source), `--no-build`, `--tool docker|podman`, `--debug`/`--no-teardown`.

## grpc/ tests

```bash
pseti test grpc all                   # all suites via qa.py
pseti test grpc lint|daq-control|daq-data|telemetry
python tests/qa.py all                # equivalent, run from grpc/
```

## Lint

```bash
pseti test lint              # ruff + mypy
pseti test lint ruff         # ruff only
pseti test lint mypy         # mypy only
```

## Hardware tests

`pseti test hw run [-k SCENARIO]` — requires real Quabos and a DAQ node; see `control/TEST.md`.

## State isolation (critical)

All tier 2+ tests MUST redirect state to a temp directory:
- `PSETI_STATE` env var → overrides `state/` root
- `pseti_workspace` pytest fixture — provides isolated `PSETI_STATE` per test
- `PSETI_*` env vars (`PSETI_LOGS`, `PSETI_RUNS_DIR`, `PSETI_TQ_DIR`, etc.) — fine-grained overrides, all fall back to `PSETI_STATE`-derived paths when unset

Never let tests share `state/locks/`, `state/runs/`, or `state/transfer/` with each other or with a live observatory.

**This isolation is enforced by a session-level default, not just per-test discipline — check it's actually in place before trusting a suite.** `control/src/ci/software_only/conftest.py`'s `pytest_configure` sets `PSETI_TMP`/`PSETI_LOGS`/`PSETI_QUABOS`/`PSETI_STATE` to `/tmp/pseti_v2_test/*` defaults *before any test runs*, specifically because the suite's `autouse=True` `clear_shared_state` fixture (runs before every single test, whether or not that test requests `pseti_workspace`) calls `PanoPaths.transfer_queue_dir()` and `RunStateManager().clear_state()` directly. `PSETI_STATE` was missing from that default list until this was caught: every test lacking its own `pseti_workspace` fixture was resolving those paths to the *real* `control/state/` tree and `rmtree`-ing the real transfer queue (pending/active/completed/failed) and observing-run ledger on every `--dev`-mode run — this is what silently emptied a live observatory's transfer queue mid-session. Diagnostic signature if you suspect this has recurred: real `state/logs/transfer_daemon/<hostname>/*.log` containing test-only job names or IPs (e.g. `robust_e65d1aea.pffd`, `192.168.117.10`) that were never real run names. If you add a new autouse fixture that touches `PanoPaths`-resolved paths, verify `PSETI_STATE` isolation is already active in that suite's `pytest_configure` — don't assume the per-test `pseti_workspace` fixture alone covers it, since autouse fixtures apply regardless of which fixtures a given test explicitly requests.

Note: `control/src/ci/software_only/` *is* the current (v2) suite despite the directory name lacking a `_v2` suffix — `software_only_v2/` referenced in some docs doesn't exist as a separate path on disk; `pseti test sw`/`sw2` both resolve here.

## Recurring failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Redis assertion flaps | `RedisBatcher` flush latency | Poll with timeout, not `time.sleep` |
| `OSError: [Errno 5]` in Docker | overlay2 EIO on deleted log file | See grpc/CLAUDE.md Key Gotchas |
| Compose project collision | Non-unique `-p` name | Each suite must use unique compose project name |
| Hashpipe `TPACKET_V3` error | Docker vNIC | Always set `BINDHOST=lo` in Docker CI |
| Cleanup blocked after crash | Stale `hashpipe_pid` | Call `StopDaq` before `CleanupData` unconditionally |
| tier3_fleet/tier4_chaos tests fail with "N/4 threads" / stuck hashpipe messages | `fake_hashpipe.py` stub is single-threaded; new daq_control health check expects 4 real OS threads | Stub now spawns 3 dummy named worker threads (`net_thread`/`compute_thread`/`output_thread`) to match — if you copy/replace this stub, keep that |
| Real transfer queue emptied after running `--dev` tests | Missing `PSETI_STATE` isolation, autouse `clear_shared_state` fixture hit real paths | See State isolation section above |
| `AttributeError: 'tuple' object has no attribute 'get'` calling a gRPC client method | `DaqControlClient` methods like `StatusDaq` return `(success: bool, result: dict)`, not a bare dict | Unpack both: `success, result = client.StatusDaq(...)` |

## Full references

- `control/TEST.md` — tier model, isolation mandate, Docker runner
- `control/DEBUGGING.md` — lock/Loki pipeline troubleshooting
- `control/src/ci/software_only_v2/README.md` — network topology, full CI setup
- `grpc/CLAUDE.md` — Testing Infrastructure + Key Gotchas sections
