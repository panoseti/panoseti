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
- `PSETI_*` env vars (`PSETI_LOGS_DIR`, `PSETI_RUNS_DIR`, etc.) — fine-grained overrides

Never let tests share `state/locks/`, `state/runs/`, or `state/transfer/` with each other or with a live observatory.

## Recurring failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Redis assertion flaps | `RedisBatcher` flush latency | Poll with timeout, not `time.sleep` |
| `OSError: [Errno 5]` in Docker | overlay2 EIO on deleted log file | See grpc/CLAUDE.md Key Gotchas |
| Compose project collision | Non-unique `-p` name | Each suite must use unique compose project name |
| Hashpipe `TPACKET_V3` error | Docker vNIC | Always set `BINDHOST=lo` in Docker CI |
| Cleanup blocked after crash | Stale `hashpipe_pid` | Call `StopDaq` before `CleanupData` unconditionally |

## Full references

- `control/TEST.md` — tier model, isolation mandate, Docker runner
- `control/DEBUGGING.md` — lock/Loki pipeline troubleshooting
- `control/src/ci/software_only_v2/README.md` — network topology, full CI setup
- `grpc/CLAUDE.md` — Testing Infrastructure + Key Gotchas sections
