# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

PANOSETI is an observatory control and data analysis system for a telescope array at Palomar Observatory. The repo contains:
- `control/` — the primary instrument control system (Python)
- `analysis/` — data analysis framework
- `anomaly-detection/`, `cloud-detection/`, `adc-to-pe/` — ML-based data processing pipelines
- `util/` — shared C++ utilities (PFF parser, image processing)
- `daq/`, `grpc/`, `web/` — git submodules (DAQ pipeline, gRPC interface, web dashboard)

---

## Common Commands

### Install dependencies
```bash
# Runtime
pip install -r control/packages/requirements.txt

# Development (tests, lint, type-check)
cd control && pip install -e ".[dev]"

# gRPC submodule
cd grpc && pip install -e ".[dev]"
```

### Observing run lifecycle (CLI)
```bash
pseti session-start     # power on, get UIDs, calibrate, start daemons
pseti start             # configure quabos, start DAQ recording
pseti status            # check recording status and disk usage
pseti stop              # stop recording; enqueues transfer job
pseti session-stop      # power off, stop daemons
```

### Run tests
```bash
# v2 test suite (software_only_v2/ — current)
pseti test sw2 unit        # Tier 1: fast logic + config unit tests
pseti test sw2 logic       # Tier 2: state-machine logic
pseti test sw2 fleet       # Tier 3: testcontainers fleet
pseti test sw2 chaos       # Tier 4: fault injection
pseti test sw2 integration # Tier 5: real Hashpipe + tcpreplay
# pseti test sw v2 <suite> is also valid (legacy alias)

# v1 test suite (software_only/ — sunset in progress; tier1 unit tests moved to sw2)
pseti test sw logic
pseti test sw integration
pseti test sw chaos

# Lint (Ruff + MyPy)
pseti test lint

# Hardware-in-the-loop tests (real Quabos + DAQ node required)
pseti test hw check-env   # verify connectivity
pseti test hw run         # full HW suite (env_check, boot_sequence, happy_path)
pseti test hw run -k boot_sequence   # single scenario

# gRPC service layer tests
pseti test grpc all
```

### Lint and type-check
```bash
cd control
ruff check src/control/
mypy src/control/utils/config_file.py --ignore-missing-imports
```

---

## Architecture

### Hardware topology

```
Head Node (control PC)
  └── SSH → DAQ Nodes (run Hashpipe, receive UDP science data, write PFF files)
  └── UDP → Quabos (detector boards, 4 per module, ~192.168.3.x)
  └── HTTP → Web Power Switches (WPS)
  └── SSH → Telescope Mount
```

Each **module** has 4 **quabos** (256-pixel detector boards). Module ID (0–255) is derived from the IP address: `module_id = (ip_octet3 * 256 + ip_octet4) >> 2 & 0xFF`. Each quabo has a `BOARDLOC = module_id * 4 + quadrant_index`.

Quabos communicate on:
- UDP port 60000–60003 — commands (one port per quabo in the module)
- UDP port 60002 — housekeeping packets (emitted every ~3 s)
- UDP port 60001 — science data (received by DAQ nodes, not head node)

### Configuration system

All configs live in `control/configs/` (symlinked to site-specific subdirs like `configs/palomar/`). The six config files and their roles:

| File | Purpose |
|------|---------|
| `obs_config.json` | Observatory layout: domes → modules → quabos, timing mode (WR or GNSS), detector overvoltage |
| `daq_config.json` | DAQ node IPs and which module IDs each node receives |
| `data_config.json` | Data products: integration time, PE threshold, image/PH mode, interleaving states |
| `network_config.json` | Port forwarding for VPN/gateway access |
| `daemons.json` | Which background daemons to enable |
| `firmware.json` | Quabo firmware binary mappings by hardware version |

Configs are loaded and validated via `control/utils/config_file.py`. Validation has three tiers:
1. **Pydantic schema** (`utils/pydantic_config_models.py`) — field types, ranges, constraints
2. **Cross-config rules** (`utils/global_validator.py`) — e.g., overvoltage consensus, port collision detection, firmware coverage
3. **Network reachability** (`utils/config_validator.py`) — parallel TCP checks against all hardware IPs

Loading pattern used throughout the codebase:
```python
from utils.config_file import get_obs_config, get_daq_config, get_data_config
obs_config = get_obs_config()    # returns validated dict
daq_config = get_daq_config()    # expands "128-250" ranges to lists
data_config = get_data_config()
```

### Observing run flow

1. `session_start.py` — powers on quabos, runs `get_uids.py` (discovers hardware UIDs), reboots firmware, sets HV, calibrates MAROC registers and PH baselines, starts Redis/InfluxDB daemons
2. `start.py` — configures quabos for the chosen data mode, tells quabos where to send science packets (DAQ node IPs), creates run directories on head and DAQ nodes, starts Hashpipe on DAQ nodes via gRPC (`DaqControlClient.StartDaq()`) — previously SSH
3. `stop.py` — kills Hashpipe via gRPC (`DaqControlClient.StopDaq()`), stops housekeeping recording, then **enqueues a transfer job** in `tmp/transfer_queue/pending/` and transitions the ledger to `RECORDING_ENDED` in seconds. Bulk I/O (rsync, manifest verify, selective cleanup) is handled by the Transfer Daemon (`daemons/transfer_daemon.py`) running out-of-band.

### Data flow

```
Quabos (UDP science packets) → DAQ Nodes (Hashpipe: net→compute→output threads) → PFF files
Quabos (UDP HK packets)      → Head Node (capture_hk.py) → Redis → InfluxDB → Grafana
```

**Hashpipe** is a multi-threaded pipeline (net thread → compute thread → output thread) that writes science data. Files roll at 1 GB.

### PFF file format

Files are named: `start_{ISO8601}.dp_{data_product}.bpp_{bytes}.dome_{N}.module_{N}.seqno_{N}.pff`

Data products: `ph256` (1 quabo, 256 pixels), `ph1024` (4 quabos), `img8`, `img16`

Each PFF file is a sequence of blocks:
- **JSON header block**: starts with `{`, ends with `\n\n`, fixed-size (padded with spaces), describes packet metadata (`quabo_num`, `pkt_num`, `pkt_tai`, `pkt_nsec`, `tv_sec`, `tv_usec`)
- **Binary image block**: preceded by `*` character, then raw pixel data

PFF parsing utilities: `control/utils/pff.py` (Python), `util/pff.cpp` (C++)

### Interleaving mode

`data_config.json` can define multiple named modes (e.g., `image_8bit`, `pulse_height_uhe`) and an `interleave` block that sequences through them with per-state durations. Quabos take ~100 ms to transition, so partial frames occur at boundaries. Constraint: movie mode and multi-pixel triggers cannot be active simultaneously in the same interleave state.

### Daemons

`control/daemons/` contains 40+ long-running processes started by `session_start.py` via `util.start_daemon()`. Key ones:
- `capture_hk.py` — receives quabo housekeeping UDP packets, stores to Redis
- `capture_gps.py` — reads GPS serial port, stores to Redis
- `capture_mount_ssh.py` — polls telescope mount pointing via SSH
- `capture_guider.py` — guider camera frames
- `storeInfluxDB.py` — reads Redis keys, writes to InfluxDB for Grafana dashboards
- `capture_telemetry_service.py` — runs the gRPC Telemetry service (see below)

Daemons are tracked by PID file; `util.stop_daemon()` sends SIGTERM.

### gRPC submodule (`grpc/` → `panoseti_grpc`)

The `grpc/` submodule (source at `../panoseti_grpc`) provides four async gRPC services that replace direct SSH/subprocess management of DAQ nodes with clean RPC boundaries. Install it with:

```bash
cd ../panoseti_grpc
pip install -e ".[dev]"
```

If you modify `.proto` files, regenerate the Python bindings:
```bash
python scripts/compile_protos.py
# Generated files land in src/panoseti_grpc/generated/ — do not edit these by hand
```

Run the gRPC test suite (requires Docker for Redis/InfluxDB/Loki):
```bash
cd ../panoseti_grpc
bash scripts/run-ci/run-daq-data-ci-test.sh
```

**The services (all hosted on one unified server):**

| Service | Module | Status | Purpose |
|---------|--------|--------|---------|
| DAQ Data | `panoseti_grpc.daq_data` | Production | Streams real-time science images from Hashpipe shared memory |
| DAQ Control | `panoseti_grpc.daq_control` | Production | Start/stop/status Hashpipe on DAQ nodes, generate manifests, clean up run data |
| Telemetry | `panoseti_grpc.telemetry` | Beta | Device status → Redis/InfluxDB; log shipping via Grafana Alloy → Loki |
| U-blox Control | `panoseti_grpc.ublox_control` | 🔴 Deprecated | GNSS chip control — disabled by default; use `Telemetry.ReportStatus` with `GnssPayload` instead |

All three active services are hosted on a single port via the unified server:
```bash
pseti-grpc server                                          # all enabled services
pseti-grpc server --profile daq_node --port-env DAQNODE_GRPC_PORT    # daq_data (edge) + daq_control
pseti-grpc server --profile headnode --port-env HEADNODE_GRPC_PORT   # telemetry + daq_data (gateway)
pseti-grpc server --list-services                          # show registered services (with [DEPRECATED] tags)
```

**gRPC port resolution — single source of truth.** The bind port is never hardcoded in a bundled profile TOML; it resolves at startup via `resolve_bind_port()` (`grpc/src/panoseti_grpc/unified_main.py`), highest priority first: `--port` > `os.getenv(--port-env's value)` > whatever the TOML/`GRPC_PORT` env resolved to > 50051. Every `docker-compose*.yml` `command:` and every bare-metal `start_grpc.sh` invocation passes `--port-env HEADNODE_GRPC_PORT` or `--port-env DAQNODE_GRPC_PORT` explicitly — the deployment names which role it is, so one `PanosetiServerConfig` shape serves two roles without profile-sniffing. Client-side, `control.utils.util.resolve_grpc_port(role, explicit=...)` applies the *identical* precedence (explicit per-node config > role env var > legacy `GRPC_PORT`/`DAQ_DATA_GATEWAY_PORT` > 50051) so server and client can't silently desync. `HEADNODE_GRPC_PORT`/`DAQNODE_GRPC_PORT` must differ when the head node and a DAQ node are co-located on one machine (both run `network_mode: host`); `pseti health` checks this before any container is touched.

**Gotcha — two independent CLI entry points.** `pseti-grpc server` (the real `[project.scripts]` console command everyone actually runs) is `panoseti_grpc.cli:standalone_app`, which lazily dispatches its `server` subcommand to `panoseti_grpc._cli.server:app` — a **separate**, Typer-based reimplementation of "load config, apply --services, run PanosetiServer". `unified_main.py`'s `main()` (argparse-based) is only reached via `python -m panoseti_grpc`, which nothing in this repo's deployment path invokes. These two duplicate the config-load/port-resolve/run sequence and have already drifted apart once (confirmed live against real hardware: a `--port-env` fix landed only in `unified_main.py` and was silently dead code, crash-looping the real DAQ node container with "No such option: --port-env"). Both call the same shared `resolve_bind_port()` so the precedence logic itself can't drift again, but **any new CLI flag on the server command must be added to `_cli/server.py`, not just `unified_main.py`** — `tests/unified_server/unit/test_config.py::test_cli_server_app_exposes_port_env_option` is a regression guard for exactly this (invokes `_cli/server.py`'s real Typer app via `CliRunner`).

**DAQ Data service** — the most actively used service. Streams `PanoImage` objects from Hashpipe shared memory to any subscriber. Client usage:
```python
from panoseti_grpc.daq_data.client import DaqDataClient
async with DaqDataClient(host, port) as client:
    await client.init_hp_io(run_dir, module_id)
    async for image in client.stream_images():
        process(image)
```
For local testing without hardware, `panoseti_grpc.daq_data.simulate` generates synthetic image streams.

**DAQ Control service** — replaces the SSH calls in `control/daq_scripts/` with RPCs. Key RPCs: `StartDaq()` (launches Hashpipe subprocess), `StopDaq()` (SIGINT), `StatusDaq()` (PID + disk usage), `CleanupData()` (full or selective), `GenerateManifest()` (blake3/xxhash checksums of run files), `GetManifest()` (streaming). The server (`python -m panoseti_grpc.daq_control.server`) runs as a systemd service on each DAQ node (port 50051). `control/daq_scripts/start_daq.py`, `stop_daq.py`, `status_daq.py` are **deprecated** in favor of this service.

`CleanupData` supports two modes (set via `mode` field):
- `CLEANUP_FULL` (default, legacy) — `rmtree` the entire run directory
- `CLEANUP_SELECTIVE` — delete only files matching `delete_patterns`, preserving those matching `preserve_patterns`; used by the Transfer Daemon. When called with `mode=CLEANUP_SELECTIVE`, the server **requires** a `manifest_digest` field (SHA-256 of the manifest file content) and refuses with `FAILED_PRECONDITION` if it doesn't match — guaranteeing no DAQ data is deleted without head-node integrity confirmation.

**Telemetry service** — consumed by `control/daemons/capture_telemetry_service.py`. Supports two storage paths:
- **Device status** (`ReportStatus` RPC): validated Pydantic payloads (e.g., `GnssPayload`, `DewPayload`) → permanent Redis HASH (hot) + InfluxDB (cold) → Grafana dashboards. Production devices use strict Pydantic schemas; `DEV_`-prefixed keys get a 24 h TTL in Redis only.
- **Log shipping** (shadow period): logs are written to `{service}.jsonl` files (structured JSON, one record per line) under `$PANOSETI_LOG_DIR/` by `get_logger()`, then shipped to Loki by **Grafana Alloy** (`alloy/config.alloy`). The legacy gRPC `Log` RPC continues running in parallel during the migration window.

To get a structured logger:
```python
from panoseti_grpc.telemetry.logger import get_logger
logger = get_logger("my_service", log_dir="/var/log/panoseti")
# writes {service}.log (plain text) + {service}.jsonl (Alloy → Loki) + gRPC
logger.info("message", extra={"git_commit": "abc1234", "run_id": "run_001"})
```

**Shared gRPC machinery (`grpc_utils`)** — all three active services share:
- `grpc_utils.exceptions` — typed `PanosetiRpcError` subclasses (`UnavailableError`, `DeadlineExceededError`, `FailedPreconditionError`, …)
- `grpc_utils.decorators` — `@grpc_call` wraps async/sync/generator methods, maps `grpc.RpcError → PanosetiRpcError`, never suppresses `asyncio.CancelledError`
- `grpc_utils.health` — `register_health()` (auto-called by unified server) + `HealthClient` wrapping `grpc.health.v1`; replaces the old `daq_data.Ping` RPC
- `grpc_utils.retries` — `build_retry_service_config()` for declarative transport-level retry policy

**Proto files** live in `../panoseti_grpc/protos/`. The `panoseti_util/` sub-package inside `panoseti_grpc` re-exports PFF reading/writing (`pff.py`) and config utilities (`config_file.py`) for use within the gRPC servers — prefer these over duplicating logic.

### Deployment & orchestration (`pseti admin`, `pseti health`)

`pseti admin deploy/build/down/status/attach/logs <node-or-headnode-or-all> --mode docker|bare-metal` (`control/src/control/admin/cli.py`) drives the containerized (or bare-metal) stack per node, over per-node Docker contexts for DAQ nodes (`docker_context` field in `daq_config.json`, falling back to `pseti-daq-<ip-with-dashes>`) and locally (no SSH) for the head node.

- **Env propagation is deterministic, not just inherited.** Every compose invocation materializes the resolved env (`_write_compose_env_file()`) to a temp file under `PanoPaths.tmp_dir()` and passes it via `--env-file`, so `PSETI_ENV_FILE`-selected values reach compose interpolation regardless of the caller's CWD (compose also auto-reads its own `.env` from the project directory, which would otherwise silently compete). `_daq_compose_env()`/`_compose_prefix()` centralize this so no new call site can forget it the way `down()`/`status()` originally did.
- **Bare-metal env propagation** (`--mode bare-metal`): `_write_remote_env_file()` writes/updates `/etc/panoseti/grpc.env` on the remote node over SSH before restarting `panoseti_grpc`/`panoseti_alloy` — the systemd units read it via `EnvironmentFile=-` (see `grpc/scripts/setup_panoseti_grpc.sh`). Without this there is no path for a head-node `.env` change to reach a bare-metal node at all.
- **Co-location** (head + DAQ node on one machine, e.g. Lick): both unified servers run `network_mode: host`, so `HEADNODE_GRPC_PORT`/`DAQNODE_GRPC_PORT` must differ, and `DAQ_DATA_DIR`/`PSETI_DATA_DIR` must not overlap. `pseti admin deploy` auto-skips the DAQ node's standalone Alloy container when that node `is_local()` to the head (the head's own compose stack already runs Alloy).

`pseti health` (`control/src/control/health.py`) is the single all-systems-green check, consolidating what `pseti val`/`pseti stat`/`pseti admin status`/`pseti test hw check-env` otherwise cover separately: config validity, WPS, Quabo TFTP reachability, a **co-located port/data-dir collision check** (pure, no network, runs first), head+DAQ gRPC health probed on the exact endpoint a real client resolves to (not a hardcoded port), container status, and the transfer daemon. Container status checks must pass `-p <project_name>` matching `pseti admin`'s exact project-naming convention (`pseti-headnode` / `pseti-daqnode-<host>`) — `docker compose ps` without it silently queries the wrong default project and always reports no containers running, even when they demonstrably are.

**Grafana provisioning.** `control/deploy/docker-compose.headnode.yml`'s `grafana` service bind-mounts dashboards/datasources straight from the `grafana/` submodule (`grafana_provisioning/dashboards|datasources`, read-only) and persists Grafana's own sqlite DB (dashboards/alerts/auth state) at `${PSETI_DATA_DIR}/grafana`. `dashboard.yml` sets `editable: true` + `allowUiUpdates: true`, so operators can edit provisioned dashboards from the web UI (edits land in the sqlite DB, not back in the JSON files — hand-edit the JSON under `grafana/grafana_provisioning/` and redeploy for source-controlled changes). The `grafana/grafana:latest` image runs as UID 472 GID 0 by default (not root); the compose service pins `user: "472:0"` to match. If Docker ever auto-creates `${PSETI_DATA_DIR}/grafana` as root:root (a fresh/never-provisioned data dir), the container crash-loops on `mkdir: can't create directory '/var/lib/grafana/plugins': Permission denied` — fix with `chown -R 472:0 ${PSETI_DATA_DIR}/grafana` on the host (same pattern as the `loki` service's own documented `10001:10001` fix in that file).

### Timing

Two precision timing sources:
- **White Rabbit (WR)**: Fiber-based, ~ns precision. WR switch deployed at observatory. Quabo 0 of each module runs WR firmware loaded via TFTP (`control/wr/wrpc_filesys`).
- **GNSS**: Alternative, less accurate. Configured per-module in `obs_config.json` via `timing_mode`.

Science packets carry both `tv_usec` (UNIX, from DAQ node NTP) and `pkt_nsec` (WR/GNSS nanoseconds since last UTC second). When combining: if `|tv_usec/1000 - pkt_nsec/1e6| > 25 ms`, adjust `tv_sec` by ±1 to get precise event time.

---

## Key Conventions

### Config directory selection
The active site config is a symlink: `control/configs/` → `control/configs/palomar/` (or `lick/`, `ucb/`). Never edit the symlink target directly for testing; create a separate config variant.

### Module/quabo IP math
```python
# Quabo IPs within a module (base IP ends in multiple of 4):
quabo_ip = f"{base_ip[:-1]}{int(base_ip.split('.')[-1]) + quabo_index}"  # quabo_index 0–3

# Module ID from IP:
parts = ip.split('.')
module_id = (int(parts[2]) * 256 + int(parts[3])) >> 2 & 0xFF
```

### Hardware driver
`control/driver/quabo_driver.py` — UDP socket-based. `QUABO(ip, port)` object sends binary command packets and receives HK packets. `DAQ_PARAMS` encapsulates data acquisition mode bits. The 829-byte serial command encodes MAROC register configuration for the SiPM readout ASICs.

### SSH to remote nodes
`control/utils/file_xfer.py` handles SSH/rsync to DAQ nodes. All remote commands use `subprocess.run(['ssh', f'{user}@{host}', cmd], ...)` (list form, not shell string) to avoid injection. `control/utils/collect.py` is **deprecated** — it prints a deprecation warning at import time. Use `utils/transfer/rsync_worker.py` instead.

### Transfer Daemon
`control/daemons/transfer_daemon.py` is a long-running daemon started by `session_start.py`. It drains jobs from `tmp/transfer_queue/pending/` and drives them through a 5-stage state machine:

1. **MANIFEST_GENERATING** — `GenerateManifest` RPC per module on each DAQ node (blake3 checksums via `asyncio.TaskGroup`)
2. **TRANSFERRING** — rsync each node's run directory to the head node (up to 3 attempts with exponential backoff: 5 s, 30 s)
3. **VERIFYING** — calls `verify_manifest()` on every `manifest.{blake3,xxh3_128,sha256}` file on the head node; any digest mismatch → `VERIFY_FAILED`; cleanup is **skipped** to preserve DAQ-side data for manual recovery
4. **CLEANING** — `CleanupData(mode=CLEANUP_SELECTIVE, delete_patterns=["*.pff"], preserve_patterns=["*.json","*.log","*.toml"])` per node; the server enforces a `manifest_digest` precondition and refuses deletion if the digest doesn't match
5. **ARCHIVED** — write `run_complete` marker

On startup the daemon sweeps `active/` for jobs stranded by a prior crash (SC-TX-005) and moves them back to `pending/` before entering the main loop.

The daemon holds `tmp/panoseti_transfer.lock` (flock) as a singleton guard. `stop.py` holds only `tmp/panoseti_control.lock` during the hardware teardown phase (seconds), never during bulk I/O.

### Data config validation constraints
- `integration_time_usec` must be a multiple of 10 and evenly divide 1,000,000
- `pe_threshold` ≥ 2.0 (pulse height mode), ≥ 1.0 (image/movie mode)
- `run_type` max 14 chars, no `.`, `_`, or spaces
- `detector_overvoltage` must match between `obs_config.json` and `data_config.json`

---

## Testing Infrastructure

### Python version requirement
`control/pyproject.toml` sets `requires-python = ">=3.14"`.

### Test locations
- `control/src/ci/software_only_v2/tier1_unit/` — pure logic, Pydantic, parsing (no hardware, no Docker)
- `control/src/ci/software_only_v2/tier2_logic/` — state-machine logic with isolated workspace
- `control/src/ci/software_only_v2/tier3_fleet/` — multi-node E2E with testcontainers (`DaqNodeSimContainer`)
- `control/src/ci/software_only_v2/tier4_chaos/` — fault injection (process kill, disk fill, gRPC proxy, netem)
- `control/src/ci/software_only_v2/tier5_integration/` — real Hashpipe binary + tcpreplay (static compose)
- `control/src/ci/software_only/` — v1 test suite (being sunset; runs in parallel with v2 during soak period)
- `control/src/ci/hardware_software/` — hardware-in-the-loop tests (requires real Quabos + DAQ node)
- `control/src/ci/Dockerfile.ci` — multi-stage image for all test suites
- `control/src/ci/test_cli.py` — unified `pseti test` CLI (invoked via `pseti test sw/hw/grpc/lint`)

### Integration test topology

```
headnode_net (10.0.1.0/24)
  test-runner   10.0.1.5        pytest runner
  gateway       10.0.1.254      socat TCP bridge → daqnode_net
  redis         10.0.1.20       log queue (logs:ingress)
  loki          10.0.1.21       log aggregation
  storeloki     10.0.1.22       Redis→Loki daemon

daqnode_net (192.168.0.0/24)
  daqnode       192.168.0.10    daq_control gRPC + real hashpipe binary
  daqnode-data  192.168.0.11    daq_data gRPC server
  daqnode-2     192.168.0.20    second daq_control node (two-node tests)
  gateway       192.168.0.254   socat bridge (dual-homed)
```

**Critical CI notes:**
- All containers use `BINDHOST=lo` (loopback). Docker virtual NICs don't support the `TPACKET_V3` mode that hashpipe's net_thread requires on `eth0`.
- `daqnode-2` has its own isolated `daq_data_2` volume to prevent `module.config` write races.
- The `integration-daqnode` entrypoint copies `hashpipe.so` from `/usr/local/lib/` to `/data/` at startup to survive the Docker volume initialization race.
- Cleanup fixtures must call `StopDaq` unconditionally before `CleanupData` — the server checks `hashpipe_pid > 0` (not liveness), so a crashed hashpipe leaves a stale PID that blocks cleanup.

Key env vars used by the test suite:

| Variable | Default | Purpose |
|---|---|---|
| `DAQNODE_DIRECT_HOST` | 192.168.0.10 | direct daqnode IP |
| `DAQNODE_DATA_HOST` | 192.168.0.11 | daq_data gRPC server IP |
| `DAQNODE2_HOST` | 192.168.0.20 | second DAQ node IP |
| `DAQNODE_GATEWAY_HOST` | 10.0.1.254 | socat gateway IP |
| `BINDHOST` | lo | hashpipe network interface (always `lo` in Docker CI) |
| `RUN_REAL_DATA_TESTS` | (unset) | set to `1` to enable tcpreplay/hashpipe tests |
| `ENABLE_TELEMETRY_TESTS` | (unset) | set to `1` to enable Telemetry gRPC tests |

### Upgrade plan
Full five-phase upgrade plan: `docs/plan/control-upgrade-plan.md`

### daq_scripts/ deprecation
`control/daq_scripts/start_daq.py`, `stop_daq.py`, `status_daq.py` are superseded by the `panoseti_grpc.daq_control` gRPC server. They remain for reference during transition but should not be modified.
