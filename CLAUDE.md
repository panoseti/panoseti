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
```

### Config validation (no hardware required)
```bash
cd control
python start.py --validate-only    # validate all configs without touching hardware
```

### Observing run lifecycle
```bash
cd control
python session_start.py            # power on, get UIDs, calibrate, start daemons
python start.py                    # configure quabos, start DAQ recording
python status.py                   # check recording status and disk usage
python stop.py                     # stop recording, collect data
python session_stop.py             # power off, stop daemons
```

### Run tests
```bash
cd control
pip install -e ".[dev]"

# Unit tests (460 tests, no hardware required)
pytest ci-tests/unit/ -v --tb=short

# With coverage report
pytest ci-tests/unit/ --cov=utils --cov-report=term-missing

# Via Docker CI — unit suite (parallel with -n auto, ~10s)
bash ci-tests/run.sh unit

# Via Docker CI — full integration suite (43 passing, 7 skipped)
bash ci-tests/run.sh integration

# Integration: single test group
bash ci-tests/run.sh integration -- -k "TestDaqLifecycle"

# Enable Loki/Redis telemetry tests
ENABLE_TELEMETRY_TESTS=1 bash ci-tests/run.sh integration

# Real hashpipe + tcpreplay tests (requires RUN_REAL_DATA_TESTS=1)
RUN_REAL_DATA_TESTS=1 bash ci-tests/run.sh integration -- -k "real_data"
```

### Lint and type-check
```bash
ruff check control/utils/ control/driver/
mypy control/utils/config_file.py --ignore-missing-imports
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
3. `stop.py` — kills Hashpipe via gRPC (`DaqControlClient.StopDaq()`), stops housekeeping recording, rsync's PFF files from DAQ nodes to head node

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
bash scripts/run-ci-tests/run-daq-data-ci-test.sh
```

**The four services:**

| Service | Module | Default port | Purpose |
|---------|--------|-------------|---------|
| DAQ Data | `panoseti_grpc.daq_data` | 50051 | Streams real-time science images from Hashpipe shared memory |
| DAQ Control | `panoseti_grpc.daq_control` | 50051 | Start/stop/status Hashpipe on DAQ nodes, clean up run data |
| U-blox Control | `panoseti_grpc.ublox_control` | 50051 | Configure ZED-F9T GNSS timing receivers, stream UBX messages |
| Telemetry | `panoseti_grpc.telemetry` | 50051 | Centralized health/metadata: logs → Loki, status → Redis/InfluxDB |

All servers follow the same launch pattern:
```bash
python -m panoseti_grpc.<service>.server   # e.g., panoseti_grpc.daq_data.server
GRPC_PORT=50052 python -m panoseti_grpc.telemetry.server
```

**DAQ Data service** — the most actively used service. Streams `PanoImage` objects from Hashpipe shared memory to any subscriber. Client usage:
```python
from panoseti_grpc.daq_data.client import DaqDataClient
async with DaqDataClient(host, port) as client:
    await client.init_hp_io(run_dir, module_id)
    async for image in client.stream_images():
        process(image)
```
For local testing without hardware, `panoseti_grpc.daq_data.simulate` generates synthetic image streams.

**DAQ Control service** — replaces the SSH calls in `control/daq_scripts/` with RPCs. Key RPCs: `StartDaq()` (launches Hashpipe subprocess), `StopDaq()` (SIGINT), `StatusDaq()` (PID + disk usage), `CleanupData()`. The server (`python -m panoseti_grpc.daq_control.server`) runs as a systemd service on each DAQ node (port 50051). `control/daq_scripts/start_daq.py`, `stop_daq.py`, `status_daq.py` are **deprecated** in favor of this service.

**Telemetry service** — consumed by `control/daemons/capture_telemetry_service.py`. Configured via `control/daemons/capture_telemetry_service/telemetry_config.toml`. Supports two storage modes:
- **Strict** (production): validated Pydantic payloads (e.g., `GnssPayload`, `DewPayload`) → permanent Redis + InfluxDB
- **Flexible** (dev): arbitrary JSON under `DEV_`-prefixed keys → 24 h TTL in Redis only

To get a structured logger that forwards to the Telemetry service:
```python
from panoseti_grpc.telemetry.logger import get_logger
logger = get_logger("my_service")  # injects git commit, PID, hostname, thread
logger.info("message")
```

**Proto files** live in `../panoseti_grpc/protos/`. The `panoseti_util/` sub-package inside `panoseti_grpc` re-exports PFF reading/writing (`pff.py`) and config utilities (`config_file.py`) for use within the gRPC servers — prefer these over duplicating logic.

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
`control/utils/file_xfer.py` and `control/utils/collect.py` handle SSH/SCP to DAQ nodes. All remote commands should use `subprocess.run(['ssh', f'{user}@{host}', cmd], ...)` (list form, not shell string) to avoid injection.

### Data config validation constraints
- `integration_time_usec` must be a multiple of 10 and evenly divide 1,000,000
- `pe_threshold` ≥ 2.0 (pulse height mode), ≥ 1.0 (image/movie mode)
- `run_type` max 14 chars, no `.`, `_`, or spaces
- `detector_overvoltage` must match between `obs_config.json` and `data_config.json`

---

## Testing Infrastructure

### Python version requirement
`control/pyproject.toml` sets `requires-python = ">=3.9"`. Target migration to 3.14+ syntax incrementally.

### Test locations
- `control/ci-tests/unit/` — hardware-agnostic Python unit tests (460 tests, 10 modules)
- `control/ci-tests/integration/` — end-to-end Docker integration tests (43 passing, 7 skipped)
- `control/ci-tests/Dockerfile.ci` — multi-stage image for all test suites
- `control/ci-tests/run.sh` — unified runner (`unit` or `integration`)

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
