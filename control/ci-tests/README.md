# PANOSETI Control — CI Test Suite

All tests live under `control/ci-tests/` and run inside Docker via a single
multi-stage `Dockerfile.ci`.

**Current status:** 460 unit tests passing · integration suite: ~50 tests passing (telemetry enabled by default)

---

## Quick Start

```bash
# From control/
bash ci-tests/run.sh unit          # hardware-agnostic unit tests (~10s, parallel)
bash ci-tests/run.sh integration   # full end-to-end suite (telemetry tests on by default)

# Pass extra pytest args after the suite name
bash ci-tests/run.sh unit -- -k test_pff
bash ci-tests/run.sh integration -- -k "TestDaqLifecycle" --timeout=30

# Real hashpipe + tcpreplay data flow tests
RUN_REAL_DATA_TESTS=1 bash ci-tests/run.sh integration -- -k "real_data"

# Skip telemetry tests (faster, no headnode Telemetry service required locally)
ENABLE_TELEMETRY_TESTS=0 bash ci-tests/run.sh integration
```

---

## Integration Test Architecture

The integration suite simulates a Palomar-like VPN topology. Each daqnode runs
the **unified panoseti-server** (daq_data + daq_control on a single port) and
forwards logs to the **headnode** Telemetry gRPC service.

```mermaid
graph TB
    subgraph headnode_net["headnode_net (10.0.1.0/24)"]
        TR["test-runner<br/>10.0.1.5<br/>(pytest)"]
        GW_H["gateway<br/>10.0.1.254"]
        RD["redis<br/>10.0.1.20"]
        LK["loki<br/>10.0.1.21"]
        HN["headnode<br/>10.0.1.22<br/>(Telemetry gRPC + storeLoki)"]
    end

    subgraph daqnode_net["daqnode_net (192.168.0.0/24)"]
        DN["daqnode<br/>192.168.0.10<br/>(daq_data + daq_control + hashpipe)"]
        DN2["daqnode-2<br/>192.168.0.20<br/>(daq_data + daq_control)"]
        GW_D["gateway<br/>192.168.0.254"]
    end

    TR -- "gRPC direct" --> DN
    TR -- "gRPC via NAT" --> GW_H --> GW_D --> DN
    TR -- "gRPC" --> DN2
    DN -- "gRPC log records" --> HN
    DN2 -- "gRPC log records" --> HN
    HN -- "RPUSH logs:ingress" --> RD
    HN -- "storeLoki" --> LK

    style headnode_net fill:#e8f4e8
    style daqnode_net fill:#e8e8f4
```

**Shared volumes:**
- `daq_data` (`/data`) — PFF files shared between `daqnode` and `test-runner`
- `daq_data_2` (`/data`) — private volume for `daqnode-2` (prevents `module.config` conflicts)

### Unified server: one port for daq_data + daq_control

The old topology had a separate `daqnode-daqdata-snoop` container (192.168.0.11)
running only `daq_data.server` to avoid the port-50051 conflict with
`daq_control.server`. The **unified panoseti-server** (`panoseti-server --config
ci-tests/integration/configs/daqnode/server.toml`) hosts both services on port
50051 within the same process. This eliminates the second container, the
`hashpipe_uds` shared volume, and the startup race it caused.

`DAQNODE_DATA_HOST` now defaults to `DAQNODE_DIRECT_HOST` (192.168.0.10).

### Why `BINDHOST=lo`?

Docker virtual NICs (`eth0`) don't support the `TPACKET_V3` ring-buffer mode
that panoseti_daq's net_thread requires. Loopback (`lo`) does. All CI runs
use `BINDHOST=lo`; production uses the real interface name.

### hashpipe.so startup copy

The `integration-daqnode` image stores `hashpipe.so` at `/usr/local/lib/panoseti_hashpipe.so`
(outside the shared volume). The container CMD copies it to `/data/hashpipe.so` at startup,
surviving the Docker volume initialization race on the shared `daq_data` volume.

### Headnode: Telemetry gRPC + storeLoki

The `headnode` container (10.0.1.22) runs two processes:
1. `panoseti-server --config ci-tests/integration/configs/headnode/server.toml` — Telemetry gRPC (port 50051)
2. `storeLoki.py` — ships `logs:ingress` Redis queue to Loki

DAQ nodes are configured with `grpc_logging=true` + `HEADNODE_IP=10.0.1.22`, so
all daq_control log records are forwarded to the headnode Telemetry service, then
on to Redis and Loki. `test_hashpipe_logs.py` verifies this end-to-end path.

### Cleanup fixtures

Always call `StopDaq` unconditionally before `CleanupData`. The server checks
`hashpipe_pid > 0` (not liveness) — a crashed hashpipe leaves a stale PID that
blocks cleanup. `StopDaq` is idempotent and resets the PID.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DAQNODE_DIRECT_HOST` | 192.168.0.10 | unified gRPC server IP (daq_data + daq_control) |
| `DAQNODE_DATA_HOST` | `DAQNODE_DIRECT_HOST` | daq_data gRPC IP (same as direct in unified topology) |
| `DAQNODE2_HOST` | 192.168.0.20 | second DAQ node IP |
| `DAQNODE_GATEWAY_HOST` | 10.0.1.254 | socat gateway IP |
| `HEADNODE_HOST` | 10.0.1.22 | headnode Telemetry gRPC IP |
| `GRPC_PORT` | 50051 | gRPC port for all services |
| `DAQ_DATA_DIR` | /data | data directory on daqnode |
| `HEAD_DATA_DIR` | /data/head | data destination on headnode |
| `DAQNODE_CONTAINER_NAME` | ctl-int-daqnode-1 | Docker container name for pause/unpause tests |
| `BINDHOST` | lo | hashpipe network interface (`lo` required in Docker CI) |
| `ENABLE_TELEMETRY_TESTS` | **1** (in compose) | set to `0` to skip `test_hashpipe_logs.py` |
| `RUN_REAL_DATA_TESTS` | 0 | set to `1` to run tcpreplay/hashpipe tests |

---

## Test Files

### Unit (`ci-tests/unit/`) — 460 tests

No hardware or networking required. Unit tests run in parallel with `pytest-xdist -n auto`.

| File | Coverage |
|---|---|
| `test_pydantic_models.py` | Config Pydantic schemas |
| `test_config_file.py` | Config loading, range expansion |
| `test_global_validator.py` | Cross-config rules |
| `test_config_validator.py` | Network reachability checks |
| `test_pff.py` | PFF file parser |
| `test_util.py` | Utility helpers |
| `test_redis_utils.py` | Redis key/stream utilities |
| `test_image_quantiles.py` | Image statistics |
| `test_quabo_driver.py` | UDP quabo driver: packet construction, opcodes, field encoding |
| `test_quabo_driver_protocol.py` | Protocol math: BOARDLOC/IP, HK packet layout, HV conversion, MAROC structure, trigger mask |

### Integration (`ci-tests/integration/`)

Requires the full Docker compose stack.

| File | Coverage |
|---|---|
| `test_config_validation.py` | CI config files pass Pydantic + cross-config rules |
| `test_daq_lifecycle.py` | Start/Stop/Status lifecycle; disk usage; run dir isolation |
| `test_data_collection.py` | Collect + cleanup transaction; failure recovery; edge cases |
| `test_concurrent_daq_operations.py` | Concurrent start serialization; rapid Start→Stop cycles |
| `test_gateway_topology.py` | Gateway forwarding and state consistency |
| `test_two_node_direct.py` | Two independent DAQ nodes; isolation guarantees |
| `test_science_streaming.py` | daq_data gRPC simulation path (init_sim + stream_images) |
| `test_loki_pipeline.py` | Redis→Loki log shipping; severity; large payloads; burst |
| `test_hashpipe_logs.py` | Hashpipe log forwarding via Telemetry gRPC (on by default in compose) |
| `test_real_data_flow.py` | tcpreplay→hashpipe→daq_data→headnode (requires `RUN_REAL_DATA_TESTS=1`) |

---

## Real Hashpipe Tests

`test_real_data_flow.py` tests the full data path using tcpreplay to inject
PCAP packets into a live hashpipe process.  Skipped by default.

```bash
RUN_REAL_DATA_TESTS=1 bash ci-tests/run.sh integration -- -k "real_data"
```

Requirements (satisfied in the `integration-daqnode` Docker image):
- `hashpipe` binary + `hashpipe.so` plugin at `/data/hashpipe.so`
- `tcpreplay` in PATH
- PCAP file at `/app/ci-tests/integration/data/*.pcapng`

---

## Local Development (without Docker)

```bash
cd control
pip install -e ".[dev]"

# Unit tests (fast)
pytest ci-tests/unit/ -v --tb=short

# Single test module
pytest ci-tests/unit/test_quabo_driver_protocol.py -v
```

---

## Requirements

- Docker Engine 24+
- Docker Compose v2 (`docker compose`, no hyphen)
- ~1 GB disk for the test images (Python 3.14 slim + hashpipe compilation)
