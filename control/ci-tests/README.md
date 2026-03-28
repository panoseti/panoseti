# PANOSETI Control — CI Test Suite

All tests live under `control/ci-tests/` and run inside Docker via a single
multi-stage `Dockerfile.ci`.

---

## Quick Start

```bash
# From control/
bash ci-tests/run.sh unit          # hardware-agnostic unit tests
bash ci-tests/run.sh integration   # full end-to-end integration suite
```

Pass extra pytest args after the suite name:

```bash
bash ci-tests/run.sh unit -- -k test_pff
bash ci-tests/run.sh integration -- -k "TestDaqLifecycle" --timeout=30
```

---

## Integration Test Architecture

The integration suite simulates a Palomar-like VPN topology using Docker networks:

```
headnode_net (10.0.1.0/24)
  test-runner  10.0.1.5        — pytest
  gateway      10.0.1.254      — socat TCP bridge → daqnode_net
  redis        10.0.1.20       — telemetry log queue
  loki         10.0.1.21       — log aggregation
  storeloki    10.0.1.22       — Redis→Loki daemon under test

daqnode_net (192.168.0.0/24)
  daqnode      192.168.0.10    — daq_control gRPC + real hashpipe binary
  daqnode-data 192.168.0.11    — daq_data gRPC server
  daqnode-2    192.168.0.20    — second daq_control node (two-node tests)
  gateway      192.168.0.254   — socat bridge (same container, two NICs)

Shared volumes
  daq_data     /data           — PFF files (daqnode ↔ daqnode-data ↔ test-runner)
  hashpipe_uds /tmp            — hashpipe UDS sockets (daqnode ↔ daqnode-data)
```

### Why two daqnode containers?

`daq_control.server` and `daq_data.server` both hardcode TCP port 50051.
Running them in the same container causes a port conflict.  The workaround
is to give each server its own IP address so both can own port 50051.

The long-term fix (upstream `panoseti_grpc` change) is to read a `DATA_PORT`
env var in `daq_data/server.py` and `daq_data/client.py`.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DAQNODE_DIRECT_HOST` | 192.168.0.10 | direct daq_control IP |
| `DAQNODE_DATA_HOST` | 192.168.0.11 | daq_data gRPC server IP |
| `DAQNODE2_HOST` | 192.168.0.20 | second DAQ node IP |
| `DAQNODE_GATEWAY_HOST` | 10.0.1.254 | socat gateway IP |
| `GRPC_PORT` | 50051 | gRPC port for all services |
| `DAQ_DATA_DIR` | /data | data directory on daqnode |
| `HEAD_DATA_DIR` | /data/head | data destination on headnode |
| `DAQNODE_CONTAINER_NAME` | ctl-int-daqnode-1 | Docker container name for pause/unpause tests |
| `RUN_REAL_DATA_TESTS` | (unset) | set to `1` to run tcpreplay/hashpipe tests |
| `ENABLE_TELEMETRY_TESTS` | (unset) | set to `1` to run Telemetry gRPC pipeline tests |

---

## Test Files

### Unit (`ci-tests/unit/`)

No hardware or networking required.  Tests run against `fakeredis`.

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
| `test_quabo_driver.py` | UDP quabo driver |

### Integration (`ci-tests/integration/`)

Requires the full Docker compose stack.

| File | Coverage |
|---|---|
| `test_config_validation.py` | CI config files pass Pydantic + cross-config rules |
| `test_daq_lifecycle.py` | Start/Stop/Status via direct and gateway paths |
| `test_data_collection.py` | Collect + cleanup transaction; failure recovery |
| `test_gateway_topology.py` | Gateway-specific: forwarding, state consistency |
| `test_two_node_direct.py` | Two independent DAQ nodes |
| `test_science_streaming.py` | daq_data gRPC: simulation path (init_sim + stream_images) |
| `test_loki_pipeline.py` | Redis→Loki log shipping |
| `test_hashpipe_logs.py` | Hashpipe log forwarding (ENABLE_TELEMETRY_TESTS=1) |
| `test_real_data_flow.py` | tcpreplay→hashpipe→daq_data→headnode (RUN_REAL_DATA_TESTS=1) |

---

## Real Hashpipe Tests

`test_real_data_flow.py` tests the full data path using tcpreplay to inject
PCAP packets into a live hashpipe process.  Skipped by default.

```bash
# Run with real hashpipe data injection
RUN_REAL_DATA_TESTS=1 bash ci-tests/run.sh integration -- -k "real_data"
```

Requirements (already satisfied in the `integration-daqnode` Docker image):
- `hashpipe` binary + `hashpipe.so` plugin at `/data/hashpipe.so`
- `tcpreplay` in PATH
- PCAP file at `/app/ci-tests/integration/data/*.pcapng`
- `hashpipe_uds` volume shared at `/tmp` between `daqnode` and `daqnode-data`

---

## Local Development (without Docker)

```bash
cd control
pip install -e ".[dev]"

# Unit tests
pytest ci-tests/unit/ -v --tb=short

# Single test module
pytest ci-tests/unit/test_pff.py -v
```

---

## Requirements

- Docker Engine 24+
- Docker Compose v2 (`docker compose`, no hyphen)
- ~1 GB disk for the test images (Python 3.14 slim + hashpipe compilation)
