# PANOSETI Control — CI Test Suite

All tests live under `control/ci-tests/` and run inside Docker via a single
multi-stage `Dockerfile.ci`.

**Current status:** 460 unit tests passing · integration suite: ~60 tests passing (telemetry enabled by default)

---

## Quick Start

```bash
# From control/
bash ci-tests/run.sh unit          # hardware-agnostic unit tests (~10s, parallel)
bash ci-tests/run.sh integration   # full end-to-end suite (telemetry tests on by default)

# Pass extra pytest args after the suite name
bash ci-tests/run.sh unit -k test_pff
bash ci-tests/run.sh integration -k "TestDaqLifecycle" --timeout=30

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
        GW["gateway<br/>10.0.1.254<br/>(socat TCP bridge)"]
        REDIS["redis<br/>10.0.1.20"]
        LOKI["loki<br/>10.0.1.21"]
        HN["headnode<br/>10.0.1.22<br/>(Telemetry + storeLoki)"]
    end

    subgraph daqnode_net["daqnode_net (192.168.0.0/24)"]
        DN1["daqnode<br/>192.168.0.10<br/>(Hashpipe + gRPC)"]
        DN2["daqnode-2<br/>192.168.0.20<br/>(gRPC only)"]
    end

    TR -->|grpc| GW
    TR -->|grpc direct| DN1
    GW -->|forward 50051| DN1
    DN1 -.->|grpc log push| HN
    HN -->|RPUSH| REDIS
    HN -->|HTTP POST| LOKI
```

### CI Optimizations & Test Lifecycle
* **Test Runner Grouping:** The test runner executes as part of the Docker Compose stack (`up --attach`) rather than an ephemeral `run` container. This allows proper shutdown of the topology using `--abort-on-container-exit`.
* **BuildKit Layer Caching:** The CI pipeline injects `BUILDKIT_INLINE_CACHE=1` and `COMPOSE_BAKE=true`, persisting Python wheels and the compiled C-Hashpipe binaries between GitHub Actions runs.
* **Loki Fast-Boot:** Loki 3.7.0 runs in Single Binary mode with an in-memory ring, bypassing cluster consensus wait times for instant CI availability.
* **Gateway Healthchecks:** Alpine's native `nc` utility is used to health-check the `socat` gateway (`nc -z 127.0.0.1 50051`), ensuring the test runner waits for the port bridge to open before firing gRPC requests.

---

## Test Files

| File | Covers |
|------|--------|
| `test_daq_lifecycle.py` | Start→Status→Stop transitions; idempotency |
| `test_data_collection.py` | `copy_run_dir` mechanics; `CleanupData` invariants |
| `test_concurrent_daq_operations.py` | Concurrent start serialization; rapid Start→Stop cycles |
| `test_gateway_topology.py` | Gateway forwarding and state consistency |
| `test_two_node_direct.py` | Two independent DAQ nodes; isolation guarantees |
| `test_science_streaming.py` | daq_data gRPC simulation path (init_sim + stream_images) |
| `test_loki_pipeline.py` | Redis→Loki log shipping; severity; large payloads; burst |
| `test_hashpipe_logs.py` | Hashpipe log forwarding via Telemetry gRPC |
| `test_real_data_flow.py` | Basic tcpreplay→hashpipe→daq_data→headnode streaming |
| `test_real_data_advanced.py` | Streaming stress tests: inter-frame timing, module consistency, and concurrent gRPC client loads |
| `test_real_data_extended.py` | State resilience: Gateway data routing, client disconnect recovery, and rapid subscription cycling |

---

## Real Hashpipe Tests

The `real_data` test files validate the full data path using `tcpreplay` to inject PCAP packets into a live hashpipe process's shared memory ring buffers. These are skipped by default to save time during standard development.

```bash
bash ci-tests/run.sh integration -k "real_data"
```

### The "Loopback Shortcut" Requirement
When running `tcpreplay` in Docker Desktop (Mac/Windows), the virtualized network stack blindly passes packets regardless of their Destination MAC Address. However, **native Linux (GitHub Actions)** strictly enforces hardware networking rules. Injecting packets with foreign MAC addresses onto the `eth0` bridge will cause the Linux kernel to drop them before Hashpipe can read them.

To guarantee packets reach Hashpipe in CI, the tests enforce **Loopback injection**:
1. Hashpipe is bound to the loopback interface (`BINDHOST=lo` / `127.0.0.1`).
2. `tcpreplay` injects packets directly into `lo` (`--intf1=lo`).
3. The Linux kernel uses the loopback shortcut to instantly route the packets from the transmit queue to the receive queue, entirely bypassing MAC filtering and MTU restrictions.

Requirements (satisfied automatically in the `integration-daqnode` Docker image):
- `hashpipe` binary + `hashpipe.so` plugin at `/data/hashpipe.so`
- `tcpreplay` in PATH
- PCAP file at `/app/ci-tests/integration/data/*.pcapng`

---

## Local Development (without Docker)

```bash
cd control
pip install -e ".[dev]"

# Unit tests (fast)
pytest ci-tests/unit/
```