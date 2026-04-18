# PANOSETI Control — CI Test Suite

All tests live under `control/ci/` and run inside Docker via a single
multi-stage `Dockerfile.ci` using `uv` for high-performance builds.

**Current status:** 460 unit tests passing · integration suite: 65 tests passing

---

## Quick Start

The new CI infrastructure uses **persistent background daemons**. You start the stack once, and then run tests instantly without waiting for container boot times.

```bash
# 1. Start the background infrastructure
python ci/qa.py up

# 2. Run tests instantly
python ci/qa.py unit           # Parallel unit tests (~3s)
python ci/qa.py integration    # E2E integration tests (~75s)
python ci/qa.py lint           # Ruff & MyPy (concurrent)

# 3. Targeted debugging (pass any pytest args)
python ci/qa.py unit -k test_pff
python ci/qa.py integration -k TestDaqLifecycle

# 4. Infrastructure management
python ci/qa.py build          # Rebuild images (uv cached)
python ci/qa.py restart        # Restart daemons
python ci/qa.py down           # Tear down everything
```

---

## Modern CI Architecture

We have migrated to **Python 3.14** and **uv**. Our containers are designed to be "inner-loop" friendly:

*   **Persistent Daemons:** Containers run `sleep infinity` and are reused across test runs.
*   **Live Mounting:** The local `control/` directory is volume-mounted into `/app`. Edits you make on your host are instantly available for the next test run.
*   **Venv Isolation:** Python dependencies live in `/opt/venv`, safely isolated from your local volume mounts.
*   **Blazing Fast Builds:** We use BuildKit cache mounts (`--mount=type=cache,target=/root/.cache/uv`) and `uv sync` layers to ensure dependencies are only re-evaluated when `pyproject.toml` or `uv.lock` changes.

### Integration Topology (Fleet Testing)

The integration suite simulates a Palomar-like VPN topology. We use two distinct topologies depending on whether we are testing E2E high-throughput data or distributed control logic.

#### 1. Loopback Data Path (E2E Integration)
Used in `real_data` tests to verify the high-throughput pipeline (`tcpreplay` -> `hashpipe`).
- **Isolation:** Each DAQ node runs its own **dedicated local `tcpreplay` instance** inside its container.
- **Path:** Science packets never leave the container; they flow through the local `lo` (loopback) interface to bypass MTU and MAC filtering overhead.

```mermaid
graph BT
    %% Sinks
    subgraph sinks1 ["Telemetry Sinks"]
        direction LR
        LOKI1["loki<br/>(cold storage)"]
        REDIS1["redis<br/>(hot storage)"]
    end

    %% Headnode
    subgraph headnode_net1 ["Headnode Network (10.0.1.0/24)"]
        direction LR
        HN1["headnode<br/>(Telemetry gRPC)"]
        TR1["int-tester<br/>(pytest/start.py)"]
    end

    %% DAQ Nodes (Local Loopback)
    subgraph daqnode_net1 ["DAQ Fleet (192.168.0.0/24)"]
        direction LR
        subgraph DN0_BOX ["daq-0 Container"]
            DN0["hashpipe"]
            PCAP0["tcpreplay"]
            PCAP0 ---->|"UDP Science (lo)"| DN0
        end
        subgraph DN1_BOX ["daq-1 Container"]
            DN1["hashpipe"]
            PCAP1["tcpreplay"]
            PCAP1 ---->|"UDP Science (lo)"| DN1
        end
    end

    %% Global Flows
    DN0_BOX  -.->|"gRPC Log Push"| HN1
    DN1_BOX  -.->|"gRPC Log Push"| HN1
    HN1      ====> REDIS1
    HN1      ====> LOKI1
    TR1      ===>|"gRPC / SSH"| DN0_BOX
    TR1      ===>|"gRPC / SSH"| DN1_BOX
```

#### 2. Distributed Control Path (Chaos/Logic)
Used in chaos scenarios to verify `start.py`/`stop.py` transaction integrity and rollback logic.
- **Shared Service:** A single **`mock-quabo` service** simulates a full 4-quabo module (e.g. Module 200).
- **Path:** Command packets flow from `start.py` (Headnode) to `mock-quabo` (External), and telemetry flows back up. Science packets can be triggered to any node in the fleet via the external `eth0` network.

```mermaid
graph BT
    %% Sinks
    subgraph sinks2 ["Telemetry Sinks"]
        direction LR
        LOKI2["loki<br/>(cold storage)"]
        REDIS2["redis<br/>(hot storage)"]
    end

    %% Headnode
    subgraph headnode_net2 ["Headnode Network (10.0.1.0/24)"]
        direction LR
        HN2["headnode<br/>(Telemetry gRPC)"]
        TR2["int-tester<br/>(pytest/start.py)"]
    end

    %% DAQ Fleet
    subgraph daqnode_net2 ["DAQ Fleet Network (192.168.0.0/24)"]
        direction LR
        DN_FLEET["N-Node Fleet<br/>(daq-0 ... daq-N)"]
        MQ2["mock-quabo<br/>(Module Simulator)"]
    end

    %% Logic/Telemetry Flows
    MQ2  ---->|"UDP Science (eth0)"| DN_FLEET
    MQ2  -.->|"UDP Housekeeping"| HN2
    DN_FLEET -.->|"gRPC Log Push"| HN2
    HN2  ====> REDIS2
    HN2  ====> LOKI2

    %% Orchestration
    TR2  ===>|"gRPC / SSH"| DN_FLEET
    TR2  ===>|"UDS Control"| MQ2
```

---

## Real Hashpipe Tests

The `real_data` test files validate the full data path using `tcpreplay` to inject PCAP packets into a live hashpipe process. These are now integrated into the standard suite and pass under the new architecture.

```bash
# Run only real data tests
python ci/qa.py integration -k "real_data"
```

### The "Loopback Shortcut" Requirement
To guarantee packets reach Hashpipe in CI, the tests enforce **Loopback injection**:
1. Hashpipe is bound to the loopback interface (`BINDHOST=lo`).
2. `tcpreplay` injects packets directly into `lo`.
3. The Linux kernel uses the loopback shortcut to bypass MAC filtering and MTU restrictions.

---

## Local Development (without Docker)

If you prefer to run tests natively, ensure you have `uv` installed:

```bash
# Sync environment
uv sync --all-extras

# Run tests
uv run pytest ci/unit/
```
