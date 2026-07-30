![PANOSETI CI](https://github.com/panoseti/panoseti/actions/workflows/ci-tests.yml/badge.svg)
# PANOSETI Software

Software for the [PANOSETI Project](https://oirlab.ucsd.edu/PANOSETI.html) — a wide-field optical/near-infrared telescope array searching for nanosecond-scale transients (ETI signals, gamma-ray bursts, fast radio bursts).

Full documentation: [PANOSETI Wiki](https://github.com/panoseti/panoseti/wiki)

---

## Repository Structure

```
panoseti-software/
├── control/            # Instrument control system (Python) — primary dev area
├── grpc/               # Submodule: gRPC service layer (Python) — panoseti_grpc
├── analysis/           # Data analysis framework (Python + Jupyter)
├── anomaly-detection/  # ML-based anomaly detection pipeline
├── cloud-detection/    # ML-based cloud/weather detection
├── adc-to-pe/          # ADC count → photoelectron calibration
├── util/               # Shared C++ utilities (PFF parser, image processing)
├── dome-mount-control/ # Dome and telescope mount control
├── dgnss/              # Differential GNSS timing utilities
├── daq/                # Submodule: Hashpipe DAQ C plugin
├── web/                # Submodule: observatory web dashboard
├── grafana/             # Submodule: Grafana dashboard/datasource provisioning
├── alloy/               # Grafana Alloy log-shipping config (jsonl → Loki)
├── pypff/               # Submodule: PFF file format Python bindings
└── wiki_docs/            # Submodule: this wiki's source (GitHub wiki repo)
```

---

## System Overview

An observatory consists of one or more domes, each containing detector **modules** (4 quabo SiPM boards each). A **head node** runs the control system; one or more **DAQ nodes** receive UDP science packets from quabos and write [PFF files](https://github.com/panoseti/panoseti/wiki/Data-file-format).

```
Head Node
  ├── UDP → Quabos (detector boards, ports 60000–60003)
  ├── gRPC (unified server, default port 50051) → DAQ Nodes (Hashpipe DAQ pipeline)
  ├── HTTP → Web Power Switches
  └── SSH → Telescope Mount
```

Both the head node and every DAQ node run a single **unified `panoseti_grpc` server** hosting multiple services (DAQ Control, DAQ Data, Telemetry) on one port — bind ports are env-driven (`HEADNODE_GRPC_PORT`/`DAQNODE_GRPC_PORT`), not hardcoded, so a co-located head+DAQ node deployment and a multi-node fleet use the same images and compose files. See [Deploying the Modernized Control System](https://github.com/panoseti/panoseti/wiki/Deploying-the-Modernized-Control-System).

See: [DAQ system overview](https://github.com/panoseti/panoseti/wiki/daq-system-overview) · [Nodes and modules](https://github.com/panoseti/panoseti/wiki/Nodes-and-modules)

---

## Control System (`control/`)

Manages the full observing session lifecycle through the unified `pseti` CLI (Python ≥3.14). See [Control system implementation](https://github.com/panoseti/panoseti/wiki/control-system-implementation) and the [`pseti` CLI Reference](https://github.com/panoseti/panoseti/wiki/PSETI-CLI-Reference).

### Install

```bash
cd control
pip install -e ".[dev]"          # dev install (includes test/lint tools)
```

### Observing session

```bash
pseti session-start     # power on, get UIDs, calibrate, start daemons
pseti start             # configure quabos, start DAQ recording
pseti stat               # check recording status and disk usage
pseti stop               # stop recording; enqueues background transfer
pseti session-stop       # power off, stop daemons
```

See: [Observing runs](https://github.com/panoseti/panoseti/wiki/observing-runs) · [Configuration files](https://github.com/panoseti/panoseti/wiki/Configuration-files) · [Observing Run Transactions](https://github.com/panoseti/panoseti/wiki/Observing-Run-Transactions)

### Config validation (no hardware required)

```bash
pseti val
```

### Deployment (`pseti admin`)

Deploys the containerized (or bare-metal) gRPC server + Hashpipe stack to the head node and every DAQ node, from one place — every node's job runs concurrently, not sequentially:

```bash
pseti admin deploy all --mode docker    # head node + every DAQ node in daq_config.json
pseti admin status all
pseti health                            # all-systems-green check: config, WPS, Quabos, gRPC, containers
```

### Tests

```bash
pseti test sw unit          # Tier 1: fast unit tests, no Docker
pseti test sw all           # Tiers 1-5 (unit, logic, fleet, chaos, integration)
pseti test lint             # Ruff + MyPy
pseti test hw run           # hardware-in-the-loop (real Quabos + DAQ node required)
```

See [Testing the Control System](https://github.com/panoseti/panoseti/wiki/Testing-the-Control-System), [Hardware-in-the-Loop Testing](https://github.com/panoseti/panoseti/wiki/Hardware-in-the-Loop-Testing), and [CI Testing Infrastructure](https://github.com/panoseti/panoseti/wiki/CI-Testing-Infrastructure).

---

## gRPC Services (`grpc/` → `panoseti_grpc`)

Structured RPC layer between the control system and DAQ/GNSS nodes, replacing SSH-based control. Python ≥3.14.

```bash
cd ../panoseti_grpc && pip install -e ".[dev]"
```

| Service | Status | Purpose |
|---------|--------|---------|
| DAQ Control | Production | Start/stop/status Hashpipe on DAQ nodes, manifest generation, selective cleanup |
| DAQ Data | Production | Stream real-time science images from Hashpipe shared memory |
| Telemetry | Beta | Device status → Redis/InfluxDB/Grafana; log shipping via Grafana Alloy → Loki |
| U-blox Control | 🔴 Deprecated | GNSS chip control — disabled by default; use `Telemetry.ReportStatus` with `GnssPayload` instead |

All active services run on one unified server process (`pseti-grpc server`). See [Deploying the Modernized Control System](https://github.com/panoseti/panoseti/wiki/Deploying-the-Modernized-Control-System) for the full deployment model.

---

## Data Analysis (`analysis/`)

Framework for iterative algorithm development with versioned results. See [Analysis framework](https://github.com/panoseti/panoseti/wiki/Analysis-framework).

Covers: image pulse finding, PH coincidence between domes, pixel statistics, movie generation, repeated event searches.

---

## ML Pipelines

| Directory | Purpose |
|-----------|---------|
| `anomaly-detection/` | VAE-based anomaly detection on science frames |
| `cloud-detection/` | CNN classifier for cloud/weather conditions |
| `adc-to-pe/` | ADC-to-photoelectron calibration curves |

---

## Data Format

Science data is stored in [PFF (PanoSETI File Format)](https://github.com/panoseti/panoseti/wiki/Data-file-format) files. Each file is a sequence of JSON header blocks + binary image blocks. Data products: `img8`, `img16` (movie modes), `ph256`, `ph1024` (pulse-height modes).

See: [File names](https://github.com/panoseti/panoseti/wiki/Data-file-names) · [Pixel indexing](https://github.com/panoseti/panoseti/wiki/Pixel-indexing)

---

## Timing

Two precision timing sources:
- **White Rabbit** — fiber-based, ~ns precision ([WR intro](https://github.com/panoseti/panoseti/wiki/White-Rabbit-introduction))
- **GNSS (ZED-F9T)** — satellite-based ([UBX messages](https://github.com/panoseti/panoseti/wiki/Necessary-UBX-Messages))

---

## Observability

Housekeeping, GPS, and White Rabbit telemetry flow: quabos/daemons → Redis (hot) → InfluxDB (time series) → Grafana (dashboards). Structured logs (`{service}.jsonl`) are shipped by Grafana Alloy to Loki. Both stacks are provisioned as part of the head node's Docker Compose deployment — see [Deploying the Modernized Control System](https://github.com/panoseti/panoseti/wiki/Deploying-the-Modernized-Control-System).

---

## Development Roadmap

[`plans/control-upgrade-plan.md`](plans/control-upgrade-plan.md) has the original control-system modernization motivation and initial phase plan (Python packaging, gRPC migration, test coverage). Current state, past what that document originally scoped:

- [x] `pyproject.toml` packaging, Python ≥3.14
- [x] 5-tier software test suite (unit/logic/fleet/chaos/integration) + hardware-in-the-loop suite
- [x] SSH → gRPC migration (`start`/`stop`/`stat` drive DAQ nodes over the unified gRPC server, not SSH)
- [x] Containerized deployment (`pseti admin deploy/build`, concurrent across nodes), Grafana/Loki/Alloy observability stack
- [ ] Multi-site fleet rollout (Palomar, Lick, UCB) on the modernized stack
