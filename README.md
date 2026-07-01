![PANOSETI gRPC CI](https://github.com/panoseti/panoseti/actions/workflows/ci.yml/badge.svg)
# PANOSETI Software
 
Software for the [PANOSETI Project](https://oirlab.ucsd.edu/PANOSETI.html) — a wide-field optical/near-infrared telescope array searching for nanosecond-scale transients (ETI signals, gamma-ray bursts, fast radio bursts).

Full documentation: [PANOSETI Wiki](https://github.com/panoseti/panoseti/wiki)

---

## Repository Structure

```
panoseti-software/
├── control/            # Instrument control system (Python) — primary dev area
├── analysis/           # Data analysis framework (Python + Jupyter)
├── anomaly-detection/  # ML-based anomaly detection pipeline
├── cloud-detection/    # ML-based cloud/weather detection
├── adc-to-pe/          # ADC count → photoelectron calibration
├── util/               # Shared C++ utilities (PFF parser, image processing)
├── dome-mount-control/ # Dome and telescope mount control
├── dgnss/              # Differential GNSS timing utilities
├── daq/                # Submodule: Hashpipe DAQ C plugin
├── grpc/               # Submodule: gRPC service layer (Python)
└── web/                # Submodule: observatory web dashboard
```

---

## System Overview

An observatory consists of one or more domes, each containing detector **modules** (4 quabo SiPM boards each). A **head node** runs the control system; one or more **DAQ nodes** receive UDP science packets from quabos and write [PFF files](https://github.com/panoseti/panoseti/wiki/Data-file-format).

```
Head Node
  ├── UDP → Quabos (detector boards, ports 60000–60003)
  ├── gRPC (50051) → DAQ Nodes (Hashpipe DAQ pipeline)
  ├── HTTP → Web Power Switches
  └── SSH → Telescope Mount
```

See: [DAQ system overview](https://github.com/panoseti/panoseti/wiki/daq-system-overview) · [Nodes and modules](https://github.com/panoseti/panoseti/wiki/Nodes-and-modules)

---

## Control System (`control/`)

Manages the full observing session lifecycle. See [Control system implementation](https://github.com/panoseti/panoseti/wiki/control-system-implementation).

### Install

```bash
cd control
pip install -e ".[dev]"          # dev install (includes test/lint tools)
```

### Observing session

```bash
cd control
python session_start.py    # power on, calibrate, start daemons
python start.py            # configure quabos, start DAQ recording
python status.py           # check status and disk usage
python stop.py             # stop recording, collect data
python session_stop.py     # power off, stop daemons
```

See: [Observing runs](https://github.com/panoseti/panoseti/wiki/observing-runs) · [Configuration files](https://github.com/panoseti/panoseti/wiki/Configuration-files)

### Config validation (no hardware required)

```bash
cd control && python start.py --validate-only
```

### Tests

```bash
cd control
pytest ci/unit/ -v --tb=short          # 460 unit tests, no hardware needed
bash ci/run.sh unit                    # same, via Docker (parallel with -n auto)
bash ci/run.sh integration             # end-to-end: 43 passing, 7 skipped
```

---

## gRPC Services (`grpc/` → `panoseti_grpc`)

Structured RPC layer between the control system and DAQ/GNSS nodes, replacing SSH-based control.

```bash
cd ../panoseti_grpc && pip install -e ".[dev]"
```

| Service | Purpose |
|---------|---------|
| DAQ Control | Start/stop/status Hashpipe on DAQ nodes |
| DAQ Data | Stream real-time science images from Hashpipe shared memory |
| U-blox Control | Configure ZED-F9T GNSS timing receivers |
| Telemetry | Logs → Loki; status → Redis/InfluxDB/Grafana |

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

## Development Roadmap

See [`docs/plan/control-upgrade-plan.md`](docs/plan/control-upgrade-plan.md) for the full control system upgrade plan.

- [x] Phase 0 — `pyproject.toml` packaging, Python ≥ 3.9
- [x] Phase 1 — Unit tests for all utility modules (460 tests)
- [x] Phase 1b — Integration test suite (Docker, real hashpipe, gRPC end-to-end; 43 tests)
- [ ] Phase 2 — Python 3.9→3.14 modernization
- [ ] Phase 3 — SSH → gRPC migration (`start.py`/`stop.py`/`status.py`)
- [ ] Phase 4 — Telemetry pipeline integration tests (requires Telemetry service in compose)
