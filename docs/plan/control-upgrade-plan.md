# PANOSETI Control System Upgrade Plan

> **Status:** Active — initiated March 2026
> **Scope:** `control/` directory (instrument control system)
> **Branches:** `mount-metadata` → `claude-refactor`

---

## Motivation

The `control/` directory is the primary instrument control system for the PANOSETI observatory array at Palomar. With the system moving toward **robotic observation**, the current architecture has several blockers:

1. **No automated tests** — the only safety net before hardware runs is a `--validate-only` flag and ad-hoc `test_scripts/`. Any refactoring is risky without a regression baseline.
2. **SSH-based DAQ control** — `start.py`, `stop.py`, and `status.py` spawn SSH processes to remote DAQ nodes. This is brittle (key management, network topology changes), hard to test, and incompatible with distributed orchestration.
3. **No Python package metadata** — no `pyproject.toml`, no declared Python version constraint, no dev dependency management. Tooling (`pytest`, `mypy`, `ruff`) is not standardized.
4. **Legacy code patterns** — `typing.List/Dict` (deprecated in 3.9+), `%` string formatting, `os.system()` calls, sparse type annotations.

---

## Goals

| # | Goal | Outcome |
|---|------|---------|
| 0 | Python package setup | `pyproject.toml` with `python_requires=">=3.9"`, dev tools configured |
| 1 | Unit tests | pytest suite covering all 12 utility modules, no hardware required |
| 2 | Python 3.9→3.14 migration | Modern type hints, f-strings, subprocess, match/case |
| 3 | SSH → gRPC DAQ control | `start.py`/`stop.py`/`status.py` use `DaqControlClient` |
| 4 | Integration tests | End-to-end session lifecycle with mocked hardware |
| 5 | Packaging improvements | Optional: rename imports to `panoseti_control.utils.*` |

---

## Phase 0: Python Package Setup

**Files:**
- `control/pyproject.toml` ← **created in this campaign**
- `control/packages/requirements.txt` ← retained for backwards compatibility

**Key settings in `pyproject.toml`:**

```toml
[project]
name = "panoseti-control"
requires-python = ">=3.9"

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-asyncio>=0.23", "pytest-mock>=3.12",
       "fakeredis>=2.26", "mypy>=1.9", "ruff>=0.4", "numpy>=1.24"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]   # makes "from utils.X import ..." work in tests

[tool.ruff]
target-version = "py39"
select = ["E","F","UP","B","I"]
```

**Install:** `cd control && pip install -e ".[dev]"`

> **Principle:** Import paths (`from utils.config_file import ...`) are **not** renamed. Existing scripts continue to run as before. The `pyproject.toml` is purely additive tooling infrastructure.

---

## Phase 1: Unit Tests

### Directory structure

```
control/
├── pyproject.toml
└── tests/
    ├── __init__.py
    ├── conftest.py                   # sys.path setup + shared fixtures
    ├── unit/
    │   ├── __init__.py
    │   ├── test_pydantic_models.py   # DataConfig, ObsConfig, DaqConfig validators
    │   ├── test_config_file.py       # expand_ranges, ip_addr_to_module_id, etc.
    │   ├── test_global_validator.py  # Cross-config rules
    │   ├── test_pff.py               # parse_name, wr_to_unix, round-trip I/O
    │   ├── test_util.py              # ip_addr_str_to_bytes, get_daemons
    │   ├── test_redis_utils.py       # store_in_redis, get_casted_redis_value
    │   └── test_image_quantiles.py   # get_values, get_quantiles
    └── integration/                  # (Phase 4 — future)
        └── .gitkeep
```

### Running tests

```bash
cd control
pip install -e ".[dev]"

# All unit tests (no hardware required)
pytest tests/unit/ -v --tb=short

# With coverage
pytest tests/unit/ --cov=utils --cov-report=term-missing

# Via Docker (all dependencies provided)
bash run-ci-tests/run-unit-tests.sh
```

### Coverage targets per module

| Module | Key test cases |
|--------|---------------|
| `pydantic_config_models.py` | Every field boundary, every cross-field validator, interleave constraint (movie+PH trigger mutual exclusion) |
| `config_file.py` | `expand_ranges` with all formats, `ip_addr_to_module_id` IP math, `string_to_list`, `assign_numbers`, `quabo_ip_addr`, `module_id_to_daq_node` |
| `global_validator.py` | Overvoltage mismatch, port collision, firmware coverage gap, geospatial baseline, WPS reference |
| `pff.py` | `parse_name` for all data products, `wr_to_unix` boundary cases (d=0,1,1023,desync), `write_image_1D`+`read_image` round-trip |
| `util.py` | `ip_addr_str_to_bytes` valid/invalid, `get_daemons`/`get_permanent_daemons` logic |
| `redis_utils.py` | `store_in_redis`, `get_casted_redis_value` (int/float/string cast), using `fakeredis` |
| `image_quantiles.py` | `get_quantiles` with a synthetic PFF file |

---

## Phase 2: Python 3.9–3.14 Migration

**Automated (ruff `--fix`):**
- `UP006`/`UP007`: `List[X]` → `list[X]`, `Optional[X]` → `X | None` (with `from __future__ import annotations` for 3.9)
- `UP032`: `%`-format → f-string

**Manual:**
- Replace `os.system(cmd)` with `subprocess.run(...)` (structured error capture)
- Add return-type annotations to all public functions in `utils/`
- Add `from __future__ import annotations` to all utils files

**Future (3.10+, deferred):**
- `match/case` for data product routing in `pff.py`
- Remove `from __future__ import annotations` once min version ≥ 3.10
- `tomllib` (3.11 stdlib) for TOML config files

---

## Phase 3: SSH → gRPC DAQ Control

### Architecture change

```
Before:
  start.py ──SSH──▶ start_daq.py   (on DAQ node, launches hashpipe)
  stop.py  ──SSH──▶ stop_daq.py
  status.py──SSH──▶ status_daq.py

After:
  start.py ──gRPC──▶ DaqControlServicer.StartDaq()   (port 50051)
  stop.py  ──gRPC──▶ DaqControlServicer.StopDaq()
  status.py──gRPC──▶ DaqControlServicer.StatusDaq()
```

### Prerequisites

1. **Deploy `panoseti_grpc.daq_control.server` on each DAQ node** as a systemd service:

```ini
# /etc/systemd/system/panoseti-daq-control.service
[Unit]
Description=PANOSETI DAQ Control gRPC Server
After=network.target

[Service]
User=panoseti
ExecStart=/usr/bin/python3 -m panoseti_grpc.daq_control.server
Environment=GRPC_PORT=50051
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

2. **Add `grpc_port` to `daq_config.json`** (optional, defaults to 50051):
```json
{ "daq_nodes": [{ "ip_addr": "...", "grpc_port": 50051, ... }] }
```

3. **Update `DaqNodeValidator`** in `pydantic_config_models.py` to allow optional `grpc_port: int = 50051`.

### Code changes

**`start.py` `start_recording()` function** — replace `os.system(ssh_cmd)` loop:
```python
from panoseti_grpc.daq_control.client import DaqControlClient

for node in daq_config['daq_nodes']:
    host = node.get('port_forwarding', {}).get('gw_ip', node['ip_addr'])
    client = DaqControlClient(host=host, port=node.get('grpc_port', 50051))
    ok = client.StartDaq({
        "data_dir": node['data_dir'],
        "daq_ip_addr": node['ip_addr'],
        "bindhost": node.get('bindhost', '0.0.0.0'),
        "max_file_size_mb": max_file_size_mb,
        "group_ph_frames": daq_params.do_group_ph_frames,
        "run_dir": run_name,
        "obs": obs_config['name'],
        "module_id": [ip_addr_to_module_id(m['ip_addr']) for m in node['modules']],
    })
    if not ok:
        raise RuntimeError(f"StartDaq failed on {host}")
```

**`stop.py` `stop_recording()`** — replace SSH calls with `client.StopDaq(...)`.

**`status.py`** — replace SSH calls with `client.StatusDaq(...)`.

**`control/daq_scripts/`** — scripts become dead code after migration:
- `start_daq.py`, `stop_daq.py`, `status_daq.py` → deprecated (delete after validation)
- `video_daq.py` → retained until video mode is migrated

**`collect.py`** — keep SSH/rsync; file collection of large PFF files is appropriate over rsync.

### Health check

Add to `session_start.py`: verify gRPC connectivity (`Ping` RPC) to each DAQ node before proceeding with hardware configuration.

---

## Phase 4: Integration Tests

**Location:** `control/tests/integration/`

| Test | Validates |
|------|-----------|
| `test_config_validation.py` | Load actual Palomar configs through all three validation tiers (Pydantic + global, skip network) |
| `test_session_lifecycle.py` | Mock `DaqControlClient` + `QUABO` driver; exercise full `session_start`→`start`→`stop`→`session_stop` |
| `test_interleave.py` | Interleave state sequencing, transition enforcement, constraint checking |
| `test_grpc_daq_control.py` | Spin up `DaqControlServicer` in-process; full StartDaq/StopDaq/StatusDaq lifecycle |

**Mocking strategy:**
- `pytest-mock` for `QUABO` UDP driver
- `fakeredis` for Redis-dependent tests
- In-process gRPC server for DAQ control (no Docker needed)

---

## Docker CI Setup

```
control/run-ci-tests/
├── docker-compose.yml        # Redis sidecar + test runner
├── Dockerfile.test           # Python 3.12 + panoseti-control[dev]
├── run-unit-tests.sh         # One-shot: start → test → tear down
├── run-integration-tests.sh  # Integration variant
└── README.md                 # Usage
```

**Quick start:**
```bash
cd control
bash run-ci-tests/run-unit-tests.sh
```

---

## File Change Summary

| File | Type | Change |
|------|------|--------|
| `control/pyproject.toml` | **NEW** | Package metadata, dev deps, pytest config |
| `control/tests/` | **NEW** | Full test suite |
| `control/run-ci-tests/` | **NEW** | Docker CI scripts |
| `control/start.py` | Modify | Replace SSH with `DaqControlClient.StartDaq()` |
| `control/stop.py` | Modify | Replace SSH with `DaqControlClient.StopDaq()` |
| `control/status.py` | Modify | Replace SSH with `DaqControlClient.StatusDaq()` |
| `control/utils/pydantic_config_models.py` | Modify | Add `grpc_port: int = 50051` to `DaqNodeValidator` |
| `control/utils/util.py` | Modify | Add type annotations; modernize subprocess calls |
| `control/daq_scripts/{start,stop,status}_daq.py` | Deprecate | Replaced by gRPC server |
| `docs/plan/control-upgrade-plan.md` | **NEW** | This document |
| `README.md` | Modify | Software structure, submodule descriptions |
| `CLAUDE.md` | Modify | Test commands, gRPC status, Python version |

---

## Existing Utilities to Reuse (Do Not Duplicate)

| Utility | Location | Use in |
|---------|----------|--------|
| `DaqControlClient` | `panoseti_grpc.daq_control.client` | `start.py`, `stop.py`, `status.py` |
| `ip_addr_to_module_id()` | `panoseti_grpc.panoseti_util.config_file` | gRPC server configs (already re-exported) |
| `expand_ranges()` | `control/utils/config_file.py:108` | Global validator, tests |
| `DataConfigValidator` | `control/utils/pydantic_config_models.py:97` | All config loading, all tests |

---

## Verification Checklist

```bash
# 1. Unit tests pass (no hardware)
cd control && pip install -e ".[dev]"
pytest tests/unit/ -v --tb=short

# 2. Config validation (no hardware)
python start.py --validate-only

# 3. Lint
ruff check utils/ start.py stop.py status.py

# 4. Type check
mypy utils/config_file.py utils/pydantic_config_models.py --ignore-missing-imports

# 5. Docker CI (self-contained)
bash run-ci-tests/run-unit-tests.sh

# 6. gRPC connectivity (requires DAQ node)
python -c "
from panoseti_grpc.daq_control.client import DaqControlClient
c = DaqControlClient('daq-node-ip', 50051)
ok, status = c.StatusDaq({'data_dir': '/data', 'check_hashpipe_running': True,
                           'check_disk_usage': False, 'check_run_dirs': False})
print('Connected:', ok, status)
"
```
