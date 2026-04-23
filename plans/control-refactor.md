# Plan: Refactor `control/` into a Production-Grade Repo

**Created:** 2026-03-26
**Branch:** `add-ci`

## Context

The `control/` directory manages a live observatory instrument at Palomar. It has a well-structured config/validation layer (Pydantic models, two-tier cross-config validation) but the operational layer has significant technical debt: no tests, no CI, shell injection vulnerabilities, duplicated logging patterns, and 25+ bare `except:` blocks that silently swallow errors during live observations.

The `grpc/` submodule (`panoseti_grpc`) provides four async gRPC services that are gradually replacing direct SSH-based DAQ management — the refactoring plan must stay compatible with this ongoing migration.

Goal: improve reliability and maintainability without disrupting ongoing observations.

---

## Phase 1: Foundation (do first — zero observable behavior change)

### 1.1 Fix Shell Injection (1 day, security-critical)

Replace all `os.system('ssh %s@%s "%s"' % ...)` calls with `subprocess.run(list_args)`.

Files to fix:
- `control/utils/file_xfer.py` — `copy_file_to_node()`, `copy_dir_from_node()`, `make_remote_dirs()`
- `control/utils/collect.py` — `cleanup_daq()`
- `control/start.py` — `make_run_dirs()`, `start_recording()`
- `control/stop.py` — `stop_recording()`
- `control/power.py` — replace `os.system('curl ...')` with `requests` (already in requirements)

Safe pattern:
```python
# BEFORE (vulnerable — user-controlled values in shell string):
cmd = 'ssh %s@%s "%s"' % (username, ip_addr, rcmd)
ret = os.system(cmd)

# AFTER (safe — no shell interpolation):
result = subprocess.run(['ssh', f'{username}@{ip_addr}', rcmd], capture_output=True, text=True)
if result.returncode != 0:
    raise RemoteCommandError(ip_addr, rcmd[:50], result.returncode)
```

Note: the DAQ Control gRPC service (`panoseti_grpc.daq_control`) is designed to replace these SSH calls entirely long-term. As that service matures, the subprocess calls in `daq_scripts/` become redundant. Fix them now for safety; migrate to RPC later.

### 1.2 Exception Hierarchy (2 days)

Create `control/utils/exceptions.py`:
```python
class PanosetiError(Exception): ...
class ConfigError(PanosetiError): ...
class HardwareError(PanosetiError): ...
class QuaboError(HardwareError): ...       # includes ip_addr
class RemoteCommandError(HardwareError): ...  # includes host, returncode
class RunStateError(PanosetiError): ...
```

Fix top-priority bare `except:` blocks that silently swallow errors during data collection:
- `utils/util.py:13-16` — import guard bare except
- `utils/file_xfer.py:69,89,110` — `copy_dir_from_node()` bare excepts
- `utils/util.py:495` — `write_log()` bare except
- `driver/quabo_driver.py:316,355,373` — UDP receive bare excepts

### 1.3 Logging Consolidation (3 days)

The `print()` wrapper with UTC timestamps is implemented 4 separate times (start.py, stop.py, status.py, power.py), all writing to the same hardcoded path `/mnt/data11/data/palomar/L0/{date}/obslogs/`.

The `panoseti_grpc` Telemetry service already provides a structured gRPC-based logger (`panoseti_grpc.telemetry.logger.get_logger()`). The new `obs_logging` module should be designed to optionally forward to it.

Create `control/utils/obs_logging.py` with:
- `get_logger(name: str, telemetry_client=None) -> logging.Logger` — configures `RotatingFileHandler` + `StreamHandler`; if a `telemetry_client` is provided, attaches a gRPC log handler
- `configure_root_logging(log_dir: str | None = None)` — called once at entry point startup

Remove all `builtins.print = ...` monkey-patching from entry points. Replace with `logger = obs_logging.get_logger('panoseti.start')`.

Migration order (one file at a time):
1. `utils/obs_logging.py` (new)
2. `utils/util.py` — refactor `create_logger()` to delegate
3. `power.py` → `status.py` → `stop.py` → `start.py`


---

## Phase 2: Testing Infrastructure (2–3 weeks)

### Layout

```
control/tests/
  __init__.py
  conftest.py                   # shared fixtures
  fixtures/
    sample_obs_config.json      # minimal realistic config (derived from palomar/)
    sample_daq_config.json
    sample_data_config.json
    sample_network_config.json
    sample_daemons.json
    sample_firmware.json
  test_pydantic_models.py       # ← write FIRST
  test_config_file.py           # ← write SECOND
  test_global_validator.py      # ← write THIRD
  test_util.py
  test_file_xfer.py             # SSH mocked via pytest-mock
  test_quabo_driver.py          # socket mocked via pytest-mock
```

### conftest.py key fixtures

```python
@pytest.fixture(scope="session")
def fixtures_dir():
    return pathlib.Path(__file__).parent / "fixtures"

@pytest.fixture
def sample_data_config(fixtures_dir):
    with open(fixtures_dir / "sample_data_config.json") as f:
        return json.load(f)
# ... similar for obs, daq, network configs

@pytest.fixture(autouse=True)
def set_config_dir(fixtures_dir, monkeypatch):
    # Prevent tests from ever reading live Palomar configs
    monkeypatch.setattr("utils.config_file._config_dir", str(fixtures_dir))
```

### What to test (priority order)

**Tier 1 — Pydantic validators** (`test_pydantic_models.py`): Zero hardware, zero mocking.
- `run_type` with invalid chars raises `ValidationError`
- `pe_threshold < 2.0` raises `ValidationError`
- `integration_time_usec` not dividing 1,000,000 evenly raises `ValidationError`
- `two_pixel_trigger=1` with `image` mode raises `ValidationError`
- Interleave state with no valid mode raises `ValidationError`
- Valid configs pass without error

**Tier 2 — Config utility functions** (`test_config_file.py`): Pure math, no I/O.
```python
def test_ip_addr_to_module_id():
    assert ip_addr_to_module_id("192.168.3.248") == 254

def test_quabo_ip_addr():
    assert quabo_ip_addr("192.168.3.248", 2) == "192.168.3.250"

def test_string_to_list():
    assert string_to_list("0-2, 5-6") == [0, 1, 2, 5, 6]

def test_expand_ranges():
    config = {"daq_nodes": [{"module_ids": "253"}]}
    expand_ranges(config)
    assert config["daq_nodes"][0]["module_ids"] == [253]

def test_load_and_validate_bad_json(tmp_path):
    (tmp_path / "bad.json").write_text("{invalid}")
    with pytest.raises(ValueError, match="JSON Parse Error"):
        load_and_validate(DataConfig, "bad.json", str(tmp_path), "Test")
```

**Tier 3 — GlobalConfigValidator** (`test_global_validator.py`): Business rule regressions.
- Overvoltage mismatch between obs_config and data_config → error
- Missing firmware for a hw_type present in obs_config → error
- Port collision (two modules, same gateway, same port) → error
- Science run with stim enabled → warning
- Dome baseline > 2 km → warning

**Tier 4 — Hardware mocking** (`test_quabo_driver.py`): Socket mocked via pytest-mock.
- Verify correct bit patterns in UDP packets for `send_daq_params()`
- Verify socket is released on `close()`
- Verify context manager `__enter__`/`__exit__` works

### Coverage targets
- `utils/config_file.py`: 60% line coverage
- `utils/pydantic_config_models.py`: 80% line coverage
- `utils/global_validator.py`: 60% line coverage
- Pure utility functions (IP math, `string_to_list`): 80%+
- Do NOT chase coverage on `start.py`, `stop.py`, or daemon scripts

### gRPC submodule tests

The `panoseti_grpc` package has its own test suite in `../panoseti_grpc/tests/`. Do not duplicate those. The `control/` tests should only test the thin integration points:
- `control/daemons/capture_telemetry_service.py` — verify it starts the Telemetry server with the right config
- Any future code that instantiates a `DaqDataClient` or `DaqControlClient` — mock the gRPC channel

---

## Phase 3: CI/CD (1 week, parallel with end of Phase 2)

Create `.github/workflows/ci.yml` at repo root:

```yaml
name: CI
on:
  push:
    branches: [add-ci, main]
  pull_request:
    branches: [main]

jobs:
  control-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          pip install pytest pytest-mock pydantic rich haversine ruff mypy
          # Deliberately exclude hardware deps: redis, influxdb, tftpy, psutil
      - name: Lint
        run: ruff check control/utils/ control/driver/
      - name: Type check
        run: |
          mypy control/utils/config_file.py \
               control/utils/pydantic_config_models.py \
               control/utils/global_validator.py \
               --ignore-missing-imports --no-strict-optional
      - name: Unit tests
        run: cd control && python -m pytest tests/ -v --tb=long
        env:
          PYTHONPATH: .

  grpc-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: true
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install panoseti_grpc
        run: cd grpc && pip install -e ".[dev]"
      - name: Run gRPC unit tests
        run: cd grpc && python -m pytest tests/ -v --tb=short -m "not integration"
        # Integration tests require Docker (Redis, InfluxDB, Loki) — run separately
```

Add `control/pyproject.toml`:
```toml
[tool.ruff]
target-version = "py310"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "B", "S", "UP"]
ignore = ["S603", "S101"]

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["S"]
```

The `S` (bandit) rules auto-flag `os.system()` calls (`S605`) and bare `except` (`S110`) — enforcing the Phase 1 fixes going forward.

---

## Phase 4: Architecture Improvements (after CI is green)

### 4.1 Constants Module

Create `control/utils/constants.py` to consolidate hardcoded values currently scattered across 8+ files:
```python
DEFAULT_HK_DEST_IP = "192.168.1.100"
DEFAULT_CMD_PORT = 60000
DEFAULT_HK_PORT = 60002
DEFAULT_REBOOT_PORT = 69
QUABO_SOCKET_TIMEOUT_S = 0.5
QUABOS_PER_MODULE = 4
DEFAULT_OBS_LOG_ROOT = "/mnt/data11/data/palomar/L0"
```

### 4.2 Socket Context Managers

Add `__enter__`/`__exit__` to `QUABO` class in `driver/quabo_driver.py`:
```python
def __enter__(self): return self
def __exit__(self, *_): self.close(); return False
```

Update callers in `start.py`, `stop.py` to use `with quabo_driver.QUABO(...) as q:`.

### 4.3 Daemon Base Class

Create `control/utils/daemon_base.py` with `PanosetiDaemon(ABC)` providing:
- Standard `setup()`, `loop_body()`, `teardown()` lifecycle
- SIGINT/SIGTERM handler → sets `_running = False`
- Structured logging via `obs_logging`

Apply to new daemons and when actively modifying existing ones. Do NOT mass-migrate all 43 daemons.

The new `capture_telemetry_service.py` daemon already follows a similar pattern using the gRPC server's own lifecycle — it does not need to inherit from `PanosetiDaemon`.

### 4.4 Dependency Injection for Config Loading

Add `set_config_dir(d: str)` to `config_file.py` so tests can redirect away from live Palomar configs without filesystem mocking.

### 4.5 Type Hints

Add to highest-traffic functions first:
- `config_file.py`: `ip_addr_to_module_id`, `quabo_ip_addr`, `get_boardloc`, `string_to_list`
- `util.py`: `now_str`, `local_ip`, `ip_addr_str_to_bytes`, `is_script_running`

### 4.6 DAQ Control Migration (long-term)

As `panoseti_grpc.daq_control` matures, migrate `control/daq_scripts/start_daq.py`, `stop_daq.py`, and `status_daq.py` to be thin wrappers over the DAQ Control gRPC client instead of SSH subprocess calls. This removes the last direct SSH dependency in the DAQ path. Migration gate: the DAQ Control service must be marked stable in the `panoseti_grpc` changelog.

---

## Prioritization (strict order for small team)

| # | Task | Effort | Why |
|---|------|--------|-----|
| 1 | Shell injection fix (subprocess list args) | 1 day | Security |
| 2 | Exception hierarchy + top-5 bare excepts | 2 days | Observability during ops |
| 3 | Logging consolidation (remove print monkey-patch) | 3 days | Reliability |
| 5 | Write Pydantic model tests | 3 days | Regression protection |
| 6 | Write config utility tests | 2 days | IP math correctness |
| 7 | Write global validator tests | 2 days | Business rule regressions |
| 8 | GitHub Actions CI (control + grpc jobs) | 1 day | Automated gates |
| 9 | Ruff + mypy in CI | 2 days | Quality enforcement |
| 10 | Type hints on utility functions | 2 days | Maintainability |
| 11 | Socket context managers | 1 day | Resource safety |
| 12 | Constants module | 1 day | Readability |
| 13 | Daemon base class | 3 days | Long-term structure |
| 14 | DAQ Control gRPC migration | 3 days | Remove SSH from DAQ path |

## What NOT to touch yet

- `quabo_driver.py` MAROC register bit-packing — works, complex, requires hardware to verify
- `config_file.py:associate()` mutation pattern — too invasive to change safely
- The 43 daemon scripts (mass migration) — only touch when already editing for another reason
- `global_validator.py` reflection pattern (`dir(self)` → `_check_*`) — clever, not harmful
- `panoseti_grpc` internals — that package has its own test suite and release cycle

## Verification

- Phase 1: Run `python start.py --validate-only` after each change; no behavior regression
- Phase 2: `cd control && python -m pytest tests/ -v`; all tests green before moving to CI
- Phase 3: Push to `add-ci` branch; both `control-tests` and `grpc-tests` GitHub Actions jobs must show green
- Phase 4: Run `mypy --strict` on refactored files; zero errors
