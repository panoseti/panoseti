"""
conftest.py — Shared pytest fixtures for the panoseti-control test suite.

sys.path is managed by pyproject.toml [tool.pytest.ini_options] pythonpath=["."],
which adds control/ to the path so "from utils.X import ..." works.

We also add control/utils/ for modules that use bare `import pff` style imports
(e.g. image_quantiles.py).
"""

import contextlib
import copy
import io
import json
import os
import pathlib
import shutil
import struct
import time
import tomllib
import uuid
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from panoseti_grpc.daq_control.client import DaqControlClient
from panoseti_grpc.daq_data.client import DaqDataClient

from ci.fixtures.factories import (
    make_mock_daq_config,
    make_transfer_job,
    simulate_daq_filesystem,
)
from ci.paths import PanoPathsTest
from control.utils.pydantic_config_models import (
    DaqConfig,
    NetworkConfig,
    ObsConfig,
)


def pytest_configure_node(node: Any) -> None:
    """Called by xdist controller to configure each worker before it starts."""
    if not hasattr(node.config, "_tc_run_uuid"):
        import uuid as _uuid
        node.config._tc_run_uuid = _uuid.uuid4().hex[:8]
    node.workerinput["tc_run_uuid"] = node.config._tc_run_uuid


def pytest_configure(config: Any) -> None:
    """
    Set environment variable overrides to isolate the test environment.
    This ensures PanoPaths resolves to test-specific directories instead of
    production code directories, preventing state leakage.
    """
    # 1. Route configs to the integration test configs (default to direct for unit tests)
    if "PSETI_CONFIG" not in os.environ:
        os.environ["PSETI_CONFIG"] = str(PanoPathsTest.integration_configs("direct"))

    # 2. Route state to isolated test directories (fallback defaults)
    if "PSETI_TMP" not in os.environ:
        os.environ["PSETI_TMP"] = "/tmp/pseti_test/tmp"
    if "PSETI_LOGS" not in os.environ:
        os.environ["PSETI_LOGS"] = "/tmp/pseti_test/logs"
    if "PSETI_QUABOS" not in os.environ:
        os.environ["PSETI_QUABOS"] = "/tmp/pseti_test/quabos"

    # 3. Ensure directories exist
    os.makedirs(os.environ["PSETI_TMP"], exist_ok=True)
    os.makedirs(os.environ["PSETI_LOGS"], exist_ok=True)
    os.makedirs(os.environ["PSETI_QUABOS"], exist_ok=True)

    # 4. Give each xdist worker its own testcontainers Ryuk session so parallel
    #    workers don't collide on the shared TC_SESSION_ID (409 Conflict).
    import uuid as _uuid
    if hasattr(config, "workerinput"):
        # xdist worker process
        worker_id = config.workerinput.get("workerid", "master")
        run_uuid = config.workerinput.get("tc_run_uuid", _uuid.uuid4().hex[:8])
    else:
        # single-process run (fleet/chaos suites run without xdist)
        worker_id = "solo"
        run_uuid = _uuid.uuid4().hex[:8]

    os.environ["TC_SESSION_ID"] = f"tc-{worker_id}-{run_uuid}"

    # Ensure testcontainers internal state is updated
    try:
        import testcontainers.core.utils
        if hasattr(testcontainers.core.utils, "setup_default_session_id"):
            testcontainers.core.utils.setup_default_session_id()
        else:
            testcontainers.core.utils.SESSION_ID = os.environ["TC_SESSION_ID"]
    except (ImportError, AttributeError):
        pass


@pytest.fixture(scope="session")
def worker_id(request: Any) -> str:
    """Returns the xdist worker ID or 'master' if not running in parallel."""
    if hasattr(request.config, "workerinput"):
        return request.config.workerinput["workerid"]
    return "master"


@pytest.fixture(scope="session", autouse=True)
def auto_isolate(
    tmp_path_factory: pytest.TempPathFactory, 
    worker_id: str
) -> Iterator[pathlib.Path]:
    """
    Autouse session-scoped fixture that provides isolation for configs, transient state,
    and telemetry databases.
    
    - Redirects PSETI_STATE, PSETI_CONTROL, PSETI_CONFIG, and PSETI_TMP to subdirs in a session tmp_path.
    - Assigns unique Redis DB indices and Loki Tenant IDs based on worker_id.
    - Guarantees that any 'sed -i' or ledger writes stay within the session scope.
    """
    tmp_path = tmp_path_factory.mktemp(f"session_{worker_id}")

    # 1. Setup isolated directories inside tmp_path
    cfg_tmp = tmp_path / "configs"
    state_tmp = tmp_path / "state"
    ctl_tmp = tmp_path / "control"
    tmp_tmp = tmp_path / "tmp"
    
    for d in [cfg_tmp, state_tmp, ctl_tmp, tmp_tmp]:
        d.mkdir(parents=True, exist_ok=True)
        
    # 2. Populate configs from current PSETI_CONFIG
    src_cfg = os.environ.get("PSETI_CONFIG")
    head_prefix = os.environ.get("HEAD_NET_PREFIX", "10.0.1")
    daq_prefix = os.environ.get("DAQ_NET_PREFIX", "192.168.0")
    quabo_prefix = os.environ.get("QUABO_NET_PREFIX", "192.168.3")

    if src_cfg and os.path.exists(src_cfg):
        for item in pathlib.Path(src_cfg).iterdir():
            try:
                # Copy everything EXCEPT the topological configs we are about to overwrite
                if item.name not in ["daq_config.json", "obs_config.json", "network_config.json", "quabo_uids.json"]:
                    if item.is_file():
                        shutil.copy2(item, cfg_tmp)
                    elif item.is_dir():
                        shutil.copytree(item, cfg_tmp / item.name, dirs_exist_ok=True)
            except Exception:
                pass
                
    # 2.5 Generate Pristine Topological Configs
    from control.topology.fleet import generate_ci_topology
    daq_cfg, quabo_uids, net_cfg, obs_cfg = generate_ci_topology(head_prefix, daq_prefix, quabo_prefix)
    
    (cfg_tmp / "daq_config.json").write_text(daq_cfg.model_dump_json(indent=2))
    (cfg_tmp / "obs_config.json").write_text(obs_cfg.model_dump_json(indent=2))
    (cfg_tmp / "network_config.json").write_text(net_cfg.model_dump_json(indent=2))
    (cfg_tmp / "quabo_uids.json").write_text(quabo_uids.model_dump_json(indent=2))
    (tmp_tmp / "quabo_uids.json").write_text(quabo_uids.model_dump_json(indent=2)) # Chaos legacy

    # 3. Apply overrides for the duration of the session
    os.environ["PSETI_CONFIG"] = str(cfg_tmp)
    os.environ["PSETI_STATE"] = str(state_tmp)
    os.environ["PSETI_CONTROL"] = str(ctl_tmp)
    os.environ["PSETI_TMP"] = str(tmp_tmp)

    # Expose isolated data dirs
    if "HEAD_DATA_DIR" not in os.environ:
        head_data_tmp = tmp_path / "head_data"
        head_data_tmp.mkdir(parents=True, exist_ok=True)
        os.environ["HEAD_DATA_DIR"] = str(head_data_tmp)
    
    if "DAQ_DATA_DIR" not in os.environ:
        daq_data_tmp = tmp_path / "daq_data"
        daq_data_tmp.mkdir(parents=True, exist_ok=True)
        os.environ["DAQ_DATA_DIR"] = str(daq_data_tmp)

    # BROAD PERMISSIONS for Docker volume mapping
    for d in [cfg_tmp, state_tmp, ctl_tmp, tmp_tmp, 
              pathlib.Path(os.environ["HEAD_DATA_DIR"]), 
              pathlib.Path(os.environ["DAQ_DATA_DIR"])]:
        try:
            os.chmod(str(d), 0o777)
        except OSError:
            pass
    
    # Refresh Pydantic's perspective of the environment
    import importlib

    from control.utils import config_file
    importlib.reload(config_file)

    # 4. Telemetry and Database Isolation
    try:
        db_index = int("".join(filter(str.isdigit, worker_id))) if worker_id != "master" else 0
    except ValueError:
        db_index = 0
        
    os.environ["REDIS_DB"] = str(db_index)
    os.environ["LOKI_TENANT_ID"] = f"test_tenant_{db_index}"
    
    # 5. Ensure PanoPaths and RunStateManager are fresh
    from control.utils.paths import PanoPaths
    from control.utils.run_state import RunStateManager
    PanoPaths.ensure_state_dirs()
    RunStateManager().clear_state()
    
    yield tmp_path

# ---------------------------------------------------------------------------
# Shared Factories & Mocks (Tier 2-4 Infrastructure)
# ---------------------------------------------------------------------------

@pytest.fixture
def transfer_job_factory():
    """Factory for creating valid TransferJob models."""
    return make_transfer_job

@pytest.fixture
def daq_fs_simulator():
    """Helper to populate a mock DAQ filesystem structure."""
    return simulate_daq_filesystem

@pytest.fixture
def daq_config_factory():
    """Factory for creating valid DaqConfig models."""
    return make_mock_daq_config

@pytest.fixture(scope="session")
def topology_templates() -> dict[str, dict[str, Any]]:
    """Loads all TOML topology templates from ci/test_topologies/."""
    templates = {}
    template_dir = pathlib.Path(__file__).parent / "test_topologies"
    if template_dir.exists():
        for toml_file in template_dir.glob("*.toml"):
            with open(toml_file, "rb") as f:
                templates[toml_file.stem] = tomllib.load(f)
    return templates


@pytest.fixture
def minimal_obs_config(topology_templates) -> dict[str, Any]:
    """Smallest valid obs_config dict: one dome, one module."""
    return copy.deepcopy(topology_templates.get("base_obs", {}))


@pytest.fixture
def two_dome_obs_config(topology_templates) -> dict[str, Any]:
    """Two-dome obs config for geospatial checks."""
    cfg = copy.deepcopy(topology_templates.get("base_obs", {}))
    # Add a second dome ~111 m apart — within 2 km baseline
    cfg["domes"].append({
        "name": "dome1",
        "obslat": 33.358,
        "obslon": -116.866,
        "obsalt": 1706.0,
        "modules": [
            {
                "mobo_serialno": "SN002",
                "quabo_version": "bga",
                "ip_addr": "192.168.3.204",
                "wps": "wps",
            }
        ],
    })
    return cfg


@pytest.fixture
def minimal_daq_config(topology_templates) -> dict[str, Any]:
    """Smallest valid daq_config dict: one DAQ node."""
    return copy.deepcopy(topology_templates.get("base_daq", {}))


@pytest.fixture
def minimal_data_config(topology_templates) -> dict[str, Any]:
    """Smallest valid data_config dict: image mode only."""
    return copy.deepcopy(topology_templates.get("base_data", {}))


@pytest.fixture
def minimal_firmware_config(topology_templates) -> dict[str, Any]:
    """Firmware config listing hardware variants."""
    return copy.deepcopy(topology_templates.get("base_firmware", {}))

@pytest.fixture
def mock_daq_config() -> DaqConfig:
    """Fully valid Pydantic model for DAQ configuration."""
    baseline = {
        "head_node_data_dir": "/data/head",
        "head_node_ip_addr": "10.0.0.1",
        "head_node_container": False,
        "daq_nodes": [
            {
                "ip_addr": "10.0.0.2",
                "data_dir": "/data",
                "username": "panoseti",
                "module_ids": [200],
                "bindhost": "lo"
            }
        ]
    }
    return DaqConfig(**baseline)

@pytest.fixture
def mock_network_config() -> NetworkConfig:
    """Fully valid Pydantic model for network configuration."""
    baseline = {
        "modules": [
            {
                "ip_addr": "192.168.3.200",
                "port_forwarding": {
                    "status": False,
                    "gw_ip": "10.200.146.11",
                    "reboot_port": [60004, 60005, 60006, 60007],
                    "cmd_port": [60000, 60001, 60002, 60003]
                }
            }
        ],
        "daq_nodes": [
            {
                "ip_addr": "10.0.0.2",
                "port_forwarding": {
                    "status": False,
                    "gw_ip": "10.200.146.11",
                    "port": 22
                }
            }
        ]
    }
    return NetworkConfig(**baseline)

@pytest.fixture
def mock_obs_config() -> ObsConfig:
    """Fully valid Pydantic model for observatory configuration."""
    baseline = {
        "name": "test_obs",
        "comment": "Test Observatory",
        "wps": {
            "url": "http://192.168.1.2",
            "quabo_socket": 1
        },
        "wr_ip_addr": "192.168.1.254",
        "gps_port": "/dev/ttyUSB0",
        "detector_overvoltage": 2,
        "domes": [
            {
                "obslat": 33.3533,
                "obslon": -116.8622,
                "obsalt": 1693.0,
                "name": "dome0",
                "modules": [
                    {
                        "mobo_serialno": "M11",
                        "quabo_version": "qfp",
                        "ip_addr": "192.168.3.200",
                        "wps": "wps",
                        "timing_mode": "gnss",
                        "azimuth": 77.0,
                        "elevation": 77.0,
                        "position_angle": 77.0
                    }
                ]
            }
        ]
    }
    return ObsConfig(**baseline)


# ---------------------------------------------------------------------------
# PFF file helpers
# ---------------------------------------------------------------------------

def _make_pff_json_header(quabo_num: int = 0, pkt_num: int = 0,
                           pkt_tai: int = 613, pkt_nsec: int = 0,
                           tv_sec: int = 1_000_000, tv_usec: int = 0) -> bytes:
    """Build a PFF JSON header block (ends with \\n\\n)."""
    # For img16 / img8 style: use a quabo_0 sub-dict
    payload = {
        "quabo_0": {
            "quabo_num": quabo_num,
            "pkt_num": pkt_num,
            "pkt_tai": pkt_tai,
            "pkt_nsec": pkt_nsec,
            "tv_sec": tv_sec,
            "tv_usec": tv_usec,
        }
    }
    s = json.dumps(payload) + "\n\n"
    return s.encode()


def _make_pff_image_block_16bit(width: int = 32) -> bytes:
    """Build a 16-bit image block (32x32 pixels, all zeros)."""
    n = width * width
    return b"*" + struct.pack(f"{n}H", *([0] * n))


def make_minimal_pff_bytes(n_frames: int = 3, tv_sec_start: int = 1_000_000) -> bytes:
    """Return bytes of a minimal PFF file with n_frames of img16 data."""
    buf = io.BytesIO()
    for i in range(n_frames):
        header = _make_pff_json_header(tv_sec=tv_sec_start + i)
        buf.write(header)
        buf.write(_make_pff_image_block_16bit())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Fixed-size PFF file factory (for img_info / time_seek tests)
# All frames have identical header sizes — required by img_info's frame_size math.
# pkt_tai is set so that d = (tv_sec - pkt_tai + 37) % 1024 == 0 for every frame.
# ---------------------------------------------------------------------------

_FIXED_HEADER_JSON_LEN = 120  # pad all JSON to this many bytes before '\n\n'


def _make_fixed_header(tv_sec: int, pkt_num: int = 0, nested: bool = True) -> bytes:
    """
    Build a PFF JSON header padded to _FIXED_HEADER_JSON_LEN + 2 bytes total.
    nested=True  → img16/img8 style   {"quabo_0": {...}}
    nested=False → ph256 style        {...}
    d=0 guaranteed: pkt_tai = (tv_sec + 37) % 1024.
    """
    pkt_tai = (tv_sec + 37) % 1024
    inner = {
        "quabo_num": 0,
        "pkt_num": pkt_num,
        "pkt_tai": pkt_tai,
        "pkt_nsec": 0,
        "tv_sec": tv_sec,
        "tv_usec": 0,
    }
    payload = {"quabo_0": inner} if nested else inner
    s = json.dumps(payload)
    # Pad with spaces so every header has the same byte length
    s = s + " " * max(0, _FIXED_HEADER_JSON_LEN - len(s))
    return (s + "\n\n").encode()


def make_pff_file(
    n_frames: int = 3,
    tv_sec_start: int = 1_000_000,
    tv_sec_values: list[int] | None = None,
    nested_header: bool = True,
    img_size: int = 32,
    bpp: int = 2,
) -> io.BytesIO:
    """
    Write an in-memory PFF file and return a seeked-to-start BytesIO.

    All frames have identical fixed-size headers (padded to _FIXED_HEADER_JSON_LEN).
    tv_sec_values, if provided, overrides per-frame tv_sec; must be length n_frames.
    """
    if tv_sec_values is None:
        tv_sec_values = [tv_sec_start + i for i in range(n_frames)]
    assert len(tv_sec_values) == n_frames, "tv_sec_values length must match n_frames"

    n_pixels = img_size * img_size
    fmt = f"{n_pixels}{'H' if bpp == 2 else 'B'}"
    image_bytes = b"*" + struct.pack(fmt, *([0] * n_pixels))

    buf = io.BytesIO()
    for i, tv_sec in enumerate(tv_sec_values):
        buf.write(_make_fixed_header(tv_sec, pkt_num=i, nested=nested_header))
        buf.write(image_bytes)
    buf.seek(0)
    return buf


@pytest.fixture
def pff_file_factory() -> Callable[..., io.BytesIO]:
    """Fixture that returns the make_pff_file() helper."""
    return make_pff_file


# ---------------------------------------------------------------------------
# DaqControlClient fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def session_fleet(auto_isolate) -> Iterator[Any]:
    """Start a 2-node testcontainers fleet and yield (fleet, daq_cfg_dict).

    The daq_cfg dict is built from a validated Pydantic DaqConfig so
    all port-forwarding metadata is guaranteed correct before any test runs.
    """
    import json

    from ci.fixtures.fleet import make_fleet, setup_docker_host

    # 1. Configure Docker host (macOS Docker Desktop socket detection).
    setup_docker_host()

    # 2. Build and start the fleet.
    fleet = make_fleet(n=2)
    try:
        fleet.start()
        fleet.wait_healthy(timeout=90.0)
    except Exception as exc:
        fleet.tear_down()
        raise RuntimeError(f"Fleet failed to start or become healthy: {exc}") from exc

    # 3. Materialise a validated Pydantic DaqConfig and serialise to dict.
    #    to_daq_config() injects port_forwarding blocks with the dynamic
    #    mapped ports so clients use 127.0.0.1:<port> automatically.
    daq_config = fleet.to_daq_config()
    daq_cfg = json.loads(daq_config.model_dump_json())

    # PERSIST: Write the dynamic config to disk in the isolated directory
    # so tools like stop_run (which reload from disk) see the mapped ports.
    cfg_dir = pathlib.Path(os.environ["PSETI_CONFIG"])
    daq_config_path = cfg_dir / "daq_config.json"
    fleet.write_daq_config(daq_config_path)

    yield fleet, daq_cfg

    fleet.tear_down()

@pytest.fixture(scope="session")
def daq_control_direct(session_fleet) -> DaqControlClient:
    """Client connected directly to the first daqnode."""
    fleet, _daq_cfg = session_fleet
    spec = fleet.specs[0]
    return DaqControlClient(host=spec.container_host_ip, port=spec.mapped_port)


@pytest.fixture(scope="session")
def daq_control_node2(session_fleet) -> DaqControlClient:
    """DaqControlClient connected to the second DAQ node (two-node tests)."""
    fleet, _daq_cfg = session_fleet
    spec = fleet.specs[1]
    return DaqControlClient(host=spec.container_host_ip, port=spec.mapped_port)


@pytest.fixture(scope="session")
def daqnode_container(session_fleet) -> Any:
    """Returns the Docker SDK Container for the first fleet daqnode."""
    fleet, _ = session_fleet
    return fleet.containers[0].get_wrapped_container()


@pytest.fixture(scope="session")
def daq_control_gateway(session_fleet) -> DaqControlClient:
    """Client connected via the gateway. For local testcontainers, it's just the first node."""
    fleet, _daq_cfg = session_fleet
    spec = fleet.specs[0]
    return DaqControlClient(host=spec.container_host_ip, port=spec.mapped_port)


@pytest.fixture(scope="session")
def daq_data_client(session_fleet) -> Iterator[DaqDataClient]:
    """Session-scoped DaqDataClient connected to the fleet."""
    _fleet, daq_cfg = session_fleet
    with DaqDataClient(daq_cfg, network_config=None) as client:
        yield client


# ---------------------------------------------------------------------------
# Run parameters fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def run_params(session_fleet) -> dict[str, Any]:
    """Fresh run parameters for each module — daq_ip_addr from the fleet node."""
    fleet, _ = session_fleet
    return {
        "data_dir":         "/data",
        "daq_ip_addr":      fleet.node_ip(0),
        "bindhost":         "lo",
        "max_file_size_mb": 1,
        "group_ph_frames":  True,
        "run_dir":          f"ci_run_{uuid.uuid4().hex[:8]}.pffd",
        "obs":              "citest",
        "module_id":        [250, 254],
    }


# ---------------------------------------------------------------------------
# Auto-cleanup: stop any lingering hashpipe after each test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=False)
def ensure_clean_daq_state(daq_control_direct: DaqControlClient, run_params: dict[str, Any]) -> Iterator[None]:
    """Stop hashpipe and clean up if a test leaves it running."""
    yield
    # Always call StopDaq unconditionally — it's idempotent and handles
    # the case where hashpipe crashed (leaving a stale hashpipe_pid on the
    # server) so CleanupData isn't blocked by the stale pid check.
    with contextlib.suppress(Exception):
        daq_control_direct.StopDaq({
            "data_dir": run_params["data_dir"],
            "run_dir":  run_params["run_dir"],
        })
    
    # Use helper from tier3_fleet.conftest if needed, but we define it here if possible or just use a simple wait.
    # For now, we'll assume wait_hashpipe_stopped is available or we'll just sleep.
    time.sleep(1) 

    with contextlib.suppress(Exception):
        daq_control_direct.CleanupData({
            "data_dir":  run_params["data_dir"],
            "run_dir":   run_params["run_dir"],
            "module_id": run_params["module_id"],
        })


# ---------------------------------------------------------------------------
# Machine-readable test summary for qa.py
# ---------------------------------------------------------------------------

def pytest_terminal_summary(terminalreporter: Any, exitstatus: int, config: Any) -> None:
    """
    Hook to print a JSON-formatted summary of test results at the very end
    of the pytest run. qa.py parses this to build its metrics table.
    """
    stats = terminalreporter.stats
    # terminalreporter.stats is a dict mapping status -> list of reports
    summary = {
        "passed": len(stats.get("passed", [])),
        "failed": len(stats.get("failed", [])),
        "skipped": len(stats.get("skipped", [])),
        "error": len(stats.get("error", [])),
        "xfail": len(stats.get("xfail", [])),
        "xpass": len(stats.get("xpass", [])),
    }
    # Print with a unique prefix so it's easy to grep from the stream
    print(f"\nTEST_METRICS_JSON: {json.dumps(summary)}")
