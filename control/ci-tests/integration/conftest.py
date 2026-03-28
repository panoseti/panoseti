"""
conftest.py — Shared fixtures for the PANOSETI integration test suite.

Environment variables (set by docker-compose.integration.yml):
    DAQNODE_DIRECT_HOST   — IP of the daqnode container (direct access)
    DAQNODE_GATEWAY_HOST  — IP of the gateway container (forwarded access)
    GRPC_PORT             — gRPC port (default 50051)
    LOKI_URL              — Loki HTTP base URL
    REDIS_HOST            — Redis hostname
    DAQ_DATA_DIR          — data dir on the daqnode (and shared volume mount point)
    HEAD_DATA_DIR         — headnode data destination dir
    DAQNODE_CONTAINER_NAME — Docker container name for pause/unpause tests
    CONFIG_DIR            - Directory to integration test configuration files
"""
from __future__ import annotations

import os
import sys
import pathlib
import shutil
import subprocess
import time
import uuid

import pytest

from panoseti_grpc.daq_control.client import DaqControlClient
from panoseti_grpc.daq_data.client import DaqDataClient

# ---------------------------------------------------------------------------
# Environment / connection parameters
# ---------------------------------------------------------------------------

DAQNODE_DIRECT_HOST  = os.getenv("DAQNODE_DIRECT_HOST",  "localhost")
DAQNODE_GATEWAY_HOST = os.getenv("DAQNODE_GATEWAY_HOST", "localhost")
DAQNODE_DATA_HOST    = os.getenv("DAQNODE_DATA_HOST",    "localhost")
DAQNODE2_HOST        = os.getenv("DAQNODE2_HOST",        "localhost")
GRPC_PORT            = int(os.getenv("GRPC_PORT", "50051"))
LOKI_URL             = os.getenv("LOKI_URL",   "http://localhost:3100")
REDIS_HOST           = os.getenv("REDIS_HOST", "localhost")
DAQ_DATA_DIR         = os.getenv("DAQ_DATA_DIR", "/data")
HEAD_DATA_DIR        = os.getenv("HEAD_DATA_DIR", "/data/head")
DAQNODE_CONTAINER    = os.getenv("DAQNODE_CONTAINER_NAME", "ctl-int-daqnode-1")
# BINDHOST is the network interface name on the daqnode for receiving science packets.
# In Docker CI containers the primary interface is always "eth0".
BINDHOST             = os.getenv("BINDHOST", "eth0")

CONTROL_DIR = pathlib.Path(__file__).parent.parent.parent   # control/
CONFIG_DIR = pathlib.Path(__file__).parent / "configs"      # config/ci-tests/integration/configs/

# ---------------------------------------------------------------------------
# Portforwarding fixtures
# ---------------------------------------------------------------------------
DIRECT_CONFIG = CONFIG_DIR / "direct"
GATEWAY_CONFIG = CONFIG_DIR / "gateway"

# 1. Point sys.path to the root 'control' directory, NOT the 'utils' directory.
# 'conftest.py' is in control/ci-tests/integration/, so we go up two levels.
control_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if control_root not in sys.path:
    sys.path.insert(0, control_root)
    
from utils import config_file

def get_daq_and_network_config(kind="direct") -> tuple[dict, dict | None]:
    """(daq_config.json, network_config.json) for clients connected:
        1. Directly to the daqnode (bypasses gateway).
        2. Via the socat gateway (simulates VPN/NAT topology)
    """
    match kind:
        case "direct": 
            cfg_dir = DIRECT_CONFIG
            net_cfg = None
        case "gateway": 
            cfg_dir = GATEWAY_CONFIG
            net_cfg = config_file.get_network_config(cfg_dir)
        case _:
            raise ValueError(f"Invalid {kind=}. Must be 'direct' or 'gateway'")

    daq_cfg = config_file.get_daq_config(cfg_dir)
    return daq_cfg, net_cfg


# ---------------------------------------------------------------------------
# Session setup — ensure shared data directories exist before any test runs
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def create_data_dirs():
    """Create expected data directories on the shared volume at session start.

    /data/head is referenced by daq_config.json (head_node_data_dir) and must
    exist for global_validator's Headnode Disk Space check to pass.
    """
    pathlib.Path(HEAD_DATA_DIR).mkdir(parents=True, exist_ok=True)
    pathlib.Path(DAQ_DATA_DIR).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# DaqControlClient fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def daq_control_direct() -> DaqControlClient:
    """Client connected directly to the daqnode (bypasses gateway)."""
    return DaqControlClient(host=DAQNODE_DIRECT_HOST, port=GRPC_PORT)


@pytest.fixture(scope="session")
def daq_control_gateway() -> DaqControlClient:
    """Client connected via the socat gateway (simulates VPN/NAT topology)."""
    return DaqControlClient(host=DAQNODE_GATEWAY_HOST, port=GRPC_PORT)


@pytest.fixture(scope="session")
def daq_control_node2() -> DaqControlClient:
    """DaqControlClient connected to the second DAQ node (two-node tests)."""
    return DaqControlClient(host=DAQNODE2_HOST, port=GRPC_PORT)


@pytest.fixture(scope="session")
def daq_data_client() -> DaqDataClient:
    """Session-scoped DaqDataClient connected to daqnode-data.

    The connection is established once for the whole test session.
    Each test is responsible for calling init_sim() or init_hp_io()
    to configure server state — do NOT share hp_io state between tests.
    """
    daq_cfg = {
        "daq_nodes": [{"ip_addr": DAQNODE_DATA_HOST, "data_dir": DAQ_DATA_DIR}]
    }
    with DaqDataClient(daq_cfg, network_config=None) as client:
        yield client


# ---------------------------------------------------------------------------
# Run parameters fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def run_params() -> dict:
    """Fresh run parameters for each test — unique run_dir per test."""
    return {
        "data_dir":         DAQ_DATA_DIR,
        "daq_ip_addr":      DAQNODE_DIRECT_HOST,
        "bindhost":         BINDHOST,
        "max_file_size_mb": 100,
        "group_ph_frames":  False,
        "run_dir":          f"ci_run_{uuid.uuid4().hex[:8]}.pffd",
        "obs":              "citest",
        "module_id":        [200],
    }


# ---------------------------------------------------------------------------
# Data directory fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def daq_data_dir() -> pathlib.Path:
    """Root data directory on the daqnode (also mounted in test-runner)."""
    return pathlib.Path(DAQ_DATA_DIR)


@pytest.fixture
def head_data_dir() -> pathlib.Path:
    """Head node data directory (where collected data lands)."""
    p = pathlib.Path(HEAD_DATA_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Docker container handle (for pause/unpause in failure-simulation tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def daqnode_container():
    """
    Returns a thin wrapper around the daqnode Docker container.
    Requires /var/run/docker.sock to be mounted in the test-runner.
    Skips gracefully if docker SDK is unavailable.
    """
    try:
        import docker
        client = docker.from_env()
        container = client.containers.get(DAQNODE_CONTAINER)
        return container
    except Exception as e:
        pytest.skip(f"Docker SDK unavailable or container not found: {e}")


# ---------------------------------------------------------------------------
# Helper: simulate data copy (rsync equivalent using shared volume)
# ---------------------------------------------------------------------------

def copy_run_dir(run_params: dict, dst: pathlib.Path) -> bool:
    """
    Simulate rsync from daqnode to headnode using the shared Docker volume.
    Copies module_{id}/{run_dir}/ from daq_data_dir to dst/{run_dir}/.
    Returns True on success, False if source data is missing.
    """
    src_root = pathlib.Path(run_params["data_dir"])
    run_dir  = run_params["run_dir"]
    success  = True

    dst_run = dst / run_dir
    dst_run.mkdir(parents=True, exist_ok=True)

    for module_id in run_params["module_id"]:
        src = src_root / f"module_{module_id}" / run_dir
        if not src.exists():
            success = False
            continue
        dst_module = dst_run / f"module_{module_id}"
        if dst_module.exists():
            shutil.rmtree(dst_module)
        shutil.copytree(src, dst_module)

    return success


def start_copy_background(run_params: dict, dst: pathlib.Path) -> subprocess.Popen:
    """
    Start a copy in the background using cp -r (subprocess).
    Returns the Popen handle so tests can pause containers mid-copy.
    """
    src_root = pathlib.Path(run_params["data_dir"])
    run_dir  = run_params["run_dir"]
    src = str(src_root / f"module_{run_params['module_id'][0]}" / run_dir)
    dst_dir = str(dst / run_dir)
    os.makedirs(dst_dir, exist_ok=True)
    return subprocess.Popen(["cp", "-r", src, dst_dir])


# Expose helpers as fixtures too
@pytest.fixture
def copy_run_dir_fn():
    return copy_run_dir


@pytest.fixture
def start_copy_background_fn():
    return start_copy_background


# ---------------------------------------------------------------------------
# Auto-cleanup: stop any lingering hashpipe after each test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def ensure_clean_daq_state(daq_control_direct, run_params):
    """Stop hashpipe and clean up if a test leaves it running."""
    yield
    try:
        ok, status = daq_control_direct.StatusDaq({
            "data_dir":               run_params["data_dir"],
            "check_hashpipe_running": True,
            "check_disk_usage":       False,
            "check_run_dirs":         False,
        })
        if ok and status.get("hashpipe_running"):
            daq_control_direct.StopDaq({
                "data_dir": run_params["data_dir"],
                "run_dir":  run_params["run_dir"],
            })
            time.sleep(1)
        # Best-effort cleanup
        daq_control_direct.CleanupData({
            "data_dir":  run_params["data_dir"],
            "run_dir":   run_params["run_dir"],
            "module_id": run_params["module_id"],
        })
    except Exception:
        pass  # cleanup best-effort only
