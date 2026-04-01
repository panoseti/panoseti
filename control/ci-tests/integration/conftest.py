"""
conftest.py — Shared fixtures for the PANOSETI integration test suite.

Environment variables (set by docker-compose.integration.yml):
    DAQNODE_DIRECT_HOST    — IP of the daqnode container (direct access)
    DAQNODE_GATEWAY_HOST   — IP of the gateway container (forwarded access)
    DAQNODE_DATA_HOST      — IP for daq_data gRPC (defaults to DAQNODE_DIRECT_HOST;
                             unified server hosts daq_data + daq_control on the same port)
    DAQNODE2_HOST          — IP of the second DAQ node
    HEADNODE_HOST          — IP of the headnode Telemetry gRPC service
    GRPC_PORT              — gRPC port (default 50051)
    LOKI_URL               — Loki HTTP base URL
    REDIS_HOST             — Redis hostname
    DAQ_DATA_DIR           — data dir on the daqnode (and shared volume mount point)
    HEAD_DATA_DIR          — headnode data destination dir
    DAQNODE_CONTAINER_NAME — Docker container name for pause/unpause tests
    CONFIG_DIR             — Directory to integration test configuration files
    ENABLE_TELEMETRY_TESTS — set to "1" to run test_hashpipe_logs.py
"""
from __future__ import annotations

import os
import json
import sys
import pathlib
import shutil
import subprocess
import time
import uuid
from typing import Callable

import pytest

from panoseti_grpc.daq_control.client import DaqControlClient
from panoseti_grpc.daq_data.client import DaqDataClient

# ---------------------------------------------------------------------------
# Environment / connection parameters
# ---------------------------------------------------------------------------

DAQNODE_DIRECT_HOST  = os.getenv("DAQNODE_DIRECT_HOST",  "localhost")
DAQNODE_GATEWAY_HOST = os.getenv("DAQNODE_GATEWAY_HOST", "localhost")
# Unified server: daq_data + daq_control share the same port on the same IP.
# DAQNODE_DATA_HOST defaults to DAQNODE_DIRECT_HOST (no separate container needed).
DAQNODE_DATA_HOST    = os.getenv("DAQNODE_DATA_HOST", DAQNODE_DIRECT_HOST)
DAQNODE2_HOST        = os.getenv("DAQNODE2_HOST",        "localhost")
HEADNODE_HOST        = os.getenv("HEADNODE_HOST",        "localhost")
GRPC_PORT            = int(os.getenv("GRPC_PORT", "50051"))
LOKI_URL             = os.getenv("LOKI_URL",   "http://localhost:3100")
REDIS_HOST           = os.getenv("REDIS_HOST", "localhost")
DAQ_DATA_DIR         = os.getenv("DAQ_DATA_DIR", "/data")
HEAD_DATA_DIR        = os.getenv("HEAD_DATA_DIR", "/data/head")
DAQNODE_CONTAINER    = os.getenv("DAQNODE_CONTAINER_NAME", "ctl-int-daqnode-1")
BINDHOST             = os.getenv("BINDHOST", "lo")
ENABLE_TELEMETRY_TESTS = os.getenv("ENABLE_TELEMETRY_TESTS", "0") == "1"

CONTROL_DIR = pathlib.Path(__file__).parent.parent.parent   # control/
CONFIG_DIR = pathlib.Path(__file__).parent / "configs"      # config/ci-tests/integration/configs/


# ---------------------------------------------------------------------------
# Real Hashpipe Test module-scoped fixture: start hashpipe + tcpreplay, tear down after all tests
# ---------------------------------------------------------------------------

# Path to PCAP file *inside* the daqnode container (after COPY . .)
PCAP_GLOB = "/app/ci-tests/integration/data/*.pcapng"

# hp_io_cfg for real (non-simulated) hashpipe mode
REAL_HP_IO_CFG = {
    "update_interval_seconds": 0.1,
    "simulate_daq": False,
    "force": True,
    "module_ids": [],   # stream from all active modules
}

HASHPIPE_READY_RETRIES = 20


@pytest.fixture(scope="module")
def hashpipe_pcap_session(daqnode_container, daq_control_direct, run_params):
    """
    Start hashpipe via daq_control gRPC, inject PCAP packets via docker exec
    tcpreplay, then yield.  Tears down hashpipe on exit.

    Function-scoped: each test gets its own fresh hashpipe run so tests are
    fully independent (test_data_collectible_after_stop stops hashpipe mid-test).
    """
    # 0. Verify PCAP exists so tcpreplay doesn't silently fail
    if daqnode_container.exec_run(f"sh -c 'ls {PCAP_GLOB}'").exit_code != 0:
        pytest.fail(f"PCAP missing in container at {PCAP_GLOB}")
    
    # 1. Start hashpipe via gRPC (bindhost=lo so it receives loopback packets)
    lp = {**run_params, "bindhost": "lo"}
    try:
        daq_control_direct.StartDaq(lp)
    except Exception as e:
        pytest.fail(f"Failed to start hashpipe via gRPC: {e}")

    # 2. Wait for hashpipe to be confirmed running
    if not wait_hashpipe_running(
        daq_control_direct, run_params["data_dir"], timeout=HASHPIPE_READY_RETRIES
    ):
        pytest.fail(f"hashpipe did not start within {HASHPIPE_READY_RETRIES}s")
    
    # Forces the native Linux veth to accept the foreign MAC addresses from the PCAP
    daqnode_container.exec_run("ip link set lo promisc on")

    # 3. Run tcpreplay inside daqnode container (loop=5, low rate to avoid overflow)
    replay_cmd = f"sh -c 'tcpreplay --mbps=0.1 --loop=0 --intf1=lo {PCAP_GLOB}'"
    # daqnode_container.exec_run(replay_cmd, detach=True)
    daqnode_container.exec_run(replay_cmd, detach=True)

    yield run_params
    
    # 4. Teardown
    # Kill TCPREPLAY first to stop the packet flood
    daqnode_container.exec_run("pkill -9 tcpreplay", detach=False)

    # 5. Teardown
    try:
        daq_control_direct.StopDaq({
            "data_dir": run_params["data_dir"],
            "run_dir":  run_params["run_dir"],
        })
    except Exception:
        pass
    assert wait_hashpipe_stopped(daq_control_direct, run_params["data_dir"], timeout=8)


# ---------------------------------------------------------------------------
# Helper: daq_data client configured for real (non-simulated) mode
# ---------------------------------------------------------------------------

@pytest.fixture
def real_daq_data_client(hashpipe_pcap_session):
    """
    DaqDataClient connected to the unified daqnode gRPC server.
    daq_data and daq_control share a process, so hashpipe UDS sockets
    at /tmp are directly accessible — no shared volume required.
    """
    run_params = hashpipe_pcap_session
    daq_cfg = {
        "daq_nodes": [{"ip_addr": DAQNODE_DATA_HOST, "data_dir": run_params["data_dir"]}]
    }
    with DaqDataClient(daq_cfg, network_config=None) as client:
        ok = client.init_hp_io(hosts=None, hp_io_cfg=REAL_HP_IO_CFG)
        if not ok:
            pytest.skip(
                "init_hp_io(simulate_daq=False) failed — "
                "check that hashpipe started and UDS sockets are present at /tmp."
            )
        yield client


# ---------------------------------------------------------------------------
# Polling helpers — replace time.sleep with condition-based waits
# ---------------------------------------------------------------------------

def wait_until(
    condition: Callable[[], bool],
    *,
    timeout: float = 10.0,
    interval: float = 0.2,
) -> bool:
    """Poll condition() until it returns truthy or timeout expires.

    Exceptions from condition() are swallowed (treated as False) so callers
    don't need try/except around transient gRPC errors.

    Returns True on success, False on timeout.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if condition():
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False


_HP_STATUS_PARAMS: dict = {
    "check_hashpipe_running": True,
    "check_disk_usage":       False,
    "check_run_dirs":         False,
}


def wait_hashpipe_running(
    client: DaqControlClient,
    data_dir: str,
    *,
    timeout: float = 10.0,
) -> bool:
    """Poll StatusDaq until hashpipe_running=True or timeout."""
    return wait_until(
        lambda: client.StatusDaq({"data_dir": data_dir, **_HP_STATUS_PARAMS})[1].get(
            "hashpipe_running"
        )
        is True,
        timeout=timeout,
    )


def wait_hashpipe_stopped(
    client: DaqControlClient,
    data_dir: str,
    *,
    timeout: float = 10.0,
) -> bool:
    """Poll StatusDaq until hashpipe_running is False/None or timeout."""
    return wait_until(
        lambda: client.StatusDaq({"data_dir": data_dir, **_HP_STATUS_PARAMS})[1].get(
            "hashpipe_running"
        )
        is not True,
        timeout=timeout,
    )


def wait_grpc_reachable(client, data_dir: str, *, timeout: float = 15.0) -> bool:
    """Poll until a StatusDaq RPC succeeds (server is back after restart/pause)."""
    return wait_until(
        lambda: client.StatusDaq({
            "data_dir":               data_dir,
            "check_hashpipe_running": False,
            "check_disk_usage":       False,
            "check_run_dirs":         False,
        })[0] is True,
        timeout=timeout,
    )

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
            with open(cfg_dir / "network_config.json", 'rb') as f:
                net_cfg_raw = json.load(f)
                net_cfg = config_file.NetworkConfigValidator(**net_cfg_raw).model_dump(mode='json', exclude_unset=True)
        case _:
            raise ValueError(f"Invalid {kind=}. Must be 'direct' or 'gateway'")

    with open(cfg_dir / "daq_config.json", 'rb') as f:
        daq_cfg_raw = json.load(f)
        daq_cfg = config_file.DaqConfigValidator(**daq_cfg_raw).model_dump(mode='json', exclude_unset=True)
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
    # daq_cfg = {
    #     "daq_nodes": [{"ip_addr": DAQNODE_DATA_HOST, "data_dir": DAQ_DATA_DIR}]
    # }
    daq_cfg, net_cfg = get_daq_and_network_config(kind="gateway")
    with DaqDataClient(daq_cfg, network_config=net_cfg) as client:
        yield client


# ---------------------------------------------------------------------------
# Run parameters fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def run_params() -> dict:
    """Fresh run parameters for each test — unique run_dir per test."""
    return {
        "data_dir":         DAQ_DATA_DIR,
        "daq_ip_addr":      DAQNODE_DIRECT_HOST,
        "bindhost":         BINDHOST,
        "max_file_size_mb": 1,
        "group_ph_frames":  True,
        "run_dir":          f"ci_run_{uuid.uuid4().hex[:8]}.pffd",
        "obs":              "citest",
        "module_id":        [250, 254],
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

@pytest.fixture(autouse=False)
def ensure_clean_daq_state(daq_control_direct, run_params):
    """Stop hashpipe and clean up if a test leaves it running."""
    yield
    # Always call StopDaq unconditionally — it's idempotent and handles
    # the case where hashpipe crashed (leaving a stale hashpipe_pid on the
    # server) so CleanupData isn't blocked by the stale pid check.
    try:
        daq_control_direct.StopDaq({
            "data_dir": run_params["data_dir"],
            "run_dir":  run_params["run_dir"],
        })
    except Exception:
        pass
    wait_hashpipe_stopped(daq_control_direct, run_params["data_dir"], timeout=8)
    try:
        daq_control_direct.CleanupData({
            "data_dir":  run_params["data_dir"],
            "run_dir":   run_params["run_dir"],
            "module_id": run_params["module_id"],
        })
    except Exception:
        pass
