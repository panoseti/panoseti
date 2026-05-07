"""
conftest.py — Tier 5 Heavy Integration fixtures (v2).

Connects to the STATIC Docker Compose stack (docker-compose.integration.yml).
All tests skip automatically if the compose stack is not reachable.

Network topology (from the compose file):
    headnode_net  10.0.1.0/24
    daqnode_net   192.168.0.0/24
    quabo_net     192.168.3.0/24

Key env vars (all have compose-stack defaults):
    DAQNODE_DIRECT_HOST    DAQ node 1 (daqnode_net IP)
    DAQNODE2_HOST          DAQ node 2 (daqnode_net IP)
    GRPC_PORT              gRPC port (default 50051)
    REDIS_HOST             Redis (headnode_net IP)
    LOKI_URL               Loki endpoint
    HEAD_DATA_DIR          Data dir inside the headnode/tester containers
    DAQ_DATA_DIR           Data dir shared with daqnode containers
    DAQNODE_CONTAINER_NAME Docker container name for primary daqnode
"""

from __future__ import annotations

import contextlib
import importlib
import os
import pathlib
import socket
import time
import uuid
from collections.abc import Generator, Iterator
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Static IP defaults (match docker-compose.integration.yml)
# ---------------------------------------------------------------------------

DAQNODE1_HOST = os.getenv("DAQNODE_DIRECT_HOST", "192.168.0.10")
DAQNODE2_HOST = os.getenv("DAQNODE2_HOST", "192.168.0.20")
GRPC_PORT = int(os.getenv("GRPC_PORT", "50051"))
REDIS_HOST = os.getenv("REDIS_HOST", "10.0.1.20")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
LOKI_URL = os.getenv("LOKI_URL", "http://10.0.1.21:3100")
HEAD_DATA_DIR = os.getenv("HEAD_DATA_DIR", "/data/head")
DAQ_DATA_DIR = os.getenv("DAQ_DATA_DIR", "/data")
DAQNODE_CONTAINER_NAME = os.getenv("DAQNODE_CONTAINER_NAME", "ctl-int-daqnode-1")

# Path to PCAP files inside the daqnode container
PCAP_CONTAINER_DIR = "/app/src/ci/fixtures/data/"
PCAP_GLOB = "*.pcapng"


# ---------------------------------------------------------------------------
# Stack-availability guard
# ---------------------------------------------------------------------------

def _tcp_reachable(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _compose_stack_running() -> bool:
    """Return True if both daqnodes are TCP-reachable on their gRPC port."""
    return (
        _tcp_reachable(DAQNODE1_HOST, GRPC_PORT)
        and _tcp_reachable(DAQNODE2_HOST, GRPC_PORT)
    )


requires_compose_stack = pytest.mark.skipif(
    not _compose_stack_running(),
    reason=(
        f"Compose stack not running — {DAQNODE1_HOST}:{GRPC_PORT} or "
        f"{DAQNODE2_HOST}:{GRPC_PORT} unreachable"
    ),
)


# ---------------------------------------------------------------------------
# Polling helpers
# ---------------------------------------------------------------------------

def _wait_until(
    condition: Any,
    *,
    timeout: float = 10.0,
    interval: float = 0.2,
) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if condition():
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False


def wait_hashpipe_running(
    client: Any,
    data_dir: str,
    *,
    timeout: float = 10.0,
) -> bool:
    """Poll StatusDaq until hashpipe_running=True or timeout."""
    params = {
        "data_dir": data_dir,
        "check_hashpipe_running": True,
        "check_disk_usage": False,
        "check_run_dirs": False,
    }
    return _wait_until(
        lambda: client.StatusDaq(params, timeout=2.0)[1].get("hashpipe_running") is True,
        timeout=timeout,
    )


def wait_hashpipe_stopped(
    client: Any,
    data_dir: str,
    *,
    timeout: float = 10.0,
) -> bool:
    """Poll StatusDaq until hashpipe_running is False/None or timeout."""
    params = {
        "data_dir": data_dir,
        "check_hashpipe_running": True,
        "check_disk_usage": False,
        "check_run_dirs": False,
    }
    return _wait_until(
        lambda: client.StatusDaq(params, timeout=2.0)[1].get("hashpipe_running") is not True,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Config workspace (session-scoped)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def t5_workspace(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Session-scoped workspace matching the static compose stack topology.

    Uses FleetSpec.two_node_ci() to generate all 7 validated Pydantic configs
    and materialize them to a tmp dir, then sets PSETI_* env vars for the
    duration of the session.
    """
    from ci.software_only_v2.infra.spec import FleetSpec
    from ci.software_only_v2.infra.materialize import write_all
    from ci.software_only_v2.infra.workspace import StateProbe, Workspace
    from control.utils.paths import PanoPaths

    head_prefix = os.getenv("HEAD_NET_PREFIX", "10.0.1")
    daq_prefix = os.getenv("DAQ_NET_PREFIX", "192.168.0")
    quabo_prefix = os.getenv("QUABO_NET_PREFIX", "192.168.3")

    spec = FleetSpec.two_node_ci(
        head_prefix=head_prefix,
        daq_prefix=daq_prefix,
        quabo_prefix=quabo_prefix,
        tier="tier5",
    )
    tmp_path = tmp_path_factory.mktemp("t5_workspace")

    env_dirs = [
        ("PSETI_CONFIG", "configs"),
        ("PSETI_STATE", "state"),
        ("PSETI_TMP", "tmp"),
        ("PSETI_LOGS", "state/logs"),
        ("PSETI_QUABOS", "quabos"),
        ("PSETI_FIRMWARE", "firmware"),
    ]
    original_env: dict[str, str | None] = {}
    for key, sub in env_dirs:
        original_env[key] = os.environ.get(key)
        path = tmp_path / sub
        path.mkdir(parents=True, exist_ok=True)
        os.chmod(path, 0o777)
        os.environ[key] = str(path)

    topology = spec.build()
    write_all(topology, PanoPaths.config_dir())

    # get_quabo_uids() reads from tmp_dir — write there too
    uids_json = topology.quabo_uids.model_dump_json(indent=2)  # type: ignore[union-attr]
    (tmp_path / "tmp" / "quabo_uids.json").write_text(uids_json)

    PanoPaths.ensure_state_dirs()
    importlib.reload(importlib.import_module("control.utils.config_file"))

    workspace = Workspace(root=tmp_path, topology=topology, state_probe=StateProbe())
    yield workspace

    for key, orig in original_env.items():
        if orig is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = orig


# ---------------------------------------------------------------------------
# gRPC client fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def daq_control_node1() -> Any:
    """DaqControlClient for the primary static daqnode (daqnode_net IP)."""
    from panoseti_grpc.daq_control.client import DaqControlClient
    return DaqControlClient(host=DAQNODE1_HOST, port=GRPC_PORT)


@pytest.fixture(scope="session")
def daq_control_node2() -> Any:
    """DaqControlClient for the secondary static daqnode."""
    from panoseti_grpc.daq_control.client import DaqControlClient
    return DaqControlClient(host=DAQNODE2_HOST, port=GRPC_PORT)


# ---------------------------------------------------------------------------
# Telemetry fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def redis_client() -> Iterator[Any]:
    """Redis client connected to the static compose Redis service."""
    import redis  # type: ignore[import]
    r = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=int(os.getenv("REDIS_DB", "0")),
        decode_responses=True,
    )
    yield r


# ---------------------------------------------------------------------------
# Docker container handle (for exec_run / chaos ops)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def daqnode_docker_container() -> Any:
    """Docker SDK Container for the primary daqnode (exec_run / chaos ops)."""
    import docker  # type: ignore[import]
    client = docker.from_env()
    try:
        return client.containers.get(DAQNODE_CONTAINER_NAME)
    except docker.errors.NotFound:
        pytest.skip(
            f"Container '{DAQNODE_CONTAINER_NAME}' not found. "
            "Run: docker compose -f docker-compose.integration.yml up -d"
        )


# ---------------------------------------------------------------------------
# Chaos accessor for static compose containers
# ---------------------------------------------------------------------------

@pytest.fixture
def chaos() -> Any:
    """Chaos accessor bound to the static compose stack containers.

    Use container name strings directly:
        with chaos.net.latency(DAQNODE_CONTAINER_NAME, latency_ms=200): ...
    """
    from ci.software_only_v2.fixtures.chaos import Chaos

    class _StaticFleet:
        pass

    return Chaos(_StaticFleet())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Run parameters + isolation
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def run_params() -> dict[str, Any]:
    """Run parameters for tier-5 integration tests."""
    return {
        "data_dir": DAQ_DATA_DIR,
        "daq_ip_addr": DAQNODE1_HOST,
        "bindhost": os.getenv("BINDHOST", "lo"),
        "max_file_size_mb": 10,
        "group_ph_frames": False,
        "run_dir": f"t5_run_{uuid.uuid4().hex[:8]}.pffd",
        "obs": "tier5",
        # Module 250 is hardcoded: only the bundled PCAP has module-250 data
        "module_id": [200, 201, 250],
    }


@pytest.fixture
def head_data_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """Per-test isolated head-node data directory."""
    d = tmp_path / "head_data"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Hashpipe + tcpreplay session fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hashpipe_pcap_session(
    daqnode_docker_container: Any,
    daq_control_node1: Any,
    run_params: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    """Start hashpipe via gRPC + tcpreplay; yield run_params; teardown."""
    # Verify PCAP exists
    res = daqnode_docker_container.exec_run(
        f"sh -c 'ls {PCAP_CONTAINER_DIR}/{PCAP_GLOB}'"
    )
    if res.exit_code != 0:
        pytest.skip(
            f"PCAP missing in container at {PCAP_CONTAINER_DIR}/{PCAP_GLOB} — "
            "real-data tests require the compose daqnode with /app mounted"
        )

    params = {**run_params, "bindhost": "lo"}
    try:
        daq_control_node1.StartDaq(params)
    except Exception as exc:
        pytest.fail(f"Failed to start hashpipe via gRPC: {exc}")

    if not wait_hashpipe_running(daq_control_node1, params["data_dir"], timeout=20):
        pytest.fail("hashpipe did not start within 20s")

    daqnode_docker_container.exec_run("ip link set lo promisc on")
    replay_cmd = (
        f"sh -c 'tcpreplay --mbps=0.1 --loop=0 --intf1=lo {PCAP_GLOB}'"
    )
    daqnode_docker_container.exec_run(replay_cmd, detach=True, workdir=PCAP_CONTAINER_DIR)

    yield run_params

    daqnode_docker_container.exec_run("pkill -9 tcpreplay", detach=False)
    with contextlib.suppress(Exception):
        daq_control_node1.StopDaq({
            "data_dir": params["data_dir"],
            "run_dir": params["run_dir"],
        })
    wait_hashpipe_stopped(daq_control_node1, params["data_dir"], timeout=15)


# ---------------------------------------------------------------------------
# Clean-state autouse fixture
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def ensure_clean_daq_state(
    daq_control_node1: Any,
    daq_control_node2: Any,
) -> Generator[None, None, None]:
    """Ensure no hashpipe is running before and after each test."""

    def _stop_all() -> None:
        for client in (daq_control_node1, daq_control_node2):
            with contextlib.suppress(Exception):
                client.StopDaq({"data_dir": DAQ_DATA_DIR, "run_dir": ""}, timeout=70.0)
            wait_hashpipe_stopped(client, DAQ_DATA_DIR, timeout=10)
        from control.utils.run_state import RunStateManager
        RunStateManager().clear_state()

    _stop_all()
    yield
    _stop_all()
