"""
conftest.py — Tier 5 Heavy Integration fixtures.

Connects to the STATIC Docker Compose stack (docker-compose.integration.yml).
These tests require Hashpipe and high Linux capabilities.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import docker
import pytest
from docker.models.containers import Container
from panoseti_grpc.daq_control.client import DaqControlClient
from panoseti_grpc.daq_data.client import DaqDataClient

from ci.software_only.conftest import (
    REAL_HP_IO_CFG,
)

# Static IPs from docker-compose.integration.yml
DAQNODE_DIRECT_HOST = os.getenv("DAQNODE_DIRECT_HOST", "192.168.0.10")
DAQNODE2_DIRECT_HOST = os.getenv("DAQNODE2_DIRECT_HOST", "192.168.0.20")
GRPC_PORT = int(os.getenv("GRPC_PORT", "50051"))
DAQNODE_CONTAINER_NAME = os.getenv("DAQNODE_CONTAINER_NAME", "pseti-integration-daqnode-1")

@pytest.fixture(scope="session")
def daq_control_direct():

    """Client connected directly to the first static daqnode."""
    return DaqControlClient(host=DAQNODE_DIRECT_HOST, port=GRPC_PORT)

@pytest.fixture(scope="session")
def daq_control_node2():
    """Client connected directly to the second static daqnode."""
    return DaqControlClient(host=DAQNODE2_DIRECT_HOST, port=GRPC_PORT)

@pytest.fixture(autouse=True)
def _ensure_clean_daq_state(ensure_clean_daq_state):
    """Make the shared clean-up fixture autouse for Tier 5."""
    pass

@pytest.fixture(scope="session")
def redis_client() -> Iterator[Any]:
    """Static Redis client connected to the compose service."""
    import redis
    r = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        db=int(os.getenv("REDIS_DB", "0")),
        decode_responses=True
    )
    yield r

@pytest.fixture(scope="session")
def daq_data_client():
    """DaqDataClient for the static integration environment."""
    # Build a minimal daq_config that points to the static IPs
    daq_cfg = {
        "daq_nodes": [
            {"ip_addr": DAQNODE_DIRECT_HOST, "data_dir": "/data"},
            {"ip_addr": DAQNODE2_DIRECT_HOST, "data_dir": "/data"}
        ]
    }
    with DaqDataClient(daq_cfg, network_config=None) as client:
        yield client

@pytest.fixture(scope="module")
def run_params():
    """Static run parameters for Tier 5 tests."""
    return {
        "data_dir": "/data",
        "daq_ip_addr": DAQNODE_DIRECT_HOST,
        "bindhost": os.getenv("BINDHOST", "lo"),
        "max_file_size_mb": 10,
        "group_ph_frames": False,
        "run_dir": "tier5_integration_test.pffd",
        "obs": "tier5",
        "module_id": [200, 201, 250], # MUST INCLUDE MODULE 250: hardcoded bc the pcapng file only has module 250 data
    }

@pytest.fixture(scope="session")
def head_data_dir() -> str:
    """Path to the head node data directory inside the tester container."""
    return "/data/head"

@pytest.fixture(scope="session")
def daqnode_container() -> Container:
    """Returns the Docker SDK Container for the primary static daqnode."""
    client = docker.from_env()
    try:
        return client.containers.get(DAQNODE_CONTAINER_NAME)
    except docker.errors.NotFound:
        pytest.fail(f"Static daqnode container '{DAQNODE_CONTAINER_NAME}' not found. "
                    "Tier 5 tests require the Docker Compose stack to be running.")

# ---------------------------------------------------------------------------
# Helper: daq_data client configured for real (non-simulated) mode
# ---------------------------------------------------------------------------


@pytest.fixture
def real_daq_data_client(hashpipe_pcap_session: dict[str, Any], daqnode_num: int = 1) -> Iterator[DaqDataClient]:
    """
    DaqDataClient connected to the unified daqnode gRPC server.
    daq_data and daq_control share a process, so hashpipe UDS sockets
    at /tmp are directly accessible — no shared volume required.
    """
    run_params = hashpipe_pcap_session
    daqnode_host = DAQNODE_DIRECT_HOST# if daqnode_num == 1 else DAQNODE2_DIRECT_HOST
    daq_cfg = {
        "daq_nodes": [{"ip_addr": daqnode_host, "data_dir": run_params["data_dir"]}]
    }
    with DaqDataClient(daq_cfg, network_config=None) as client:
        ok = client.init_hp_io(hosts=None, hp_io_cfg=REAL_HP_IO_CFG)
        # ok = client.init_sim(hosts=None)
        if not ok:
            pytest.skip(
                "init_hp_io(simulate_daq=False) failed — "
                "check that hashpipe started and UDS sockets are present at /tmp."
            )
        yield client
        
    


@contextmanager
def env_var(key, value):
    """
    # Usage
    with env_var("DATABASE_URL", "postgres://user:pass@localhost/db"):
        # Code in this block sees the new environment variable
        print(os.environ["DATABASE_URL"])
    """
    original_value = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if original_value is None:
            del os.environ[key]
        else:
            os.environ[key] = original_value

