"""
conftest.py — Tier 5 Heavy Integration fixtures.

Connects to the STATIC Docker Compose stack (docker-compose.integration.yml).
These tests require Hashpipe and high Linux capabilities.
"""

import contextlib
import os
from collections.abc import Iterator
from typing import Any

import docker
import pytest
from panoseti_grpc.daq_control.client import DaqControlClient
from panoseti_grpc.daq_data.client import DaqDataClient

from ci.tier3_fleet.conftest import (
    wait_hashpipe_running,
    wait_hashpipe_stopped,
)

# Static IPs from docker-compose.integration.yml
DAQNODE_DIRECT_HOST = os.getenv("DAQNODE_DIRECT_HOST", "192.168.0.10")
DAQNODE2_DIRECT_HOST = os.getenv("DAQNODE2_DIRECT_HOST", "192.168.0.20")
GRPC_PORT = int(os.getenv("GRPC_PORT", "50051"))
DAQNODE_CONTAINER_NAME = os.getenv("DAQNODE_CONTAINER_NAME", "pseti-integration-daqnode-1")
PCAP_GLOB = "/app/wr/raw/*.pcap"

@pytest.fixture(scope="session")
def daq_control_direct():
    """Client connected directly to the first static daqnode."""
    return DaqControlClient(host=DAQNODE_DIRECT_HOST, port=GRPC_PORT)

@pytest.fixture(scope="session")
def daq_control_node2():
    """Client connected directly to the second static daqnode."""
    return DaqControlClient(host=DAQNODE2_DIRECT_HOST, port=GRPC_PORT)

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
        "group_ph_frames": True,
        "run_dir": "tier5_integration_test.pffd",
        "obs": "tier5",
        "module_id": [200, 201],
    }

@pytest.fixture(scope="session")
def head_data_dir() -> str:
    """Path to the head node data directory inside the tester container."""
    return "/data/head"

@pytest.fixture(scope="session")
def daqnode_container() -> Any:
    """Returns the Docker SDK Container for the primary static daqnode."""
    client = docker.from_env()
    try:
        return client.containers.get(DAQNODE_CONTAINER_NAME)
    except docker.errors.NotFound:
        pytest.fail(f"Static daqnode container '{DAQNODE_CONTAINER_NAME}' not found. "
                    "Tier 5 tests require the Docker Compose stack to be running.")

@pytest.fixture(scope="module")
def hashpipe_pcap_session(daqnode_container: Any, daq_control_direct: DaqControlClient, run_params: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """
    Start hashpipe via daq_control gRPC, inject PCAP packets via docker exec
    tcpreplay, then yield.  Tears down hashpipe on exit.
    """
    # 1. Start hashpipe via gRPC
    lp = {**run_params, "bindhost": "lo"}
    daq_control_direct.StartDaq(lp)

    # 2. Wait for hashpipe to be confirmed running
    if not wait_hashpipe_running(daq_control_direct, run_params["data_dir"], timeout=20):
        pytest.fail("hashpipe did not start within 20s")
    
    # Enable promisc mode for the virtual interface
    daqnode_container.exec_run("ip link set lo promisc on")

    # 3. Run tcpreplay inside daqnode container
    replay_cmd = f"sh -c 'tcpreplay --mbps=0.1 --loop=0 --intf1=lo {PCAP_GLOB}'"
    daqnode_container.exec_run(replay_cmd, detach=True)

    yield run_params
    
    # 4. Teardown
    daqnode_container.exec_run("pkill -9 tcpreplay", detach=False)
    with contextlib.suppress(Exception):
        daq_control_direct.StopDaq({
            "data_dir": run_params["data_dir"],
            "run_dir":  run_params["run_dir"],
        })
    wait_hashpipe_stopped(daq_control_direct, run_params["data_dir"], timeout=8)
