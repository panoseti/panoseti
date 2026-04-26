"""
conftest.py — Tier 5 Heavy Integration fixtures.

Connects to the STATIC Docker Compose stack (docker-compose.integration.yml).
These tests require Hashpipe and high Linux capabilities.
"""

import os

import pytest
from panoseti_grpc.daq_control.client import DaqControlClient
from panoseti_grpc.daq_data.client import DaqDataClient

# Static IPs from docker-compose.integration.yml
DAQNODE_DIRECT_HOST = os.getenv("DAQNODE_DIRECT_HOST", "192.168.0.10")
DAQNODE2_DIRECT_HOST = os.getenv("DAQNODE2_DIRECT_HOST", "192.168.0.20")
GRPC_PORT = int(os.getenv("GRPC_PORT", "50051"))

@pytest.fixture(scope="session")
def daq_control_direct():
    """Client connected directly to the first static daqnode."""
    return DaqControlClient(host=DAQNODE_DIRECT_HOST, port=GRPC_PORT)

@pytest.fixture(scope="session")
def daq_control_node2():
    """Client connected directly to the second static daqnode."""
    return DaqControlClient(host=DAQNODE2_DIRECT_HOST, port=GRPC_PORT)

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
