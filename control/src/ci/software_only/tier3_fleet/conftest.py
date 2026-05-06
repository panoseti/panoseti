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
"""
from __future__ import annotations

import json
import os
import pathlib
from typing import Any

import pytest
from panoseti_grpc.daq_control.client import DaqControlClient

from ci.paths import PanoPathsTest
from ci.software_only.conftest import (
    wait_until,
)
from control.utils import config_file

# ---------------------------------------------------------------------------
# Environment / connection parameters
# ---------------------------------------------------------------------------

DAQNODE_DIRECT_HOST: str  = os.getenv("DAQNODE_DIRECT_HOST") or ""
DAQNODE_GATEWAY_HOST: str = os.getenv("DAQNODE_GATEWAY_HOST") or ""
DAQNODE_DATA_HOST: str    = os.getenv("DAQNODE_DATA_HOST", DAQNODE_DIRECT_HOST) or ""
DAQNODE2_HOST: str        = os.getenv("DAQNODE2_HOST") or ""
HEADNODE_HOST: str        = os.getenv("HEADNODE_HOST") or ""
REDIS_HOST: str           = os.getenv("REDIS_HOST") or ""
GRPC_PORT: int            = int(os.getenv("GRPC_PORT", "50051"))

LOKI_URL: str             = os.getenv("LOKI_URL",   "http://localhost:3100")
DAQ_DATA_DIR: str         = os.getenv("DAQ_DATA_DIR", "/data")
HEAD_DATA_DIR: str        = os.getenv("HEAD_DATA_DIR", "/data/head")
DAQNODE_CONTAINER: str    = os.getenv("DAQNODE_CONTAINER_NAME", "ctl-int-daqnode-1")
BINDHOST: str             = os.getenv("BINDHOST") or "lo"


CONTROL_DIR = PanoPathsTest.base_dir()
CONFIG_DIR = PanoPathsTest.integration_configs_root()


# ---------------------------------------------------------------------------
# Clean-up autouse
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _ensure_clean_daq_state(ensure_clean_daq_state):
    """Make the shared clean-up fixture autouse for Tier 3."""
    pass


# ---------------------------------------------------------------------------
# Polling helpers — (REMOVED: imported from parent)
# ---------------------------------------------------------------------------

def wait_grpc_reachable(client: DaqControlClient, data_dir: str, *, timeout: float = 15.0) -> bool:
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

@pytest.fixture
def direct_config_dir() -> pathlib.Path:
    """Returns the isolated directory for direct-connect configs."""
    # In CI, PSETI_CONFIG already points to the isolated variant dir (direct or chaos)
    return pathlib.Path(os.environ["PSETI_CONFIG"])

@pytest.fixture
def gateway_config_dir() -> pathlib.Path:
    """Returns the isolated directory for gateway configs."""
    # If we are in chaos mode, we use the current isolated PSETI_CONFIG.
    # Otherwise, fallback to the gateway template root (but this shouldn't happen often)
    p = pathlib.Path(os.environ["PSETI_CONFIG"])
    if (p / "network_config.json").exists():
        return p
    return PanoPathsTest.integration_configs("gateway")



def get_daq_and_network_config(kind: str = "direct") -> tuple[dict[str, Any], dict[str, Any] | None]:
    """(daq_config.json, network_config.json) for clients connected:
        1. Directly to the daqnode (bypasses gateway).
        2. Via the socat gateway (simulates VPN/NAT topology)
    """
    match kind:
        case "direct": 
            cfg_dir = pathlib.Path(os.environ["PSETI_CONFIG"])
            net_cfg = None
        case "gateway": 
            cfg_dir = pathlib.Path(os.environ["PSETI_CONFIG"])
            # If network_config.json is missing (not in chaos mode), 
            # fallback to gateway template
            if not (cfg_dir / "network_config.json").exists():
                cfg_dir = PanoPathsTest.integration_configs("gateway")
                
            with open(cfg_dir / "network_config.json", 'rb') as f:
                net_cfg_raw = json.load(f)
                net_cfg = config_file.NetworkConfig(**net_cfg_raw).model_dump(mode='json', exclude_unset=True)
        case _:
            raise ValueError(f"Invalid {kind=}. Must be 'direct' or 'gateway'")

    with open(cfg_dir / "daq_config.json", 'rb') as f:
        daq_cfg_raw = json.load(f)
        daq_cfg = config_file.DaqConfig(**daq_cfg_raw).model_dump(mode='json', exclude_unset=True)
    return daq_cfg, net_cfg


# ---------------------------------------------------------------------------
# Docker container handle (for pause/unpause in failure-simulation tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def docker_client() -> Any:
    """Returns a Docker SDK client handle. Skips if unavailable."""
    try:
        import docker
        return docker.from_env()
    except Exception as e:
        pytest.skip(f"Docker SDK unavailable: {e}")
