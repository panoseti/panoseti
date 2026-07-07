"""
test_single_node_headnode.py — Tests for a single node acting as both headnode and daqnode.
"""

from __future__ import annotations

import os
from ipaddress import ip_address

import pytest

from ci.software_only.infra.spec import FleetSpec
from ci.software_only.infra.workspace import Workspace
from ci.software_only.orchestrator.fleet import Fleet
from ci.software_only.tier3_fleet.conftest import requires_docker
from control.utils.pydantic_config_models import RunStatus

pytestmark = pytest.mark.tier3

SINGLE_NODE_IP = "192.168.10.10"
HEADNODE_PORT = 50052
DAQNODE_PORT = 50051

single_node_spec = (
    FleetSpec(seed=3, name="single_node_headnode", tier="tier3")
    .with_headnode(ip=SINGLE_NODE_IP, data_dir="/data")
    .add_dome("dome0", lat=37.342, lon=-121.637, alt=1283.0)
    .add_module(200, version="qfp", timing="wr", ip="192.168.3.32")
    .add_daq_node(ip=SINGLE_NODE_IP, modules=[200], bindhost="lo", data_dir="/data")
)

@requires_docker
@pytest.mark.parametrize(
    "pseti_workspace",
    [single_node_spec],
    indirect=True,
)
class TestSingleNodeHeadnode:
    def test_single_node_config_validation(self, pseti_workspace: Workspace):
        """Test that topology building passes without infinite loop errors."""
        topology = pseti_workspace.topology
        assert topology is not None
        assert topology.daq.head_node_ip_addr == ip_address(SINGLE_NODE_IP)
        assert topology.daq.daq_nodes[0].ip_addr == ip_address(SINGLE_NODE_IP)

    def test_single_node_lifecycle(self, pseti_workspace: Workspace):
        """Test that headnode and daqnode can start concurrently on the same machine."""
        # Override the headnode gateway port for this test
        os.environ["DAQ_DATA_GATEWAY_PORT"] = str(HEADNODE_PORT)
        
        fleet = Fleet.from_topology(
            pseti_workspace.topology, 
            pseti_workspace,
            healthcheck_timeout=120.0
        )
        
        with fleet:
            fleet.wait_healthy()
            
            assert fleet.n_nodes == 1
            cfg = fleet.live_daq_config
            assert cfg is not None
            
            # Ensure the port forwarding picked up a valid mapped port
            pf = cfg.daq_nodes[0].port_forwarding
            assert pf is not None
            assert pf.grpc_port > 1024
