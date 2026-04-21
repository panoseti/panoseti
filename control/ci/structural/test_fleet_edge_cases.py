"""
test_fleet_edge_cases.py

Edge case testing for programmatic fleet generation.
"""

from __future__ import annotations

from control.topology.fleet import generate_fleet_configs


def test_fleet_100_percent_subnet():
    """Verify fleet generation when every node is in a subnet."""
    daq, _uids = generate_fleet_configs(num_daq_nodes=5, subnet_probability=1.0)
    for node in daq.daq_nodes:
        assert node.port_forwarding is not None
        assert node.port_forwarding.status is True


def test_fleet_0_percent_subnet():
    """Verify fleet generation when no nodes are in a subnet."""
    daq, _uids = generate_fleet_configs(num_daq_nodes=5, subnet_probability=0.0)
    for node in daq.daq_nodes:
        assert node.port_forwarding is None


def test_fleet_large_scale():
    """Verify generation of a 100-node fleet (stress test)."""
    num_nodes = 100
    daq, uids = generate_fleet_configs(num_daq_nodes=num_nodes, modules_per_node=1)
    assert len(daq.daq_nodes) == num_nodes
    # 100 modules * 1 module/node
    total_modules = sum(len(d.modules) for d in uids.domes)
    assert total_modules == num_nodes


def test_fleet_zero_nodes():
    """Verify behavior with 0 nodes."""
    daq, uids = generate_fleet_configs(num_daq_nodes=0)
    assert len(daq.daq_nodes) == 0
    total_modules = sum(len(d.modules) for d in uids.domes)
    assert total_modules == 0
