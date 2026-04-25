"""
ci/tier1_unit/test_topology_gen.py

Unit tests for the programmatic topology generation utilities.
Verifies that generated matrices satisfy Pydantic strictness and reflect real sites.
"""

from __future__ import annotations

from control.topology.fleet import generate_fleet_configs, generate_palomar_topology
from control.utils.pydantic_config_models import DaqConfig, NetworkConfig, ObsConfig, QuaboUids


def test_when_fleet_generated_then_models_are_valid():
    """
    Intent: Ensure generate_fleet_configs produces a full set of valid Pydantic models.
    Assertion: All 4 models (Daq, UIDs, Network, Obs) instantiate without ValidationError.
    """
    daq, uids, net, obs = generate_fleet_configs(num_daq_nodes=2, modules_per_node=2)
    
    assert isinstance(daq, DaqConfig)
    assert isinstance(uids, QuaboUids)
    assert isinstance(net, NetworkConfig)
    assert isinstance(obs, ObsConfig)
    
    assert len(daq.daq_nodes) == 2
    assert len(uids.domes[0].modules) == 4

def test_when_palomar_topology_generated_then_matches_site_docs():
    """
    Intent: Verify the Palomar generator reflects the documentation (4 sites, port forwarding).
    Assertion: 4 DAQ nodes, all with port forwarding enabled.
    """
    daq, uids, net, obs = generate_palomar_topology()
    
    assert str(daq.head_node_ip_addr) == "10.200.146.1"
    assert len(daq.daq_nodes) == 4
    
    # Verify Gattini site (from docs)
    gattini = next(n for n in daq.daq_nodes if str(n.ip_addr) == "192.168.0.4")
    assert gattini.port_forwarding.status is True
    assert str(gattini.port_forwarding.gw_ip) == "10.200.146.11"
    
    # Verify NetworkConfig has matching entries
    net_mod = next(m for m in net.modules if str(m.ip_addr) == "192.168.3.248")
    assert net_mod.port_forwarding.status is True
    assert net_mod.port_forwarding.reboot_port == [69, 60004, 60005, 60006]

def test_when_large_fleet_generated_then_ids_are_unique():
    """
    Intent: Validate ID assignment logic for high-node-count simulations.
    Assertion: All module IDs across all nodes are unique and sequential.
    """
    daq, uids, net, obs = generate_fleet_configs(num_daq_nodes=10, modules_per_node=1)
    
    all_mids = []
    for node in daq.daq_nodes:
        all_mids.extend(node.module_ids)
        
    assert len(all_mids) == 10
    assert len(set(all_mids)) == 10
    assert sorted(all_mids) == list(range(1, 11))
