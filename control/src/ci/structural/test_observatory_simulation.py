"""
test_observatory_simulation.py

Parameterized structural simulations of various observatory fleet sizes and network topologies.
"""

from __future__ import annotations

import networkx as nx
import pytest

from control.topology.fleet import generate_fleet_configs
from control.topology.graph_builder import GraphBuilder


@pytest.mark.parametrize("num_modules", [1, 2, 4])
def test_simulated_observatory_structure(num_modules: int):
    """
    Simulates an observatory with n modules, each having its own DAQ node.
    Each node has a 50% chance of being in a subnet (requiring port forwarding).
    """
    # 1. Generate the configuration
    # One DAQ node per module as requested
    daq_config, quabo_uids = generate_fleet_configs(
        num_daq_nodes=num_modules, 
        modules_per_node=1,
        subnet_probability=0.5
    )
    
    # 2. Build the Graph
    builder = GraphBuilder()
    graph = builder.build_from_configs(daq_config, quabo_uids)
    
    # 3. Verify Structural Invariants
    head_ip = str(daq_config.head_node_ip_addr)
    
    # Total expected nodes:
    # 1 HeadNode
    # n DAQNodes
    # m Gateways (count where pf is active)
    # n Modules
    # n * 4 Quabos
    gateways = [n for n, d in graph.nodes(data=True) if d.get("role") == "gateway"]
    expected_node_count = 1 + num_modules + len(gateways) + num_modules + (num_modules * 4)
    
    assert len(graph.nodes) == expected_node_count
    assert nx.is_directed_acyclic_graph(graph), "Topology must be a DAG"

    # 4. Verify Reachability (Logical Start/Stop Path)
    # If the Head Node can reach every Quabo, it means a Start/Stop command 
    # has a valid propagation path through the network.
    reachable_from_head = nx.descendants(graph, head_ip)
    
    quabo_nodes = [n for n, d in graph.nodes(data=True) if d.get("role") == "quabo"]
    for q in quabo_nodes:
        assert q in reachable_from_head, f"Quabo {q} is unreachable from Head Node {head_ip}"

    # Verify every DAQ node is reachable
    daq_nodes = [n for n, d in graph.nodes(data=True) if d.get("role") == "daqnode"]
    for d_node in daq_nodes:
        assert d_node in reachable_from_head, f"DAQ Node {d_node} is unreachable from Head Node"

    # 5. Verify Subnet Logic
    # If a DAQ node is behind a gateway, its module must also be reachable via that gateway
    for node in daq_config.daq_nodes:
        if node.port_forwarding and node.port_forwarding.status:
            gw_ip = str(node.port_forwarding.gw_ip)
            assert gw_ip in graph.nodes, f"Gateway {gw_ip} missing from graph"
            
            # Find the module for this node
            # (In our generator, node i manages module i+1)
            # We can check descendants of the gateway
            gw_descendants = nx.descendants(graph, gw_ip)
            module_found = False
            for desc in gw_descendants:
                if graph.nodes[desc].get("role") == "module":
                    module_found = True
                    break
            assert module_found, f"No module found downstream of gateway {gw_ip}"
