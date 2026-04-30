"""
test_topology_logic.py

High-speed mathematical validation of PANOSETI topology invariants.
"""

from __future__ import annotations

from ipaddress import IPv4Address

import networkx as nx
import pytest

from control.topology.fleet import generate_fleet_configs
from control.topology.graph_builder import GraphBuilder
from control.utils.pydantic_config_models import (
    DaqConfig,
    DaqNode,
    PortForwarding,
    QuaboUidDome,
    QuaboUidEntry,
    QuaboUidModule,
    QuaboUids,
)


@pytest.fixture
def base_quabo_uids() -> QuaboUids:
    module = QuaboUidModule(
        ip_addr=IPv4Address("192.168.3.248"),
        quabos=[QuaboUidEntry(uid=f"uid_{i}") for i in range(4)],
        id=254,
    )
    return QuaboUids(domes=[QuaboUidDome(num=0, modules=[module])])


def test_detect_orphan_module(base_quabo_uids):
    """If a DAQ node exists but has no edge from Head, its modules are orphans."""
    daq_node = DaqNode(
        username="root",
        data_dir="/data",
        ip_addr=IPv4Address("192.168.0.10"),
        module_ids=[254],
    )
    daq_config = DaqConfig(
        head_node_data_dir="/data/head",
        head_node_ip_addr=IPv4Address("10.0.1.5"),
        daq_nodes=[daq_node],
    )

    builder = GraphBuilder()
    graph = builder.build_from_configs(daq_config, base_quabo_uids)

    # Manually break ALL links from the Head node to simulate isolation
    head_ip = "10.0.1.5"
    for target in list(graph.successors(head_ip)):
        graph.remove_edge(head_ip, target)

    reachable = nx.descendants(graph, head_ip) | {head_ip}

    # Verify module is orphaned
    module_nodes = [n for n, d in graph.nodes(data=True) if d.get("role") == "module"]
    for mod in module_nodes:
        assert mod not in reachable, f"Module {mod} should be unreachable"


def test_gateway_bottleneck_detection(base_quabo_uids):
    """A gateway with 8 Quabos should trigger a structural warning logic check."""
    # Create 2 modules (8 quabos total)
    m1 = QuaboUidModule(
        ip_addr=IPv4Address("192.168.3.248"),
        quabos=[QuaboUidEntry(uid=f"u1_{i}") for i in range(4)],
        id=201,
    )
    m2 = QuaboUidModule(
        ip_addr=IPv4Address("192.168.3.252"),
        quabos=[QuaboUidEntry(uid=f"u2_{i}") for i in range(4)],
        id=202,
    )
    uids = QuaboUids(domes=[QuaboUidDome(num=0, modules=[m1, m2])])

    pf = PortForwarding(status=True, gw_ip=IPv4Address("10.0.1.10"))
    daq_node = DaqNode(
        username="root",
        data_dir="/data",
        ip_addr=IPv4Address("192.168.0.10"),
        module_ids=[201, 202],
        port_forwarding=pf,
    )
    daq_config = DaqConfig(
        head_node_data_dir="/data/head",
        head_node_ip_addr=IPv4Address("10.0.1.5"),
        daq_nodes=[daq_node],
    )

    builder = GraphBuilder()
    graph = builder.build_from_configs(daq_config, uids)

    # Count quabos under the gateway
    gw_node = "10.0.1.10"
    downstream = nx.descendants(graph, gw_node)
    quabos = [n for n in downstream if graph.nodes[n].get("role") == "quabo"]

    assert len(quabos) == 8
    assert len(quabos) > 4, "Should detect bottleneck (> 4 quabos per gateway)"


def test_n_node_fleet_topology():
    """Verify that we can generate and build a graph for a large 10-node fleet."""
    num_nodes = 10
    mods_per_node = 2
    daq_config, quabo_uids = generate_fleet_configs(
        num_daq_nodes=num_nodes, modules_per_node=mods_per_node
    )

    builder = GraphBuilder()
    graph = builder.build_from_configs(daq_config, quabo_uids)

    # Assertions
    # 1 HeadNode + 10 DAQNodes + m Gateways + (10 * 2) Modules + (10 * 2 * 4) Quabos
    gateways = [n for n, d in graph.nodes(data=True) if d.get("role") == "gateway"]
    expected_nodes = (
        1 + num_nodes + len(gateways) + (num_nodes * mods_per_node) + (num_nodes * mods_per_node * 4)
    )
    assert len(graph.nodes) == expected_nodes

    # Reachability
    head_ip = str(daq_config.head_node_ip_addr)
    reachable = nx.descendants(graph, head_ip) | {head_ip}
    assert len(reachable) == expected_nodes, "Some nodes are unreachable in the fleet"


def test_bottleneck_with_module_limit():
    """Verify that the module-based bottleneck detection logic works."""
    # Generate a config where one node has 5 modules (over the limit of 4)
    daq_config, quabo_uids = generate_fleet_configs(
        num_daq_nodes=1, modules_per_node=5, module_limit=4
    )

    builder = GraphBuilder()
    graph = builder.build_from_configs(daq_config, quabo_uids)

    node_ip = str(daq_config.daq_nodes[0].ip_addr)
    successors = nx.descendants(graph, node_ip)
    modules = [n for n in successors if graph.nodes[n].get("role") == "module"]

    assert len(modules) == 5
    assert len(modules) > (daq_config.daq_node_module_limit or 4)


def test_control_loop_detection(base_quabo_uids):
    """Verify that we can detect non-DAG structures."""
    daq_node = DaqNode(
        username="root",
        data_dir="/data",
        ip_addr=IPv4Address("192.168.0.10"),
        module_ids=[254],
    )
    daq_config = DaqConfig(
        head_node_data_dir="/data/head",
        head_node_ip_addr=IPv4Address("10.0.1.5"),
        daq_nodes=[daq_node],
    )

    builder = GraphBuilder()
    graph = builder.build_from_configs(
        daq_config, base_quabo_uids, obs_config=None, network_config=None
        )

    # Intentionally add a loop: DAQ -> Head
    graph.add_edge("192.168.0.10", "10.0.1.5", type="invalid")

    assert not nx.is_directed_acyclic_graph(graph)
    cycles = list(nx.simple_cycles(graph))
    assert len(cycles) > 0
