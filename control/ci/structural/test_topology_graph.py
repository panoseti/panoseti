"""
test_topology_graph.py

Unit tests for the NetworkX topology graph builder.
"""

from __future__ import annotations

from ipaddress import IPv4Address

import pytest

from control.topology.graph_builder import GraphBuilder
from control.utils.pydantic_config_models import (
    DaqConfigValidator,
    DaqNodeValidator,
    PortForwarding,
    QuaboUidDome,
    QuaboUidEntry,
    QuaboUidModule,
    QuaboUidsValidator,
)


@pytest.fixture
def mock_quabo_uids() -> QuaboUidsValidator:
    """Creates a basic Quabo UID configuration with one module (ID 254)."""
    module = QuaboUidModule(
        ip_addr=IPv4Address("192.168.3.248"),
        quabos=[
            QuaboUidEntry(uid="uid_0"),
            QuaboUidEntry(uid="uid_1"),
            QuaboUidEntry(uid="uid_2"),
            QuaboUidEntry(uid="uid_3"),
        ],
        id=254
    )
    dome = QuaboUidDome(num=0, modules=[module])
    return QuaboUidsValidator(domes=[dome])


def test_build_direct_topology(mock_quabo_uids: QuaboUidsValidator):
    """Verify graph structure for a direct-connection topology."""
    daq_node = DaqNodeValidator(
        username="root",
        data_dir="/data",
        ip_addr=IPv4Address("192.168.0.10"),
        module_ids=[254]
    )
    daq_config = DaqConfigValidator(
        head_node_data_dir="/data/head",
        head_node_ip_addr=IPv4Address("10.0.1.5"),
        daq_nodes=[daq_node]
    )

    # Note: associate() is needed to link the IDs correctly if derived from objects
    # but our builder currently uses node.module_ids and module.id
    
    builder = GraphBuilder()
    graph = builder.build_from_configs(daq_config, mock_quabo_uids)

    # Assertions
    assert "10.0.1.5" in graph.nodes, "Head node missing"
    assert "192.168.0.10" in graph.nodes, "DAQ node missing"
    
    # Check roles
    assert graph.nodes["10.0.1.5"]["role"] == "headnode"
    assert graph.nodes["192.168.0.10"]["role"] == "daqnode"

    # Check edges
    assert graph.has_edge("10.0.1.5", "192.168.0.10")
    assert graph.edges["10.0.1.5", "192.168.0.10"]["type"] == "control"

    # Check Module and Quabos
    module_node = next(n for n, d in graph.nodes(data=True) if d.get("role") == "module")
    assert "192.168.3.248" in module_node
    assert graph.has_edge("192.168.0.10", module_node)
    
    quabos = [n for n, d in graph.nodes(data=True) if d.get("role") == "quabo"]
    assert len(quabos) == 4
    for q in quabos:
        assert graph.has_edge(module_node, q)


def test_build_gateway_topology(mock_quabo_uids: QuaboUidsValidator):
    """Verify graph structure for a gateway/port-forwarding topology."""
    pf = PortForwarding(
        status=True,
        gw_ip=IPv4Address("10.0.1.10"),
        grpc_port=50051
    )
    daq_node = DaqNodeValidator(
        username="root",
        data_dir="/data",
        ip_addr=IPv4Address("192.168.0.10"),
        module_ids=[254],
        port_forwarding=pf
    )
    daq_config = DaqConfigValidator(
        head_node_data_dir="/data/head",
        head_node_ip_addr=IPv4Address("10.0.1.5"),
        daq_nodes=[daq_node]
    )

    builder = GraphBuilder()
    graph = builder.build_from_configs(daq_config, mock_quabo_uids)

    # Assertions
    assert "10.0.1.10" in graph.nodes, "Gateway node missing"
    assert graph.nodes["10.0.1.10"]["role"] == "gateway"

    # Check edge sequence: Head -> DAQ -> Gateway -> Module
    # Check edge sequence: Head -> Gateway --> DAQ -> Module
    assert graph.has_edge("10.0.1.5", "10.0.1.10") # Control path
    assert graph.has_edge("10.0.1.10", "192.168.0.10") # Network path (tunnel)
    
    module_node = next(n for n, d in graph.nodes(data=True) if d.get("role") == "module")
    assert graph.has_edge("192.168.0.10", module_node), "Module should be downstream of Gateway"
