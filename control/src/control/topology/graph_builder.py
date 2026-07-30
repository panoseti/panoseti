"""
graph_builder.py

NetworkX-based topology generator for the PANOSETI DAQ fleet.
"""

from __future__ import annotations

from typing import Any

import networkx as nx

from control.utils import config_file, util
from control.utils.pydantic_config_models import (
    DaqConfig,
    NetworkConfig,
    ObsConfig,
    QuaboUids,
)


class GraphBuilder:
    """
    Constructs a NetworkX representation of the observatory hardware and network topology.
    """

    def __init__(self) -> None:
        self.graph = nx.DiGraph()

    def build_from_configs(
        self,
        daq_config: DaqConfig,
        quabo_uids: QuaboUids,
        obs_config: ObsConfig | None = None,
        network_config: NetworkConfig | None = None,
    ) -> nx.DiGraph:
        """
        Builds the graph from Pydantic configuration models.
        """
        self.graph.clear()
        if network_config is not None:
            util.attach_daq_config(daq_config, network_config)

        # 1. Add Head Node
        head_ip = str(daq_config.head_node_ip_addr)
        self.graph.add_node(
            head_ip,
            role="headnode",
            ip=head_ip,
            layer=0,
            data_dir=daq_config.head_node_data_dir,
            label=f"Head Node\n({head_ip})"
        )

        # 2. Add DAQ Nodes and Gateways
        for node in daq_config.daq_nodes:
            daq_ip = str(node.ip_addr)
            
            # Add DAQ Node
            self.graph.add_node(
                daq_ip,
                role="daqnode",
                ip=daq_ip,
                layer=2,
                username=node.username,
                data_dir=node.data_dir,
                label=f"DAQ Node\n({daq_ip})"
            )
            
            # Control Path Edge
            upstream_ips = {
                'daq': daq_ip
            }

            # Check for Gateway (via Port Forwarding)
            if node.port_forwarding and node.port_forwarding.status:
                gw_ip = str(node.port_forwarding.gw_ip)
                self.graph.add_node(
                    gw_ip,
                    role="gateway",
                    ip=gw_ip,
                    layer=1,
                    label=f"Gateway\n({gw_ip})"
                )

                # Headnode -> Gateway
                self.graph.add_edge(head_ip, gw_ip, type="network", label="")
                # Gateway -> Daqnode
                self.graph.add_edge(gw_ip, daq_ip, type="network", label="")
                upstream_ips['control'] = gw_ip
            else:
                # Headnode -> Daqnode
                self.graph.add_edge(head_ip, daq_ip, type="network", label="")
                upstream_ips['control'] = head_ip
                

            # 3. Add Quabos (linked to this DAQ node)
            # We need to find which modules are assigned to this node.
            # This information is often injected by config_file.associate() 
            # but we can also derive it from module_ids in DaqNode.
            
            # Find modules in quabo_uids that match this node's module_ids
            for dome in quabo_uids.domes:
                for module in dome.modules:
                    # module.id is injected at runtime, but we can check if it exists
                    # or derive it. For now, we assume associate() might have been called
                    # or we match by ID.
                    mid = getattr(module, 'id', None)
                    if mid is not None and mid in node.module_ids:
                        self._add_module_to_graph(module, dome.num, upstream_ips)
        
        return self.graph

    def _add_module_to_graph(self, module: Any, dome_num: int | None, upstream_ips: dict[str, str]) -> None:
        """Helper to add module and its 4 quabos to the graph."""
        module_ip = str(module.ip_addr)
        module_id = getattr(module, 'id', 'unknown')
        
        module_node_id = f"mod_{module_id}_{module_ip}"
        self.graph.add_node(
            module_node_id,
            role="module",
            ip=module_ip,
            module_id=module_id,
            dome=dome_num,
            layer=3,
            label=f"Module {module_id}\n({module_ip})"
        )
        
        # Edge from Upstream (DAQ Node or Gateway) to Module
        for node_type, ip in upstream_ips.items():
            if node_type == 'daq':
                self.graph.add_edge(ip, module_node_id, type="data", label="UDP")
            elif node_type == 'control':
                self.graph.add_edge(ip, module_node_id, type="control", label="UDP")

        # Add 4 Quabos
        for i, q_entry in enumerate(module.quabos):
            q_uid = q_entry.uid[:7] if q_entry.uid else f"{module_id}_{i}"
            # Quabo IP is derived from module base IP
            q_ip = config_file.quabo_ip_addr(module_ip, i)
            
            q_node_id = f"q_{q_uid}"
            self.graph.add_node(
                q_node_id,
                role="quabo",
                ip=q_ip,
                uid=q_uid,
                index=i,
                layer=4,
                label=f"Q{i}\n({q_uid})"
            )
            self.graph.add_edge(module_node_id, q_node_id, type="logical")
