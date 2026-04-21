"""
fleet.py

Utilities to programmatically generate n-node PANOSETI fleet configurations.
"""

from __future__ import annotations

from ipaddress import IPv4Address

from control.utils.pydantic_config_models import (
    DaqConfigValidator,
    DaqNodeValidator,
    QuaboUidDome,
    QuaboUidEntry,
    QuaboUidModule,
    QuaboUidsValidator,
)


import random
from control.utils.pydantic_config_models import (
    DaqConfigValidator,
    DaqNodeValidator,
    QuaboUidsValidator,
    QuaboUidDome,
    QuaboUidModule,
    QuaboUidEntry,
    PortForwarding,
)


def generate_fleet_configs(
    num_daq_nodes: int, 
    modules_per_node: int = 1,
    subnet_probability: float = 0.5,
    head_node_ip: str = "10.0.1.5",
    daq_base_ip: str = "192.168.0.100",
    module_base_ip: str = "192.168.3.10",
    module_limit: int = 4
) -> tuple[DaqConfigValidator, QuaboUidsValidator]:
    """
    Programmatically creates a set of configurations for an n-node fleet.

    Args:
        num_daq_nodes: Number of DAQ nodes in the fleet.
        modules_per_node: How many modules each node should manage.
        subnet_probability: Probability (0-1) that a node is in a subnet (requires port forwarding).
        head_node_ip: IP address of the head node.
        daq_base_ip: Base IP for DAQ nodes (increments from here).
        module_base_ip: Base IP for modules (increments from here).
        module_limit: Bottleneck threshold for the structural validator.
    """
    daq_nodes = []
    all_modules = []

    head_ip_obj = IPv4Address(head_node_ip)
    daq_start = int(IPv4Address(daq_base_ip))
    module_start = int(IPv4Address(module_base_ip))

    current_module_id = 1

    for i in range(num_daq_nodes):
        node_ip = str(IPv4Address(daq_start + i))

        # Decide if this node is in a subnet
        pf = None
        if random.random() < subnet_probability:
            # Create a gateway IP in a different range
            gw_ip = f"10.0.2.{100 + i}"
            pf = PortForwarding(
                status=True,
                gw_ip=IPv4Address(gw_ip),
                grpc_port=50051 + i
            )

        managed_module_ids = []
        for _j in range(modules_per_node):
            mod_ip = str(IPv4Address(module_start + (current_module_id - 1) * 4))

            module = QuaboUidModule(
                ip_addr=IPv4Address(mod_ip),
                quabos=[QuaboUidEntry(uid=f"q_{current_module_id}_{k}") for k in range(4)],
                id=current_module_id
            )
            all_modules.append(module)
            managed_module_ids.append(current_module_id)
            current_module_id += 1

        daq_node = DaqNodeValidator(
            username="panoseti",
            data_dir="/data",
            ip_addr=IPv4Address(node_ip),
            module_ids=managed_module_ids,
            bindhost="0.0.0.0",
            port_forwarding=pf
        )
        daq_nodes.append(daq_node)

    daq_config = DaqConfigValidator(
        head_node_data_dir="/data/head",
        head_node_ip_addr=head_ip_obj,
        head_node_container=False,
        daq_node_module_limit=module_limit,
        daq_nodes=daq_nodes
    )

    quabo_uids = QuaboUidsValidator(
        domes=[QuaboUidDome(num=0, modules=all_modules)]
    )

    return daq_config, quabo_uids

