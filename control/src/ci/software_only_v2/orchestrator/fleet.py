"""
orchestrator/fleet.py — Fleet: FleetSpec → live containers + typed handles.

The Fleet translates a validated Topology into a set of running testcontainers,
discovers their dynamic host ports, and patches the DaqConfig with PortForwarding
blocks so every gRPC client can find each node via the Boot-and-Discover pattern.

Usage::

    with Fleet.from_topology(topology, workspace) as fleet:
        fleet.wait_healthy()
        client = fleet.daq_control_client(0)
        ...
"""

from __future__ import annotations

import os
import pathlib
import shutil
import tempfile
import uuid
from ipaddress import IPv4Address
from typing import TYPE_CHECKING, Any

from ci.software_only_v2.containers.daqnode_sim import DaqNodeSimContainer
from ci.software_only_v2.containers.headnode import HeadnodeContainer
from ci.software_only_v2.orchestrator.lifecycle import (
    start_all,
    tear_down_all,
    wait_all_healthy,
)
from ci.software_only_v2.orchestrator.network import (
    SharedNetwork,
    placeholder_subnet,
    setup_docker_host,
)
from control.utils.pydantic_config_models import DaqConfig, DaqNode, PortForwarding

if TYPE_CHECKING:
    from ci.software_only_v2.fixtures.chaos import Chaos
    from ci.software_only_v2.infra.spec import Topology
    from ci.software_only_v2.infra.workspace import Workspace

_PLACEHOLDER_OFFSET = 10
_HEADNODE_IP = "10.0.1.5"


class Fleet:
    """Manages live testcontainers for a v2 test session.

    Attributes:
        topology:       The validated Topology that was used to build this fleet.
        headnode:       The HeadnodeContainer instance.
        daq_nodes:      Ordered list of DaqNodeSimContainer instances.
        live_daq_config: A patched DaqConfig with real host IPs + mapped ports
                         injected into each PortForwarding block after start().
    """

    def __init__(
        self,
        topology: "Topology",
        workspace: "Workspace",
        *,
        headnode_command: str = "sleep infinity",
        healthcheck_timeout: float = 90.0,
    ) -> None:
        self.topology = topology
        self.workspace = workspace
        self._headnode_command = headnode_command
        self._healthcheck_timeout = healthcheck_timeout

        self._network: SharedNetwork | None = None
        self._headnode_container: HeadnodeContainer | None = None
        self._daqnode_containers: list[DaqNodeSimContainer] = []
        self._temp_dirs: list[str] = []
        self.live_daq_config: DaqConfig | None = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_topology(
        cls,
        topology: "Topology",
        workspace: "Workspace",
        **kwargs: Any,
    ) -> "Fleet":
        """Build a Fleet from a Topology + Workspace without starting containers."""
        return cls(topology, workspace, **kwargs)

    # ------------------------------------------------------------------
    # Typed handles
    # ------------------------------------------------------------------

    @property
    def headnode(self) -> HeadnodeContainer:
        if self._headnode_container is None:
            raise RuntimeError("Fleet not started — call fleet.start() first")
        return self._headnode_container

    @property
    def daq_nodes(self) -> list[DaqNodeSimContainer]:
        return list(self._daqnode_containers)

    @property
    def n_nodes(self) -> int:
        return len(self._daqnode_containers)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> "Fleet":
        """Create and start all containers; patch DaqConfig with live ports."""
        setup_docker_host()
        tc_id = os.environ.get("TC_SESSION_ID", "solo")
        subnet = placeholder_subnet()

        # 1. Shared Docker network
        self._network = SharedNetwork(f"pseti-v2-{tc_id}")
        self._network.create()

        # Docker Network object (not just the name string) — testcontainers
        # requires a network SDK object with a .name attribute.
        docker_network = self._network._network
        if docker_network is None:
            raise RuntimeError(
                f"Failed to create Docker network '{self._network.name}'. "
                "Is the Docker daemon running?"
            )

        # 2. HeadnodeContainer
        self._headnode_container = HeadnodeContainer(
            name=f"pseti-v2-headnode-{tc_id}",
            command=self._headnode_command,
            config_dir=self.workspace.config_dir,
            state_dir=self.workspace.root / "state",
            network=docker_network,
        )

        # 3. DaqNodeSimContainers — one per DaqNode in the topology
        daq_nodes_spec = self.topology.daq.daq_nodes
        for i, node_spec in enumerate(daq_nodes_spec):
            data_dir = tempfile.mkdtemp(
                prefix=f"pseti_daq_{uuid.uuid4().hex[:8]}_"
            )
            os.chmod(data_dir, 0o777)
            self._temp_dirs.append(data_dir)

            sim = DaqNodeSimContainer(
                name=f"pseti-v2-daqnode-{tc_id}-{i}",
                module_ids=list(node_spec.module_ids),
                headnode_ip=_HEADNODE_IP,
                network=docker_network,
            )
            sim._volume(data_dir, "/data", "rw")
            self._daqnode_containers.append(sim)

        # 4. Start all containers (headnode first, then daq nodes)
        all_containers: list[Any] = [self._headnode_container] + self._daqnode_containers  # type: ignore[list-item]
        start_all(all_containers)  # type: ignore[arg-type]

        # 5. Patch DaqConfig with live ports (Boot-and-Discover)
        self.live_daq_config = self._build_live_daq_config(subnet)

        return self

    def wait_healthy(self, timeout: float | None = None) -> None:
        """Block until every DAQ node's gRPC channel reaches READY."""
        t = timeout or self._healthcheck_timeout
        # Only wait on daqnodes — headnode runs sleep infinity by default
        daqnode_containers: list[Any] = self._daqnode_containers  # type: ignore[list-item]
        wait_all_healthy(daqnode_containers, timeout=t)  # type: ignore[arg-type]

    def tear_down(self) -> None:
        """Stop all containers and clean up host-side temp directories."""
        all_containers: list[Any] = list(self._daqnode_containers)  # type: ignore[list-item]
        if self._headnode_container:
            all_containers.append(self._headnode_container)  # type: ignore[arg-type]
        tear_down_all(all_containers, temp_dirs=self._temp_dirs)  # type: ignore[arg-type]
        self._temp_dirs.clear()
        self._daqnode_containers.clear()
        self._headnode_container = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "Fleet":
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.tear_down()

    # ------------------------------------------------------------------
    # Live DaqConfig construction (Boot-and-Discover)
    # ------------------------------------------------------------------

    def _build_live_daq_config(self, subnet: str) -> DaqConfig:
        """Build a DaqConfig with real host IPs + mapped gRPC ports.

        The placeholder ip_addr (192.168.x.y) is kept for Pydantic validation.
        Clients route via port_forwarding.gw_ip + grpc_port.
        """
        live_nodes: list[DaqNode] = []
        for i, (sim, orig_node) in enumerate(
            zip(self._daqnode_containers, self.topology.daq.daq_nodes)
        ):
            pf = PortForwarding(
                status=True,
                gw_ip=IPv4Address(sim.grpc_host),
                port=2222,  # placeholder SSH port (>= 1024, not used)
                grpc_port=sim.grpc_port,
            )
            live_nodes.append(DaqNode(
                username=orig_node.username,
                data_dir=orig_node.data_dir,
                ip_addr=IPv4Address(f"{subnet}.{_PLACEHOLDER_OFFSET + i}"),
                module_ids=orig_node.module_ids,
                bindhost="lo",
                port_forwarding=pf,
            ))

        head_data = os.environ.get("HEAD_DATA_DIR", "/data/head")
        return DaqConfig(
            head_node_data_dir=head_data,
            head_node_ip_addr=IPv4Address(_HEADNODE_IP),
            head_node_container=True,
            daq_nodes=live_nodes,
        )

    # ------------------------------------------------------------------
    # Chaos accessor
    # ------------------------------------------------------------------

    @property
    def chaos(self) -> "Chaos":
        """Fault-injection accessor for this fleet's containers."""
        from ci.software_only_v2.fixtures.chaos import Chaos
        return Chaos(self)

    # ------------------------------------------------------------------
    # Client factories
    # ------------------------------------------------------------------

    def daq_control_client(self, node_index: int) -> Any:
        """Return a connected DaqControlClient for the given DAQ node index."""
        from panoseti_grpc.daq_control.client import DaqControlClient
        sim = self._daqnode_containers[node_index]
        return DaqControlClient(host=sim.grpc_host, port=sim.grpc_port)

    def daq_data_client(self, node_index: int) -> Any:
        """Return a connected DaqDataClient for the given DAQ node index."""
        from panoseti_grpc.daq_data.client import AioDaqDataClient
        sim = self._daqnode_containers[node_index]
        return AioDaqDataClient(host=sim.grpc_host, port=sim.grpc_port)
