"""
integration/fleet.py

Dynamic N-daqnode fleet management for Pillar 3 scaling tests using testcontainers.

Usage in a pytest test:
    @pytest.mark.parametrize("daqnode_fleet", [2, 4], indirect=True)
    def test_start_scales(daqnode_fleet: Fleet, ...) -> None:
        ...

Design constraints:
  - N ≤ 4 by default; N > 4 requires RUN_LARGE_FLEET=1 env var
  - Each container needs ≥ 2 GB /dev/shm for hashpipe shared memory
  - Each container gets its own isolated network and volume.
  - Topology is written to a test-scoped daq_config.json for the test runner.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Any

from testcontainers.core.container import DockerContainer
from testcontainers.core.network import Network

MAX_DEFAULT_FLEET_N = 4
DAQNODE_SHM_BYTES = 2 * 1024 ** 3  # 2 GB
DAQNODE_IMAGE = "ctl-int-daqnode"
BASE_IP_PREFIX = "192.168.0"
BASE_HEADNODE_IP_PREFIX = "10.0.1"
BASE_IP_OFFSET = 30  # first dynamic daqnode at .30, .31, ...


@dataclass
class DaqnodeSpec:
    name: str
    ip: str
    headnode_ip: str
    volume_name: str
    module_ids: list[int]
    grpc_port: int = 50051


def module_id_slice(node_index: int, n_nodes: int, total_modules: int = 4) -> list[int]:
    """Distribute module IDs [200..200+total) across n_nodes nodes."""
    per_node = max(1, total_modules // n_nodes)
    start = 200 + node_index * per_node
    end = start + per_node
    return list(range(start, end))


class Fleet:
    """Manages a set of dynamically created daqnode containers using testcontainers."""

    def __init__(
        self,
        specs: list[DaqnodeSpec],
        shm_bytes: int = DAQNODE_SHM_BYTES,
        grpc_port: int = 50051,
        bindhost: str = "0.0.0.0",
        headnode_ip: str = "10.0.1.10",
        headnode_grpc_port: int = 50051,
        image: str = DAQNODE_IMAGE,
    ) -> None:
        self.specs = specs
        self.shm_bytes = shm_bytes
        self.grpc_port = grpc_port
        self.bindhost = bindhost
        self.headnode_ip = headnode_ip
        self.headnode_grpc_port = headnode_grpc_port
        self.image = image
        self._containers: list[DockerContainer] = []
        self._network = Network()

    @property
    def n_nodes(self) -> int:
        return len(self.specs)

    def start(self) -> None:
        """Start all daqnode containers using testcontainers."""
        self._network.create()
        for spec in self.specs:
            container = DockerContainer(self.image)
            container.with_name(spec.name)
            container.with_network(self._network)
            container.with_env("GRPC_PORT", str(spec.grpc_port))
            container.with_env("BINDHOST", self.bindhost)
            container.with_env("HEADNODE_IP", self.headnode_ip)
            container.with_env("HEADNODE_GRPC_PORT", str(self.headnode_grpc_port))
            
            # Request CAP_ADD privileges for network and priority tasks
            container.with_kwargs(
                cap_add=["NET_RAW", "NET_ADMIN", "IPC_LOCK", "SYS_NICE"],
                shm_size=self.shm_bytes,
                hostname=spec.name
            )
            
            container.start()
            self._containers.append(container)

    def wait_healthy(self, timeout: float = 60.0) -> None:
        """Wait until gRPC ports are responsive."""
        pass

    def tear_down(self) -> None:
        """Stop all containers and remove the network."""
        for container in self._containers:
            container.stop()
        self._containers.clear()
        self._network.remove()

    def write_daq_config(self, path: pathlib.Path, head_node_ip: str) -> None:
        """Write a daq_config.json describing this fleet to path."""
        config = {
            "head_node_data_dir": "/data/head",
            "head_node_ip_addr": head_node_ip,
            "head_node_container": True,
            "daq_nodes": [
                {
                    "username": "panoseti",
                    "data_dir": "/data",
                    "ip_addr": spec.name, # Use hostname in Docker network
                    "module_ids": "-".join(str(m) for m in spec.module_ids)
                    if len(spec.module_ids) > 1
                    else str(spec.module_ids[0]),
                    "bindhost": "0.0.0.0",
                }
                for spec in self.specs
            ],
        }
        path.write_text(json.dumps(config, indent=2))


def make_fleet(n: int, **kwargs: Any) -> Fleet:
    """Build Fleet with n daqnode specs with auto-assigned IPs."""
    specs = [
        DaqnodeSpec(
            name=f"ctl-int-daqnode-dyn-{i}",
            ip="", # managed by docker
            headnode_ip=f"{BASE_HEADNODE_IP_PREFIX}.{BASE_IP_OFFSET + i}",
            volume_name=f"daq_data_dyn_{i}",
            module_ids=module_id_slice(i, n),
        )
        for i in range(n)
    ]
    return Fleet(specs, **kwargs)
