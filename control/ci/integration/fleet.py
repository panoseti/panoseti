"""
integration/fleet.py

Dynamic N-daqnode fleet management for Pillar 3 scaling tests.

Usage in a pytest test:
    @pytest.mark.parametrize("daqnode_fleet", [2, 4], indirect=True)
    def test_start_scales(daqnode_fleet: Fleet, ...):
        ...

Design constraints:
  - N ≤ 4 by default; N > 4 requires RUN_LARGE_FLEET=1 env var
  - Each container needs ≥ 2 GB /dev/shm for hashpipe shared memory
  - Each container gets its own Docker volume (no module.config race)
  - Topology is written to a test-scoped daq_config.json for the test runner
"""

from __future__ import annotations

import json
import os
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any

MAX_DEFAULT_FLEET_N = 4
DAQNODE_SHM_BYTES = 2 * 1024 ** 3  # 2 GB
DAQNODE_IMAGE = "integration-daqnode"
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
    """Manages a set of dynamically created daqnode containers."""

    def __init__(
        self,
        docker_client: Any,
        specs: list[DaqnodeSpec],
        shm_bytes: int = DAQNODE_SHM_BYTES,
        grpc_port: int = 50051,
        bindhost: str = "lo",
        headnode_ip: str = "10.0.1.10",
        headnode_grpc_port: int = 50051,
        image: str = DAQNODE_IMAGE,
    ) -> None:
        self.client = docker_client
        self.specs = specs
        self.shm_bytes = shm_bytes
        self.grpc_port = grpc_port
        self.bindhost = bindhost
        self.headnode_ip = headnode_ip
        self.headnode_grpc_port = headnode_grpc_port
        self.image = image
        self._containers: list[Any] = []
        self._volumes: list[Any] = []

    @property
    def n_nodes(self) -> int:
        return len(self.specs)

    def start(self) -> None:
        """Start all daqnode containers."""
        for spec in self.specs:
            vol = self._ensure_volume(spec.volume_name)
            self._volumes.append(vol)
            container = self.client.containers.run(
                self.image,
                name=spec.name,
                detach=True,
                remove=True,
                shm_size=self.shm_bytes,
                cap_add=["NET_RAW", "NET_ADMIN", "IPC_LOCK", "SYS_NICE"],
                volumes={
                    spec.volume_name: {"bind": "/data", "mode": "rw"},
                },
                environment={
                    "GRPC_PORT": str(spec.grpc_port),
                    "BINDHOST": self.bindhost,
                    "HEADNODE_IP": self.headnode_ip,
                    "HEADNODE_GRPC_PORT": str(self.headnode_grpc_port),
                },
                network_mode="host",  # simplification for dynamic fleet
            )
            self._containers.append(container)

    def _ensure_volume(self, name: str) -> Any:
        try:
            return self.client.volumes.get(name)
        except Exception:
            return self.client.volumes.create(name)

    def wait_healthy(self, timeout: float = 60.0) -> None:
        """Poll each container until gRPC port is open or timeout."""
        import socket as _socket
        deadline = time.monotonic() + timeout
        for spec in self.specs:
            while time.monotonic() < deadline:
                try:
                    s = _socket.create_connection((spec.ip, spec.grpc_port), timeout=1.0)
                    s.close()
                    break
                except OSError:
                    time.sleep(0.5)

    def verify_shm(self) -> None:
        """Fail fast if any container has insufficient /dev/shm."""
        for i, container in enumerate(self._containers):
            result = container.exec_run("df -B1 /dev/shm | tail -1 | awk '{print $2}'")
            avail = int((result.output or b"0").decode().strip() or "0")
            if avail < self.shm_bytes:
                spec = self.specs[i]
                raise RuntimeError(
                    f"Container {spec.name} has {avail} bytes of /dev/shm; "
                    f"need {self.shm_bytes}. Pass shm_size correctly."
                )

    def stop_and_remove(self) -> None:
        """Stop and remove all managed containers (best-effort)."""
        for container in self._containers:
            try:
                container.stop(timeout=5)
            except Exception:
                pass
        self._containers.clear()

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
                    "ip_addr": spec.ip,
                    "module_ids": "-".join(str(m) for m in spec.module_ids)
                    if len(spec.module_ids) > 1
                    else str(spec.module_ids[0]),
                    "bindhost": "0.0.0.0",
                }
                for spec in self.specs
            ],
        }
        path.write_text(json.dumps(config, indent=2))


def make_fleet(docker_client: Any, n: int, **kwargs: Any) -> Fleet:
    """Build Fleet with n daqnode specs with auto-assigned IPs and volumes."""
    specs = [
        DaqnodeSpec(
            name=f"ctl-int-daqnode-dyn-{i}",
            ip=f"{BASE_IP_PREFIX}.{BASE_IP_OFFSET + i}",
            headnode_ip=f"{BASE_HEADNODE_IP_PREFIX}.{BASE_IP_OFFSET + i}",
            volume_name=f"daq_data_dyn_{i}",
            module_ids=module_id_slice(i, n),
        )
        for i in range(n)
    ]
    return Fleet(docker_client, specs, **kwargs)
