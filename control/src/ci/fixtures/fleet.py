"""
ci/fixtures/fleet.py

Dynamic N-daqnode fleet management for Tier 3 scaling tests using testcontainers.

Architecture: Boot-and-Discover pattern
  1. Build a theoretical Pydantic topology (placeholder IPs, no ports yet).
  2. Start testcontainers; each container gets a random host-mapped port.
  3. Query each container for its mapped port via get_exposed_port() and its
     reachable host IP via get_container_host_ip().  On macOS the host IP is
     127.0.0.1; inside a Docker-in-Docker runner it is the bridge gateway
     (e.g. 172.17.0.1) — never assume 127.0.0.1.
  4. Inject a PortForwarding block into each DaqNode:
       gw_ip  = container.get_container_host_ip()  (dynamic, not hardcoded)
       grpc_port = <mapped_port>
     The placeholder ip_addr field retains a stable internal IP so Pydantic's
     cross-field validators never fire.
  5. model_dump_json() the validated DaqConfig into the test's isolated
     PSETI_CONFIG directory.

The gRPC clients read port_forwarding.gw_ip + port_forwarding.grpc_port,
so they connect to <host_ip>:<mapped_port> without any hardcoded addresses.

Usage:
    @pytest.fixture(scope="session")
    def my_fleet() -> Iterator[Fleet]:
        fleet = make_fleet(n=2)
        fleet.start()
        fleet.wait_healthy()
        yield fleet
        fleet.tear_down()
"""

from __future__ import annotations

import os
import pathlib
import socket
import time
from dataclasses import dataclass
from ipaddress import IPv4Address
from typing import Any

from testcontainers.core.container import DockerContainer

from control.utils.pydantic_config_models import DaqConfig, DaqNode, PortForwarding


class SharedNetwork:
    """A testcontainers-compatible Network wrapper that uses a persistent backbone."""
    def __init__(self, name: str = "pseti-shared-net"):
        self.name = name
        self._network = None

    def create(self) -> None:
        import contextlib

        import docker
        client = docker.from_env()
        try:
            self._network = client.networks.get(self.name)
        except docker.errors.NotFound:
            with contextlib.suppress(Exception):
                self._network = client.networks.create(self.name, check_duplicate=True)

    def remove(self) -> None:
        # Persistence Fix: We never remove the shared backbone during tests.
        pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_DEFAULT_FLEET_N = 4
DAQNODE_SHM_BYTES = 2 * 1024**3   # 2 GB shm for hashpipe
DAQNODE_IMAGE = "pseti-daqnode:latest"
REDIS_IMAGE = "redis:alpine"
LOKI_IMAGE = "grafana/loki:latest"
GRPC_CONTAINER_PORT = 50051
GRPC_HEALTHCHECK_TIMEOUT = 90.0   # seconds to wait for server to accept TCP

# Placeholder subnet for daq-node ip_addr fields (not routed directly;
# routing always goes via port_forwarding.gw_ip = 127.0.0.1).
_PLACEHOLDER_SUBNET = "192.168.100"
_PLACEHOLDER_OFFSET = 10           # first node gets .10, second .11, …

# Head-node IP used in generated DaqConfig (the test-runner itself).
# Must NOT match any daq-node placeholder IP to avoid the Pydantic
# check_head_node_data_dir_match validator.
_HEADNODE_IP = "10.0.1.5"


# ---------------------------------------------------------------------------
# Docker host auto-detection
# ---------------------------------------------------------------------------

def setup_docker_host() -> None:
    """Configure DOCKER_HOST for the current platform if not already set.

    On macOS with Docker Desktop the socket lives at
    ~/.docker/run/docker.sock rather than the Linux default
    /var/run/docker.sock. testcontainers (and the Docker SDK) honour the
    DOCKER_HOST env-var, so we set it once before any container is created.
    """
    if os.environ.get("DOCKER_HOST"):
        return

    macos_socket = pathlib.Path.home() / ".docker" / "run" / "docker.sock"
    if macos_socket.exists():
        os.environ["DOCKER_HOST"] = f"unix://{macos_socket}"
        return

    # Standard Linux socket — leave DOCKER_HOST unset; SDK uses default.


# ---------------------------------------------------------------------------
# Spec dataclass
# ---------------------------------------------------------------------------

@dataclass
class DaqnodeSpec:
    """Describes one daqnode container before and after it is started."""
    name: str
    module_ids: list[int]
    grpc_container_port: int = GRPC_CONTAINER_PORT
    mapped_port: int | None = None          # filled by Fleet.start()
    container_host_ip: str = "127.0.0.1"   # filled by Fleet.start(); never assume loopback


def module_id_slice(node_index: int, n_nodes: int, total_modules: int = 4) -> list[int]:
    """Distribute module IDs [200 … 200+total) evenly across n_nodes."""
    per_node = max(1, total_modules // n_nodes)
    start = 200 + node_index * per_node
    return list(range(start, start + per_node))


# ---------------------------------------------------------------------------
# Fleet
# ---------------------------------------------------------------------------

class Fleet:
    """Manages a set of ephemeral daqnode containers via testcontainers.

    Lifecycle:
        fleet = make_fleet(n=2)
        fleet.start()          # creates containers, discovers mapped ports
        fleet.wait_healthy()   # blocks until gRPC servers accept TCP
        …
        fleet.tear_down()      # stops containers
    """

    def __init__(
        self,
        specs: list[DaqnodeSpec],
        *,
        shm_bytes: int = DAQNODE_SHM_BYTES,
        image: str = DAQNODE_IMAGE,
        headnode_ip: str = "10.0.1.22",
        headnode_grpc_port: int = GRPC_CONTAINER_PORT,
    ) -> None:
        self.specs = specs
        self.shm_bytes = shm_bytes
        self.image = image
        self.headnode_ip = headnode_ip
        self.headnode_grpc_port = headnode_grpc_port
        self._containers: list[DockerContainer] = []
        self._redis_container: DockerContainer | None = None
        self._loki_container: DockerContainer | None = None
        self._temp_dirs: list[str] = []
        # Use the persistent backbone wrapper
        self._network = SharedNetwork("pseti-shared-net")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def redis_port(self) -> int:
        if self._redis_container is None:
            return 6379
        return int(self._redis_container.get_exposed_port(6379))

    @property
    def loki_port(self) -> int:
        if self._loki_container is None:
            return 3100
        return int(self._loki_container.get_exposed_port(3100))

    @property
    def n_nodes(self) -> int:
        return len(self.specs)

    @property
    def containers(self) -> list[DockerContainer]:
        """The live testcontainers DockerContainer objects (one per spec)."""
        return self._containers

    def node_ip(self, index: int) -> str:
        """Placeholder IPv4 address assigned to the node at *index*.

        Matches the ip_addr injected into the DaqConfig by to_daq_config().
        Valid for Pydantic validation; not used for direct routing (routing
        always goes through port_forwarding.gw_ip).
        """
        return f"{_PLACEHOLDER_SUBNET}.{_PLACEHOLDER_OFFSET + index}"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start all daqnode containers.  Fills spec.mapped_port for each."""
        import tempfile
        import uuid
        setup_docker_host()
        # Shared Backbone Fix: Ensure the persistent network exists.
        self._network.create()
        
        # Start sidecars if telemetry enabled
        if os.getenv("ENABLE_TELEMETRY_TESTS") == "1":
            tc_id = os.environ.get("TC_SESSION_ID", "solo")
            
            self._redis_container = DockerContainer(REDIS_IMAGE)
            self._redis_container.with_name(f"pseti-redis-{tc_id}")
            self._redis_container.with_network(self._network)
            self._redis_container.with_exposed_ports(6379)
            self._redis_container.start()

            self._loki_container = DockerContainer(LOKI_IMAGE)
            self._loki_container.with_name(f"pseti-loki-{tc_id}")
            self._loki_container.with_network(self._network)
            self._loki_container.with_exposed_ports(3100)
            self._loki_container.start()

        # Support architectural fixes by mounting local gRPC source code
        # Path is relative to control/ (the CWD of the test runner)
        grpc_source = pathlib.Path("../grpc/src/panoseti_grpc").resolve()

        for spec in self.specs:
            container = DockerContainer(self.image)
            container.with_name(spec.name)
            container.with_network(self._network)
            
            # Isolate the Container Volumes
            # Every single container must get a unique physical path.
            unique_data_dir = tempfile.mkdtemp(prefix=f"daq_data_{uuid.uuid4().hex[:8]}_")
            # Recursive chmod 0o777 to ensure container root can write/delete
            os.chmod(unique_data_dir, 0o777)
            self._temp_dirs.append(unique_data_dir)
            container.with_volume_mapping(unique_data_dir, "/data", "rw")
            
            if grpc_source.exists():
                container.with_volume_mapping(str(grpc_source), "/grpc/src/panoseti_grpc", "rw")
                # Ensure the mounted source is preferred over pre-installed site-packages
            container.with_env("PYTHONPATH", "/grpc/src")
            # Force uninstallation of the site-packages version to ensure 
            # the mounted source is the only one available.
            container.with_command("sh -c 'pip uninstall -y panoseti-grpc && python -m panoseti_grpc.daq_control.server'")
            container.with_exposed_ports(spec.grpc_container_port)
            container.with_env("GRPC_PORT", str(spec.grpc_container_port))
            # gRPC log forwarding will fail gracefully when headnode is
            # unreachable; it must not prevent the server from starting.
            container.with_env("HEADNODE_IP", self.headnode_ip)
            container.with_env("HEADNODE_GRPC_PORT", str(self.headnode_grpc_port))
            container.with_kwargs(
                cap_add=["NET_RAW", "NET_ADMIN", "IPC_LOCK", "SYS_NICE"],
                shm_size=self.shm_bytes,
                hostname=spec.name,
            )
            container.start()
            spec.mapped_port = int(container.get_exposed_port(spec.grpc_container_port))
            # get_container_host_ip() is DinD-aware: returns the Docker host's
            # reachable IP (bridge gateway inside CI, 127.0.0.1 on macOS).
            raw_host = container.get_container_host_ip()
            try:
                spec.container_host_ip = socket.gethostbyname(raw_host)
            except socket.gaierror:
                spec.container_host_ip = raw_host
            self._containers.append(container)

    def wait_healthy(self, timeout: float = GRPC_HEALTHCHECK_TIMEOUT) -> None:
        """Block until every container's gRPC channel reaches READY state.

        Two-phase check:
          1. TCP connect — port is open (fast, no grpc import needed).
          2. grpc.channel_ready_future() — HTTP/2 handshake complete and the
             channel is in READY state, so the very first RPC won't race with
             channel negotiation.

        Does not import panoseti_grpc; only raw `grpc` is needed.
        """
        import grpc as _grpc

        deadline = time.monotonic() + timeout
        for spec in self.specs:
            if spec.mapped_port is None:
                raise RuntimeError(
                    f"spec {spec.name} has no mapped_port — did you call start()?"
                )
            addr = f"{spec.container_host_ip}:{spec.mapped_port}"

            # Phase 1: TCP — wait until the port is open.
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Container {spec.name!r} did not accept TCP on "
                        f"{addr} within {timeout}s"
                    )
                try:
                    with socket.create_connection(
                        (spec.container_host_ip, spec.mapped_port), timeout=min(2.0, remaining)
                    ):
                        break
                except OSError:
                    time.sleep(0.5)

            # Phase 2: gRPC channel READY — HTTP/2 handshake and server
            # initialisation complete.  Without this the first RPC (e.g.
            # DaqDataClient.ping() with a 0.3 s deadline) can arrive before
            # the channel is READY and time out even though TCP is up.
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Container {spec.name!r}: TCP up but gRPC channel did not "
                    f"reach READY on {addr} within {timeout}s"
                )
            channel = _grpc.insecure_channel(addr)
            try:
                _grpc.channel_ready_future(channel).result(timeout=remaining)
            except _grpc.FutureTimeoutError as exc:
                raise TimeoutError(
                    f"Container {spec.name!r}: gRPC channel on {addr} did not "
                    f"reach READY within {timeout}s"
                ) from exc
            finally:
                channel.close()

    def tear_down(self) -> None:
        """Stop all containers and remove the Docker network."""
        import contextlib
        all_containers = self._containers.copy()
        if self._redis_container:
            all_containers.append(self._redis_container)
        if self._loki_container:
            all_containers.append(self._loki_container)

        for container in all_containers:
            with contextlib.suppress(Exception):
                container.stop()
        
        # Cleanup isolated volumes
        import shutil
        for d in self._temp_dirs:
            with contextlib.suppress(Exception):
                shutil.rmtree(d)
        self._temp_dirs.clear()

        self._containers.clear()
        self._redis_container = None
        self._loki_container = None
        # We NO LONGER remove the network here as it's shared across xdist workers.
        # It should be pruned by the developer or a final 'cleanup' script.

    # ------------------------------------------------------------------
    # Topology helpers
    # ------------------------------------------------------------------

    def to_daq_config(self, head_node_ip: str = _HEADNODE_IP) -> DaqConfig:
        """Return a validated DaqConfig Pydantic model for this fleet.

        Each DaqNode gets a stable placeholder ip_addr in the 192.168.100.x
        subnet (not routed directly) and a PortForwarding block whose gw_ip
        is the dynamic container host IP (127.0.0.1 locally, bridge gateway
        in CI).  DaqDataClient and DaqControlClient read port_forwarding.gw_ip
        + grpc_port for the actual connection, so tests never hardcode any
        address.

        Raises:
            RuntimeError: if start() has not been called yet.
        """
        # Container perspective: data is always at /data
        # Mapped to host via volume mapping in start()
        daq_data_dir = "/data"
        head_data_dir = os.environ.get("HEAD_DATA_DIR", "/data/head")

        daq_nodes: list[DaqNode] = []
        for i, spec in enumerate(self.specs):
            if spec.mapped_port is None:
                raise RuntimeError(
                    f"spec {spec.name} has no mapped_port — did you call start()?"
                )
            pf = PortForwarding(
                status=True,
                gw_ip=IPv4Address(spec.container_host_ip),
                port=2222,  # Dummy SSH port (must be >= 1024) for build_rsync_cmd
                grpc_port=spec.mapped_port,
            )
            node = DaqNode(
                username="panoseti",
                data_dir=daq_data_dir,
                # Placeholder — never used for routing; isolates from head_node_ip
                # so Pydantic's check_head_node_data_dir_match stays silent.
                ip_addr=IPv4Address(f"{_PLACEHOLDER_SUBNET}.{_PLACEHOLDER_OFFSET + i}"),
                module_ids=spec.module_ids,
                bindhost="lo",
                port_forwarding=pf,
            )
            daq_nodes.append(node)

        return DaqConfig(
            head_node_data_dir=head_data_dir,
            head_node_ip_addr=IPv4Address(head_node_ip),
            head_node_container=True,
            daq_nodes=daq_nodes,
        )

    def write_daq_config(self, path: pathlib.Path, head_node_ip: str = _HEADNODE_IP) -> None:
        """Serialise the validated DaqConfig to *path* as JSON.

        Prefer to_daq_config() directly when the Pydantic model is needed.
        """
        daq_config = self.to_daq_config(head_node_ip)
        path.write_text(daq_config.model_dump_json(indent=2))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_fleet(n: int, **kwargs: Any) -> Fleet:
    """Build a Fleet of *n* daqnode specs with auto-assigned module IDs."""
    if n > MAX_DEFAULT_FLEET_N and not os.getenv("RUN_LARGE_FLEET"):
        raise ValueError(
            f"n={n} exceeds the default limit of {MAX_DEFAULT_FLEET_N}. "
            "Set RUN_LARGE_FLEET=1 to override."
        )
    
    # 409 Conflict mitigation: use TC_SESSION_ID (from xdist worker) to make names unique
    tc_id = os.environ.get("TC_SESSION_ID", "solo")
    
    specs = [
        DaqnodeSpec(
            name=f"pseti-daqnode-{tc_id}-{i}",
            module_ids=module_id_slice(i, n),
        )
        for i in range(n)
    ]
    return Fleet(specs, **kwargs)
