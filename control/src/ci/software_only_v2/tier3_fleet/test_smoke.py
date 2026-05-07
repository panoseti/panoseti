"""
tier3_fleet/test_smoke.py — Fleet smoke test.

Boots a 1-headnode + 2-daqnode fleet using DaqNodeSimContainer (UdsStrategy)
and asserts that:
  1. All containers reach gRPC READY within the healthcheck timeout.
  2. fleet.live_daq_config is patched with real host IPs + mapped ports.
  3. DaqControlClient connections succeed (gRPC channel opens).
  4. The fleet tears down cleanly without exceptions.

These tests require Docker.  Mark them tier3 and skip if Docker is unavailable.
"""

from __future__ import annotations

import socket

import pytest

from ci.software_only_v2.infra.spec import FleetSpec
from ci.software_only_v2.infra.workspace import Workspace
from ci.software_only_v2.orchestrator.fleet import Fleet


pytestmark = pytest.mark.tier3


def _docker_available() -> bool:
    """Return True if Docker daemon is reachable."""
    try:
        import docker  # type: ignore[import]
        docker.from_env(timeout=5).ping()
        return True
    except Exception:
        return False


requires_docker = pytest.mark.skipif(
    not _docker_available(),
    reason="Docker daemon not available",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def two_node_workspace(tmp_path_factory):
    """Module-scoped workspace for the two-node CI topology."""
    import importlib
    import os
    spec = FleetSpec.two_node_ci(tier="tier3")
    tmp_path = tmp_path_factory.mktemp("smoke_ws")

    env_dirs = [
        ("PSETI_CONFIG", "configs"),
        ("PSETI_STATE", "state"),
        ("PSETI_TMP", "tmp"),
        ("PSETI_LOGS", "state/logs"),
        ("PSETI_QUABOS", "quabos"),
        ("PSETI_FIRMWARE", "firmware"),
        ("HEAD_DATA_DIR", "head_data"),
        ("DAQ_DATA_DIR", "daq_data"),
    ]
    original: dict[str, str | None] = {}
    for key, sub in env_dirs:
        original[key] = os.environ.get(key)
        path = tmp_path / sub
        path.mkdir(parents=True, exist_ok=True)
        os.chmod(path, 0o777)
        os.environ[key] = str(path)

    from control.utils.paths import PanoPaths
    from ci.software_only_v2.infra.materialize import write_all
    from control.utils import config_file as _cfm
    from ci.software_only_v2.infra.workspace import StateProbe

    topology = spec.build()
    write_all(topology, PanoPaths.config_dir())
    PanoPaths.ensure_state_dirs()
    importlib.reload(_cfm)

    yield Workspace(root=tmp_path, topology=topology, state_probe=StateProbe())

    for key, orig in original.items():
        if orig is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = orig


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

@requires_docker
class TestFleetSmoke:
    """Boot a 2-daqnode fleet and verify health + gRPC connectivity."""

    def test_fleet_starts_and_is_healthy(self, two_node_workspace: Workspace) -> None:
        """Fleet boot: containers reach gRPC READY within timeout."""
        fleet = Fleet.from_topology(
            two_node_workspace.topology,
            two_node_workspace,
            healthcheck_timeout=120.0,
        )
        with fleet:
            fleet.wait_healthy()
            assert fleet.n_nodes == 2

    def test_live_daq_config_has_forwarded_ports(self, two_node_workspace: Workspace) -> None:
        """live_daq_config carries real host IPs + mapped gRPC ports after start()."""
        fleet = Fleet.from_topology(two_node_workspace.topology, two_node_workspace)
        with fleet:
            cfg = fleet.live_daq_config
            assert cfg is not None
            assert len(cfg.daq_nodes) == 2
            for node in cfg.daq_nodes:
                pf = node.port_forwarding
                assert pf is not None
                assert pf.status is True
                assert pf.grpc_port > 1024, "mapped port must be a real host port"
                assert str(pf.gw_ip) in ("127.0.0.1", "localhost", "0.0.0.0") or \
                    str(pf.gw_ip) != "", "gw_ip must be set"

    def test_grpc_port_is_tcp_reachable(self, two_node_workspace: Workspace) -> None:
        """Mapped gRPC ports are TCP-reachable on the host after start()."""
        fleet = Fleet.from_topology(two_node_workspace.topology, two_node_workspace)
        with fleet:
            for sim in fleet.daq_nodes:
                host, port = sim.grpc_host, sim.grpc_port
                with socket.create_connection((host, port), timeout=5.0) as sock:
                    assert sock is not None

    def test_daq_control_client_connects(self, two_node_workspace: Workspace) -> None:
        """DaqControlClient can be instantiated for each daqnode after start()."""
        fleet = Fleet.from_topology(two_node_workspace.topology, two_node_workspace)
        with fleet:
            fleet.wait_healthy()
            for i in range(fleet.n_nodes):
                client = fleet.daq_control_client(i)
                assert client is not None

    def test_headnode_container_is_running(self, two_node_workspace: Workspace) -> None:
        """Headnode container starts and exposes the headnode property."""
        fleet = Fleet.from_topology(two_node_workspace.topology, two_node_workspace)
        with fleet:
            hn = fleet.headnode
            assert hn is not None

    def test_fleet_teardown_is_clean(self, two_node_workspace: Workspace) -> None:
        """tear_down() does not raise; containers are removed after exit."""
        fleet = Fleet.from_topology(two_node_workspace.topology, two_node_workspace)
        fleet.start()
        fleet.tear_down()
        # After tear_down, container list is cleared
        assert fleet.n_nodes == 0


@requires_docker
class TestMinimalFleet:
    """Minimal 1-daqnode fleet — fast sanity check."""

    def test_minimal_fleet_healthy(self, tmp_path, monkeypatch) -> None:
        """Minimal single-node fleet reaches READY."""
        import importlib
        import os

        spec = FleetSpec.minimal_fleet()
        env_dirs = [
            ("PSETI_CONFIG", "configs"),
            ("PSETI_STATE", "state"),
            ("PSETI_TMP", "tmp"),
            ("PSETI_LOGS", "state/logs"),
            ("PSETI_QUABOS", "quabos"),
            ("PSETI_FIRMWARE", "firmware"),
            ("HEAD_DATA_DIR", "head_data"),
            ("DAQ_DATA_DIR", "daq_data"),
        ]
        for key, sub in env_dirs:
            path = tmp_path / sub
            path.mkdir(parents=True, exist_ok=True)
            os.chmod(path, 0o777)
            monkeypatch.setenv(key, str(path))

        from control.utils.paths import PanoPaths
        from ci.software_only_v2.infra.materialize import write_all
        from control.utils import config_file as _cfm
        from ci.software_only_v2.infra.workspace import StateProbe

        topology = spec.build()
        write_all(topology, PanoPaths.config_dir())
        PanoPaths.ensure_state_dirs()
        importlib.reload(_cfm)

        workspace = Workspace(root=tmp_path, topology=topology, state_probe=StateProbe())
        fleet = Fleet.from_topology(topology, workspace, healthcheck_timeout=120.0)
        with fleet:
            fleet.wait_healthy()
            assert fleet.n_nodes == 1
            assert fleet.live_daq_config is not None
