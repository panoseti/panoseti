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

    def test_validate_all_rules_pass(self, two_node_workspace: Workspace) -> None:
        """GlobalConfigValidator passes on the configs materialized from the topology."""
        from control.utils import config_file as _cfm
        from control.utils.global_validator import GlobalConfigValidator
        validated = {
            "obs_config": _cfm.get_obs_config(),
            "daq_config": _cfm.get_daq_config(),
            "data_config": _cfm.get_data_config(),
        }
        validator = GlobalConfigValidator(validated)
        assert validator.validate_all_rules()

    def test_status_daq_returns_idle(self, two_node_workspace: Workspace) -> None:
        """StatusDaq on each sim daqnode returns success with hashpipe not running."""
        fleet = Fleet.from_topology(
            two_node_workspace.topology, two_node_workspace, healthcheck_timeout=120.0
        )
        with fleet:
            fleet.wait_healthy()
            for i in range(fleet.n_nodes):
                client = fleet.daq_control_client(i)
                ok, status = client.StatusDaq({"data_dir": "/data"})
                client.close()
                assert ok
                assert not status["hashpipe_running"]
                assert status["hashpipe_pid"] is None

    def test_n_nodes_matches_topology(self, two_node_workspace: Workspace) -> None:
        """fleet.n_nodes equals the topology's DAQ node count after start()."""
        fleet = Fleet.from_topology(two_node_workspace.topology, two_node_workspace)
        with fleet:
            assert fleet.n_nodes == len(two_node_workspace.topology.daq.daq_nodes)

    def test_generate_manifest_on_stub_run(self, two_node_workspace: Workspace) -> None:
        """GenerateManifest RPC on a stub run dir completes with success=True."""
        import docker as _docker
        fleet = Fleet.from_topology(
            two_node_workspace.topology, two_node_workspace, healthcheck_timeout=120.0
        )
        with fleet:
            fleet.wait_healthy()
            sim = fleet.daq_nodes[0]
            module_ids = list(two_node_workspace.topology.daq.daq_nodes[0].module_ids)
            _docker.from_env().containers.get(sim.name).exec_run(
                "bash -c 'mkdir -p /data/stub_run && chmod 777 /data/stub_run'"
            )
            client = fleet.daq_control_client(0)
            result = client.GenerateManifest({
                "data_dir": "/data",
                "run_dir": "stub_run",
                "module_id": [module_ids[0]],
            })
            client.close()
        assert result.get("success", False)

    def test_cleanup_data_full_succeeds(self, two_node_workspace: Workspace) -> None:
        """CleanupData CLEANUP_FULL on a stub run dir returns success."""
        import docker as _docker
        fleet = Fleet.from_topology(
            two_node_workspace.topology, two_node_workspace, healthcheck_timeout=120.0
        )
        with fleet:
            fleet.wait_healthy()
            sim = fleet.daq_nodes[0]
            module_ids = list(two_node_workspace.topology.daq.daq_nodes[0].module_ids)
            _docker.from_env().containers.get(sim.name).exec_run(
                "bash -c 'mkdir -p /data/stub_run_clean && chmod 777 /data/stub_run_clean'"
            )
            client = fleet.daq_control_client(0)
            result = client.CleanupData({
                "data_dir": "/data",
                "run_dir": "stub_run_clean",
                "module_id": [module_ids[0]],
                "mode": "CLEANUP_FULL",
                "force": True,
            })
            client.close()
        assert result.get("success", False)


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
