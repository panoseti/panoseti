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

from ci.software_only.infra.spec import FleetSpec
from ci.software_only.infra.workspace import Workspace
from ci.software_only.orchestrator.fleet import Fleet
from ci.software_only.tier3_fleet.conftest import requires_docker

pytestmark = pytest.mark.tier3


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

@requires_docker
@pytest.mark.parametrize(
    "pseti_workspace",
    [FleetSpec.two_node_ci(tier="tier3")],
    indirect=True,
)
class TestFleetSmoke:
    """Boot a 2-daqnode fleet and verify health + gRPC connectivity."""

    def test_fleet_starts_and_is_healthy(self, pseti_workspace: Workspace) -> None:
        """Fleet boot: containers reach gRPC READY within timeout."""
        fleet = Fleet.from_topology(
            pseti_workspace.topology,
            pseti_workspace,
            healthcheck_timeout=120.0,
        )
        with fleet:
            fleet.wait_healthy()
            assert fleet.n_nodes == 2
            from ci.software_only.infra.parity import run_scenario
            run_scenario("fleet_boot_and_healthy", fleet=fleet)

    def test_live_daq_config_has_forwarded_ports(self, pseti_workspace: Workspace) -> None:
        """live_daq_config carries real host IPs + mapped gRPC ports after start()."""
        fleet = Fleet.from_topology(pseti_workspace.topology, pseti_workspace)
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

    def test_grpc_port_is_tcp_reachable(self, pseti_workspace: Workspace) -> None:
        """Mapped gRPC ports are TCP-reachable on the host after start()."""
        fleet = Fleet.from_topology(pseti_workspace.topology, pseti_workspace)
        with fleet:
            for sim in fleet.daq_nodes:
                host, port = sim.grpc_host, sim.grpc_port
                with socket.create_connection((host, port), timeout=5.0) as sock:
                    assert sock is not None

    def test_daq_control_client_connects(self, pseti_workspace: Workspace) -> None:
        """DaqControlClient can be instantiated for each daqnode after start()."""
        fleet = Fleet.from_topology(pseti_workspace.topology, pseti_workspace)
        with fleet:
            fleet.wait_healthy()
            for i in range(fleet.n_nodes):
                client = fleet.daq_control_client(i)
                assert client is not None

    def test_headnode_container_is_running(self, pseti_workspace: Workspace) -> None:
        """Headnode container starts and exposes the headnode property."""
        fleet = Fleet.from_topology(pseti_workspace.topology, pseti_workspace)
        with fleet:
            hn = fleet.headnode
            assert hn is not None

    def test_fleet_teardown_is_clean(self, pseti_workspace: Workspace) -> None:
        """tear_down() does not raise; containers are removed after exit."""
        fleet = Fleet.from_topology(pseti_workspace.topology, pseti_workspace)
        fleet.start()
        fleet.tear_down()
        assert fleet.n_nodes == 0

    def test_status_daq_returns_idle(self, pseti_workspace: Workspace) -> None:
        """StatusDaq on each sim daqnode returns success with hashpipe not running."""
        fleet = Fleet.from_topology(
            pseti_workspace.topology, pseti_workspace, healthcheck_timeout=120.0
        )
        with fleet:
            fleet.wait_healthy()
            from ci.software_only.infra.parity import run_scenario
            for i in range(fleet.n_nodes):
                run_scenario("grpc_status_returns_idle", fleet=fleet, node_index=i)

    def test_generate_manifest_on_stub_run(self, pseti_workspace: Workspace) -> None:
        """GenerateManifest RPC on a stub run dir completes with success=True."""
        fleet = Fleet.from_topology(
            pseti_workspace.topology, pseti_workspace, healthcheck_timeout=120.0
        )
        with fleet:
            fleet.wait_healthy()
            module_ids = list(pseti_workspace.topology.daq.daq_nodes[0].module_ids)
            fleet.exec_in_node(0, "mkdir -p /data/stub_run && chmod 777 /data/stub_run")
            client = fleet.daq_control_client(0)
            result = client.GenerateManifest({
                "data_dir": "/data",
                "run_dir": "stub_run",
                "module_id": [module_ids[0]],
            })
            client.close()
        assert result.get("success", False)

    def test_cleanup_data_full_succeeds(self, pseti_workspace: Workspace) -> None:
        """CleanupData CLEANUP_FULL on a stub run dir returns success."""
        fleet = Fleet.from_topology(
            pseti_workspace.topology, pseti_workspace, healthcheck_timeout=120.0
        )
        with fleet:
            fleet.wait_healthy()
            module_ids = list(pseti_workspace.topology.daq.daq_nodes[0].module_ids)
            fleet.exec_in_node(0, "mkdir -p /data/stub_run_clean && chmod 777 /data/stub_run_clean")
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
@pytest.mark.parametrize(
    "pseti_workspace",
    [FleetSpec.minimal_fleet()],
    indirect=True,
)
class TestMinimalFleet:
    """Minimal 1-daqnode fleet — fast sanity check."""

    def test_minimal_fleet_healthy(self, pseti_workspace: Workspace) -> None:
        """Minimal single-node fleet reaches READY."""
        fleet = Fleet.from_topology(
            pseti_workspace.topology,
            pseti_workspace,
            healthcheck_timeout=120.0
        )
        with fleet:
            fleet.wait_healthy()
            assert fleet.n_nodes == 1
            assert fleet.live_daq_config is not None
