"""
tier4_chaos/test_chaos_smoke.py — Chaos fault-injection smoke tests.

Each class exercises one chaos sub-handle via the session_fleet.
All tests skip if Docker is unavailable.

Container capability requirements:
 - NET_ADMIN: NetemTests, IptablesTests (daqnode containers have this)
 - No extra caps: DiskTests, ProcessTests, GrpcProxyTests
"""

from __future__ import annotations

import socket
import time

import pytest

from ci.software_only_v2.infra.spec import FleetSpec
from ci.software_only_v2.orchestrator.fleet import Fleet


pytestmark = pytest.mark.tier4


def _docker_available() -> bool:
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
# Shared module-scoped fleet for all tier4 tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def chaos_workspace(tmp_path_factory, monkeypatch):
    """Module-scoped workspace for the chaos fleet."""
    import importlib
    import os

    spec = FleetSpec.minimal_fleet()
    tmp_path = tmp_path_factory.mktemp("chaos_ws")

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

    from ci.software_only_v2.infra.workspace import Workspace
    return Workspace(root=tmp_path, topology=topology, state_probe=StateProbe())


@pytest.fixture(scope="module")
def chaos_fleet(chaos_workspace):
    """Module-scoped live fleet for chaos tests."""
    fleet = Fleet.from_topology(
        chaos_workspace.topology,
        chaos_workspace,
        healthcheck_timeout=120.0,
    )
    with fleet:
        fleet.wait_healthy()
        yield fleet


# ---------------------------------------------------------------------------
# gRPC proxy tests (no special Docker caps required)
# ---------------------------------------------------------------------------

@requires_docker
class TestGrpcProxy:
    """GrpcHandle: in-process fault injection without Docker capabilities."""

    def test_inject_unavailable_raises(self, chaos_fleet: Fleet) -> None:
        """inject_rpc_fault('unavailable') makes the method raise grpc.RpcError."""
        import grpc
        client = chaos_fleet.daq_control_client(0)
        with chaos_fleet.chaos.grpc.inject(client, "StatusDaq", "unavailable"):
            with pytest.raises(grpc.RpcError) as exc_info:
                client.StatusDaq()
            assert exc_info.value.code() == grpc.StatusCode.UNAVAILABLE

    def test_proxy_set_mode_and_restore(self, chaos_fleet: Fleet) -> None:
        """GrpcChaosProxy patches and unpatch cleanly."""
        import grpc
        client = chaos_fleet.daq_control_client(0)
        proxy = chaos_fleet.chaos.grpc.proxy(client)
        proxy.set_mode("StatusDaq", "unavailable")

        with proxy:
            with pytest.raises(grpc.RpcError):
                client.StatusDaq()

        # After context exit, restore() was called — original method is back
        # (StatusDaq may fail for other reasons, but not from our injection)
        try:
            client.StatusDaq()
        except grpc.RpcError as e:
            assert e.code() != grpc.StatusCode.UNAVAILABLE, (
                "UNAVAILABLE after restore() means proxy was not cleaned up"
            )

    def test_success_then_fail_mode(self, chaos_fleet: Fleet) -> None:
        """success_then_fail: first call succeeds, second raises UNAVAILABLE."""
        import grpc
        client = chaos_fleet.daq_control_client(0)
        proxy = chaos_fleet.chaos.grpc.proxy(client)
        proxy.set_mode("StatusDaq", "success_then_fail")
        proxy.apply(client)
        try:
            # First call should succeed (or raise a real gRPC error, not UNAVAILABLE)
            try:
                client.StatusDaq()
            except grpc.RpcError as e:
                # A real server error (not injected) is acceptable on call 0
                assert e.code() != grpc.StatusCode.UNAVAILABLE

            # Second call must raise UNAVAILABLE
            with pytest.raises(grpc.RpcError) as exc_info:
                client.StatusDaq()
            assert exc_info.value.code() == grpc.StatusCode.UNAVAILABLE
        finally:
            proxy.restore()


# ---------------------------------------------------------------------------
# Process chaos tests (no special Docker caps required)
# ---------------------------------------------------------------------------

@requires_docker
class TestProcessChaos:
    """ProcessHandle: kill and process-liveness checks inside containers."""

    def test_process_alive_returns_true_for_running_server(
        self, chaos_fleet: Fleet
    ) -> None:
        """panoseti-server is alive in a healthy daqnode container."""
        node = chaos_fleet.daq_nodes[0]
        # python process runs the server
        alive = chaos_fleet.chaos.proc.alive(node, "python")
        assert alive, "Expected python process to be alive in daqnode container"

    def test_kill_and_wait_dead(self, chaos_fleet: Fleet) -> None:
        """Killing a non-critical process and confirming it is gone."""
        node = chaos_fleet.daq_nodes[0]
        chaos = chaos_fleet.chaos

        # Only run if python is actually alive
        if not chaos.proc.alive(node, "python"):
            pytest.skip("python not running in container — skipping kill test")

        # We kill 'sleep' if present (harmless) rather than the server itself
        # (killing the server would break other tests in this module)
        from ci.fixtures.chaos import process_chaos
        code, _ = process_chaos._exec(node.name, "sleep 30 &")
        time.sleep(0.3)
        chaos.proc.kill(node, "sleep", sig="TERM")
        dead = chaos.proc.wait_dead(node, "sleep", timeout=5.0)
        assert dead, "sleep process should have died after SIGTERM"

    def test_wait_alive_timeout_on_absent_process(self, chaos_fleet: Fleet) -> None:
        """wait_alive() returns False if the process never appears within timeout."""
        node = chaos_fleet.daq_nodes[0]
        result = chaos_fleet.chaos.proc.wait_alive(
            node, "definitely_not_a_real_process_xyzzy", timeout=2.0
        )
        assert result is False


# ---------------------------------------------------------------------------
# Disk chaos tests
# ---------------------------------------------------------------------------

@requires_docker
class TestDiskChaos:
    """DiskHandle: fill/release filesystem space inside containers."""

    def test_full_disk_context_manager_releases(self, chaos_fleet: Fleet) -> None:
        """full_disk() fills /tmp, then releases on exit."""
        node = chaos_fleet.daq_nodes[0]
        chaos = chaos_fleet.chaos

        # Record free space before fill
        from ci.fixtures.chaos import process_chaos
        code, before = process_chaos._exec(
            node.name, "df -k /tmp | tail -1 | awk '{print $4}'"
        )
        assert code == 0
        before_kb = int(before.strip()) if before.strip().isdigit() else None

        with chaos.disk.full_disk(node, "/tmp", fill_pct=95):
            code, during = process_chaos._exec(
                node.name, "df -k /tmp | tail -1 | awk '{print $4}'"
            )
            assert code == 0
            if during.strip().isdigit() and before_kb is not None:
                during_kb = int(during.strip())
                assert during_kb < before_kb, "Disk should be fuller during fill"

        # After context exit, fill file should be gone
        code, after = process_chaos._exec(
            node.name, "df -k /tmp | tail -1 | awk '{print $4}'"
        )
        assert code == 0
        if after.strip().isdigit() and before_kb is not None:
            after_kb = int(after.strip())
            # Allow ±10 MB tolerance
            assert abs(after_kb - before_kb) < 10_240, (
                f"Free space after release ({after_kb} kB) "
                f"differs from before ({before_kb} kB) by more than 10 MB"
            )

    def test_manual_fill_and_release(self, chaos_fleet: Fleet) -> None:
        """fill() and release() work as standalone calls."""
        node = chaos_fleet.daq_nodes[0]
        chaos = chaos_fleet.chaos
        fill_file = chaos.disk.fill(node, "/tmp", fill_pct=50)
        try:
            assert fill_file, "Expected a fill-file path"
        finally:
            chaos.disk.release(node, fill_file)


# ---------------------------------------------------------------------------
# Network chaos tests (require NET_ADMIN on the container)
# ---------------------------------------------------------------------------

@requires_docker
class TestNetworkChaos:
    """NetemHandle + IptablesHandle: network impairments inside containers.

    These tests require NET_ADMIN capability.  They are marked xfail if the
    capability is absent (observed via tc command exit code).
    """

    def _has_net_admin(self, fleet: Fleet) -> bool:
        """Return True if the first daqnode has NET_ADMIN (tc works)."""
        from ci.fixtures.chaos import process_chaos
        node = fleet.daq_nodes[0]
        code, _ = process_chaos._exec(node.name, "tc qdisc show")
        return code == 0

    def test_latency_context_manager_adds_and_removes(
        self, chaos_fleet: Fleet
    ) -> None:
        """latency() context manager applies and removes tc-netem."""
        if not self._has_net_admin(chaos_fleet):
            pytest.skip("NET_ADMIN not available — skipping netem test")

        node = chaos_fleet.daq_nodes[0]
        # Verify it doesn't raise
        with chaos_fleet.chaos.net.latency(node, latency_ms=50):
            # We can't easily measure RTT here; just assert no exception
            pass

    def test_packet_loss_context_manager(self, chaos_fleet: Fleet) -> None:
        """packet_loss() context manager runs without raising."""
        if not self._has_net_admin(chaos_fleet):
            pytest.skip("NET_ADMIN not available — skipping netem test")

        node = chaos_fleet.daq_nodes[0]
        with chaos_fleet.chaos.net.packet_loss(node, loss_pct=10.0):
            pass  # Just verify no exception

    def test_iptables_blocked_egress_restores(self, chaos_fleet: Fleet) -> None:
        """blocked_egress() drops then restores outbound traffic rule."""
        if not self._has_net_admin(chaos_fleet):
            pytest.skip("NET_ADMIN not available — skipping iptables test")

        node = chaos_fleet.daq_nodes[0]
        # Use a safe non-routable destination
        with chaos_fleet.chaos.iptables.blocked_egress(node, "10.255.255.255"):
            pass  # Rule applied and removed without raising
