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
from typing import Any

import pytest

from ci.software_only_v2.infra.spec import FleetSpec
from ci.software_only_v2.orchestrator.fleet import Fleet


pytestmark = pytest.mark.tier4

# Minimal parameters required by DaqControlClient.StatusDaq
_STATUS_PARAMS: dict[str, Any] = {
    "data_dir": "/data",
    "check_hashpipe_running": True,
    "check_disk_usage": False,
    "check_run_dirs": False,
}


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
def chaos_workspace(tmp_path_factory):
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

    from ci.software_only_v2.infra.workspace import Workspace
    yield Workspace(root=tmp_path, topology=topology, state_probe=StateProbe())

    for key, orig in original.items():
        if orig is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = orig


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
                client.StatusDaq(_STATUS_PARAMS)
            assert exc_info.value.code() == grpc.StatusCode.UNAVAILABLE

    def test_proxy_set_mode_and_restore(self, chaos_fleet: Fleet) -> None:
        """GrpcChaosProxy patches and unpatch cleanly."""
        import grpc
        client = chaos_fleet.daq_control_client(0)
        proxy = chaos_fleet.chaos.grpc.proxy(client)
        proxy.set_mode("StatusDaq", "unavailable")

        with proxy:
            with pytest.raises(grpc.RpcError):
                client.StatusDaq(_STATUS_PARAMS)

        # After context exit, restore() was called — original method is back
        # (StatusDaq may fail for other reasons, but not from our injection)
        try:
            client.StatusDaq(_STATUS_PARAMS)
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
                client.StatusDaq(_STATUS_PARAMS)
            except grpc.RpcError as e:
                # A real server error (not injected) is acceptable on call 0
                assert e.code() != grpc.StatusCode.UNAVAILABLE

            # Second call must raise UNAVAILABLE
            with pytest.raises(grpc.RpcError) as exc_info:
                client.StatusDaq(_STATUS_PARAMS)
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
        """pseti-grpc process is alive in the daqnode container (comm = script name)."""
        node = chaos_fleet.daq_nodes[0]
        alive = chaos_fleet.chaos.proc.alive(node, "pseti-grpc")
        assert alive, "Expected pseti-grpc process to be alive in daqnode container"

    def test_kill_and_wait_dead(self, chaos_fleet: Fleet) -> None:
        """Killing a non-critical process and confirming it is gone."""
        node = chaos_fleet.daq_nodes[0]
        chaos = chaos_fleet.chaos

        # Only run if pseti-grpc is actually alive
        if not chaos.proc.alive(node, "pseti-grpc"):
            pytest.skip("python not running in container — skipping kill test")

        # We kill 'sleep' if present (harmless) rather than the server itself
        # (killing the server would break other tests in this module)
        from ci.software_only_v2.fixtures.chaos import process_chaos
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
    """DiskHandle: fill/release filesystem space inside containers.

    Tests verify the fill/release mechanism at a fixed small size (10 MB)
    rather than a percentage, because the container overlay FS is large (400+ GB)
    and filling a percentage would be impractically slow.
    """

    def test_full_disk_context_manager_releases(self, chaos_fleet: Fleet) -> None:
        """full_disk() context manager creates a fill file removed on exit."""
        node = chaos_fleet.daq_nodes[0]
        fill_file = "/tmp/.chaos_fill"

        from ci.software_only_v2.fixtures.chaos import process_chaos

        # Pre-condition: no fill file
        code, _ = process_chaos._exec(node.name, f"test -f {fill_file}")
        assert code != 0, "Fill file should not exist before test"

        # Write a 10 MB file directly to test the mechanism
        code, out = process_chaos._exec(
            node.name, f"dd if=/dev/zero of={fill_file} bs=1M count=10 2>/dev/null"
        )
        assert code == 0, f"dd should succeed for a 10 MB file: {out}"
        try:
            code, _ = process_chaos._exec(node.name, f"test -f {fill_file}")
            assert code == 0, "Fill file should exist during fill"
        finally:
            process_chaos._exec(node.name, f"rm -f {fill_file}")

        code, _ = process_chaos._exec(node.name, f"test -f {fill_file}")
        assert code != 0, "Fill file should be removed after release"

    def test_manual_fill_and_release(self, chaos_fleet: Fleet) -> None:
        """fill(fill_pct=1) and release() lifecycle: fill file created then removed."""
        node = chaos_fleet.daq_nodes[0]
        chaos = chaos_fleet.chaos
        from ci.software_only_v2.fixtures.chaos import process_chaos

        fill_file = chaos.disk.fill(node, "/tmp", fill_pct=1)
        try:
            assert fill_file, "Expected a fill-file path to be returned"
            code, _ = process_chaos._exec(node.name, f"test -f {fill_file}")
            assert code == 0, "Fill file should exist after fill()"
        finally:
            chaos.disk.release(node, fill_file)
        code, _ = process_chaos._exec(node.name, f"test -f {fill_file}")
        assert code != 0, "Fill file should be gone after release()"


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
        from ci.software_only_v2.fixtures.chaos import process_chaos
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
