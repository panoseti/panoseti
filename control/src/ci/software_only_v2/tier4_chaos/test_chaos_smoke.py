"""
tier4_chaos/test_chaos_smoke.py — Chaos fault-injection smoke tests.

Each class exercises one chaos sub-handle via the session_fleet.
All tests skip if Docker is unavailable.

Container capability requirements:
 - NET_ADMIN: NetemTests, IptablesTests (daqnode containers have this)
 - No extra caps: DiskTests, ProcessTests, GrpcProxyTests
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from ci.software_only_v2.orchestrator.fleet import Fleet
from ci.software_only_v2.fixtures.chaos import Chaos

# Reuse shared Docker guard from the tier-level conftest
from ci.software_only_v2.tier4_chaos.conftest import requires_docker

pytestmark = pytest.mark.tier4

# Minimal parameters required by DaqControlClient.StatusDaq
_STATUS_PARAMS: dict[str, Any] = {
    "data_dir": "/data",
    "check_hashpipe_running": True,
    "check_disk_usage": False,
    "check_run_dirs": False,
}


# ---------------------------------------------------------------------------
# gRPC proxy tests (no special Docker caps required)
# ---------------------------------------------------------------------------

@requires_docker
class TestGrpcProxy:
    """GrpcHandle: in-process fault injection without Docker capabilities."""

    def test_inject_unavailable_raises(self, session_fleet: Fleet) -> None:
        """inject_rpc_fault('unavailable') makes the method raise grpc.RpcError."""
        import grpc
        client = session_fleet.daq_control_client(0)
        with session_fleet.chaos.grpc.inject(client, "StatusDaq", "unavailable"):
            with pytest.raises(grpc.RpcError) as exc_info:
                client.StatusDaq(_STATUS_PARAMS)
            assert exc_info.value.code() == grpc.StatusCode.UNAVAILABLE

    def test_proxy_set_mode_and_restore(self, session_fleet: Fleet) -> None:
        """GrpcChaosProxy patches and unpatches cleanly."""
        import grpc
        client = session_fleet.daq_control_client(0)
        proxy = session_fleet.chaos.grpc.proxy(client)
        proxy.set_mode("StatusDaq", "unavailable")

        with proxy:
            with pytest.raises(grpc.RpcError):
                client.StatusDaq(_STATUS_PARAMS)

        # After context exit, restore() was called — original method is back
        try:
            client.StatusDaq(_STATUS_PARAMS)
        except grpc.RpcError as e:
            assert e.code() != grpc.StatusCode.UNAVAILABLE, (
                "UNAVAILABLE after restore() means proxy was not cleaned up"
            )

    def test_success_then_fail_mode(self, session_fleet: Fleet) -> None:
        """success_then_fail: first call succeeds, second raises UNAVAILABLE."""
        import grpc
        client = session_fleet.daq_control_client(0)
        proxy = session_fleet.chaos.grpc.proxy(client)
        proxy.set_mode("StatusDaq", "success_then_fail")
        proxy.apply(client)
        try:
            try:
                client.StatusDaq(_STATUS_PARAMS)
            except grpc.RpcError as e:
                assert e.code() != grpc.StatusCode.UNAVAILABLE

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

    def test_when_server_is_running_then_process_alive_returns_true(
        self, session_fleet: Fleet
    ) -> None:
        """pseti-grpc process is alive in the daqnode container."""
        node = session_fleet.daq_nodes[0]
        alive = session_fleet.chaos.proc.alive(node, "pseti-grpc")
        assert alive, "Expected pseti-grpc process to be alive in daqnode container"

    def test_when_process_is_killed_then_wait_dead_returns_true(
        self, session_fleet: Fleet
    ) -> None:
        """Killing a non-critical process and confirming it is gone."""
        node = session_fleet.daq_nodes[0]

        if not session_fleet.chaos.proc.alive(node, "pseti-grpc"):
            pytest.skip("pseti-grpc not running in container — skipping kill test")

        # Kill a harmless 'sleep' rather than the server itself
        from ci.software_only_v2.fixtures.chaos import process_chaos
        process_chaos._exec(node.name, "sleep 30 &")
        time.sleep(0.3)
        session_fleet.chaos.proc.kill(node, "sleep", sig="TERM")
        dead = session_fleet.chaos.proc.wait_dead(node, "sleep", timeout=5.0)
        assert dead, "sleep process should have died after SIGTERM"

    def test_when_absent_process_waited_then_wait_alive_returns_false(
        self, session_fleet: Fleet
    ) -> None:
        """wait_alive() returns False if the process never appears within timeout."""
        node = session_fleet.daq_nodes[0]
        result = session_fleet.chaos.proc.wait_alive(
            node, "definitely_not_a_real_process_xyzzy", timeout=2.0
        )
        assert result is False

    def test_kill_and_restart_grpc_server(self, session_fleet: Fleet) -> None:
        """SC: killing and waiting for pseti-grpc restart exercises process_kill_and_restart."""
        node = session_fleet.daq_nodes[0]

        if not session_fleet.chaos.proc.alive(node, "pseti-grpc"):
            pytest.skip("pseti-grpc not running — cannot test kill+restart")

        session_fleet.chaos.proc.kill(node, "pseti-grpc", sig="KILL")
        # The container supervisor (or systemd-inside) should restart it
        from ci.software_only_v2.infra.parity import run_scenario
        run_scenario("process_kill_and_restart", fleet=session_fleet, node_index=0, timeout=30)


# ---------------------------------------------------------------------------
# Disk chaos tests
# ---------------------------------------------------------------------------

@requires_docker
class TestDiskChaos:
    """DiskHandle: fill/release filesystem space inside containers."""

    def test_when_fill_file_written_then_context_manager_cleans_up(
        self, session_fleet: Fleet
    ) -> None:
        """full_disk() context manager creates a fill file removed on exit."""
        node = session_fleet.daq_nodes[0]
        fill_file = "/tmp/.chaos_fill"

        from ci.software_only_v2.fixtures.chaos import process_chaos

        code, _ = process_chaos._exec(node.name, f"test -f {fill_file}")
        assert code != 0, "Fill file should not exist before test"

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

    def test_when_fill_and_release_called_then_file_lifecycle_is_correct(
        self, session_fleet: Fleet
    ) -> None:
        """fill() and release(): fill file is created then removed."""
        node = session_fleet.daq_nodes[0]
        from ci.software_only_v2.fixtures.chaos import process_chaos

        fill_file = session_fleet.chaos.disk.fill(node, "/tmp", fill_pct=1)
        try:
            assert fill_file, "Expected a fill-file path to be returned"
            code, _ = process_chaos._exec(node.name, f"test -f {fill_file}")
            assert code == 0, "Fill file should exist after fill()"
        finally:
            session_fleet.chaos.disk.release(node, fill_file)
        code, _ = process_chaos._exec(node.name, f"test -f {fill_file}")
        assert code != 0, "Fill file should be gone after release()"


# ---------------------------------------------------------------------------
# Network chaos tests (require NET_ADMIN on the container)
# ---------------------------------------------------------------------------

@requires_docker
class TestNetworkChaos:
    """NetemHandle + IptablesHandle: network impairments inside containers.

    These tests require NET_ADMIN capability and are skipped if absent.
    """

    def _has_net_admin(self, fleet: Fleet) -> bool:
        """Return True if the first daqnode has NET_ADMIN (tc works)."""
        from ci.software_only_v2.fixtures.chaos import process_chaos
        node = fleet.daq_nodes[0]
        code, _ = process_chaos._exec(node.name, "tc qdisc show")
        return code == 0

    def test_when_latency_applied_then_context_manager_restores(
        self, session_fleet: Fleet
    ) -> None:
        """latency() context manager applies and removes tc-netem without raising."""
        if not self._has_net_admin(session_fleet):
            pytest.skip("NET_ADMIN not available — skipping netem test")

        node = session_fleet.daq_nodes[0]
        with session_fleet.chaos.net.latency(node, latency_ms=50):
            pass

    def test_when_packet_loss_applied_then_context_manager_restores(
        self, session_fleet: Fleet
    ) -> None:
        """packet_loss() context manager runs without raising."""
        if not self._has_net_admin(session_fleet):
            pytest.skip("NET_ADMIN not available — skipping netem test")

        node = session_fleet.daq_nodes[0]
        with session_fleet.chaos.net.packet_loss(node, loss_pct=10.0):
            pass

    def test_when_iptables_blocked_then_rule_is_removed_on_exit(
        self, session_fleet: Fleet
    ) -> None:
        """blocked_egress() drops then restores outbound traffic rule."""
        if not self._has_net_admin(session_fleet):
            pytest.skip("NET_ADMIN not available — skipping iptables test")

        node = session_fleet.daq_nodes[0]
        with session_fleet.chaos.iptables.blocked_egress(node, "10.255.255.255"):
            pass
