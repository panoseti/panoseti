"""
fixtures/chaos.py — Chaos accessor for v2 tier-4 fault-injection tests.

Wraps the existing ci/fixtures/chaos/* modules through typed sub-handles so
tests operate on Fleet container objects instead of bare container-name strings.

Usage in tests::

    def test_latency(session_fleet):
        node = session_fleet.daq_nodes[0]
        with session_fleet.chaos.net.latency(node, latency_ms=200):
            # gRPC calls to node experience ~200 ms added latency
            ...

    def test_process_kill(session_fleet):
        node = session_fleet.daq_nodes[0]
        session_fleet.chaos.proc.kill(node, "pseti-grpc")
        assert not session_fleet.chaos.proc.alive(node, "pseti-grpc")

Or via the pytest fixture::

    def test_something(chaos, session_fleet):
        with chaos.disk.full_disk(session_fleet.daq_nodes[0], "/data"):
            ...
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from ci.software_only_v2.containers.base import PsetiContainer
    from ci.software_only_v2.orchestrator.fleet import Fleet

import pytest

# Type alias for the two accepted forms
_ContainerArg = Union["PsetiContainer", str]


def _cname(container_or_name: _ContainerArg) -> str:
    """Extract a Docker container name from a PsetiContainer or a bare string."""
    if isinstance(container_or_name, str):
        return container_or_name
    return container_or_name.name  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Sub-handles
# ---------------------------------------------------------------------------

class NetemHandle:
    """tc-netem network impairment handle (requires NET_ADMIN on target container)."""

    @contextmanager
    def latency(
        self,
        container: _ContainerArg,
        latency_ms: int,
        iface: str | None = None,
    ) -> Generator[None]:
        """Add constant outbound latency for the block duration, then restore."""
        from ci.software_only_v2.fixtures.chaos import netem
        with netem.latency(_cname(container), latency_ms=latency_ms, iface=iface):
            yield

    @contextmanager
    def packet_loss(
        self,
        container: _ContainerArg,
        loss_pct: float,
        iface: str | None = None,
    ) -> Generator[None]:
        """Add outbound packet loss for the block duration, then restore."""
        from ci.software_only_v2.fixtures.chaos import netem
        with netem.packet_loss(_cname(container), loss_pct=loss_pct, iface=iface):
            yield

    def add(
        self,
        container: _ContainerArg,
        *,
        iface: str | None = None,
        latency_ms: int = 0,
        loss_pct: float = 0.0,
        duplicate_pct: float = 0.0,
        corrupt_pct: float = 0.0,
    ) -> None:
        """Apply tc-netem impairments (caller is responsible for calling remove)."""
        from ci.software_only_v2.fixtures.chaos import netem
        netem.add_netem(
            _cname(container),
            iface=iface,
            latency_ms=latency_ms,
            loss_pct=loss_pct,
            duplicate_pct=duplicate_pct,
            corrupt_pct=corrupt_pct,
        )

    def remove(self, container: _ContainerArg, iface: str | None = None) -> None:
        """Remove tc-netem impairments (best-effort)."""
        from ci.software_only_v2.fixtures.chaos import netem
        netem.remove_netem(_cname(container), iface=iface)


class IptablesHandle:
    """iptables blackhole rules (requires NET_ADMIN on target container)."""

    @contextmanager
    def blocked_egress(
        self,
        container: _ContainerArg,
        dst_ip: str,
        port: int | None = None,
    ) -> Generator[None]:
        """Drop outbound traffic to dst_ip[:port] for the block duration."""
        from ci.software_only_v2.fixtures.chaos import iptables
        with iptables.blocked_egress(_cname(container), dst_ip=dst_ip, port=port):
            yield

    def blackhole(
        self,
        container: _ContainerArg,
        dst_ip: str,
        port: int | None = None,
    ) -> None:
        """Add OUTPUT DROP rule for dst_ip (caller responsible for cleanup)."""
        from ci.software_only_v2.fixtures.chaos import iptables
        iptables.blackhole(_cname(container), dst_ip=dst_ip, port=port)

    def unblackhole(
        self,
        container: _ContainerArg,
        dst_ip: str,
        port: int | None = None,
    ) -> None:
        """Remove OUTPUT DROP rule (best-effort)."""
        from ci.software_only_v2.fixtures.chaos import iptables
        iptables.unblackhole(_cname(container), dst_ip=dst_ip, port=port)


class DiskHandle:
    """Filesystem fault injection (ENOSPC simulation)."""

    @contextmanager
    def full_disk(
        self,
        container: _ContainerArg,
        mount_path: str,
        fill_pct: int = 99,
    ) -> Generator[None]:
        """Fill mount_path to fill_pct% for the block, then release."""
        from ci.software_only_v2.fixtures.chaos import disk_chaos
        with disk_chaos.full_disk(_cname(container), mount_path=mount_path, fill_pct=fill_pct):
            yield

    def fill(
        self,
        container: _ContainerArg,
        mount_path: str,
        fill_pct: int = 99,
    ) -> str:
        """Fill mount_path to fill_pct%. Returns fill-file path for manual cleanup."""
        from ci.software_only_v2.fixtures.chaos import disk_chaos
        return disk_chaos.fill_volume(_cname(container), mount_path=mount_path, fill_pct=fill_pct)

    def release(self, container: _ContainerArg, fill_file: str) -> None:
        """Remove the fill file created by fill()."""
        from ci.software_only_v2.fixtures.chaos import disk_chaos
        disk_chaos.release_fill(_cname(container), fill_file=fill_file)


class ProcessHandle:
    """Process-level fault injection (kill, freeze, wait)."""

    def kill(
        self,
        container: _ContainerArg,
        process_name: str,
        sig: str = "KILL",
    ) -> None:
        """Send signal to process_name inside container."""
        from ci.software_only_v2.fixtures.chaos import process_chaos
        process_chaos.kill_process(_cname(container), process_name=process_name, sig=sig)

    def alive(self, container: _ContainerArg, process_name: str) -> bool:
        """Return True if process_name is running inside container."""
        from ci.software_only_v2.fixtures.chaos import process_chaos
        return process_chaos.process_alive(_cname(container), process_name=process_name)

    def wait_dead(
        self,
        container: _ContainerArg,
        process_name: str,
        timeout: float = 10.0,
    ) -> bool:
        """Poll until process_name is gone or timeout. Returns True on success."""
        from ci.software_only_v2.fixtures.chaos import process_chaos
        return process_chaos.wait_for_process_death(
            _cname(container), process_name=process_name, timeout=timeout
        )

    def wait_alive(
        self,
        container: _ContainerArg,
        process_name: str,
        timeout: float = 10.0,
    ) -> bool:
        """Poll until process_name appears or timeout. Returns True on success."""
        from ci.software_only_v2.fixtures.chaos import process_chaos
        return process_chaos.wait_for_process_start(
            _cname(container), process_name=process_name, timeout=timeout
        )

    @contextmanager
    def freeze(
        self,
        container: _ContainerArg,
        process_name: str,
    ) -> Generator[None]:
        """SIGSTOP process_name for the block, then SIGCONT."""
        from ci.software_only_v2.fixtures.chaos import process_chaos
        with process_chaos.freeze_process(_cname(container), process_name=process_name):
            yield

    @contextmanager
    def kill_after(
        self,
        container: _ContainerArg,
        process_name: str,
        delay_s: float = 0.0,
        sig: str = "KILL",
    ) -> Generator[None]:
        """Kill process_name after delay_s seconds (context manager cancels if not fired)."""
        from ci.software_only_v2.fixtures.chaos import process_chaos
        with process_chaos.kill_after(
            _cname(container), process_name=process_name, delay_s=delay_s, sig=sig
        ):
            yield


class GrpcHandle:
    """In-process gRPC fault injection via GrpcChaosProxy."""

    def proxy(self, client: Any | None = None) -> Any:
        """Return a GrpcChaosProxy wrapping client (or an unbound proxy if None)."""
        from ci.software_only_v2.fixtures.chaos.grpc_proxy import GrpcChaosProxy
        return GrpcChaosProxy(client)

    @contextmanager
    def inject(
        self,
        client: Any,
        method: str,
        mode: str,
        timeout_s: float = 30.0,
    ) -> Generator[None]:
        """Inject a single-method fault for the block duration.

        Modes: "timeout", "unavailable", "slow_response",
               "success_then_fail", "reset_stream", "partial_response".
        """
        from ci.software_only_v2.fixtures.chaos.grpc_proxy import inject_rpc_fault
        with inject_rpc_fault(client, method=method, mode=mode, timeout_s=timeout_s):
            yield


# ---------------------------------------------------------------------------
# Chaos accessor
# ---------------------------------------------------------------------------

class Chaos:
    """Unified fault-injection accessor attached to a Fleet.

    Exposed via fleet.chaos (property) or via the chaos() pytest fixture.
    Each sub-handle accepts PsetiContainer objects (or bare container-name
    strings) so tests stay readable without managing raw Docker names.
    """

    def __init__(self, fleet: Fleet) -> None:
        self._fleet = fleet
        self._net = NetemHandle()
        self._iptables = IptablesHandle()
        self._disk = DiskHandle()
        self._proc = ProcessHandle()
        self._grpc = GrpcHandle()

    @property
    def net(self) -> NetemHandle:
        """tc-netem: latency, packet_loss, add/remove."""
        return self._net

    @property
    def iptables(self) -> IptablesHandle:
        """iptables: blocked_egress, blackhole/unblackhole."""
        return self._iptables

    @property
    def disk(self) -> DiskHandle:
        """disk: full_disk, fill/release."""
        return self._disk

    @property
    def proc(self) -> ProcessHandle:
        """process: kill, freeze, wait_dead/wait_alive, kill_after."""
        return self._proc

    @property
    def grpc(self) -> GrpcHandle:
        """gRPC: proxy(), inject()."""
        return self._grpc

    # Convenience: direct access to fleet handles
    @property
    def fleet(self) -> Fleet:
        return self._fleet


# ---------------------------------------------------------------------------
# Pytest fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def chaos(session_fleet: Any) -> Chaos:
    """Function-scoped Chaos accessor bound to the session fleet.

    Requires session_fleet to be active (tier-4 tests only).
    """
    return Chaos(session_fleet)
