"""
containers/gateway.py — socat-based port-forwarding gateway container.

Mirrors the static gateway setup in ci/fixtures/build/gateway_setup.sh but
as a dynamic testcontainer.  Used when a FleetSpec has DAQ nodes behind a
gateway (NetworkConfig.daq_nodes[*].port_forwarding.status=True).

Each forwarding rule is a socat process:
    socat TCP-LISTEN:<gw_port>,fork TCP:<target_ip>:<target_port>
"""

from __future__ import annotations

from dataclasses import dataclass

from ci.software_only.containers.base import PsetiContainer

_GATEWAY_IMAGE = "alpine/socat"
_SOCAT_BINARY = "socat"


@dataclass
class ForwardRule:
    """One socat port-forwarding rule."""
    listen_port: int
    target_ip: str
    target_port: int


class GatewayContainer(PsetiContainer):
    """A socat-based gateway that port-forwards to DAQ nodes.

    Usage::

        gw = GatewayContainer("my-gw", rules=[
            ForwardRule(listen_port=50051, target_ip="192.168.0.10", target_port=50051),
            ForwardRule(listen_port=50052, target_ip="192.168.0.20", target_port=50051),
        ])
        gw.start()
    """

    _IMAGE = _GATEWAY_IMAGE
    _GRPC_PORT = 0  # no single gRPC port — individual rules exposed below

    def __init__(
        self,
        name: str,
        *,
        rules: list[ForwardRule] | None = None,
        network=None,
    ) -> None:
        self._rules: list[ForwardRule] = rules or []
        super().__init__(name=name, network=network)

    def _configure(self) -> None:
        if not self._rules:
            self._command("sleep infinity")
            return

        # Expose all listen ports
        for rule in self._rules:
            self._expose(rule.listen_port)

        # Build a shell command that starts all socat rules in parallel
        socat_cmds = " & ".join(
            f"socat TCP-LISTEN:{r.listen_port},fork,reuseaddr TCP:{r.target_ip}:{r.target_port}"
            for r in self._rules
        )
        self._command(f"sh -c '{socat_cmds}; wait'")

    # Gateway has no single healthcheck port — skip gRPC check
    def wait_grpc_ready(self, *, timeout: float = 10.0) -> None:
        import time
        time.sleep(1.0)

    def mapped_port(self, rule: ForwardRule) -> int:
        """Return the host-mapped port for a given ForwardRule."""
        return int(self._container.get_exposed_port(rule.listen_port))
