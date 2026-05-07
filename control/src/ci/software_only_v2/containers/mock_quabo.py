"""
containers/mock_quabo.py — MockQuabo container.

Wraps the existing pseti-mock-quabo:latest image (built from
ci/mock_quabo/Dockerfile).  Exposes UDP ports 60000–60003 for quabo
command/HK traffic.  Requires NET_ADMIN to add IP aliases inside the
container.
"""

from __future__ import annotations

import os

from ci.software_only_v2.containers.base import PsetiContainer

_MOCK_QUABO_IMAGE = os.environ.get("PSETI_MOCK_QUABO_IMAGE", "pseti-mock-quabo:latest")

# UDP ports used by quabo protocol
_QUABO_PORTS = [60000, 60001, 60002, 60003]


class MockQuaboContainer(PsetiContainer):
    """A simulated Quabo module (4 UDP listeners per module IP).

    The mock server handles:
    - UDP 60000–60003: command packets
    - UDP 60002: housekeeping beacon (every ~3 s)

    Set ``hk_dest_ip`` to route HK packets to the head node's IP.
    """

    _IMAGE = _MOCK_QUABO_IMAGE
    # No gRPC port — communication is UDP only
    _GRPC_PORT = 0

    def __init__(
        self,
        name: str,
        *,
        module_id: int = 200,
        base_ip: str = "192.168.3.32",
        hk_dest_ip: str = "127.0.0.1",
        network=None,
    ) -> None:
        self._module_id = module_id
        self._base_ip = base_ip
        self._hk_dest_ip = hk_dest_ip
        super().__init__(name=name, network=network)

    def _configure(self) -> None:
        self._env("MOCK_QUABO_MODULE_ID", str(self._module_id))
        self._env("MOCK_QUABO_BASE_IP", self._base_ip)
        self._env("MOCK_QUABO_HK_DEST_IP", self._hk_dest_ip)

        # NET_ADMIN required to add IP aliases for all 4 quabo IPs
        self._kwargs(cap_add=["NET_ADMIN"])

    # Override healthcheck — this container has no TCP port; use Docker inspect
    def wait_tcp(self, *, timeout: float = 30.0) -> None:
        """MockQuabo has no TCP port — just give it a moment to start."""
        import time
        time.sleep(2.0)

    def wait_grpc_ready(self, *, timeout: float = 30.0) -> None:
        self.wait_tcp(timeout=timeout)
