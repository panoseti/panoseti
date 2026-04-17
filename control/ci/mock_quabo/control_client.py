"""
mock_quabo/control_client.py

pytest-side UDS client for controlling the mock_quabo server.
Used in test fixtures to inject topology-level faults without
simulating firmware-level misbehavior.
"""

from __future__ import annotations

import json
import os
import socket
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

UDS_SOCK_PATH = os.getenv("MOCK_QUABO_UDS", "/tmp/mock_quabo.sock")


class MockQuaboControlClient:
    """Thin synchronous UDS client for mock_quabo/server.py control socket."""

    def __init__(self, uds_path: str = UDS_SOCK_PATH) -> None:
        self.uds_path = uds_path

    def _send(self, cmd: str) -> dict[str, Any]:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(5.0)
            sock.connect(self.uds_path)
            sock.sendall((cmd + "\n").encode())
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buf += chunk
        resp: dict[str, Any] = json.loads(buf.decode().strip())
        return resp

    def set_hk_dest(self, ip: str) -> dict[str, Any]:
        """Direct HK packets to the given IP address."""
        return self._send(f"set_hk_dest {ip}")

    def report_state(self) -> dict[str, Any]:
        """Return the full server state dict."""
        return self._send("report_state")

    def reset(self) -> dict[str, Any]:
        """Reset all quabo state to defaults."""
        return self._send("reset")

    def silence(self) -> dict[str, Any]:
        """Drop all UDP responses (simulates silent quabo)."""
        return self._send("silence")

    def unsilence(self) -> dict[str, Any]:
        """Restore normal UDP response behaviour."""
        return self._send("unsilence")

    def emit_science_packet(
        self, dest_ip: str, dest_port: int = 60001, payload: bytes = b""
    ) -> dict[str, Any]:
        """Emit a single science UDP datagram to dest_ip:dest_port."""
        params = {
            "dest_ip": dest_ip,
            "dest_port": dest_port,
            "payload_hex": payload.hex(),
        }
        return self._send(f"emit_science_packet {json.dumps(params)}")

    def quabo_state(self, quabo_index: int) -> dict[str, Any]:
        """Return state dict for a specific quabo index."""
        resp = self.report_state()
        quabos: list[dict[str, Any]] = resp.get("state", {}).get("quabos", [])
        for q in quabos:
            if q.get("index") == quabo_index:
                return q
        return {}


class MockQuaboFleet:
    """
    Thin handle to the mock_quabo container.
    Wraps MockQuaboControlClient with reset-between-tests logic.
    """

    def __init__(self, uds_path: str = UDS_SOCK_PATH) -> None:
        self.client = MockQuaboControlClient(uds_path)

    def reset_all(self) -> None:
        self.client.reset()

    def silence_quabo(self, index: int | None = None) -> None:
        """Silence ALL quabos (index unused, kept for API symmetry)."""
        self.client.silence()

    def unsilence_all(self) -> None:
        self.client.unsilence()

    def all(self) -> list[dict[str, Any]]:
        resp = self.client.report_state()
        return resp.get("state", {}).get("quabos", [])

    def quabo_state(self, index: int) -> dict[str, Any]:
        return self.client.quabo_state(index)

    @staticmethod
    def attach(container_name: str | None = None, uds_path: str = UDS_SOCK_PATH) -> MockQuaboFleet:
        """
        Attach to a running mock_quabo container.
        In CI the UDS is volume-mounted at uds_path.
        """
        return MockQuaboFleet(uds_path)


@contextmanager
def silent_quabo(fleet: MockQuaboFleet) -> Generator[None]:
    """Context manager: silences mock_quabo for the duration of the block."""
    fleet.silence_quabo()
    try:
        yield
    finally:
        fleet.unsilence_all()
