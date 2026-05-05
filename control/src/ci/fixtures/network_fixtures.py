"""
ci/fixtures/network_fixtures.py

Fixtures for mocking network communication and Quabo command/HK streams.
"""

from __future__ import annotations

import socket
import pytest

class FakeSocket:
    """A simulated UDP socket for testing network drivers."""
    def __init__(self):
        self.sent_packets: list[tuple[bytes, tuple[str, int]]] = []
        self.received_packets: list[tuple[bytes, tuple[str, int]]] = []
        self.timeout: float | None = None
        self.bound_address: tuple[str, int] | None = None

    def sendto(self, data: bytes, address: tuple[str, int]) -> int:
        self.sent_packets.append((data, address))
        return len(data)

    def recvfrom(self, bufsize: int) -> tuple[bytes, tuple[str, int]]:
        if not self.received_packets:
            raise socket.timeout("Fake socket timeout")
        return self.received_packets.pop(0)

    def settimeout(self, timeout: float | None) -> None:
        self.timeout = timeout

    def bind(self, address: tuple[str, int]) -> None:
        self.bound_address = address

    def setsockopt(self, level: int, optname: int, value: int) -> None:
        pass

    def close(self) -> None:
        pass

@pytest.fixture
def mock_network(monkeypatch: pytest.MonkeyPatch) -> FakeSocket:
    """Intercepts low-level socket.socket calls with a FakeSocket."""
    fake_sock = FakeSocket()
    monkeypatch.setattr("socket.socket", lambda *a, **kw: fake_sock)
    return fake_sock
