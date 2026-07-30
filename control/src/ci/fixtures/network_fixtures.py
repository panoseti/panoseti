"""
ci/fixtures/network_fixtures.py

Fixtures for mocking network communication and Quabo command/HK streams.
"""

from __future__ import annotations

import struct
from typing import Any

import pytest

from control.driver.quabo_driver import UDP_CMD_PORT


class FakeSocket:
    """A simulated UDP socket for testing network drivers."""
    def __init__(self):
        self.sent: list[tuple[bytes, tuple]] = []
        self.received_packets: list[tuple[bytes, tuple[str, int]]] = []
        self.timeout: float | None = None
        self.bound_address: tuple[str, int] | None = None
        self.responses: dict[int, bytes] = {}  # opcode → bytes returned by recvfrom

    def sendto(self, data: bytes, address: tuple) -> int:
        self.sent.append((bytes(data), address))
        return len(data)

    def recvfrom(self, bufsize: int) -> tuple[bytes, tuple]:
        # Priority 1: Automated response based on last sent opcode
        if self.sent:
            opcode = self.sent[-1][0][0] & 0x7F
            if opcode in self.responses:
                return self.responses[opcode], ("192.168.0.100", UDP_CMD_PORT)
        
        # Priority 2: Pre-queued packets
        if self.received_packets:
             return self.received_packets.pop(0)

        raise TimeoutError("Fake socket timeout")

    def settimeout(self, timeout: float | None) -> None:
        self.timeout = timeout

    def bind(self, address: tuple) -> None:
        self.bound_address = address

    def setsockopt(self, *a: Any) -> None:
        pass

    def close(self) -> None:
        pass

    @property
    def last_cmd(self) -> bytes:
        assert self.sent, "No packets sent yet"
        return self.sent[-1][0]

    @property
    def last_dest(self) -> tuple:
        assert self.sent, "No packets sent yet"
        return self.sent[-1][1]

# ===========================================================================
# Assertion helpers (usable against FakeSocket or real captured bytes)
# ===========================================================================

def assert_opcode(data: bytes, expected: int, mask: int = 0xFF) -> None:
    """Assert the first byte (masked) equals *expected*."""
    assert (data[0] & mask) == expected, (
        f"Opcode mismatch: got 0x{data[0]:02X}, expected 0x{expected:02X}"
    )


def assert_packet_length(data: bytes, expected_len: int) -> None:
    assert len(data) == expected_len, (
        f"Packet length mismatch: got {len(data)}, expected {expected_len}"
    )


def assert_bytes_zero(data: bytes, start: int, end: int) -> None:
    """Assert that bytes [start:end] are all zero."""
    region = data[start:end]
    assert all(b == 0 for b in region), (
        f"Expected zero bytes in [{start}:{end}], got {region.hex()}"
    )


def assert_le_uint16(data: bytes, offset: int, expected: int) -> None:
    val = struct.unpack_from("<H", data, offset)[0]
    assert val == expected, f"LE uint16 at offset {offset}: got {val}, expected {expected}"

@pytest.fixture
def mock_network(monkeypatch: pytest.MonkeyPatch) -> FakeSocket:
    """Intercepts low-level socket.socket calls with a FakeSocket."""
    fake_sock = FakeSocket()
    monkeypatch.setattr("socket.socket", lambda *a, **kw: fake_sock)
    monkeypatch.setattr("socket.gethostbyname", lambda x: x)
    return fake_sock
