"""
Packet capture fixtures — promoted from tier1_unit FakeSocket pattern.

FakeSocket can be used by both software-only tests (monkeypatched) and
HITL driver_protocol tests (real FPGA echo responses checked against
the same assertion helpers).
"""

from __future__ import annotations

import socket
import struct
from typing import Any

import pytest

from control.driver.quabo_driver import UDP_CMD_PORT

# ===========================================================================
# FakeSocket
# ===========================================================================

class FakeSocket:
    """
    Minimal socket shim that captures all sendto() calls.

    For HITL tests, replace this with a real socket bound to the quabo's
    response port and use the same assertion methods.
    """

    def __init__(self):
        self.sent: list[tuple[bytes, tuple]] = []
        self._timeout: float = 0.5
        self.responses: dict[int, bytes] = {}  # opcode → bytes returned by recvfrom

    def settimeout(self, t: float) -> None:
        self._timeout = t

    def bind(self, addr: tuple) -> None:
        pass

    def close(self) -> None:
        pass

    def setsockopt(self, *a: Any) -> None:
        pass

    def sendto(self, data: bytes, addr: tuple) -> None:
        self.sent.append((bytes(data), addr))

    def recvfrom(self, size: int) -> tuple[bytes, tuple]:
        if self.sent:
            opcode = self.sent[-1][0][0] & 0x7F
            if opcode in self.responses:
                return self.responses[opcode], ("192.168.0.100", UDP_CMD_PORT)
        raise TimeoutError()

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


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def fake_socket(monkeypatch: Any) -> FakeSocket:
    """Yield a FakeSocket with socket.socket monkeypatched."""
    sock = FakeSocket()
    monkeypatch.setattr("socket.socket", lambda *a, **kw: sock)
    monkeypatch.setattr("socket.gethostbyname", lambda x: x)
    return sock


@pytest.fixture
def real_udp_capture(pytestconfig: Any):
    """
    Yield a real UDP socket bound to an ephemeral port for HITL packet capture.
    Use this in hw0_driver_protocol tests to sniff real FPGA echo responses.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(5.0)
    sock.bind(("", 0))
    yield sock
    sock.close()
