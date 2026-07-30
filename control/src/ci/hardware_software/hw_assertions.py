"""
Assertion helpers for HITL tests.
Provides HKPacketParser for raw UDP housekeeping packets,
and Redis/InfluxDB query helpers for mid- and high-level assertions.
"""

from __future__ import annotations

import struct
from typing import Any

# HK packet field offsets (64-byte packet, per capture_hk.py and quabo_driver.py)
HK_MAGIC = 0x20
HK_MAGIC_OFFSET = 0
HK_BOOTBYTE_OFFSET = 1
HK_BOARDLOC_OFFSET = 2       # 2 bytes LE
HK_PACKET_LEN = 64


class HKPacketParser:
    """
    Parse and assert fields in a raw 64-byte housekeeping packet.

    References:
    - control/daemons/capture_hk.py  (field layout and SI conversion)
    - Quabo-packet-interface.md       (bit-level definitions)
    """

    def __init__(self, raw_bytes: bytes):
        if len(raw_bytes) != HK_PACKET_LEN:
            raise ValueError(f"Expected {HK_PACKET_LEN} bytes, got {len(raw_bytes)}")
        self._data = raw_bytes

    # ── Raw field accessors ────────────────────────────────────────────────

    @property
    def magic(self) -> int:
        return self._data[HK_MAGIC_OFFSET]

    @property
    def bootbyte(self) -> int:
        return self._data[HK_BOOTBYTE_OFFSET]

    @property
    def boardloc(self) -> int:
        return struct.unpack_from("<H", self._data, HK_BOARDLOC_OFFSET)[0]

    def uint16_le(self, offset: int) -> int:
        return struct.unpack_from("<H", self._data, offset)[0]

    def int16_le(self, offset: int) -> int:
        return struct.unpack_from("<h", self._data, offset)[0]

    # ── Assertion methods ──────────────────────────────────────────────────

    def assert_magic(self) -> None:
        assert self.magic == HK_MAGIC, (
            f"Magic byte mismatch: got 0x{self.magic:02X}, expected 0x{HK_MAGIC:02X}"
        )

    def assert_bootbyte(self, expected: int) -> None:
        assert self.bootbyte == expected, (
            f"Bootbyte mismatch: got 0x{self.bootbyte:02X}, expected 0x{expected:02X}"
        )

    def assert_boardloc(self, module_id: int, quadrant: int) -> None:
        expected = module_id * 4 + quadrant
        assert self.boardloc == expected, (
            f"BOARDLOC mismatch: got {self.boardloc}, expected {expected} "
            f"(module {module_id}, quad {quadrant})"
        )

    def validate_pcb_revision(self, expected: int) -> None:
        """Assert bit 0 of byte 53 equals expected PCB revision."""
        pcb_rev = self._data[53] & 0x01
        assert pcb_rev == expected, (
            f"PCB revision mismatch: got {pcb_rev}, expected {expected}"
        )

    def assert_length(self) -> None:
        assert len(self._data) == HK_PACKET_LEN


# ===========================================================================
# Redis helpers
# ===========================================================================

def get_redis_hk(redis_client: Any, boardloc: int) -> dict[str, str]:
    """
    Fetch all HK fields for a quabo from Redis.

    Args:
        redis_client: A connected redis.Redis instance.
        boardloc: The BOARDLOC (module_id * 4 + quadrant).

    Returns:
        Dict of field → raw string value (as stored by capture_hk.py).
    """
    key = f"QUABO_{boardloc}"
    data = redis_client.hgetall(key)
    return {k.decode(): v.decode() for k, v in data.items()}


def assert_voltage_in_spec(
    hk_dict: dict[str, str],
    field: str,
    nominal: float,
    tolerance: float = 0.05,
) -> None:
    """Assert that a voltage field is within *tolerance* fraction of *nominal*."""
    val = float(hk_dict[field])
    lo, hi = nominal * (1 - tolerance), nominal * (1 + tolerance)
    assert lo <= val <= hi, (
        f"{field} out of spec: {val:.4f} not in [{lo:.4f}, {hi:.4f}]"
    )


def assert_temperature_plausible(hk_dict: dict[str, str], field: str) -> None:
    """Assert a temperature field (°C) is in a plausible range for lab operation."""
    val = float(hk_dict[field])
    assert -10 <= val <= 85, f"{field} temperature implausible: {val:.1f} °C"
