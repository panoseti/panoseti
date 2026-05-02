"""
test_quabo_driver_protocol_1.py

Protocol-level compliance tests for the PANOSETI quabo packet interface.

Tests cover:
  - BOARDLOC math: module_id ↔ IP address ↔ boardloc
  - Housekeeping packet byte layout and conversion formulas
  - HV voltage conversion math
  - MAROC packet ASIC region gap bytes and structure
  - Trigger mask multi-channel encoding
  - ACQ mode flag combinations

All tests are hardware-agnostic; no real network calls are made.
Reference: Quabo-packet-interface.md / quabo_driver.py
"""
from __future__ import annotations

import struct

import pytest

from ci.software_only.tier1_unit.conftest import _make_hk_packet, _parse_hk_field
from control.utils.config_file import get_boardloc, ip_addr_to_module_id

# ===========================================================================
# TestBoardLocMath
# ===========================================================================

class TestBoardLocMath:
    """BOARDLOC ↔ module_id ↔ IP address encoding math.

    Formula (from utils/config_file.py):
      n = int(ip[3]) + 256 * int(ip[2])
      module_id = (n >> 2) & 0xFF
      boardloc  = n + quabo_index          (where base IP has ip[3] % 4 == 0)
    """

    def test_module_id_simple_ip(self) -> None:
        """ip3=0: module_id = ip4 >> 2."""
        assert ip_addr_to_module_id("192.168.0.0") == 0
        assert ip_addr_to_module_id("192.168.0.32") == 8
        assert ip_addr_to_module_id("192.168.0.252") == 63
        assert ip_addr_to_module_id("192.168.0.200") == 50

    def test_module_id_palomar_range(self) -> None:
        """At Palomar ip3=3: module_id = (3*256 + ip4) >> 2."""
        # (3*256 + 200) >> 2 = 968 >> 2 = 242
        assert ip_addr_to_module_id("192.168.3.200") == 242
        # (3*256 + 252) >> 2 = 1020 >> 2 = 255
        assert ip_addr_to_module_id("192.168.3.252") == 255

    def test_four_consecutive_ips_share_module_id(self) -> None:
        """All four quabos in a module (base IP + 0..3) have the same module_id."""
        base_module = ip_addr_to_module_id("192.168.0.100")
        for q in range(1, 4):
            assert ip_addr_to_module_id(f"192.168.0.{100 + q}") == base_module

    def test_boardloc_zero_quadrant(self) -> None:
        """boardloc = ip3*256 + ip4 + 0 for quabo 0."""
        assert get_boardloc("192.168.0.32", 0) == 32
        assert get_boardloc("192.168.0.252", 0) == 252

    def test_boardloc_all_quadrants(self) -> None:
        """boardloc increments by 1 per quadrant."""
        for q in range(4):
            assert get_boardloc("192.168.0.32", q) == 32 + q

    def test_boardloc_encodes_module_and_quadrant(self) -> None:
        """boardloc >> 2 == module_id, boardloc & 3 == quadrant (base IP % 4 == 0)."""
        for last in [0, 32, 100, 200, 252]:
            ip = f"192.168.0.{last}"
            module_id = ip_addr_to_module_id(ip)
            for q in range(4):
                bl = get_boardloc(ip, q)
                assert bl >> 2 == module_id, f"ip={ip} q={q}: boardloc={bl} >> 2 != module_id={module_id}"
                assert bl & 3 == q

    def test_boardloc_round_trip(self) -> None:
        """Given a boardloc, recover module_id and quadrant exactly."""
        ip = "192.168.0.32"
        module_id = ip_addr_to_module_id(ip)
        for q in range(4):
            bl = get_boardloc(ip, q)
            assert bl >> 2 == module_id
            assert bl & 3 == q

    def test_boardloc_fits_uint16(self) -> None:
        """All valid boardlocs fit in a uint16 (HK packet field width)."""
        # Largest realistic value: ip3=3, ip4=255, quadrant=3 → 3*256+255+3 = 1026
        bl = get_boardloc("192.168.3.255", 3)
        assert 0 <= bl <= 0xFFFF


# ===========================================================================
# TestHousekeepingPacketParsing
# ===========================================================================

class TestHousekeepingPacketParsing:
    """Verify byte layout of the 64-byte quabo housekeeping packet.

    capture_hk.py reads all fields as 16-bit LE pairs starting at byte 2,
    with TEMP1 (array[17]) as the only signed field.
    """

    def test_packet_length(self) -> None:
        assert len(_make_hk_packet()) == 64

    def test_magic_byte(self) -> None:
        pkt = _make_hk_packet()
        assert pkt[0] == 0x20

    # --- BOARDLOC ---

    def test_boardloc_at_array0(self) -> None:
        """BOARDLOC is the first 16-bit LE pair at bytes[2:4]."""
        pkt = _make_hk_packet(boardloc=35)
        assert _parse_hk_field(pkt, 0) == 35

    def test_boardloc_module8_quadrant3(self) -> None:
        """For 192.168.0.32 quabo 3: module_id=8, boardloc=35, bytes[2:4]=35."""
        module_id = ip_addr_to_module_id("192.168.0.32")
        assert module_id == 8
        boardloc = get_boardloc("192.168.0.32", 3)
        assert boardloc == 35

        pkt = _make_hk_packet(boardloc=boardloc)
        raw = _parse_hk_field(pkt, 0)
        assert raw == 35
        assert raw >> 2 == 8    # module_id
        assert raw & 3 == 3     # quadrant

    # --- TEMP1 (PCB temperature, signed, 0.25 °C/LSB) ---

    def test_temp1_at_array17_positive(self) -> None:
        """TEMP1 stored as signed int16 at bytes[36:38]; 0.25 °C/LSB → 80 °C = raw 320."""
        pkt = _make_hk_packet(temp1_raw=320)
        val = _parse_hk_field(pkt, 17, signed=True)
        assert val == 320
        assert val * 0.25 == pytest.approx(80.0)

    def test_temp1_negative(self) -> None:
        """Negative TEMP1 survives round-trip as signed int16."""
        pkt = _make_hk_packet(temp1_raw=-40)    # -10 °C
        val = _parse_hk_field(pkt, 17, signed=True)
        assert val == -40
        assert val * 0.25 == pytest.approx(-10.0)

    def test_temp1_zero(self) -> None:
        pkt = _make_hk_packet(temp1_raw=0)
        assert _parse_hk_field(pkt, 17, signed=True) == 0

    # --- TEMP2 (FPGA temperature, unsigned, formula N/130.04 - 273.15) ---

    def test_temp2_at_array18(self) -> None:
        """TEMP2 at bytes[38:40]: N/130.04 - 273.15 ≈ 0 °C for N ≈ 35569."""
        raw = 35569
        pkt = _make_hk_packet(temp2_raw=raw)
        val = _parse_hk_field(pkt, 18)
        assert val == raw
        temp_c = val / 130.04 - 273.15
        assert temp_c == pytest.approx(0.0, abs=0.5)

    # --- HVMON0 (HV monitor, 1 LSB ≈ 1.22 mV, unsigned) ---

    def test_hvmon0_at_array1(self) -> None:
        """HVMON0 at bytes[4:6]; voltage = -raw x 1.22e-3 V."""
        raw = 61475  # ≈ -75 V at 1.22 mV/LSB
        pkt = _make_hk_packet(hvmon0=raw)
        val = _parse_hk_field(pkt, 1)
        assert val == raw
        voltage = -val * 1.22e-3
        assert voltage == pytest.approx(-75.0, abs=0.1)

    def test_hvmon_full_scale(self) -> None:
        """0xFFFF raw (≈ -80 V) is representable in the packet field."""
        pkt = _make_hk_packet(hvmon0=0xFFFF)
        val = _parse_hk_field(pkt, 1)
        assert val == 0xFFFF
        voltage = -val * 1.22e-3
        assert voltage == pytest.approx(-80.0, abs=0.1)

    # --- UID (4 x uint16 LE at bytes[44:52]) ---

    def test_uid_four_words(self) -> None:
        """UID occupies arrays [21..24] = bytes[44:52]."""
        uid = (0xABCD, 0x1234, 0xEF00, 0x5678)
        pkt = _make_hk_packet(uid=uid)
        for i, expected in enumerate(uid):
            assert _parse_hk_field(pkt, 21 + i) == expected, f"UID[{i}] mismatch"

    def test_uid_all_zeros(self) -> None:
        pkt = _make_hk_packet(uid=(0, 0, 0, 0))
        for i in range(4):
            assert _parse_hk_field(pkt, 21 + i) == 0

    def test_uid_bytes_at_correct_offsets(self) -> None:
        """UID[0] starts at byte 44 (array index 21 → 2 + 21*2 = 44)."""
        pkt = _make_hk_packet(uid=(0x00FF, 0, 0, 0))
        assert struct.unpack_from("<H", pkt, 44)[0] == 0x00FF

    # --- Status / PCBrev (array[25] = bytes[52:54]) ---

    def test_status_shutter_bit0(self) -> None:
        """SHUTTER = bit 0 of the low byte (byte 52)."""
        pkt = _make_hk_packet(status_pcbrev=0x0001)
        assert _parse_hk_field(pkt, 25) & 0x01

    def test_status_light_sensor_bit1(self) -> None:
        pkt = _make_hk_packet(status_pcbrev=0x0002)
        assert _parse_hk_field(pkt, 25) & 0x02

    def test_status_ext_10mhz_bit2(self) -> None:
        pkt = _make_hk_packet(status_pcbrev=0x0004)
        assert _parse_hk_field(pkt, 25) & 0x04

    def test_status_ext_1pps_bit3(self) -> None:
        pkt = _make_hk_packet(status_pcbrev=0x0008)
        assert _parse_hk_field(pkt, 25) & 0x08

    def test_pcbrev_bga_in_high_byte(self) -> None:
        """PCBrev in byte 53 (high byte of array[25]): 1 = BGA."""
        pkt = _make_hk_packet(status_pcbrev=0x0100)   # PCBrev=1 at bit 8
        raw = _parse_hk_field(pkt, 25)
        assert (raw >> 8) & 0x01 == 1

    def test_pcbrev_qfp_is_zero(self) -> None:
        pkt = _make_hk_packet(status_pcbrev=0x0000)
        raw = _parse_hk_field(pkt, 25)
        assert (raw >> 8) & 0x01 == 0

    def test_status_and_pcbrev_independent(self) -> None:
        """Status bits and PCBrev can be set simultaneously without aliasing."""
        pkt = _make_hk_packet(status_pcbrev=0x010F)   # PCBrev=1, all status bits set
        raw = _parse_hk_field(pkt, 25)
        assert raw & 0x0F == 0x0F       # all four status bits
        assert (raw >> 8) & 0x01 == 1   # PCBrev = BGA
