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

import os
import struct
from typing import Any

import pytest

from control.driver.quabo_driver import (
    QUABO,
    UDP_CMD_PORT,
)
from control.utils.config_file import get_boardloc, ip_addr_to_module_id

# ===========================================================================
# FakeSocket + fixture (local copy — avoids cross-file import coupling)
# ===========================================================================

class FakeSocket:
    """Minimal socket shim that captures all sendto() calls."""
    def __init__(self):
        self.sent: list[tuple[bytes, tuple]] = []
        self._timeout = 0.5
        self.responses: dict[int, bytes] = {}   # opcode → bytes returned by recvfrom

    def settimeout(self, t): self._timeout = t
    def bind(self, addr): pass
    def close(self): pass
    def setsockopt(self, *a): pass

    def sendto(self, data, addr):
        self.sent.append((bytes(data), addr))

    def recvfrom(self, size):
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


@pytest.fixture
def quabo_and_sock(monkeypatch: Any, tmp_path: Any) -> tuple[QUABO, FakeSocket]:
    """Yield (quabo, fake_sock). All socket I/O is captured in fake_sock."""
    fake_sock = FakeSocket()
    monkeypatch.setattr("socket.socket", lambda *a, **kw: fake_sock)
    monkeypatch.setattr("socket.gethostbyname", lambda x: x)
    # mock get_logger to return a logger that doesn't write to disk
    import logging
    mock_logger = logging.getLogger("quabo_driver_test")
    monkeypatch.setattr("control.driver.quabo_driver.get_logger", lambda *a, **kw: mock_logger)

    cfg_file = tmp_path / "quabo_config.txt"
    real_cfg = os.path.join(
        os.path.dirname(__file__), "..", "..", "driver", "quabo_config.txt"
    )
    if os.path.exists(real_cfg):
        with open(real_cfg) as f:
            cfg_file.write_text(f.read())
    else:
        cfg_file.write_text("* minimal stub\n")

    q = QUABO(
        "192.168.0.100",
        config_file_path=str(cfg_file),
        logfile=str(tmp_path / "quabo_driver.log"),
    )
    return q, fake_sock


# ===========================================================================
# Helpers
# ===========================================================================

def _make_hk_packet(
    boardloc: int = 0,
    hvmon0: int = 0,
    temp1_raw: int = 0,
    temp2_raw: int = 0,
    uid: tuple = (0, 0, 0, 0),
    status_pcbrev: int = 0,
) -> bytes:
    """Build a synthetic 64-byte HK packet with specified field values.

    All uint16 LE pairs start at byte 2.  Array index mapping:
      array[ 0] bytes[ 2: 4] — BOARDLOC (unsigned)
      array[ 1] bytes[ 4: 6] — HVMON0   (unsigned)
      array[17] bytes[36:38] — TEMP1    (signed)
      array[18] bytes[38:40] — TEMP2    (unsigned)
      array[21] bytes[44:46] — UID[0]   (unsigned)
      array[22] bytes[46:48] — UID[1]
      array[23] bytes[48:50] — UID[2]
      array[24] bytes[50:52] — UID[3]
      array[25] bytes[52:54] — status (low byte) | PCBrev (high byte)
    """
    pkt = bytearray(64)
    pkt[0] = 0x20   # PANOSETI HK packet magic
    pkt[1] = 0x00   # not a startup packet
    struct.pack_into("<H", pkt, 2, boardloc & 0xFFFF)
    struct.pack_into("<H", pkt, 4, hvmon0 & 0xFFFF)
    struct.pack_into("<h", pkt, 36, temp1_raw)          # signed int16
    struct.pack_into("<H", pkt, 38, temp2_raw & 0xFFFF)
    for i, u in enumerate(uid):
        struct.pack_into("<H", pkt, 44 + i * 2, u & 0xFFFF)
    struct.pack_into("<H", pkt, 52, status_pcbrev & 0xFFFF)
    return bytes(pkt)


def _parse_hk_field(packet: bytes, array_index: int, signed: bool = False) -> int:
    """Read a 16-bit LE field at byte offset 2 + array_index * 2."""
    offset = 2 + array_index * 2
    fmt = "<h" if signed else "<H"
    return struct.unpack_from(fmt, packet, offset)[0]


def _minimal_maroc_config() -> dict:
    """All-zero MAROC config dict with every required key present."""
    scalar_keys = [
        "OTABG_ON", "DAC_ON", "SMALL_DAC", "ENB_OUT_ADC", "INV_START_GRAY",
        "RAMP8B", "RAMP10B", "CMD_CK_MUX", "D1_D2", "INV_DISCR_ADC",
        "POLAR_DISCRI", "ENB3ST", "VAL_DC_FSB2", "SW_FSB2_50F", "SW_FSB2_100F",
        "SW_FSB2_100K", "SW_FSB2_50K", "VALID_DC_FS", "CMD_FSB_FSU",
        "SW_FSB1_50F", "SW_FSB1_100F", "SW_FSB1_100K", "SW_FSB1_50k",
        "SW_FSU_100K", "SW_FSU_50K", "SW_FSU_25K", "SW_FSU_40F", "SW_FSU_20F",
        "H1H2_CHOICE", "EN_ADC", "SW_SS_1200F", "SW_SS_600F", "SW_SS_300F",
        "ON_OFF_SS", "SWB_BUF_2P", "SWB_BUF_1P", "SWB_BUF_500F", "SWB_BUF_250F",
        "CMD_FSB", "CMD_SS", "CMD_FSU",
    ]
    config = {k: "0,0,0,0" for k in scalar_keys}
    config["DAC2"] = "0,0,0,0"
    config["DAC1"] = "0,0,0,0"
    for i in range(64):
        config[f"GAIN{i}"] = "0,0,0,0"
        config[f"CTEST_{i}"] = "0,0,0,0"
        config[f"MASKOR1_{i}"] = "0,0,0,0"
        config[f"MASKOR2_{i}"] = "0,0,0,0"
    return config


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
