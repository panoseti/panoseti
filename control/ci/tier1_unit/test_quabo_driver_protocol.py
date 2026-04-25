"""
test_quabo_driver_protocol.py

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
from typing import Any, ClassVar

import pytest

from control.driver.quabo_driver import (
    ACQ_IMAGE,
    ACQ_IMAGE_8BIT,
    ACQ_NO_BASELINE_SUBTRACT,
    ACQ_PULSE_HEIGHT,
    DAQ_PARAMS,
    QUABO,
    SERIAL_COMMAND_LENGTH,
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


# ===========================================================================
# TestHvVoltageConversion
# ===========================================================================

class TestHvSetMath:
    """Math tests for HV DAC encoding in hv_set().

    Conversion: voltage_V = -raw x 1.22e-3
    (1 LSB ≈ 1.22 mV; 0xFFFF ≈ -80 V)
    """

    def test_encoding_for_approx_75v(self, quabo_and_sock) -> None:
        """Raw 61475 → bytes[2:4] = 61475, decoded as ≈ -75 V."""
        q, sock = quabo_and_sock
        raw = 61475
        q.hv_set([raw, 0, 0, 0])
        encoded = struct.unpack_from("<H", sock.last_cmd, 2)[0]
        assert encoded == raw
        assert -encoded * 1.22e-3 == pytest.approx(-75.0, abs=0.1)

    def test_all_four_channels_independent(self, quabo_and_sock) -> None:
        """Four distinct raw values land in the correct 2-byte LE slots."""
        q, sock = quabo_and_sock
        values = [100, 200, 300, 400]
        q.hv_set(values)
        for i, v in enumerate(values):
            assert struct.unpack_from("<H", sock.last_cmd, 2 + i * 2)[0] == v

    def test_zero_voltage(self, quabo_and_sock) -> None:
        """All-zero raw → bytes[2:10] are zero."""
        q, sock = quabo_and_sock
        q.hv_set([0, 0, 0, 0])
        for i in range(4):
            assert struct.unpack_from("<H", sock.last_cmd, 2 + i * 2)[0] == 0

    def test_full_scale_encoding(self, quabo_and_sock) -> None:
        """0xFFFF in every channel encodes correctly (≈ -80 V)."""
        q, sock = quabo_and_sock
        q.hv_set([0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF])
        for i in range(4):
            assert struct.unpack_from("<H", sock.last_cmd, 2 + i * 2)[0] == 0xFFFF


# ===========================================================================
# TestAcqModeBitmask
# ===========================================================================

class TestAcqModeBitmask:
    """ACQ command byte[2] mode-flag combinations not covered by test_quabo_driver.py."""

    def test_all_modes_off_byte_is_zero(self, quabo_and_sock) -> None:
        """No image, no PH, bl_subtract=True → mode byte = 0."""
        q, sock = quabo_and_sock
        p = DAQ_PARAMS(do_image=False, image_us=0, image_8bit=False,
                       do_ph=False, bl_subtract=True)
        q.send_daq_params(p)
        assert sock.last_cmd[2] == 0x00

    def test_image_ph_8bit_all_set(self, quabo_and_sock) -> None:
        """image + PH + 8bit bits all present simultaneously."""
        q, sock = quabo_and_sock
        p = DAQ_PARAMS(do_image=True, image_us=50000, image_8bit=True,
                       do_ph=True, bl_subtract=True)
        q.send_daq_params(p)
        mode = sock.last_cmd[2]
        assert mode & ACQ_IMAGE
        assert mode & ACQ_PULSE_HEIGHT
        assert mode & ACQ_IMAGE_8BIT
        assert not (mode & ACQ_NO_BASELINE_SUBTRACT)

    def test_no_baseline_subtract_bit_when_disabled(self, quabo_and_sock) -> None:
        """bl_subtract=False → ACQ_NO_BASELINE_SUBTRACT inhibit bit is set."""
        q, sock = quabo_and_sock
        p = DAQ_PARAMS(do_image=True, image_us=1000, image_8bit=False,
                       do_ph=False, bl_subtract=False)
        q.send_daq_params(p)
        assert sock.last_cmd[2] & ACQ_NO_BASELINE_SUBTRACT

    def test_integration_time_10000us(self, quabo_and_sock) -> None:
        """image_us=10000 (0x2710) encoded LE at bytes[4:6]."""
        q, sock = quabo_and_sock
        p = DAQ_PARAMS(True, 10000, False, False, True)
        q.send_daq_params(p)
        encoded = struct.unpack_from("<H", sock.last_cmd, 4)[0]
        assert encoded == 10000
        assert encoded == 0x2710   # cross-check against the literal hex value

    def test_integration_time_zero(self, quabo_and_sock) -> None:
        """image_us=0 → bytes[4:6] are zero."""
        q, sock = quabo_and_sock
        p = DAQ_PARAMS(False, 0, False, True, True)
        q.send_daq_params(p)
        assert struct.unpack_from("<H", sock.last_cmd, 4)[0] == 0

    def test_integration_time_max_uint16(self, quabo_and_sock) -> None:
        """image_us=65535 (max uint16) encodes without truncation."""
        q, sock = quabo_and_sock
        p = DAQ_PARAMS(True, 65535, False, False, True)
        q.send_daq_params(p)
        assert struct.unpack_from("<H", sock.last_cmd, 4)[0] == 65535


# ===========================================================================
# TestMarocPacketStructure
# ===========================================================================

class TestMarocPacketStructure:
    """MAROC 492-byte command structural tests.

    Layout:
      byte[0]        = opcode 0x01
      bytes[ 4:108]  = ASIC 0 data (104 bytes)
      bytes[108:132] = gap / zero padding (24 bytes)
      bytes[132:236] = ASIC 1 data (104 bytes)
      bytes[236:260] = gap / zero padding
      bytes[260:364] = ASIC 2 data (104 bytes)
      bytes[364:388] = gap / zero padding
      bytes[388:492] = ASIC 3 data (104 bytes)
    """

    # ASIC region boundaries: start[i] = 4 + i * 128
    _ASIC_STARTS: ClassVar[list[int]] = [4, 132, 260, 388]
    _ASIC_LEN = 104
    _GAP_REGIONS: ClassVar[list[tuple[int, int]]] = [(108, 132), (236, 260), (364, 388)]

    def test_packet_length(self, quabo_and_sock) -> None:
        q, sock = quabo_and_sock
        q.send_maroc_params(_minimal_maroc_config())
        assert len(sock.last_cmd) == 492

    def test_opcode_is_0x01(self, quabo_and_sock) -> None:
        q, sock = quabo_and_sock
        q.send_maroc_params(_minimal_maroc_config())
        assert sock.last_cmd[0] == 0x01

    def test_serial_command_length_constant(self) -> None:
        """SERIAL_COMMAND_LENGTH = 829 bits encodes 4 MAROC shift registers."""
        assert SERIAL_COMMAND_LENGTH == 829
        # Each ASIC region (104 bytes = 832 bits) must be wide enough
        assert self._ASIC_LEN * 8 >= SERIAL_COMMAND_LENGTH

    def test_four_asic_region_boundaries(self, quabo_and_sock) -> None:
        """ASIC regions start at 4, 132, 260, 388 and each fit in the 492-byte packet."""
        q, _sock = quabo_and_sock
        q.send_maroc_params(_minimal_maroc_config())
        for start in self._ASIC_STARTS:
            assert start + self._ASIC_LEN <= 492

    def test_gap_bytes_are_zero_with_zero_config(self, quabo_and_sock) -> None:
        """With all-zero MAROC config the 24-byte gaps between ASIC regions are zero."""
        q, sock = quabo_and_sock
        q.send_maroc_params(_minimal_maroc_config())
        cmd = sock.last_cmd
        for gap_start, gap_end in self._GAP_REGIONS:
            for idx in range(gap_start, gap_end):
                assert cmd[idx] == 0, (
                    f"Gap byte at index {idx} = {cmd[idx]:#04x} (expected 0)"
                )

    def test_destination_port(self, quabo_and_sock) -> None:
        """MAROC command is sent to UDP_CMD_PORT (60000)."""
        q, sock = quabo_and_sock
        q.send_maroc_params(_minimal_maroc_config())
        assert sock.last_dest[1] == UDP_CMD_PORT


# ===========================================================================
# TestTriggerMaskMultiChannel
# ===========================================================================

class TestTriggerMaskMultiChannel:
    """send_trigger_mask() encoding for all CHANMASK_0 … CHANMASK_8.

    Layout: CHANMASK_N at bytes [4 + N*4 : 8 + N*4] as LE uint32.
    """

    def test_all_channels_enabled(self, quabo_and_sock) -> None:
        """All CHANMASK_N = 0xFFFFFFFF → every 4-byte slot is all-ones."""
        q, sock = quabo_and_sock
        config = {f"CHANMASK_{i}": 0xFFFFFFFF for i in range(9)}
        q.send_trigger_mask(config, do_flush_rx_buf=False)
        cmd = sock.last_cmd
        for i in range(9):
            encoded = struct.unpack_from("<I", cmd, 4 + i * 4)[0]
            assert encoded == 0xFFFFFFFF, f"CHANMASK_{i} not all-ones"

    def test_all_channels_disabled(self, quabo_and_sock) -> None:
        """All CHANMASK_N = 0 → bytes 4-39 all zero."""
        q, sock = quabo_and_sock
        config = {f"CHANMASK_{i}": 0 for i in range(9)}
        q.send_trigger_mask(config, do_flush_rx_buf=False)
        for i in range(9):
            assert struct.unpack_from("<I", sock.last_cmd, 4 + i * 4)[0] == 0

    def test_single_channel_no_aliasing(self, quabo_and_sock) -> None:
        """Only CHANMASK_3 non-zero → all other slots remain zero."""
        q, sock = quabo_and_sock
        config = {f"CHANMASK_{i}": 0 for i in range(9)}
        config["CHANMASK_3"] = 0xDEADBEEF
        q.send_trigger_mask(config, do_flush_rx_buf=False)
        cmd = sock.last_cmd
        assert struct.unpack_from("<I", cmd, 4 + 3 * 4)[0] == 0xDEADBEEF
        for i in [x for x in range(9) if x != 3]:
            assert struct.unpack_from("<I", cmd, 4 + i * 4)[0] == 0, (
                f"CHANMASK_{i} should be 0 but got {struct.unpack_from('<I', cmd, 4 + i*4)[0]:#010x}"
            )

    def test_last_channel_chanmask8(self, quabo_and_sock) -> None:
        """CHANMASK_8 (OR-mask / last channel) at bytes[36:40]."""
        q, sock = quabo_and_sock
        config = {f"CHANMASK_{i}": 0 for i in range(9)}
        config["CHANMASK_8"] = 0x0000FFFF
        q.send_trigger_mask(config, do_flush_rx_buf=False)
        assert struct.unpack_from("<I", sock.last_cmd, 4 + 8 * 4)[0] == 0x0000FFFF

    def test_all_distinct_values(self, quabo_and_sock) -> None:
        """Nine distinct values each encode in their own slot without overlap."""
        q, sock = quabo_and_sock
        config = {f"CHANMASK_{i}": 0xAA000000 | i for i in range(9)}
        q.send_trigger_mask(config, do_flush_rx_buf=False)
        cmd = sock.last_cmd
        for i in range(9):
            assert struct.unpack_from("<I", cmd, 4 + i * 4)[0] == (0xAA000000 | i)

    def test_opcode_and_length(self, quabo_and_sock) -> None:
        q, sock = quabo_and_sock
        q.send_trigger_mask({f"CHANMASK_{i}": 0 for i in range(9)},
                             do_flush_rx_buf=False)
        assert sock.last_cmd[0] == 0x06
        assert len(sock.last_cmd) == 64
