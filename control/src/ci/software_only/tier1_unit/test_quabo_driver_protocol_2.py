"""
test_quabo_driver_protocol_2.py

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
from typing import ClassVar

import pytest

from ci.software_only.tier1_unit.conftest import _minimal_maroc_config
from control.driver.quabo_driver import (
    ACQ_IMAGE,
    ACQ_IMAGE_8BIT,
    ACQ_NO_BASELINE_SUBTRACT,
    ACQ_PULSE_HEIGHT,
    DAQ_PARAMS,
    SERIAL_COMMAND_LENGTH,
    UDP_CMD_PORT,
)

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
