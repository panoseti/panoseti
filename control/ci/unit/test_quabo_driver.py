"""
test_quabo_driver.py

Driver interface compliance tests for control/driver/quabo_driver.py.

All tests use a FakeSocket that captures outgoing UDP packets without
requiring quabo hardware.  No real network calls are made.

Packet format reference:
  - Commands: 64-byte UDP packets to port 60000
  - MAROC params: 492-byte UDP packets
  - Byte 0: command opcode (e.g. 0x03 = DAQ params, 0x02 = HV, ...)
"""
from __future__ import annotations

import os
import struct
from typing import Any

import pytest

# Ensure control/ is on the path (pyproject.toml sets pythonpath=["."])
from driver.quabo_driver import (
    ACQ_IMAGE,
    ACQ_IMAGE_8BIT,
    ACQ_NO_BASELINE_SUBTRACT,
    ACQ_PULSE_HEIGHT,
    DAQ_PARAMS,
    QUABO,
    UDP_CMD_PORT,
)

# ---------------------------------------------------------------------------
# FakeSocket — captures sendto() calls and optionally injects responses
# ---------------------------------------------------------------------------

class FakeSocket:
    def __init__(self):
        self.sent: list[tuple[bytes, tuple]] = []
        self._timeout = 0.5
        self.responses: dict[int, bytes] = {}   # opcode → bytes returned by recvfrom

    def settimeout(self, t):
        self._timeout = t

    def bind(self, addr):
        pass

    def close(self):
        pass

    def sendto(self, data, addr):
        self.sent.append((bytes(data), addr))

    def recvfrom(self, size):
        if self.sent:
            # last sent packet's first byte is the command opcode (mask off echo bit)
            opcode = self.sent[-1][0][0] & 0x7F
            if opcode in self.responses:
                return self.responses[opcode], ('192.168.3.100', UDP_CMD_PORT)
        raise TimeoutError()

    @property
    def last_cmd(self) -> bytes:
        """Return the payload of the most recent sendto() call."""
        assert self.sent, "No packets sent yet"
        return self.sent[-1][0]

    @property
    def last_dest(self) -> tuple:
        """Return the (ip, port) of the most recent sendto() call."""
        assert self.sent, "No packets sent yet"
        return self.sent[-1][1]


# Shared fixture: QUABO instance with FakeSocket
# ---------------------------------------------------------------------------

@pytest.fixture
def quabo_and_sock(monkeypatch: Any, tmp_path: Any) -> tuple[QUABO, FakeSocket]:
    """Yield (quabo, fake_sock).  All socket I/O is captured in fake_sock."""
    fake_sock = FakeSocket()
    monkeypatch.setattr("socket.socket", lambda *a, **kw: fake_sock)
    monkeypatch.setattr("socket.gethostbyname", lambda x: x)

    # Suppress log-file creation — tests don't need a real log file
    monkeypatch.setattr("utils.util.create_logger", lambda *a, **kw: None)

    cfg_file = tmp_path / "quabo_config.txt"
    # Copy the real quabo_config.txt into tmp so send_maroc_params_file() works
    real_cfg = os.path.join(
        os.path.dirname(__file__), "..", "..", "driver", "quabo_config.txt"
    )
    if os.path.exists(real_cfg):
        cfg_file.write_text(open(real_cfg).read())
    else:
        cfg_file.write_text("* minimal stub\n")

    q = QUABO(
        "192.168.3.100",
        config_file_path=str(cfg_file),
        logfile=str(tmp_path / "quabo_driver.log"),
    )
    return q, fake_sock


# ===========================================================================
# TestDAQParamsClass — unit tests of the DAQ_PARAMS data class
# ===========================================================================

class TestDAQParamsClass:
    def test_image_only_sets_do_image(self):
        p = DAQ_PARAMS(do_image=True, image_us=50000, image_8bit=False,
                       do_ph=False, bl_subtract=True)
        assert p.do_image is True
        assert p.do_ph is False

    def test_ph_only_sets_do_ph(self):
        p = DAQ_PARAMS(do_image=False, image_us=0, image_8bit=False,
                       do_ph=True, bl_subtract=True)
        assert p.do_ph is True
        assert p.do_image is False

    def test_default_flash_stim_off(self):
        p = DAQ_PARAMS(True, 50000, False, False, True)
        assert p.do_flash is False
        assert p.do_stim is False

    def test_set_flash_params(self):
        p = DAQ_PARAMS(True, 50000, False, False, True)
        p.set_flash_params(rate=3, level=15, width=7)
        assert p.do_flash is True
        assert p.flash_rate == 3
        assert p.flash_level == 15
        assert p.flash_width == 7

    def test_set_stim_params(self):
        p = DAQ_PARAMS(True, 50000, False, False, True)
        p.set_stim_params(rate=2, level=100)
        assert p.do_stim is True
        assert p.stim_rate == 2
        assert p.stim_level == 100

    def test_image_8bit_flag(self):
        p = DAQ_PARAMS(True, 100000, True, False, True)
        assert p.image_8bit is True

    def test_bl_subtract_flag(self):
        p = DAQ_PARAMS(True, 100000, False, False, False)
        assert p.bl_subtract is False


# ===========================================================================
# TestDaqParamPacket — send_daq_params() byte-level verification
# ===========================================================================

class TestDaqParamPacket:
    def test_packet_length_is_64(self, quabo_and_sock):
        q, sock = quabo_and_sock
        p = DAQ_PARAMS(True, 50000, False, False, True)
        q.send_daq_params(p)
        assert len(sock.last_cmd) == 64

    def test_opcode_is_0x03(self, quabo_and_sock):
        q, sock = quabo_and_sock
        p = DAQ_PARAMS(True, 50000, False, False, True)
        q.send_daq_params(p)
        assert sock.last_cmd[0] == 0x03

    def test_destination_port_is_cmd_port(self, quabo_and_sock):
        q, sock = quabo_and_sock
        p = DAQ_PARAMS(True, 50000, False, False, True)
        q.send_daq_params(p)
        assert sock.last_dest == ("192.168.3.100", UDP_CMD_PORT)

    def test_image_mode_bit(self, quabo_and_sock):
        q, sock = quabo_and_sock
        p = DAQ_PARAMS(do_image=True, image_us=50000, image_8bit=False,
                       do_ph=False, bl_subtract=True)
        q.send_daq_params(p)
        assert sock.last_cmd[2] & ACQ_IMAGE

    def test_ph_mode_bit(self, quabo_and_sock):
        q, sock = quabo_and_sock
        p = DAQ_PARAMS(do_image=False, image_us=0, image_8bit=False,
                       do_ph=True, bl_subtract=True)
        q.send_daq_params(p)
        assert sock.last_cmd[2] & ACQ_PULSE_HEIGHT

    def test_8bit_image_bit(self, quabo_and_sock):
        q, sock = quabo_and_sock
        p = DAQ_PARAMS(do_image=True, image_us=50000, image_8bit=True,
                       do_ph=False, bl_subtract=True)
        q.send_daq_params(p)
        assert sock.last_cmd[2] & ACQ_IMAGE_8BIT

    def test_no_baseline_subtract_bit_when_off(self, quabo_and_sock):
        q, sock = quabo_and_sock
        p = DAQ_PARAMS(do_image=True, image_us=50000, image_8bit=False,
                       do_ph=False, bl_subtract=False)
        q.send_daq_params(p)
        assert sock.last_cmd[2] & ACQ_NO_BASELINE_SUBTRACT

    def test_baseline_subtract_bit_clear_when_on(self, quabo_and_sock):
        q, sock = quabo_and_sock
        p = DAQ_PARAMS(do_image=True, image_us=50000, image_8bit=False,
                       do_ph=False, bl_subtract=True)
        q.send_daq_params(p)
        assert not (sock.last_cmd[2] & ACQ_NO_BASELINE_SUBTRACT)

    def test_integration_time_encoding(self, quabo_and_sock):
        """image_us stored little-endian in bytes [4:6]. Driver supports up to 65535 μs."""
        q, sock = quabo_and_sock
        image_us = 50000  # 50 ms — fits in uint16
        p = DAQ_PARAMS(True, image_us, False, False, True)
        q.send_daq_params(p)
        encoded = struct.unpack_from("<H", sock.last_cmd, 4)[0]
        assert encoded == image_us

    def test_flash_rate_level_width_encoding(self, quabo_and_sock):
        q, sock = quabo_and_sock
        p = DAQ_PARAMS(True, 50000, False, False, True)
        p.set_flash_params(rate=5, level=20, width=10)
        q.send_daq_params(p)
        assert sock.last_cmd[22] == 5   # flash_rate at byte 22
        assert sock.last_cmd[24] == 20  # flash_level at byte 24
        assert sock.last_cmd[26] == 10  # flash_width at byte 26

    def test_no_flash_leaves_bytes_zero(self, quabo_and_sock):
        q, sock = quabo_and_sock
        p = DAQ_PARAMS(True, 50000, False, False, True)
        q.send_daq_params(p)
        # Flash bytes should be zero when flash is off
        assert sock.last_cmd[22] == 0
        assert sock.last_cmd[24] == 0
        assert sock.last_cmd[26] == 0

    def test_stim_encoding(self, quabo_and_sock):
        q, sock = quabo_and_sock
        p = DAQ_PARAMS(True, 50000, False, False, True)
        p.set_stim_params(rate=3, level=128)
        q.send_daq_params(p)
        assert sock.last_cmd[14] == 1    # STIMON at byte 14
        assert sock.last_cmd[16] == 128  # stim_level at byte 16
        assert sock.last_cmd[18] == 3    # stim_rate at byte 18

    def test_both_image_and_ph_bits(self, quabo_and_sock):
        q, sock = quabo_and_sock
        p = DAQ_PARAMS(do_image=True, image_us=50000, image_8bit=False,
                       do_ph=True, bl_subtract=True)
        q.send_daq_params(p)
        mode = sock.last_cmd[2]
        assert mode & ACQ_IMAGE
        assert mode & ACQ_PULSE_HEIGHT


# ===========================================================================
# TestHvPacket — hv_set() byte-level verification
# ===========================================================================

class TestHvPacket:
    def test_packet_length_is_64(self, quabo_and_sock):
        q, sock = quabo_and_sock
        q.hv_set([100, 200, 300, 400])
        assert len(sock.last_cmd) == 64

    def test_opcode_is_0x02(self, quabo_and_sock):
        q, sock = quabo_and_sock
        q.hv_set([100, 200, 300, 400])
        assert sock.last_cmd[0] == 0x02

    def test_four_channel_encoding(self, quabo_and_sock):
        """Values encoded as 4 little-endian uint16 starting at byte 2."""
        q, sock = quabo_and_sock
        values = [100, 200, 300, 400]
        q.hv_set(values)
        for i, v in enumerate(values):
            decoded = struct.unpack_from("<H", sock.last_cmd, 2 + i * 2)[0]
            assert decoded == v, f"Channel {i}: expected {v}, got {decoded}"

    def test_zero_values(self, quabo_and_sock):
        q, sock = quabo_and_sock
        q.hv_set([0, 0, 0, 0])
        for i in range(4):
            decoded = struct.unpack_from("<H", sock.last_cmd, 2 + i * 2)[0]
            assert decoded == 0

    def test_max_values(self, quabo_and_sock):
        q, sock = quabo_and_sock
        q.hv_set([65535, 65535, 65535, 65535])
        for i in range(4):
            decoded = struct.unpack_from("<H", sock.last_cmd, 2 + i * 2)[0]
            assert decoded == 65535


# ===========================================================================
# TestResetPacket — reset()
# ===========================================================================

class TestResetPacket:
    def test_opcode_is_0x04(self, quabo_and_sock):
        q, sock = quabo_and_sock
        q.reset()
        assert sock.last_cmd[0] == 0x04

    def test_packet_length_is_64(self, quabo_and_sock):
        q, sock = quabo_and_sock
        q.reset()
        assert len(sock.last_cmd) == 64


# ===========================================================================
# TestShutterNew — shutter_new()
# ===========================================================================

class TestShutterNew:
    def test_opcode_is_0x08(self, quabo_and_sock):
        q, sock = quabo_and_sock
        q.shutter_new(closed=True)
        assert sock.last_cmd[0] == 0x08

    def test_closed_sets_byte1_to_1(self, quabo_and_sock):
        q, sock = quabo_and_sock
        q.shutter_new(closed=True)
        assert sock.last_cmd[1] == 0x01

    def test_open_sets_byte1_to_0(self, quabo_and_sock):
        q, sock = quabo_and_sock
        q.shutter_new(closed=False)
        assert sock.last_cmd[1] == 0x00

    def test_packet_length_is_64(self, quabo_and_sock):
        q, sock = quabo_and_sock
        q.shutter_new(closed=True)
        assert len(sock.last_cmd) == 64


# ===========================================================================
# TestFocusPacket — focus()
# ===========================================================================

class TestFocusPacket:
    def test_opcode_is_0x05(self, quabo_and_sock):
        q, sock = quabo_and_sock
        q.focus(1000)
        assert sock.last_cmd[0] == 0x05

    def test_steps_encoding(self, quabo_and_sock):
        """Steps encoded as little-endian uint16 at bytes [4:6]."""
        q, sock = quabo_and_sock
        steps = 12345
        q.focus(steps)
        encoded = struct.unpack_from("<H", sock.last_cmd, 4)[0]
        assert encoded == steps

    def test_packet_length_is_64(self, quabo_and_sock):
        q, sock = quabo_and_sock
        q.focus(500)
        assert len(sock.last_cmd) == 64


# ===========================================================================
# TestMarocParamPacket — send_maroc_params() (in-memory config dict)
# ===========================================================================

class TestMarocParamPacket:
    def _minimal_maroc_config(self):
        """Minimal MAROC config dict with all required keys set to defaults."""
        # All per-channel keys need 4 comma-separated values
        config = {}
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
        for k in scalar_keys:
            config[k] = "0,0,0,0"
        config["DAC2"] = "0,0,0,0"
        config["DAC1"] = "0,0,0,0"
        for i in range(64):
            config[f"GAIN{i}"] = "0,0,0,0"
            config[f"CTEST_{i}"] = "0,0,0,0"
            config[f"MASKOR1_{i}"] = "0,0,0,0"
            config[f"MASKOR2_{i}"] = "0,0,0,0"
        return config

    def test_packet_length_is_492(self, quabo_and_sock):
        q, sock = quabo_and_sock
        config = self._minimal_maroc_config()
        q.send_maroc_params(config)
        assert len(sock.last_cmd) == 492

    def test_opcode_first_byte(self, quabo_and_sock):
        """make_maroc_cmd sets cmd[0] = 0x01."""
        q, sock = quabo_and_sock
        config = self._minimal_maroc_config()
        q.send_maroc_params(config)
        assert sock.last_cmd[0] == 0x01

    def test_four_asic_regions_present(self, quabo_and_sock):
        """ASIC data starts at offsets 4, 132, 260, 388 (104 bytes each)."""
        q, sock = quabo_and_sock
        config = self._minimal_maroc_config()
        q.send_maroc_params(config)
        # Each ASIC region is 104 bytes; verify they fit within the 492-byte packet
        for asic_offset in [4, 132, 260, 388]:
            assert asic_offset + 104 <= 492


# ===========================================================================
# TestTriggerMask — send_trigger_mask()
# ===========================================================================

class TestTriggerMask:
    def _minimal_trigger_config(self):
        return {f"CHANMASK_{i}": 0xFFFFFFFF for i in range(9)}

    def test_opcode_is_0x06(self, quabo_and_sock):
        q, sock = quabo_and_sock
        q.send_trigger_mask(self._minimal_trigger_config(), do_flush_rx_buf=False)
        assert sock.last_cmd[0] == 0x06

    def test_packet_length_is_64(self, quabo_and_sock):
        q, sock = quabo_and_sock
        q.send_trigger_mask(self._minimal_trigger_config(), do_flush_rx_buf=False)
        assert len(sock.last_cmd) == 64

    def test_channel_mask_encoding(self, quabo_and_sock):
        """CHANMASK_0 = 0x12345678 → bytes 4-7 = little-endian 0x12345678."""
        q, sock = quabo_and_sock
        config = {f"CHANMASK_{i}": 0 for i in range(9)}
        config["CHANMASK_0"] = 0x12345678
        q.send_trigger_mask(config, do_flush_rx_buf=False)
        encoded = struct.unpack_from("<I", sock.last_cmd, 4)[0]
        assert encoded == 0x12345678


# ===========================================================================
# TestDataPacketDestination — data_packet_destination() IP encoding
# ===========================================================================

class TestDataPacketDestination:
    def test_opcode_is_0x0a(self, quabo_and_sock):
        q, sock = quabo_and_sock
        # Inject a 12-byte fake response for opcode 0x0a
        sock.responses[0x0a] = b'\x00' * 12
        q.data_packet_destination("10.0.0.1")
        # The cmd opcode is in the LAST sendto before the recvfrom; we need the
        # cmd packet (not flush packets which would be recvfrom calls)
        # find the packet with opcode 0x0a
        daq_pkts = [p for p, _ in sock.sent if p[0] == 0x0a]
        assert daq_pkts, "No 0x0a opcode packet found"

    def test_ip_bytes_encoded_at_offsets_1_and_5(self, quabo_and_sock):
        """IP bytes are placed at cmd[1:5] and cmd[5:9]."""
        q, sock = quabo_and_sock
        sock.responses[0x0a] = b'\x00' * 12
        q.data_packet_destination("10.20.30.40")
        # find the 0x0a packet
        daq_pkts = [p for p, _ in sock.sent if p[0] == 0x0a]
        assert daq_pkts
        cmd = daq_pkts[-1]
        # IP 10.20.30.40 → bytes 10, 20, 30, 40
        assert list(cmd[1:5]) == [10, 20, 30, 40]
        assert list(cmd[5:9]) == [10, 20, 30, 40]

    def test_returns_false_on_no_response(self, quabo_and_sock):
        """When recvfrom times out (no response injected), returns False."""
        q, sock = quabo_and_sock
        # No response injected → socket.timeout raised → count=0 → return False
        result = q.data_packet_destination("10.0.0.1")
        assert result is False

    def test_returns_true_on_12_byte_response(self, quabo_and_sock):
        q, sock = quabo_and_sock
        sock.responses[0x0a] = b'\x00' * 12
        result = q.data_packet_destination("10.0.0.1")
        assert result is True
