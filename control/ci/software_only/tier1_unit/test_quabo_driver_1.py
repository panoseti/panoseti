"""
test_quabo_driver_1.py

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
from unittest.mock import MagicMock

import pytest

# Ensure control/ is on the path (pyproject.toml sets pythonpath=["."])
from control.driver.quabo_driver import (
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
    monkeypatch.setattr("control.driver.quabo_driver.get_logger", lambda *a, **kw: MagicMock())

    cfg_file = tmp_path / "quabo_config.txt"
    # Copy the real quabo_config.txt into tmp so send_maroc_params_file() works
    real_cfg = os.path.join(
        os.path.dirname(__file__), "..", "..", "driver", "quabo_config.txt"
    )
    if os.path.exists(real_cfg):
        with open(real_cfg) as f:
            cfg_file.write_text(f.read())
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
    def test_image_only_sets_do_image(self) -> None:
        p = DAQ_PARAMS(do_image=True, image_us=50000, image_8bit=False,
                       do_ph=False, bl_subtract=True)
        assert p.do_image is True
        assert p.do_ph is False

    def test_ph_only_sets_do_ph(self) -> None:
        p = DAQ_PARAMS(do_image=False, image_us=0, image_8bit=False,
                       do_ph=True, bl_subtract=True)
        assert p.do_ph is True
        assert p.do_image is False

    def test_default_flash_stim_off(self) -> None:
        p = DAQ_PARAMS(True, 50000, False, False, True)
        assert p.do_flash is False
        assert p.do_stim is False

    def test_set_flash_params(self) -> None:
        p = DAQ_PARAMS(True, 50000, False, False, True)
        p.set_flash_params(rate=3, level=15, width=7)
        assert p.do_flash is True
        assert p.flash_rate == 3
        assert p.flash_level == 15
        assert p.flash_width == 7

    def test_set_stim_params(self) -> None:
        p = DAQ_PARAMS(True, 50000, False, False, True)
        p.set_stim_params(rate=2, level=100)
        assert p.do_stim is True
        assert p.stim_rate == 2
        assert p.stim_level == 100

    def test_image_8bit_flag(self) -> None:
        p = DAQ_PARAMS(True, 100000, True, False, True)
        assert p.image_8bit is True

    def test_bl_subtract_flag(self) -> None:
        p = DAQ_PARAMS(True, 100000, False, False, False)
        assert p.bl_subtract is False


# ===========================================================================
# TestDaqParamPacket — send_daq_params() byte-level verification
# ===========================================================================

class TestDaqParamPacket:
    def test_packet_length_is_64(self, quabo_and_sock) -> None:
        q, sock = quabo_and_sock
        p = DAQ_PARAMS(True, 50000, False, False, True)
        q.send_daq_params(p)
        assert len(sock.last_cmd) == 64

    def test_opcode_is_0x03(self, quabo_and_sock) -> None:
        q, sock = quabo_and_sock
        p = DAQ_PARAMS(True, 50000, False, False, True)
        q.send_daq_params(p)
        assert sock.last_cmd[0] == 0x03

    def test_destination_port_is_cmd_port(self, quabo_and_sock) -> None:
        q, sock = quabo_and_sock
        p = DAQ_PARAMS(True, 50000, False, False, True)
        q.send_daq_params(p)
        assert sock.last_dest == ("192.168.3.100", UDP_CMD_PORT)

    def test_image_mode_bit(self, quabo_and_sock) -> None:
        q, sock = quabo_and_sock
        p = DAQ_PARAMS(do_image=True, image_us=50000, image_8bit=False,
                       do_ph=False, bl_subtract=True)
        q.send_daq_params(p)
        assert sock.last_cmd[2] & ACQ_IMAGE

    def test_ph_mode_bit(self, quabo_and_sock) -> None:
        q, sock = quabo_and_sock
        p = DAQ_PARAMS(do_image=False, image_us=0, image_8bit=False,
                       do_ph=True, bl_subtract=True)
        q.send_daq_params(p)
        assert sock.last_cmd[2] & ACQ_PULSE_HEIGHT

    def test_8bit_image_bit(self, quabo_and_sock) -> None:
        q, sock = quabo_and_sock
        p = DAQ_PARAMS(do_image=True, image_us=50000, image_8bit=True,
                       do_ph=False, bl_subtract=True)
        q.send_daq_params(p)
        assert sock.last_cmd[2] & ACQ_IMAGE_8BIT

    def test_no_baseline_subtract_bit_when_off(self, quabo_and_sock) -> None:
        q, sock = quabo_and_sock
        p = DAQ_PARAMS(do_image=True, image_us=50000, image_8bit=False,
                       do_ph=False, bl_subtract=False)
        q.send_daq_params(p)
        assert sock.last_cmd[2] & ACQ_NO_BASELINE_SUBTRACT

    def test_baseline_subtract_bit_clear_when_on(self, quabo_and_sock) -> None:
        q, sock = quabo_and_sock
        p = DAQ_PARAMS(do_image=True, image_us=50000, image_8bit=False,
                       do_ph=False, bl_subtract=True)
        q.send_daq_params(p)
        assert not (sock.last_cmd[2] & ACQ_NO_BASELINE_SUBTRACT)

    def test_integration_time_encoding(self, quabo_and_sock) -> None:
        """image_us stored little-endian in bytes [4:6]. Driver supports up to 65535 μs."""
        q, sock = quabo_and_sock
        image_us = 50000  # 50 ms — fits in uint16
        p = DAQ_PARAMS(True, image_us, False, False, True)
        q.send_daq_params(p)
        encoded = struct.unpack_from("<H", sock.last_cmd, 4)[0]
        assert encoded == image_us

    def test_flash_rate_level_width_encoding(self, quabo_and_sock) -> None:
        q, sock = quabo_and_sock
        p = DAQ_PARAMS(True, 50000, False, False, True)
        p.set_flash_params(rate=5, level=20, width=10)
        q.send_daq_params(p)
        assert sock.last_cmd[22] == 5   # flash_rate at byte 22
        assert sock.last_cmd[24] == 20  # flash_level at byte 24
        assert sock.last_cmd[26] == 10  # flash_width at byte 26

    def test_no_flash_leaves_bytes_zero(self, quabo_and_sock) -> None:
        q, sock = quabo_and_sock
        p = DAQ_PARAMS(True, 50000, False, False, True)
        q.send_daq_params(p)
        # Flash bytes should be zero when flash is off
        assert sock.last_cmd[22] == 0
        assert sock.last_cmd[24] == 0
        assert sock.last_cmd[26] == 0

    def test_stim_encoding(self, quabo_and_sock) -> None:
        q, sock = quabo_and_sock
        p = DAQ_PARAMS(True, 50000, False, False, True)
        p.set_stim_params(rate=3, level=128)
        q.send_daq_params(p)
        assert sock.last_cmd[14] == 1    # STIMON at byte 14
        assert sock.last_cmd[16] == 128  # stim_level at byte 16
        assert sock.last_cmd[18] == 3    # stim_rate at byte 18

    def test_both_image_and_ph_bits(self, quabo_and_sock) -> None:
        q, sock = quabo_and_sock
        p = DAQ_PARAMS(do_image=True, image_us=50000, image_8bit=False,
                       do_ph=True, bl_subtract=True)
        q.send_daq_params(p)
        mode = sock.last_cmd[2]
        assert mode & ACQ_IMAGE
        assert mode & ACQ_PULSE_HEIGHT
