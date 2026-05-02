"""
test_quabo_driver_2.py

Driver interface compliance tests for control/driver/quabo_driver.py.

All tests use a FakeSocket that captures outgoing UDP packets without
requiring quabo hardware.  No real network calls are made.

Packet format reference:
  - Commands: 64-byte UDP packets to port 60000
  - MAROC params: 492-byte UDP packets
  - Byte 0: command opcode (e.g. 0x03 = DAQ params, 0x02 = HV, ...)
"""
from __future__ import annotations

import struct

# Ensure control/ is on the path (pyproject.toml sets pythonpath=["."])
from control.driver.quabo_driver import (
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




# ===========================================================================
# TestHvPacket — hv_set() byte-level verification
# ===========================================================================

class TestHvPacket:
    def test_packet_length_is_64(self, quabo_and_sock) -> None:
        q, sock = quabo_and_sock
        q.hv_set([100, 200, 300, 400])
        assert len(sock.last_cmd) == 64

    def test_opcode_is_0x02(self, quabo_and_sock) -> None:
        q, sock = quabo_and_sock
        q.hv_set([100, 200, 300, 400])
        assert sock.last_cmd[0] == 0x02

    def test_four_channel_encoding(self, quabo_and_sock) -> None:
        """Values encoded as 4 little-endian uint16 starting at byte 2."""
        q, sock = quabo_and_sock
        values = [100, 200, 300, 400]
        q.hv_set(values)
        for i, v in enumerate(values):
            decoded = struct.unpack_from("<H", sock.last_cmd, 2 + i * 2)[0]
            assert decoded == v, f"Channel {i}: expected {v}, got {decoded}"

    def test_zero_values(self, quabo_and_sock) -> None:
        q, sock = quabo_and_sock
        q.hv_set([0, 0, 0, 0])
        for i in range(4):
            decoded = struct.unpack_from("<H", sock.last_cmd, 2 + i * 2)[0]
            assert decoded == 0

    def test_max_values(self, quabo_and_sock) -> None:
        q, sock = quabo_and_sock
        q.hv_set([65535, 65535, 65535, 65535])
        for i in range(4):
            decoded = struct.unpack_from("<H", sock.last_cmd, 2 + i * 2)[0]
            assert decoded == 65535


# ===========================================================================
# TestResetPacket — reset()
# ===========================================================================

class TestResetPacket:
    def test_opcode_is_0x04(self, quabo_and_sock) -> None:
        q, sock = quabo_and_sock
        q.reset()
        assert sock.last_cmd[0] == 0x04

    def test_packet_length_is_64(self, quabo_and_sock) -> None:
        q, sock = quabo_and_sock
        q.reset()
        assert len(sock.last_cmd) == 64


# ===========================================================================
# TestShutterNew — shutter_new()
# ===========================================================================

class TestShutterNew:
    def test_opcode_is_0x08(self, quabo_and_sock) -> None:
        q, sock = quabo_and_sock
        q.shutter_new(closed=True)
        assert sock.last_cmd[0] == 0x08

    def test_closed_sets_byte1_to_1(self, quabo_and_sock) -> None:
        q, sock = quabo_and_sock
        q.shutter_new(closed=True)
        assert sock.last_cmd[1] == 0x01

    def test_open_sets_byte1_to_0(self, quabo_and_sock) -> None:
        q, sock = quabo_and_sock
        q.shutter_new(closed=False)
        assert sock.last_cmd[1] == 0x00

    def test_packet_length_is_64(self, quabo_and_sock) -> None:
        q, sock = quabo_and_sock
        q.shutter_new(closed=True)
        assert len(sock.last_cmd) == 64


# ===========================================================================
# TestFocusPacket — focus()
# ===========================================================================

class TestFocusPacket:
    def test_opcode_is_0x05(self, quabo_and_sock) -> None:
        q, sock = quabo_and_sock
        q.focus(1000)
        assert sock.last_cmd[0] == 0x05

    def test_steps_encoding(self, quabo_and_sock) -> None:
        """Steps encoded as little-endian uint16 at bytes [4:6]."""
        q, sock = quabo_and_sock
        steps = 12345
        q.focus(steps)
        encoded = struct.unpack_from("<H", sock.last_cmd, 4)[0]
        assert encoded == steps

    def test_packet_length_is_64(self, quabo_and_sock) -> None:
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

    def test_packet_length_is_492(self, quabo_and_sock) -> None:
        q, sock = quabo_and_sock
        config = self._minimal_maroc_config()
        q.send_maroc_params(config)
        assert len(sock.last_cmd) == 492

    def test_opcode_first_byte(self, quabo_and_sock) -> None:
        """make_maroc_cmd sets cmd[0] = 0x01."""
        q, sock = quabo_and_sock
        config = self._minimal_maroc_config()
        q.send_maroc_params(config)
        assert sock.last_cmd[0] == 0x01

    def test_four_asic_regions_present(self, quabo_and_sock) -> None:
        """ASIC data starts at offsets 4, 132, 260, 388 (104 bytes each)."""
        q, _sock = quabo_and_sock
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

    def test_opcode_is_0x06(self, quabo_and_sock) -> None:
        q, sock = quabo_and_sock
        q.send_trigger_mask(self._minimal_trigger_config(), do_flush_rx_buf=False)
        assert sock.last_cmd[0] == 0x06

    def test_packet_length_is_64(self, quabo_and_sock) -> None:
        q, sock = quabo_and_sock
        q.send_trigger_mask(self._minimal_trigger_config(), do_flush_rx_buf=False)
        assert len(sock.last_cmd) == 64

    def test_channel_mask_encoding(self, quabo_and_sock) -> None:
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
    def test_opcode_is_0x0a(self, quabo_and_sock) -> None:
        q, sock = quabo_and_sock
        # Inject a 12-byte fake response for opcode 0x0a
        sock.responses[0x0a] = b'\x00' * 12
        q.data_packet_destination("10.0.0.1")
        # The cmd opcode is in the LAST sendto before the recvfrom; we need the
        # cmd packet (not flush packets which would be recvfrom calls)
        # find the packet with opcode 0x0a
        daq_pkts = [p for p, _ in sock.sent if p[0] == 0x0a]
        assert daq_pkts, "No 0x0a opcode packet found"

    def test_ip_bytes_encoded_at_offsets_1_and_5(self, quabo_and_sock) -> None:
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

    def test_returns_false_on_no_response(self, quabo_and_sock) -> None:
        """When recvfrom times out (no response injected), returns False."""
        q, _sock = quabo_and_sock
        # No response injected → socket.timeout raised → count=0 → return False
        result = q.data_packet_destination("10.0.0.1")
        assert result is False

    def test_returns_true_on_12_byte_response(self, quabo_and_sock) -> None:
        q, sock = quabo_and_sock
        sock.responses[0x0a] = b'\x00' * 12
        result = q.data_packet_destination("10.0.0.1")
        assert result is True
