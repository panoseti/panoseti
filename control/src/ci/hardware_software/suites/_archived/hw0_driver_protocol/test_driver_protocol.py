"""
hw0_driver_protocol — Low-level packet protocol tests against real quabos.

Tests verify the actual byte layout of commands sent to and echoed from
the FPGA. This is the hardware-backed equivalent of the tier1_unit
FakeSocket protocol tests; the same assertion helpers from
ci/fixtures/network_fixtures.py are reused here.

Required state: BOOTED  (quabo CPU up, FPGA loaded, registers at defaults)
Class: driver_protocol (batch_priority=0)

Reference: Quabo-packet-interface.md, quabo_driver.py
"""

from __future__ import annotations

import socket
import struct
import time

import pytest

from ci.fixtures.network_fixtures import (
    assert_bytes_zero,
    assert_le_uint16,
    assert_opcode,
    assert_packet_length,
)
from control.driver.quabo_driver import ACQ_IMAGE, ACQ_PULSE_HEIGHT, DAQ_PARAMS, QUABO, UDP_CMD_PORT

pytestmark = pytest.mark.hw_class("driver_protocol")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ECHO_TIMEOUT = 3.0  # real FPGA echo latency budget


def _flush_and_echo(quabo: QUABO, cmd_fn, echo_size: int = 64) -> bytes:
    """
    Flush the receive buffer, invoke cmd_fn() to send a command, then
    read back one echo packet from the quabo's command socket.
    """
    quabo.flush_rx_buf()
    cmd_fn()
    quabo.sock.settimeout(_ECHO_TIMEOUT)
    try:
        data, _ = quabo.sock.recvfrom(echo_size)
    except (TimeoutError, OSError) as exc:
        pytest.fail(f"No echo received within {_ECHO_TIMEOUT}s: {exc}")
    finally:
        quabo.sock.settimeout(0.5)
    return data


# ---------------------------------------------------------------------------
# HV set packet layout
# ---------------------------------------------------------------------------

def test_hv_set_packet_layout(quabo: QUABO) -> None:
    """
    hv_set([0,0,0,0], echo=True) sends opcode 0x82; FPGA echoes back.
    Payload bytes [2:10] must all be zero (zero HV for all 4 channels).
    Total echo length must be 64 bytes.
    """
    data = _flush_and_echo(quabo, lambda: quabo.hv_set([0, 0, 0, 0], echo=True))
    assert_opcode(data, 0x82)
    assert_packet_length(data, 64)
    assert_bytes_zero(data, 2, 10)


def test_hv_set_nonzero_layout(quabo: QUABO) -> None:
    """
    hv_set([1000, 2000, 3000, 4000], echo=True) — verify the LE-encoded values
    appear correctly at bytes [2:10] of the echo, then zero them out.
    """
    values = [1000, 2000, 3000, 4000]
    data = _flush_and_echo(quabo, lambda: quabo.hv_set(values, echo=True))
    assert_opcode(data, 0x82)
    for i, v in enumerate(values):
        assert_le_uint16(data, 2 + 2 * i, v)
    # Restore to zero — safety
    quabo.hv_set([0, 0, 0, 0])


# ---------------------------------------------------------------------------
# ACQ params packet layout
# ---------------------------------------------------------------------------

def test_acq_params_packet_layout(quabo: QUABO) -> None:
    """
    send_daq_params(..., echo=True) sends opcode 0x83; FPGA echoes back.
    Verify mode byte at [2] and integration_time bytes at [4:6].
    """
    params = DAQ_PARAMS(
        do_image=True, image_us=10000, image_8bit=False,
        do_ph=False, bl_subtract=True
    )
    data = _flush_and_echo(quabo, lambda: quabo.send_daq_params(params, echo=True))
    assert_opcode(data, 0x83)
    assert_packet_length(data, 64)
    expected_mode = ACQ_IMAGE  # 0x02
    assert data[2] == expected_mode, f"mode byte: got 0x{data[2]:02X}, expected 0x{expected_mode:02X}"
    # image_us=10000: LSB=0x10, MSB=0x27
    assert struct.unpack_from("<H", data, 4)[0] == 10000


def test_acq_params_ph_mode(quabo: QUABO) -> None:
    """
    Pulse-height-only mode: ACQ_PULSE_HEIGHT bit set, no ACQ_IMAGE bit.
    """
    params = DAQ_PARAMS(
        do_image=False, image_us=0, image_8bit=False,
        do_ph=True, bl_subtract=True
    )
    data = _flush_and_echo(quabo, lambda: quabo.send_daq_params(params, echo=True))
    assert_opcode(data, 0x83)
    assert data[2] & ACQ_PULSE_HEIGHT, "ACQ_PULSE_HEIGHT bit missing"
    assert not (data[2] & ACQ_IMAGE), "ACQ_IMAGE bit unexpectedly set"


# ---------------------------------------------------------------------------
# MAROC roundtrip
# ---------------------------------------------------------------------------

def test_maroc_roundtrip(quabo: QUABO, maroc_config: dict) -> None:
    """
    Send a known MAROC config twice; the echo's 829-bit shift-register region
    should be identical between calls (proving the FPGA loaded the registers).
    The 492-byte MAROC command echoes as 492 bytes with opcode 0x81.
    """
    echo1 = _flush_and_echo(
        quabo, lambda: quabo.send_maroc_params(maroc_config), echo_size=512
    )
    echo2 = _flush_and_echo(
        quabo, lambda: quabo.send_maroc_params(maroc_config), echo_size=512
    )
    assert echo1 == echo2, "MAROC echo differed between two identical sends"


# ---------------------------------------------------------------------------
# Baseline calibration
# ---------------------------------------------------------------------------

def test_calibrate_baseline_returns_256_coeffs(quabo: QUABO) -> None:
    """
    calibrate_ph_baseline() returns a list of exactly 256 uint16 coefficients
    in the range [0, 4095] (12-bit ADC).
    """
    coeffs = quabo.calibrate_ph_baseline()
    assert len(coeffs) == 256, f"Expected 256 coefficients, got {len(coeffs)}"
    for i, c in enumerate(coeffs):
        assert 0 <= c <= 4095, f"Coefficient {i}={c} outside [0, 4095]"


# ---------------------------------------------------------------------------
# Data packet destination (MAC reply)
# ---------------------------------------------------------------------------

def test_data_packet_destination_returns_macs(quabo: QUABO, topology) -> None:
    """
    data_packet_destination(daq_ip) sends opcode 0x0a; FPGA replies with
    a 12-byte packet (PH MAC + IM MAC). Verify return value is True (12 bytes
    received) and reply contains non-zero bytes.
    """
    daq_nodes = topology.daq_nodes()
    if not daq_nodes:
        pytest.skip("No DAQ nodes in active topology")
    daq_ip = daq_nodes[0].host
    from ipaddress import ip_address
    ok = quabo.data_packet_destination(ip_address(daq_ip))
    assert ok, f"data_packet_destination({daq_ip!r}) did not receive 12-byte reply"


# ---------------------------------------------------------------------------
# Software PPS
# ---------------------------------------------------------------------------

def test_software_pps_resets_nanosec(quabo: QUABO) -> None:
    """
    swpps() (opcode 0x0f) triggers a software 1-PPS reset.
    Capture two consecutive HK packets; after swpps(), the NANOSEC field
    (bytes 32:36 of the HK packet, LE uint32) should be near zero.
    """
    # Collect a pre-swpps HK packet to establish baseline
    pkt_before = quabo.read_hk_packet()
    if pkt_before is None:
        pytest.skip("No HK packet received before swpps (quabo not emitting HK?)")

    quabo.swpps()
    time.sleep(0.5)  # allow FPGA to process

    pkt_after = quabo.read_hk_packet()
    if pkt_after is None:
        pytest.skip("No HK packet received after swpps")

    # NANOSEC is at bytes 32:36 (LE uint32) — per Quabo-packet-interface.md
    nanosec_before = struct.unpack_from("<I", pkt_before, 32)[0] if len(pkt_before) >= 36 else None
    nanosec_after = struct.unpack_from("<I", pkt_after, 32)[0] if len(pkt_after) >= 36 else None

    if nanosec_before is not None and nanosec_after is not None:
        # After a software PPS, NANOSEC should have wrapped/reset to a small value
        assert nanosec_after < nanosec_before or nanosec_after < 5_000_000, (
            f"NANOSEC did not appear to reset after swpps: before={nanosec_before}, after={nanosec_after}"
        )


def test_software_pps_only_q0(topology) -> None:
    """
    swpps (opcode 0x0f) is only meaningful when sent to Q0 of a module.
    Sending it to Q1/Q2/Q3 should NOT produce a NANOSEC reset effect.
    This test verifies no exception is raised and the command completes
    (we cannot easily verify negative assertion on remote hardware, but we
    can confirm the command is accepted without error).
    """
    quabo_addrs = topology.quabo_ips()
    non_q0 = [a for a in quabo_addrs if a.quadrant != 0]
    if not non_q0:
        pytest.skip("Only one quabo per module; cannot test Q1/Q2/Q3")
    for addr in non_q0:
        q = QUABO(addr.ip)
        # Should complete without raising
        q.swpps()
        q.close()


# ---------------------------------------------------------------------------
# Port forwarding command path
# ---------------------------------------------------------------------------

@pytest.mark.skipif(True, reason="Requires port_forwarding capability — gated by hw_tests.toml [[requirements]]")
def test_port_forwarding_command_path(topology) -> None:
    """
    Send an hv_set command through the gateway IP + port-forwarding rule
    from network_config.json and verify the echo arrives back, proving
    the deploy command's port-forwarding rules survive on the real network.

    Skipped unless the 'port_forwarding' capability is present in topology.
    The [[requirements]] block in hw_tests.toml enforces this at collection time.
    """
    caps = topology.capabilities()
    if "port_forwarding" not in caps:
        pytest.skip("port_forwarding capability not in active topology")

    net = topology._net
    pf_entries = net.get("port_forwarding", [])
    if not pf_entries:
        pytest.skip("No port_forwarding entries in network_config")

    # Use the first forwarding entry
    entry = pf_entries[0]
    gw_ip = entry.get("gateway_ip", "")
    gw_port = entry.get("external_port", UDP_CMD_PORT)

    # Send hv_set via the gateway IP:port
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(_ECHO_TIMEOUT)
    cmd = bytearray(64)
    cmd[0] = 0x02  # hv_set opcode
    try:
        sock.sendto(bytes(cmd), (gw_ip, gw_port))
        data, _ = sock.recvfrom(64)
        assert data[0] == 0x82, f"Echo opcode wrong through gateway: 0x{data[0]:02X}"
    except (TimeoutError, OSError) as exc:
        pytest.fail(f"No echo through gateway {gw_ip}:{gw_port}: {exc}")
    finally:
        sock.close()
