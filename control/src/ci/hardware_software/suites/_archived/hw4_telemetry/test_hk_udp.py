"""
hw4_telemetry — Low-level HK packet tests (raw UDP/60002).

Validates the 64-byte housekeeping packet byte layout emitted by quabos
every ~3 seconds. Assertions reference Quabo-packet-interface.md and
control/daemons/capture_hk.py for the canonical field layout.

Required state: BOOTED
Class: telemetry (batch_priority=0)

All tests use the `hk_socket` fixture (bound to UDP/60002) and the
HKPacketParser from hw_assertions.py.
"""

from __future__ import annotations

import socket
import struct
import time
from collections import defaultdict

import pytest

from ci.hardware_software.hw_assertions import HK_PACKET_LEN, HKPacketParser

pytestmark = pytest.mark.hw_class("telemetry")

_CAPTURE_WINDOW_S = 30.0   # sliding window for rate/count tests
_HK_CADENCE_S = 3.0        # expected HK cadence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_packets(hk_socket: socket.socket, duration: float) -> list[bytes]:
    """Collect all HK packets arriving within *duration* seconds."""
    packets: list[bytes] = []
    deadline = time.monotonic() + duration
    hk_socket.settimeout(min(1.0, duration))
    while time.monotonic() < deadline:
        try:
            data, _ = hk_socket.recvfrom(HK_PACKET_LEN + 8)
            if len(data) >= HK_PACKET_LEN:
                packets.append(data[:HK_PACKET_LEN])
        except (TimeoutError, OSError):
            pass
    return packets


# ---------------------------------------------------------------------------
# Magic byte
# ---------------------------------------------------------------------------

def test_hk_magic_byte(hk_socket: socket.socket) -> None:
    """Every HK packet must start with 0x20."""
    hk_socket.settimeout(10.0)
    try:
        data, _ = hk_socket.recvfrom(HK_PACKET_LEN)
    except (TimeoutError, OSError):
        pytest.skip("No HK packet received within 10 s (quabo not booted?)")
    HKPacketParser(data).assert_magic()


# ---------------------------------------------------------------------------
# Bootbyte on first packet after boot
# ---------------------------------------------------------------------------

def test_hk_bootbyte_first_after_boot(topology) -> None:
    """
    After a fresh power-on, exactly the first HK packet from each quabo
    must have bootbyte 0xAA; all subsequent packets must have 0x00.

    This test does NOT power-cycle the hardware; it relies on the fixture
    set leaving quabos in BOOTED state.  If the quabo has been running for
    a while, the first packet we capture may already have bootbyte 0x00 —
    we assert that case too (consistent 0x00 is also valid).
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(10.0)
    sock.bind(("", 60002))
    try:
        packets = _collect_packets(sock, 12.0)
    finally:
        sock.close()

    if not packets:
        pytest.skip("No HK packets captured")

    bootbytes = [p[1] for p in packets]
    # Either the first is 0xAA and the rest are 0x00,
    # or all are 0x00 (quabo has been up for a while).
    if bootbytes[0] == 0xAA:
        for i, bb in enumerate(bootbytes[1:], 1):
            assert bb == 0x00, f"Packet {i}: expected bootbyte 0x00 after first, got 0x{bb:02X}"
    else:
        for i, bb in enumerate(bootbytes):
            assert bb == 0x00, f"Packet {i}: unexpected bootbyte 0x{bb:02X}"


# ---------------------------------------------------------------------------
# BOARDLOC matches obs_config
# ---------------------------------------------------------------------------

def test_hk_boardloc_matches_obs_config(topology, hk_socket: socket.socket) -> None:
    """
    BOARDLOC (bytes 2:4, LE uint16) must equal module_id * 4 + quadrant
    for every quabo, matching the active obs_config.
    """
    expected = {a.boardloc for a in topology.quabo_ips()}
    seen: set[int] = set()
    packets = _collect_packets(hk_socket, _CAPTURE_WINDOW_S)
    for pkt in packets:
        parser = HKPacketParser(pkt)
        seen.add(parser.boardloc)

    # Every packet must have a BOARDLOC declared in the active obs_config
    unexpected = seen - expected
    assert not unexpected, (
        f"HK packets with unexpected BOARDLOC: {unexpected}. "
        f"Expected set: {expected}"
    )


# ---------------------------------------------------------------------------
# Inter-packet interval
# ---------------------------------------------------------------------------

def test_hk_inter_packet_interval(hk_socket: socket.socket, topology) -> None:
    """
    Sliding window over 30 s: mean inter-packet interval per quabo must be
    3 s ± 0.5 s (based on the quabo's internal 3-second HK cadence).
    """
    timestamps: dict[int, list[float]] = defaultdict(list)
    deadline = time.monotonic() + _CAPTURE_WINDOW_S
    hk_socket.settimeout(1.0)
    while time.monotonic() < deadline:
        try:
            data, _ = hk_socket.recvfrom(HK_PACKET_LEN)
            if len(data) >= 4:
                boardloc = struct.unpack_from("<H", data, 2)[0]
                timestamps[boardloc].append(time.monotonic())
        except (TimeoutError, OSError):
            pass

    if not timestamps:
        pytest.skip("No HK packets captured during interval test")

    for boardloc, ts_list in timestamps.items():
        if len(ts_list) < 3:
            continue
        deltas = [ts_list[i + 1] - ts_list[i] for i in range(len(ts_list) - 1)]
        mean_delta = sum(deltas) / len(deltas)
        assert abs(mean_delta - _HK_CADENCE_S) <= 0.5, (
            f"BOARDLOC {boardloc}: mean HK interval {mean_delta:.2f}s "
            f"outside {_HK_CADENCE_S} ± 0.5 s"
        )


# ---------------------------------------------------------------------------
# Packet length
# ---------------------------------------------------------------------------

def test_hk_packet_length_exact(hk_socket: socket.socket) -> None:
    """
    recvfrom(128) on the HK port must return exactly 64 bytes.
    """
    hk_socket.settimeout(10.0)
    try:
        data, _ = hk_socket.recvfrom(128)
    except (TimeoutError, OSError):
        pytest.skip("No HK packet within 10 s")
    assert len(data) == HK_PACKET_LEN, f"HK packet is {len(data)} bytes, expected {HK_PACKET_LEN}"


# ---------------------------------------------------------------------------
# UID matches quabo_uids.json
# ---------------------------------------------------------------------------

def test_hk_uid_matches_quabo_uids_json(hk_socket: socket.socket, quabo_uids: dict, topology) -> None:
    """
    The UID field decoded from HK packets must match the entries in quabo_uids.json.
    """
    packets = _collect_packets(hk_socket, _CAPTURE_WINDOW_S)
    if not packets:
        pytest.skip("No HK packets captured")

    for pkt in packets:
        boardloc = struct.unpack_from("<H", pkt, 2)[0]
        # UID: bytes 44..52 (4 x LE uint16) per capture_hk.py
        parts = [struct.unpack_from("<H", pkt, 44 + 2 * i)[0] for i in range(4)]
        uid_int = parts[0] + (parts[1] << 16) + (parts[2] << 32) + (parts[3] << 48)
        uid_hex = f"0x{uid_int:016X}".lower()

        if str(boardloc) in quabo_uids:
            expected_uid = quabo_uids[str(boardloc)].lower()
            assert uid_hex == expected_uid or expected_uid in uid_hex, (
                f"BOARDLOC {boardloc}: UID mismatch: packet={uid_hex!r}, config={expected_uid!r}"
            )


# ---------------------------------------------------------------------------
# PCB revision
# ---------------------------------------------------------------------------

def test_hk_pcb_revision(hk_socket: socket.socket, topology) -> None:
    """
    Bit 0 of byte 53 is the PCBrev_N field. For QFP quabos it must be 1;
    for BGA quabos it must be 0. Cross-check against obs_config quabo_version.
    """
    version_map = {}
    for dome in topology._obs.domes:
        for module in dome.modules:
            base_ip = str(module.ip_addr)
            hw_ver = module.quabo_version or "qfp"
            parts = base_ip.split(".")
            for q in range(4):
                mid = (int(parts[2]) * 256 + int(parts[3])) >> 2 & 0xFF
                version_map[mid * 4 + q] = hw_ver

    packets = _collect_packets(hk_socket, _CAPTURE_WINDOW_S)
    if not packets:
        pytest.skip("No HK packets captured")

    for pkt in packets:
        boardloc = struct.unpack_from("<H", pkt, 2)[0]
        pcb_rev = pkt[53] & 0x01
        hw_ver = version_map.get(boardloc)
        if hw_ver == "qfp":
            assert pcb_rev == 1, f"BOARDLOC {boardloc} is QFP but PCBrev_N={pcb_rev}"
        elif hw_ver == "bga":
            assert pcb_rev == 0, f"BOARDLOC {boardloc} is BGA but PCBrev_N={pcb_rev}"


# ---------------------------------------------------------------------------
# FWVER matches firmware.json
# ---------------------------------------------------------------------------

def test_hk_fwver_matches_firmware_json(hk_socket: socket.socket, topology) -> None:
    """
    The FWVER ASCII string decoded from HK bytes must match the version
    declared in firmware.json for the corresponding hardware version.
    """
    from control.utils.config_file import get_firmware_config
    fw_config = get_firmware_config()

    version_map: dict[int, str] = {}
    for dome in topology._obs.domes:
        for module in dome.modules:
            base_ip = str(module.ip_addr)
            hw_ver = module.quabo_version or "qfp"
            parts = base_ip.split(".")
            for q in range(4):
                mid = (int(parts[2]) * 256 + int(parts[3])) >> 2 & 0xFF
                version_map[mid * 4 + q] = hw_ver

    packets = _collect_packets(hk_socket, _CAPTURE_WINDOW_S)
    if not packets:
        pytest.skip("No HK packets captured")

    for pkt in packets:
        boardloc = struct.unpack_from("<H", pkt, 2)[0]
        raw29 = struct.unpack_from("<H", pkt, 60)[0]
        raw30 = struct.unpack_from("<H", pkt, 62)[0]
        try:
            fwver = bytes.fromhex(f'{raw30:04x}{raw29:04x}').decode("ASCII")
        except (ValueError, UnicodeDecodeError):
            continue  # non-ASCII FWVER not checked

        hw_ver = version_map.get(boardloc)
        if hw_ver and hw_ver in fw_config:
            expected_fwver = fw_config[hw_ver].get("version", "")
            if expected_fwver:
                assert fwver == expected_fwver, (
                    f"BOARDLOC {boardloc}: FWVER {fwver!r} != expected {expected_fwver!r}"
                )


# ---------------------------------------------------------------------------
# EXT status consistency
# ---------------------------------------------------------------------------

def test_hk_ext_status_consistency(hk_socket: socket.socket) -> None:
    """
    If EXT_10MHz_STATUS == 0, then EXT_1PPS_STATUS must also be 0.
    (1 PPS is only valid when 10 MHz reference is locked.)
    This invariant is enforced in capture_hk.py:151.
    """
    packets = _collect_packets(hk_socket, _CAPTURE_WINDOW_S)
    if not packets:
        pytest.skip("No HK packets captured")

    for pkt in packets:
        # array[25] is at bytes 52:54 (LE uint16)
        raw25 = struct.unpack_from("<H", pkt, 52)[0]
        ext_10mhz = (raw25 & 0x08) >> 3
        ext_1pps = (raw25 & 0x10) >> 4
        if ext_10mhz == 0:
            assert ext_1pps == 0, (
                f"EXT_1PPS_STATUS=1 but EXT_10MHz_STATUS=0 (raw25=0x{raw25:04X})"
            )


# ---------------------------------------------------------------------------
# Packet count per quabo
# ---------------------------------------------------------------------------

def test_hk_packet_count_per_quabo(hk_socket: socket.socket, topology) -> None:
    """
    Over 30 s, each quabo in the active topology must emit 8-12 HK packets
    (3 s cadence -> ~10 expected, +/-2 for clock drift and startup jitter).
    """
    counts: dict[int, int] = defaultdict(int)
    packets = _collect_packets(hk_socket, _CAPTURE_WINDOW_S)
    for pkt in packets:
        boardloc = struct.unpack_from("<H", pkt, 2)[0]
        counts[boardloc] += 1

    expected_boardlocs = {a.boardloc for a in topology.quabo_ips()}
    for bl in expected_boardlocs:
        n = counts.get(bl, 0)
        assert 8 <= n <= 12, (
            f"BOARDLOC {bl}: expected 8-12 HK packets in {_CAPTURE_WINDOW_S}s, got {n}"
        )
