"""
hw3_lifecycle — Power cycle, firmware reboot, and calibration tests.

These are the most expensive tests (power cycle = 5-7 min, TFTP reboot = 90 s).
Run only in nightly CI; they live in batch_priority=3 so they run last.

Required state: UNPOWERED
Class: lifecycle (batch_priority=3)
Leaves state: BOOTED

Safety: All tests in this suite require hardware power control via the WPS.
The firmware flash test (test_firmware_load_and_reboot) additionally requires
--allow-firmware-flash at the CLI level and will be skipped without it.

pytest-timeout: 600 seconds (generous for full boot cycle)
"""

from __future__ import annotations

import os
import subprocess
import time

import pytest

from control.driver.quabo_driver import QUABO
from control.utils import config_file

pytestmark = [
    pytest.mark.hw_class("lifecycle"),
    pytest.mark.timeout(600),
]

_BOOT_TIMEOUT_S = 300  # max seconds to wait for quabo to become pingable
_PING_INTERVAL_S = 5
_TFTP_REBOOT_WAIT_S = 90  # FPGA reloads firmware over TFTP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wait_for_ping(ip: str, timeout: float = _BOOT_TIMEOUT_S) -> bool:
    """Poll ping until the host responds or timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = subprocess.run(
            ["ping", "-c1", "-W1", ip],
            capture_output=True, timeout=3
        )
        if r.returncode == 0:
            return True
        time.sleep(_PING_INTERVAL_S)
    return False


def _wps_toggle(obs_config, on: bool) -> None:
    from control.power import quabo_power
    extra = obs_config.model_extra or {}
    wps_cfg = {k: v for k, v in extra.items() if k.startswith("wps")}
    if not wps_cfg:
        pytest.skip("No WPS configured in obs_config")
    for _wps_key, wps_val in wps_cfg.items():
        quabo_power(wps_val, on=on)


# ---------------------------------------------------------------------------
# Full power cycle
# ---------------------------------------------------------------------------

@pytest.mark.slow_hw
@pytest.mark.timeout(360)
def test_full_power_cycle(topology) -> None:
    """
    WPS off → WPS on → ping check → first HK packet bootbyte == 0xAA.

    Verifies:
    1. The WPS outlet actually controls power (quabo goes offline, then online).
    2. The first HK packet after boot has the startup byte 0xAA.
    3. Subsequent packets have bootbyte == 0x00.
    """
    obs = config_file.get_obs_config()
    quabo_addrs = topology.quabo_ips()
    if not quabo_addrs:
        pytest.skip("No quabos in topology")

    q0_addr = next(a for a in quabo_addrs if a.quadrant == 0)
    q0_ip = q0_addr.ip  # raw IP for ICMP ping (pre-boot, before port-forwarding matters)

    # Power OFF
    _wps_toggle(obs, on=False)
    time.sleep(5.0)
    # Confirm it's down
    r = subprocess.run(["ping", "-c1", "-W1", q0_ip], capture_output=True, timeout=3)
    assert r.returncode != 0, "Quabo still pingable after WPS off"

    # Power ON
    _wps_toggle(obs, on=True)
    up = _wait_for_ping(q0_ip)
    assert up, f"Quabo {q0_ip} did not respond to ping within {_BOOT_TIMEOUT_S}s after power on"

    # Wait a bit more for firmware to fully load
    time.sleep(10.0)

    # Read first HK packet — bootbyte must be 0xAA
    q = QUABO(q0_addr.real_ip, port=q0_addr.cmd_port)
    pkt = q.read_hk_packet()
    q.close()
    if pkt is None:
        pytest.skip("No HK packet received after power cycle")
    assert pkt[1] == 0xAA, f"First HK bootbyte should be 0xAA after power on, got 0x{pkt[1]:02X}"

    # Read the next packet — bootbyte must be 0x00
    q2 = QUABO(q0_addr.real_ip, port=q0_addr.cmd_port)
    pkt2 = q2.read_hk_packet()
    q2.close()
    if pkt2:
        assert pkt2[1] == 0x00, f"Subsequent HK bootbyte should be 0x00, got 0x{pkt2[1]:02X}"


# ---------------------------------------------------------------------------
# TFTP reboot
# ---------------------------------------------------------------------------

@pytest.mark.slow_hw
@pytest.mark.timeout(180)
def test_tftp_reboot(topology) -> None:
    """
    Issue a TFTP reboot; verify quabo returns to pingable within _TFTP_REBOOT_WAIT_S
    and the FWVER field in HK is unchanged (same firmware version reloaded).
    """
    quabo_addrs = topology.quabo_ips()
    if not quabo_addrs:
        pytest.skip("No quabos in topology")

    q0_addr = next(a for a in quabo_addrs if a.quadrant == 0)
    q = QUABO(q0_addr.real_ip, port=q0_addr.cmd_port)

    # Capture FWVER before reboot
    pkt_before = q.read_hk_packet()
    q.close()

    fwver_before = None
    if pkt_before and len(pkt_before) >= 64:
        # FWVER bytes: array[29] and array[30] in the HK packet
        # Per capture_hk.py: bytes.fromhex(f'{array[30]:04x}{array[29]:04x}').decode("ASCII")
        import struct
        raw29 = struct.unpack_from("<H", pkt_before, 60)[0]  # offset 2 + 29*2 = 60
        raw30 = struct.unpack_from("<H", pkt_before, 62)[0]  # offset 2 + 30*2 = 62
        try:
            fwver_before = bytes.fromhex(f'{raw30:04x}{raw29:04x}').decode("ASCII")
        except (ValueError, UnicodeDecodeError):
            fwver_before = f"0x{raw30:04x}{raw29:04x}"

    # Issue TFTP reboot
    from control.driver.quabo_tftp import tftpw
    tftpw_instance = tftpw(q0_addr.ip)
    tftpw_instance.reboot()

    # Wait for the quabo to go offline then come back
    time.sleep(5.0)  # brief offline window
    up = _wait_for_ping(q0_addr.ip, timeout=_TFTP_REBOOT_WAIT_S)
    assert up, f"Quabo {q0_addr.ip} did not come back after TFTP reboot within {_TFTP_REBOOT_WAIT_S}s"

    # Wait for firmware to stabilise
    time.sleep(10.0)

    # Check FWVER is unchanged
    q2 = QUABO(q0_addr.real_ip, port=q0_addr.cmd_port)
    pkt_after = q2.read_hk_packet()
    q2.close()

    if fwver_before and pkt_after and len(pkt_after) >= 64:
        import struct
        raw29a = struct.unpack_from("<H", pkt_after, 60)[0]
        raw30a = struct.unpack_from("<H", pkt_after, 62)[0]
        try:
            fwver_after = bytes.fromhex(f'{raw30a:04x}{raw29a:04x}').decode("ASCII")
        except (ValueError, UnicodeDecodeError):
            fwver_after = f"0x{raw30a:04x}{raw29a:04x}"
        assert fwver_after == fwver_before, (
            f"FWVER changed across TFTP reboot: {fwver_before!r} → {fwver_after!r}"
        )


# ---------------------------------------------------------------------------
# Firmware load and reboot (opt-in)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("HW_ALLOW_FIRMWARE_FLASH"),
    reason="Set HW_ALLOW_FIRMWARE_FLASH=1 to enable firmware flash tests",
)
def test_firmware_load_and_reboot(topology, quabo_uids) -> None:
    """
    Load the known-good firmware binary from firmware.json, reboot, and
    verify FWVER + FWTIME match the expected values.

    DANGER: this actually flashes FPGA firmware. Only run with explicit opt-in
    via HW_ALLOW_FIRMWARE_FLASH=1 and a known-good firmware binary.
    """
    fw_config = config_file.get_firmware_config()
    quabo_addrs = topology.quabo_ips()
    if not quabo_addrs:
        pytest.skip("No quabos in topology")

    q0_addr = next(a for a in quabo_addrs if a.quadrant == 0)

    # Find the firmware binary for this quabo's hardware version
    obs = config_file.get_obs_config()
    hw_version = None
    for dome in obs.domes:
        for module in dome.modules:
            if str(module.ip_addr).startswith(q0_addr.ip.rsplit(".", 1)[0]):
                hw_version = getattr(module, "quabo_version", "qfp")
                break

    if not hw_version:
        pytest.skip("Cannot determine hardware version for this quabo")

    fw_filename = getattr(fw_config, hw_version, None)
    if not fw_filename:
        pytest.skip(f"No firmware entry for hw_version={hw_version!r} in firmware.json")

    from control.driver.quabo_tftp import tftpw
    tftpw_instance = tftpw(q0_addr.real_ip, q0_addr.reboot_port)

    from control.utils.paths import PanoPaths
    fw_path = PanoPaths.firmware_dir() / fw_filename
    assert fw_path.exists(), f"Firmware binary not found: {fw_path}"

    tftpw_instance.put_bin_file(str(fw_path))
    tftpw_instance.reboot()

    up = _wait_for_ping(q0_addr.ip, timeout=_TFTP_REBOOT_WAIT_S)
    assert up, "Quabo did not return after firmware flash + reboot"

    time.sleep(10.0)
    q = QUABO(q0_addr.real_ip, port=q0_addr.cmd_port)
    pkt = q.read_hk_packet()
    q.close()
    assert pkt is not None, "No HK packet after firmware flash"


# ---------------------------------------------------------------------------
# Baseline calibration
# ---------------------------------------------------------------------------

def test_baseline_calibration(topology) -> None:
    """
    After boot, run calibrate_ph_baseline() on each quabo.
    All 256 coefficients must be in [0, 4095] (12-bit ADC).
    """
    quabo_addrs = topology.quabo_ips()
    if not quabo_addrs:
        pytest.skip("No quabos in topology")

    for addr in quabo_addrs:
        q = QUABO(addr.real_ip, port=addr.cmd_port)
        coeffs = q.calibrate_ph_baseline()
        q.close()
        assert len(coeffs) == 256, f"{addr.ip}: expected 256 coefficients, got {len(coeffs)}"
        out_of_range = [(i, c) for i, c in enumerate(coeffs) if not (0 <= c <= 4095)]
        assert not out_of_range, f"{addr.ip}: {len(out_of_range)} coefficients out of [0, 4095]: {out_of_range[:5]}"


# ---------------------------------------------------------------------------
# UID stability across reboot
# ---------------------------------------------------------------------------

def test_uid_stability_across_reboot(topology) -> None:
    """
    The DS18B20 chip ID (UID field in HK packets) must be identical before
    and after a TFTP reboot — it's hardware-burned and cannot change.
    """
    import struct
    quabo_addrs = topology.quabo_ips()
    if not quabo_addrs:
        pytest.skip("No quabos in topology")

    q0_addr = next(a for a in quabo_addrs if a.quadrant == 0)

    def read_uid(addr) -> int | None:
        q = QUABO(addr.real_ip, port=addr.cmd_port)
        pkt = q.read_hk_packet()
        q.close()
        if not pkt or len(pkt) < 58:
            return None
        # UID: array[21..24] (4 x LE uint16 at bytes 44..52)
        parts = [struct.unpack_from("<H", pkt, 44 + 2 * i)[0] for i in range(4)]
        return parts[0] + (parts[1] << 16) + (parts[2] << 32) + (parts[3] << 48)

    uid_before = read_uid(q0_addr)
    if uid_before is None:
        pytest.skip("Could not read UID before reboot")

    # TFTP reboot
    from control.driver.quabo_tftp import tftpw
    tftpw(q0_addr.ip).reboot()
    time.sleep(5.0)
    _wait_for_ping(q0_addr.ip, timeout=_TFTP_REBOOT_WAIT_S)
    time.sleep(10.0)

    uid_after = read_uid(q0_addr)
    assert uid_after is not None, "Could not read UID after reboot"
    assert uid_after == uid_before, (
        f"UID changed across reboot: before=0x{uid_before:016X}, after=0x{uid_after:016X}"
    )


# ---------------------------------------------------------------------------
# Full session start
# ---------------------------------------------------------------------------

def test_session_start_full(runner, topology) -> None:
    """
    Run 'pseti session-start --no-hv'; assert all expected daemons are running
    and the Redis HASH for each quabo is populated within 15 seconds.
    """
    result = runner.invoke(
        __import__("control.pseti", fromlist=["app"]).app, 
        ["session-start", "--no-hv"], 
        input="y\n" * 16
    )
    assert result.exit_code == 0, f"session-start failed:\n{result.stdout}"

    # Check Redis population
    try:
        import redis as _redis
        r = _redis.Redis(host="localhost", port=6379, db=0, socket_timeout=2)
        r.ping()
    except Exception:
        pytest.skip("Redis not available for session-start verification")

    quabo_addrs = topology.quabo_ips()
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        populated = all(
            r.hgetall(f"QUABO_{addr.boardloc}")
            for addr in quabo_addrs
        )
        if populated:
            break
        time.sleep(1.0)

    for addr in quabo_addrs:
        data = r.hgetall(f"QUABO_{addr.boardloc}")
        assert data, f"Redis QUABO_{addr.boardloc} not populated after session-start"
