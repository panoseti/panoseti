"""
hw_boot — Boot-sequence validation split into one function per stage.

Each function is an independent pytest item, so failures are pinpointed to
the exact stage and subsequent stages are cascade-skipped (via
pytest_plugin.pytest_runtest_setup).

Boot stages:
  test_boot_00  — pseti power off: guarantee a clean baseline.
  test_boot_01  — WPS power on + outlet confirmation.
  test_boot_02  — Boot wait (60 s default): timed sleep with 15 s progress logs.
  test_boot_03  — UID discovery via TFTP: per-quabo table with real_ip:port.
  test_boot_04  — pseti cfg reboot: TFTP firmware load; skip-check; write BOOTED.
  test_boot_05  — Post-reboot command-port reachability.
  test_boot_06  — Echo check: each quabo echoes back an hv_set command.
  test_boot_07  — pseti cfg hk-dest: route HK packets to head node.
  test_boot_08  — pseti cfg redis-daemons: start capture_hk and Redis clients.
  test_boot_09  — pseti cfg maroc-config: program MAROC ASICs.
  test_boot_10  — pseti cfg mask-config: load trigger masks.
  test_boot_11  — pseti cfg calibrate-ph: PH calibration + coefficient check.

Required state: UNPOWERED (test_boot_00 powers off first regardless).
Leaves state:   PH_CALIBRATED.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import time

import pytest

from ci.hardware_software.hw_utils.topology import HwTopology
from ci.hardware_software.hw_utils.driver_ops import check_all_reachable
from control.pseti import app
from control.utils import util

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.hw_class("boot_sequence"),
    pytest.mark.slow_hw,
    pytest.mark.timeout(900),
]

_BOOT_WAIT_S = int(os.environ.get("HW_TEST_QUABO_BOOT_WAIT", 60))




def _log_topology_targets(topo: HwTopology) -> None:
    for a in topo.quabo_ips():
        logger.info(
            "[BOOT] quabo map: raw=%-18s  real=%-18s  cmd_port=%-6d  reboot_port=%d",
            a.ip, a.real_ip, a.cmd_port, a.reboot_port,
        )

# ---------------------------------------------------------------------------
# Stage 00 — Force power off
# ---------------------------------------------------------------------------

def test_boot_00_power_off(runner, topology) -> None:
    """Guarantee a clean hardware baseline by powering off first."""
    quabo_addrs = topology.quabo_ips()
    if not quabo_addrs:
        pytest.skip("No quabos in active topology")

    _log_topology_targets(topology)

    logger.info("[BOOT] Stage 00: pseti power off")
    r = runner.invoke(app, ["power", "off"])
    assert r.exit_code == 0, f"[BOOT] Stage 00 FAILED: pseti power off:\n{r.output}"
    logger.info("[BOOT] Stage 00 passed")
    time.sleep(5)


# ---------------------------------------------------------------------------
# Stage 01 — WPS power on
# ---------------------------------------------------------------------------

def test_boot_01_wps_power_on(runner, topology) -> None:
    """Power on all WPS outlets and confirm each reports ON."""
    from ci.hardware_software.hw_utils.driver_ops import wps_power_on
    from control.power import quabo_power_query

    logger.info("[BOOT] Stage 01: WPS power on")
    wps_power_on()

    wps_errors = []
    for wps in topology.wps_outlets():
        state = quabo_power_query({"url": wps.url, "quabo_socket": wps.quabo_socket})
        if state != "true":
            wps_errors.append(
                f"{wps.name} ({wps.url}): power query returned {state!r} (expected 'true')"
            )
        else:
            logger.info("[BOOT] Stage 01: %s confirmed ON", wps.name)

    assert not wps_errors, (
        f"[BOOT] Stage 01 FAILED: WPS did not confirm ON for {len(wps_errors)} outlet(s):\n"
        + "\n".join(wps_errors)
    )
    logger.info("[BOOT] Stage 01 passed")


# ---------------------------------------------------------------------------
# Stage 02 — Boot wait
# ---------------------------------------------------------------------------

def test_boot_02_boot_wait(runner, topology) -> None:
    """Wait for quabos to enter TFTP bootloader (configurable via HW_TEST_QUABO_BOOT_WAIT)."""
    logger.info("[BOOT] Stage 02: waiting %ds for TFTP bootloader", _BOOT_WAIT_S)
    elapsed = 0
    while elapsed < _BOOT_WAIT_S:
        chunk = min(15, _BOOT_WAIT_S - elapsed)
        time.sleep(chunk)
        elapsed += chunk
        logger.info("[BOOT] Stage 02: %ds / %ds elapsed", elapsed, _BOOT_WAIT_S)
    logger.info("[BOOT] Stage 02 passed")


# ---------------------------------------------------------------------------
# Stage 03 — UID discovery via TFTP
# ---------------------------------------------------------------------------

def test_boot_03_uid_discovery(runner, topology) -> None:
    """Discover quabo UIDs via TFTP and assert all are populated."""
    import control.get_uids as get_uids
    from control.utils import config_file
    from control.utils import util as _util

    logger.info("[BOOT] Stage 03: get_uids (TFTP to FPGA bootloader)")
    obs_config = config_file.get_obs_config()
    network_config = config_file.get_network_config()

    quabo_uids = get_uids.get_uids(obs_config, network_config)

    uid_rows: list[str] = []
    uid_errors: list[str] = []
    for dome in quabo_uids.domes:
        for module in dome.modules:
            for quadrant, entry in enumerate(module.quabos):
                ip_ports = _util.get_quabo_ip_port(module.ip_addr, quadrant, network_config)
                status = f"UID={entry.uid!r}" if entry.uid else "OFFLINE (empty UID)"
                uid_rows.append(
                    f"  module {module.ip_addr} Q{quadrant} "
                    f"→ {ip_ports.ip_addr}:{ip_ports.reboot_port}  {status}"
                )
                if not entry.uid:
                    uid_errors.append(
                        f"module {module.ip_addr} Q{quadrant}: UID empty — quabo offline or "
                        f"TFTP unreachable at {ip_ports.ip_addr}:{ip_ports.reboot_port}"
                    )

    logger.info("[BOOT] Stage 03 UID table:\n%s", "\n".join(uid_rows))
    assert not uid_errors, (
        f"[BOOT] Stage 03 FAILED: get_uids did not populate UIDs for "
        f"{len(uid_errors)} quabo(s):\n" + "\n".join(uid_errors)
    )
    logger.info("[BOOT] Stage 03 passed: all %d UIDs populated", len(topology.quabo_ips()))


# ---------------------------------------------------------------------------
# Stage 04 — TFTP reboot (loads main firmware) + skip check
# ---------------------------------------------------------------------------

def test_boot_04_tftp_reboot(runner, topology) -> None:
    """Load main firmware via TFTP reboot and assert no quabos were skipped."""
    import control.config as config
    from ci.hardware_software.hw_utils.cli import _STATE_FILE
    from ci.hardware_software.hw_utils.state_machine import _write_state
    from control.utils import config_file

    obs_config = config_file.get_obs_config()
    network_config = config_file.get_network_config()
    modules = config_file.get_modules(obs_config)
    quabo_uids_cached = config_file.get_quabo_uids()

    logger.info("[BOOT] Stage 04: TFTP reboot via config.do_reboot")
    config.do_reboot(modules, quabo_uids_cached, network_config)

    skipped = [
        f"module {m.ip_addr} Q{q_idx}: UID was empty → do_reboot skipped it"
        for d in quabo_uids_cached.domes
        for m in d.modules
        for q_idx, e in enumerate(m.quabos)
        if not e.uid
    ]
    assert not skipped, (
        f"[BOOT] Stage 04 FAILED: do_reboot skipped {len(skipped)} quabo(s):\n"
        + "\n".join(skipped)
    )

    _write_state(_STATE_FILE, "BOOTED")
    logger.info("[BOOT] Stage 04 passed")


# ---------------------------------------------------------------------------
# Stage 05 — Post-reboot command-port reachability
# ---------------------------------------------------------------------------

def test_boot_05_post_reboot_reachability(runner, topology) -> None:
    """Assert every quabo is reachable on its command port after firmware load."""
    logger.info("[BOOT] Stage 05: post-reboot reachability check")
    errors = check_all_reachable(topology)
    assert not errors, (
        f"[BOOT] Stage 05 FAILED: {len(errors)} quabo(s) unreachable after reboot:\n"
        + "\n".join(errors)
    )
    logger.info("[BOOT] Stage 05 passed: all %d quabo(s) reachable", len(topology.quabo_ips()))


# ---------------------------------------------------------------------------
# Stage 06 — Echo check
# ---------------------------------------------------------------------------

def test_boot_06_echo_check(runner, topology) -> None:
    """Assert every quabo echoes back an hv_set command."""
    from control.driver.quabo_driver import QUABO

    logger.info("[BOOT] Stage 06: echo check")
    quabo_addrs = topology.quabo_ips()
    echo_errors = []
    for a in quabo_addrs:
        q = QUABO(a.real_ip, port=a.cmd_port)
        try:
            q.sock.settimeout(3.0)
            q.flush_rx_buf()
            q.hv_set([0, 0, 0, 0], echo=True)
            data, _ = q.sock.recvfrom(64)
            if not data:
                echo_errors.append(f"{a.ip} (loc={a.boardloc}): empty echo response")
        except (TimeoutError, OSError) as exc:
            echo_errors.append(f"{a.ip} (loc={a.boardloc}): echo timeout: {exc}")
        finally:
            q.close()

    assert not echo_errors, (
        f"[BOOT] Stage 06 FAILED: {len(echo_errors)} quabo(s) did not echo:\n"
        + "\n".join(echo_errors)
    )
    logger.info("[BOOT] Stage 06 passed: all %d quabo(s) echo-responsive", len(quabo_addrs))


# ---------------------------------------------------------------------------
# Stage 07 — Route HK packets
# ---------------------------------------------------------------------------

def test_boot_07_hk_dest(runner, topology) -> None:
    """Route housekeeping packets to the head node via pseti cfg hk-dest."""
    from ci.hardware_software.hw_utils.cli import _STATE_FILE
    from ci.hardware_software.hw_utils.state_machine import _write_state

    logger.info("[BOOT] Stage 07: pseti cfg hk-dest")
    r = runner.invoke(app, ["cfg", "hk-dest"])
    assert r.exit_code == 0, f"[BOOT] Stage 07 FAILED: pseti cfg hk-dest:\n{r.output}"
    _write_state(_STATE_FILE, "HK_ROUTED")
    logger.info("[BOOT] Stage 07 passed")


# ---------------------------------------------------------------------------
# Stage 08 — Start Redis daemons
# ---------------------------------------------------------------------------

def test_boot_08_redis_daemons(runner, topology) -> None:
    """Start capture_hk and Redis daemons; verify Redis is accepting connections."""
    logger.info("[BOOT] Stage 08: pseti cfg redis-daemons")
    r = runner.invoke(app, ["cfg", "redis-daemons"])
    assert r.exit_code == 0, f"[BOOT] Stage 08 FAILED: pseti cfg redis-daemons:\n{r.output}"

    try:
        import redis as _redis
        _redis.Redis(host="127.0.0.1", port=6379, socket_timeout=3).ping()
        logger.info("[BOOT] Stage 08: Redis ping OK")
    except Exception as exc:
        pytest.fail(f"[BOOT] Stage 08 FAILED: Redis not reachable after starting daemons: {exc}")

    logger.info("[BOOT] Stage 08 passed")


# ---------------------------------------------------------------------------
# Stage 09 — MAROC config
# ---------------------------------------------------------------------------

def test_boot_09_maroc_config(runner, topology) -> None:
    """Program MAROC ASICs via pseti cfg maroc-config."""
    from ci.hardware_software.hw_utils.cli import _STATE_FILE
    from ci.hardware_software.hw_utils.state_machine import _write_state

    logger.info("[BOOT] Stage 09: pseti cfg maroc-config")
    # "Use default calibration file?" prompt appears when UID is not in quabo_info.json.
    r = runner.invoke(app, ["cfg", "maroc-config"], input="Y\nY\nY\nY\n")
    assert r.exit_code == 0, f"[BOOT] Stage 09 FAILED: pseti cfg maroc-config:\n{r.output}"
    _write_state(_STATE_FILE, "MAROC_CONFIGURED")
    logger.info("[BOOT] Stage 09 passed")


# ---------------------------------------------------------------------------
# Stage 10 — Mask config
# ---------------------------------------------------------------------------

def test_boot_10_mask_config(runner, topology) -> None:
    """Load trigger masks via pseti cfg mask-config."""
    from ci.hardware_software.hw_utils.cli import _STATE_FILE
    from ci.hardware_software.hw_utils.state_machine import _write_state

    logger.info("[BOOT] Stage 10: pseti cfg mask-config")
    r = runner.invoke(app, ["cfg", "mask-config"])
    assert r.exit_code == 0, f"[BOOT] Stage 10 FAILED: pseti cfg mask-config:\n{r.output}"
    _write_state(_STATE_FILE, "MASKS_CONFIGURED")
    logger.info("[BOOT] Stage 10 passed")


# ---------------------------------------------------------------------------
# Stage 11 — PH calibration
# ---------------------------------------------------------------------------

def test_boot_11_calibrate_ph(runner, topology) -> None:
    """Run PH calibration and verify saved coefficients are in [0, 4095]."""
    import json

    from ci.hardware_software.hw_utils.cli import _STATE_FILE
    from ci.hardware_software.hw_utils.state_machine import _write_state
    from control.utils.config_file import quabo_ph_baseline_filename
    from control.utils.paths import PanoPaths

    logger.info("[BOOT] Stage 11: pseti cfg calibrate-ph")
    r = runner.invoke(app, ["cfg", "calibrate-ph"])
    assert r.exit_code == 0, f"[BOOT] Stage 11 FAILED: pseti cfg calibrate-ph:\n{r.output}"

    baseline_path = PanoPaths.calibration_file(quabo_ph_baseline_filename)
    assert baseline_path.exists(), (
        f"[BOOT] Stage 11 FAILED: baseline file not written: {baseline_path}"
    )
    with baseline_path.open() as f:
        baseline = json.load(f)

    ph_errors = []
    for entry in baseline.get("quabos", []):
        uid = entry.get("uid", "?")
        coeffs = entry.get("coefs", [])
        out_of_range = [(i, c) for i, c in enumerate(coeffs) if not (0 <= c <= 4095)]
        if len(coeffs) != 256:
            ph_errors.append(f"uid={uid}: expected 256 coefficients, got {len(coeffs)}")
        elif out_of_range:
            ph_errors.append(
                f"uid={uid}: {len(out_of_range)} coeff(s) out of [0,4095]: {out_of_range[:5]}"
            )
        else:
            logger.info("[BOOT] Stage 11: uid=%s PH coefficients OK", uid)

    assert not ph_errors, (
        "[BOOT] Stage 11 FAILED: PH coefficients out of range:\n" + "\n".join(ph_errors)
    )

    _write_state(_STATE_FILE, "PH_CALIBRATED")
    logger.info("[BOOT] All stages passed — hardware is in PH_CALIBRATED state")
