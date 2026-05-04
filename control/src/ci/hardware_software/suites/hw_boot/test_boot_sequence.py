"""
hw_boot — Annotated boot-sequence validation from UNPOWERED to PH_CALIBRATED.

This suite is the gate for all other test classes.  It runs the full
session_start.py golden path via the pseti CLI with assertions at every
stage, so failures are diagnosed at the exact step that broke.

Boot stages:
  Stage 0  — pseti power off: guarantee a clean baseline regardless of prior state.
  Stage 1  — WPS power on + outlet confirmation.
  Stage 2  — Boot wait (60s): timed sleep with 15s progress logs.
  Stage 3  — UID discovery via TFTP: per-quabo table with real_ip:port.
  Stage 4a — pseti cfg reboot: TFTP firmware load via do_reboot.
  Stage 4b — Reboot skip check: asserts zero quabos skipped (empty UID).
  Stage 4c — Post-reboot command-port reachability.
  Stage 5  — Echo check: each quabo echoes back an hv_set command.
  Stage 6  — pseti cfg hk-dest: route HK packets to head node.
  Stage 7  — pseti cfg redis-daemons: start capture_hk and Redis clients.
  Stage 8  — pseti cfg maroc-config: program MAROC ASICs.
  Stage 9  — pseti cfg mask-config: load trigger masks.
  Stage 10 — pseti cfg calibrate-ph: baseline PH calibration; assert coefficients.

Required state: UNPOWERED (the test powers off first regardless).
Leaves state: PH_CALIBRATED.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import time

import pytest

from ci.hardware_software.hw_utils.topology import HwTopology
from control.pseti import app
from control.utils import util

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.hw_class("boot_sequence"),
    pytest.mark.timeout(900),
]

_BOOT_WAIT_S = int(os.environ.get("HW_TEST_QUABO_BOOT_WAIT", 60))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_all_reachable(topo: HwTopology) -> list[str]:
    """Return error strings for each quabo that fails util.ping (command port)."""
    errors = []
    for a in topo.quabo_ips():
        try:
            if not util.ping(ipaddress.ip_address(a.real_ip), a.cmd_port):
                errors.append(f"{a.ip} (loc={a.boardloc}, real={a.real_ip}:{a.cmd_port}) not reachable")
        except Exception as exc:
            errors.append(f"{a.ip} (loc={a.boardloc}) ping error: {exc}")
    return errors


def _log_topology_targets(topo: HwTopology) -> None:
    for a in topo.quabo_ips():
        logger.info(
            "[BOOT] quabo map: raw=%-18s  real=%-18s  cmd_port=%-6d  reboot_port=%d",
            a.ip, a.real_ip, a.cmd_port, a.reboot_port,
        )


# ---------------------------------------------------------------------------
# Boot sequence test
# ---------------------------------------------------------------------------

@pytest.mark.slow_hw
def test_annotated_boot_sequence(runner, topology) -> None:
    """
    Drive hardware from any state to PH_CALIBRATED using the same CLI
    commands as pseti session-start.  Assertions at every stage.
    """
    import control.config as config
    import control.get_uids as get_uids
    from ci.hardware_software.hw_utils.cli import _STATE_FILE
    from ci.hardware_software.hw_utils.driver_ops import wps_power_on
    from ci.hardware_software.hw_utils.state_machine import _write_state
    from control.utils import config_file

    obs_config = config_file.get_obs_config()
    network_config = config_file.get_network_config()
    modules = config_file.get_modules(obs_config)
    quabo_addrs = topology.quabo_ips()

    if not quabo_addrs:
        pytest.skip("No quabos in active topology")

    _log_topology_targets(topology)

    # ── Stage 0: Force power off ──────────────────────────────────────────────
    # Guarantees a known-clean state regardless of what happened before.
    # cmd 0x04 (soft reset) does NOT re-enter the TFTP bootloader, so the only
    # reliable reset is a WPS power cycle.
    logger.info("[BOOT] Stage 0: pseti power off")
    r = runner.invoke(app, ["power", "off"])
    assert r.exit_code == 0, f"[BOOT] Stage 0 FAILED: pseti power off:\n{r.output}"
    logger.info("[BOOT] Stage 0 passed: WPS outlets off")
    time.sleep(5)  # let outlets settle

    # ── Stage 1: WPS power on ─────────────────────────────────────────────────
    logger.info("[BOOT] Stage 1: WPS power on")
    wps_power_on()

    from control.power import quabo_power_query
    wps_errors = []
    for wps in topology.wps_outlets():
        state = quabo_power_query({"url": wps.url, "quabo_socket": wps.quabo_socket})
        if state != "true":
            wps_errors.append(f"{wps.name} ({wps.url}): power query returned {state!r} (expected 'true')")
        else:
            logger.info("[BOOT] Stage 1: %s confirmed ON", wps.name)
    assert not wps_errors, (
        f"[BOOT] Stage 1 FAILED: WPS did not confirm ON for {len(wps_errors)} outlet(s):\n"
        + "\n".join(wps_errors)
    )

    # ── Stage 2: Boot wait ────────────────────────────────────────────────────
    logger.info("[BOOT] Stage 2: waiting %ds for TFTP bootloader", _BOOT_WAIT_S)
    elapsed = 0
    while elapsed < _BOOT_WAIT_S:
        chunk = min(15, _BOOT_WAIT_S - elapsed)
        time.sleep(chunk)
        elapsed += chunk
        logger.info("[BOOT] Stage 2: %ds / %ds elapsed", elapsed, _BOOT_WAIT_S)

    # ── Stage 3: UID discovery via TFTP ──────────────────────────────────────
    logger.info("[BOOT] Stage 3: get_uids (TFTP to FPGA bootloader)")
    quabo_uids = get_uids.get_uids(obs_config, network_config)

    uid_rows: list[str] = []
    uid_errors: list[str] = []
    for dome in quabo_uids.domes:
        for module in dome.modules:
            for quadrant, entry in enumerate(module.quabos):
                ip_ports = util.get_quabo_ip_port(module.ip_addr, quadrant, network_config)
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

    logger.info("[BOOT] Stage 3 UID table:\n%s", "\n".join(uid_rows))
    assert not uid_errors, (
        f"[BOOT] Stage 3 FAILED: get_uids did not populate UIDs for "
        f"{len(uid_errors)} quabo(s):\n" + "\n".join(uid_errors)
    )
    logger.info("[BOOT] Stage 3 passed: all %d UIDs populated", len(quabo_addrs))

    # ── Stage 4a: TFTP reboot (loads main firmware) ───────────────────────────
    logger.info("[BOOT] Stage 4a: TFTP reboot via config.do_reboot")
    quabo_uids_cached = config_file.get_quabo_uids()
    config.do_reboot(modules, quabo_uids_cached, network_config)

    # ── Stage 4b: Reboot skip check ───────────────────────────────────────────
    skipped = [
        f"module {m.ip_addr} Q{q_idx}: UID was empty → do_reboot skipped it"
        for d in quabo_uids_cached.domes
        for m in d.modules
        for q_idx, e in enumerate(m.quabos)
        if not e.uid
    ]
    assert not skipped, (
        f"[BOOT] Stage 4b FAILED: do_reboot skipped {len(skipped)} quabo(s):\n"
        + "\n".join(skipped)
    )

    # ── Stage 4c: Post-reboot command-port reachability ───────────────────────
    post_reboot_errors = _check_all_reachable(topology)
    assert not post_reboot_errors, (
        f"[BOOT] Stage 4c FAILED: {len(post_reboot_errors)} quabo(s) unreachable after reboot:\n"
        + "\n".join(post_reboot_errors)
    )
    logger.info("[BOOT] Stage 4c passed: all %d quabo(s) reachable", len(quabo_addrs))

    # ── Stage 5: Echo check ───────────────────────────────────────────────────
    logger.info("[BOOT] Stage 5: echo check")
    from control.driver.quabo_driver import QUABO
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
        f"[BOOT] Stage 5 FAILED: {len(echo_errors)} quabo(s) did not echo:\n"
        + "\n".join(echo_errors)
    )
    logger.info("[BOOT] Stage 5 passed: all %d quabo(s) echo-responsive", len(quabo_addrs))
    _write_state(_STATE_FILE, "BOOTED")

    # ── Stage 6: Route HK ────────────────────────────────────────────────────
    logger.info("[BOOT] Stage 6: pseti cfg hk-dest")
    r = runner.invoke(app, ["cfg", "hk-dest"])
    assert r.exit_code == 0, f"[BOOT] Stage 6 FAILED: pseti cfg hk-dest:\n{r.output}"
    logger.info("[BOOT] Stage 6 passed: HK routed")
    _write_state(_STATE_FILE, "HK_ROUTED")

    # ── Stage 7: Start Redis daemons ──────────────────────────────────────────
    logger.info("[BOOT] Stage 7: pseti cfg redis-daemons")
    r = runner.invoke(app, ["cfg", "redis-daemons"])
    assert r.exit_code == 0, f"[BOOT] Stage 7 FAILED: pseti cfg redis-daemons:\n{r.output}"
    # Verify Redis is actually accepting connections
    try:
        import redis as _redis
        _redis.Redis(host="127.0.0.1", port=6379, socket_timeout=3).ping()
        logger.info("[BOOT] Stage 7: Redis ping OK")
    except Exception as exc:
        pytest.fail(f"[BOOT] Stage 7 FAILED: Redis not reachable after starting daemons: {exc}")

    # ── Stage 8: MAROC config ─────────────────────────────────────────────────
    # "Use default calibration file?" prompt appears when UID is not in quabo_info.json.
    # Auto-accept with "Y\n" so the test is non-interactive.
    logger.info("[BOOT] Stage 8: pseti cfg maroc-config")
    r = runner.invoke(app, ["cfg", "maroc-config"], input="Y\nY\nY\nY\n")
    assert r.exit_code == 0, f"[BOOT] Stage 8 FAILED: pseti cfg maroc-config:\n{r.output}"
    logger.info("[BOOT] Stage 8 passed: MAROC configured")
    _write_state(_STATE_FILE, "MAROC_CONFIGURED")

    # ── Stage 9: Mask config ──────────────────────────────────────────────────
    logger.info("[BOOT] Stage 9: pseti cfg mask-config")
    r = runner.invoke(app, ["cfg", "mask-config"])
    assert r.exit_code == 0, f"[BOOT] Stage 9 FAILED: pseti cfg mask-config:\n{r.output}"
    logger.info("[BOOT] Stage 9 passed: masks configured")
    _write_state(_STATE_FILE, "MASKS_CONFIGURED")

    # ── Stage 10: PH calibration ──────────────────────────────────────────────
    logger.info("[BOOT] Stage 10: pseti cfg calibrate-ph")
    r = runner.invoke(app, ["cfg", "calibrate-ph"])
    assert r.exit_code == 0, f"[BOOT] Stage 10 FAILED: pseti cfg calibrate-ph:\n{r.output}"

    # Spot-check saved coefficients from the file written by calibrate-ph.
    # Reading the file avoids re-triggering hardware calibration a second time.
    import json
    from control.utils.config_file import quabo_ph_baseline_filename
    from control.utils.paths import PanoPaths
    baseline_path = PanoPaths.calibration_file(quabo_ph_baseline_filename)
    assert baseline_path.exists(), f"[BOOT] Stage 10 FAILED: baseline file not written: {baseline_path}"
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
            logger.info("[BOOT] Stage 10: uid=%s PH coefficients OK", uid)
    assert not ph_errors, "[BOOT] Stage 10 FAILED: PH coefficients out of range:\n" + "\n".join(ph_errors)

    _write_state(_STATE_FILE, "PH_CALIBRATED")
    logger.info("[BOOT] All stages passed — hardware is in PH_CALIBRATED state")
