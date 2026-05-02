"""
hw_boot — Annotated boot-sequence validation from UNPOWERED state.

This suite is the gate for all other test classes: if the system cannot be
reliably brought from UNPOWERED to a verified BOOTED state, nothing else is
worth testing.  It runs first (batch_priority = -1) and uses the same
production functions as pseti session-start, so test failures pinpoint the
exact stage where the boot breaks.

Boot stages (matching session_start.py golden path):
  Stage 1 — WPS power on: outlets energised; FPGA bootloader starts.
            Asserts WPS reports ON after the command.
  Stage 2 — Boot wait (60s): timed sleep with 15s progress logs.
            TFTP bootloader becomes ready; command ports are NOT yet active.
  Stage 3 — UID discovery via TFTP: each quabo's real IP/port is logged so
            port-forwarding issues are immediately visible.  Fails with a
            per-quabo table showing which ones responded.
  Stage 4a — TFTP reboot: config.do_reboot loads main firmware.
  Stage 4b — Reboot skip check: asserts zero quabos were silently skipped
             because of an empty UID (symptom: bootloader was unreachable at
             Stage 3 but do_reboot would have swallowed that silently).
  Stage 4c — Post-reboot command-port reachability: all quabos respond to
             util.ping on their command port.
  Stage 5  — Echo check: each quabo echoes back an hv_set command, confirming
             the main firmware is processing commands correctly.

Required state: UNPOWERED (test drives hardware to BOOTED itself).
Leaves state: BOOTED (written to the state file on success).
"""

from __future__ import annotations

import ipaddress
import logging
import os
import time

import pytest

from ci.hardware_software.hw_utils.topology import HwTopology
from control.utils import util

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.hw_class("boot_sequence"),
    pytest.mark.timeout(600),
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
    """Log each quabo's raw IP and its resolved real_ip:port for easy diagnosis."""
    for a in topo.quabo_ips():
        logger.info(
            "[BOOT] quabo map: raw=%-18s  real=%-18s  cmd_port=%-6d  reboot_port=%d",
            a.ip, a.real_ip, a.cmd_port, a.reboot_port,
        )


# ---------------------------------------------------------------------------
# Boot sequence test
# ---------------------------------------------------------------------------

@pytest.mark.slow_hw
def test_annotated_boot_sequence(topology) -> None:
    """
    Drives hardware from UNPOWERED to BOOTED using the same code as
    pseti session-start, with assertions at each stage.

    Any failure is diagnosed at the specific stage that broke, not buried
    in a state-machine exception.
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

    # ── Stage 1: WPS power on ─────────────────────────────────────────────────
    logger.info("[BOOT] Stage 1: WPS power on")
    wps_power_on()

    from control.power import quabo_power_query
    wps_errors = []
    for wps in topology.wps_outlets():
        state = quabo_power_query({"url": wps.url, "quabo_socket": wps.quabo_socket})
        # quabo_power_query returns 'true' when outlet is energised
        if state != "true":
            wps_errors.append(f"{wps.name} ({wps.url}): power query returned {state!r} (expected 'true')")
        else:
            logger.info("[BOOT] Stage 1: %s power confirmed ON (state=%r)", wps.name, state)
    assert not wps_errors, (
        f"[BOOT] Stage 1 FAILED: WPS did not confirm ON for {len(wps_errors)} outlet(s):\n"
        + "\n".join(wps_errors)
    )

    # ── Stage 2: Boot wait ────────────────────────────────────────────────────
    # Mirrors session_start.py line 73-74: flat sleep before get_uids.
    # After power-on the quabo runs the TFTP bootloader (port 69/6000x), not
    # the main firmware; command ports are NOT yet active.
    logger.info("[BOOT] Stage 2: waiting %ds for TFTP bootloader to be ready", _BOOT_WAIT_S)
    elapsed = 0
    while elapsed < _BOOT_WAIT_S:
        chunk = min(15, _BOOT_WAIT_S - elapsed)
        time.sleep(chunk)
        elapsed += chunk
        logger.info("[BOOT] Stage 2: %ds / %ds elapsed", elapsed, _BOOT_WAIT_S)

    # ── Stage 3: UID discovery via TFTP ──────────────────────────────────────
    # Log per-quabo target so port-forwarding issues are immediately visible.
    logger.info("[BOOT] Stage 3: get_uids (TFTP to FPGA bootloader)")
    quabo_uids = get_uids.get_uids(obs_config, network_config)

    uid_rows: list[str] = []
    uid_errors: list[str] = []
    for dome in quabo_uids.domes:
        for module in dome.modules:
            for quadrant, entry in enumerate(module.quabos):
                from control.utils import util as _util
                ip_ports = _util.get_quabo_ip_port(module.ip_addr, quadrant, network_config)
                status = f"UID={entry.uid!r}" if entry.uid else "OFFLINE (empty UID)"
                uid_rows.append(
                    f"  module {module.ip_addr} Q{quadrant} "
                    f"→ {ip_ports.ip_addr}:{ip_ports.reboot_port}  {status}"
                )
                if not entry.uid:
                    uid_errors.append(
                        f"module {module.ip_addr} quadrant {quadrant}: "
                        f"UID empty — quabo offline or TFTP unreachable at "
                        f"{ip_ports.ip_addr}:{ip_ports.reboot_port}"
                    )

    logger.info("[BOOT] Stage 3 UID table:\n%s", "\n".join(uid_rows))
    assert not uid_errors, (
        f"[BOOT] Stage 3 FAILED: get_uids did not populate valid UIDs for "
        f"{len(uid_errors)} quabo(s):\n" + "\n".join(uid_errors)
    )
    logger.info("[BOOT] Stage 3 passed: all %d UIDs populated", len(quabo_addrs))

    # ── Stage 4a: TFTP reboot (loads main firmware) ───────────────────────────
    # config.do_reboot handles its own post-reboot ping-wait per quabo.
    # After this returns, main firmware is loaded and command ports are active.
    logger.info("[BOOT] Stage 4a: TFTP reboot via config.do_reboot")
    quabo_uids_cached = config_file.get_quabo_uids()
    config.do_reboot(modules, quabo_uids_cached, network_config)

    # ── Stage 4b: Reboot skip check ───────────────────────────────────────────
    # do_reboot silently skips quabos with empty UIDs.  Catch that here so the
    # failure message names the quabo and its cause rather than showing a
    # generic "unreachable after reboot".
    skipped = [
        f"module {m.ip_addr} Q{q_idx}: UID was empty → do_reboot skipped it"
        for d in quabo_uids_cached.domes
        for m in d.modules
        for q_idx, e in enumerate(m.quabos)
        if not e.uid
    ]
    assert not skipped, (
        f"[BOOT] Stage 4b FAILED: do_reboot skipped {len(skipped)} quabo(s) "
        f"because their UIDs were empty:\n" + "\n".join(skipped)
    )

    # ── Stage 4c: Post-reboot command-port reachability ───────────────────────
    post_reboot_errors = _check_all_reachable(topology)
    assert not post_reboot_errors, (
        f"[BOOT] Stage 4c FAILED: {len(post_reboot_errors)} quabo(s) unreachable "
        f"via command port after TFTP reboot:\n" + "\n".join(post_reboot_errors)
    )
    logger.info("[BOOT] Stage 4c passed: all %d quabo(s) reachable after reboot", len(quabo_addrs))

    # ── Stage 5: Echo responsiveness ─────────────────────────────────────────
    logger.info("[BOOT] Stage 5: echo check on each quabo")
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
        f"[BOOT] Stage 5 FAILED: {len(echo_errors)} quabo(s) did not echo commands "
        f"after TFTP reboot:\n" + "\n".join(echo_errors)
    )
    logger.info("[BOOT] Stage 5 passed: all %d quabo(s) echo-responsive", len(quabo_addrs))

    # ── Write BOOTED to state file ────────────────────────────────────────────
    _write_state(_STATE_FILE, "BOOTED")
    logger.info("[BOOT] All stages passed — hardware is in BOOTED state")
