"""
Boot sequence helper for session fixtures.

Provides _state_file_says() and _quabos_responsive() to let the
booted_calibrated fixture short-circuit when hardware is already in the
expected state.
"""

from __future__ import annotations

import ipaddress
import logging

logger = logging.getLogger(__name__)


def state_file_says(expected: str) -> bool:
    """Return True if the HITL state file records *expected* state."""
    from ci.hardware_software.hw_utils.cli import _STATE_FILE
    from ci.hardware_software.hw_utils.state_machine import read_state
    return read_state(_STATE_FILE) == expected


def quabos_responsive() -> bool:
    """Return True if all quabos in the active topology respond to util.ping."""
    try:
        from ci.hardware_software.hw_utils.topology import HwTopology
        from control.utils import util
        topo = HwTopology()
        return all(util.ping(ipaddress.ip_address(a.real_ip), a.cmd_port) for a in topo.quabo_ips())
    except Exception as exc:
        logger.debug("quabos_responsive check failed: %s", exc)
        return False
