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


def quabos_responsive(retries: int = 5, delay_s: float = 5.0) -> bool:
    """Return True if all quabos in the active topology respond to util.ping.
    
    Performs parallel pings across all quabos in the active topology.
    If any quabo fails to respond, it retries the entire batch up to `retries` times
    with `delay_s` between attempts.
    """
    try:
        import ipaddress
        from concurrent.futures import ThreadPoolExecutor
        from ci.hardware_software.hw_utils.topology import HwTopology
        from control.utils import util
        
        topo = HwTopology()
        quabo_addrs = list(topo.quabo_ips())
        if not quabo_addrs:
            return True

        for attempt in range(retries):
            with ThreadPoolExecutor(max_workers=len(quabo_addrs)) as executor:
                # Dispatch all pings in parallel
                futures = {
                    executor.submit(util.ping, ipaddress.ip_address(a.real_ip), a.cmd_port): a 
                    for a in quabo_addrs
                }
                
                results = {}
                for future in futures:
                    a = futures[future]
                    try:
                        results[a.ip] = future.result()
                    except Exception as exc:
                        logger.debug("Ping task failed for %s: %s", a.ip, exc)
                        results[a.ip] = False

            if all(results.values()):
                if attempt > 0:
                    logger.info("quabos_responsive: all quabos responsive after %d retry(s)", attempt)
                return True

            failed_ips = [ip for ip, res in results.items() if not res]
            if attempt < retries - 1:
                logger.warning(
                    "quabos_responsive: %d quabo(s) [%s] not responsive. Retrying in %.1fs... (attempt %d/%d)",
                    len(failed_ips), ", ".join(failed_ips), delay_s, attempt + 1, retries
                )
                import time
                time.sleep(delay_s)
            else:
                logger.error("quabos_responsive: check FAILED. Unresponsive: %s", ", ".join(failed_ips))
        
        return False
    except Exception as exc:
        logger.debug("quabos_responsive check failed: %s", exc)
        return False
