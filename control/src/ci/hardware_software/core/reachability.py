"""
Quabo reachability helper for HITL fixtures.

Provides wait_until_all_quabos_reachable(), which polls all quabos via the
same util.ping() primitive that pseti start uses internally, but with
configurable retries.  Call this after any cfg command that touches all
quabos (maroc-config, mask-config, calibrate-ph) and before invoking
pseti start, to absorb port-forwarding and UDP latency variance on Q3.
"""

from __future__ import annotations

import logging
import time
from ipaddress import ip_address

from control.utils.util import ping

logger = logging.getLogger(__name__)


def wait_until_all_quabos_reachable(
    topology,
    timeout: float = 30,
    retry_every: float = 2,
) -> None:
    """Poll all quabos until every one responds to util.ping.

    Raises AssertionError with a per-quabo failure summary if any quabo
    remains unresponsive after `timeout` seconds.
    """
    deadline = time.monotonic() + timeout
    last_errors: dict[str, str] = {}

    while time.monotonic() < deadline:
        addrs = topology.quabo_ips()
        last_errors = {}
        for a in addrs:
            try:
                if not ping(ip_address(a.real_ip), a.cmd_port):
                    last_errors[a.ip] = f"{a.real_ip}:{a.cmd_port} no response"
            except Exception as exc:
                last_errors[a.ip] = repr(exc)

        if not last_errors:
            logger.info("[REACHABILITY] all %d quabo(s) reachable", len(addrs))
            return

        logger.info(
            "[REACHABILITY] %d/%d quabo(s) not yet ready: %s — retrying in %.1fs",
            len(last_errors), len(addrs), last_errors, retry_every,
        )
        time.sleep(retry_every)

    raise AssertionError(
        f"Quabos still unreachable after {timeout}s: "
        + ", ".join(f"{k}: {v}" for k, v in last_errors.items())
    )
