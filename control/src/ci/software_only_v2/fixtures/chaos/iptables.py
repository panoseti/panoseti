"""
chaos/iptables.py

iptables blackhole rules for network fault injection inside Docker containers.

Requires NET_ADMIN capability (already present on daqnode containers).
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Generator
from contextlib import contextmanager

from ci.software_only_v2.fixtures.chaos import process_chaos as _pc

logger = logging.getLogger(__name__)


def _ipt(container_name: str, action: str, dst_ip: str, port: int | None = None) -> None:
    parts = ["iptables", "-", action, "OUTPUT", "-d", dst_ip]
    if port:
        parts += ["-p", "tcp", "--dport", str(port)]
    parts += ["-j", "DROP"]
    cmd = " ".join(parts)
    code, out = _pc._exec(container_name, cmd)
    if code != 0 and action == "A":
        raise RuntimeError(f"iptables {action} failed: {out}")


def blackhole(container_name: str, dst_ip: str, port: int | None = None) -> None:
    """Add an OUTPUT DROP rule for dst_ip (and optionally dst_port) in container."""
    _ipt(container_name, "A", dst_ip, port)
    logger.info(f"Blackholed {dst_ip}{f':{port}' if port else ''} in {container_name}")


def unblackhole(container_name: str, dst_ip: str, port: int | None = None) -> None:
    """Remove the OUTPUT DROP rule (best-effort)."""
    with contextlib.suppress(Exception):
        _ipt(container_name, "D", dst_ip, port)


@contextmanager
def blocked_egress(
    container_name: str,
    dst_ip: str,
    port: int | None = None,
) -> Generator[None]:
    """Drop outbound traffic to dst_ip[:port] for the block duration."""
    blackhole(container_name, dst_ip, port)
    try:
        yield
    finally:
        unblackhole(container_name, dst_ip, port)
