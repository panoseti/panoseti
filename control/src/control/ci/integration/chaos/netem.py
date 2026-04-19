"""
chaos/netem.py

tc-netem wrappers for network fault injection inside Docker containers.

Requires NET_ADMIN capability on the target container (already present on
daqnode containers in docker-compose.integration.yml).

All helpers are context managers — they restore the original qdisc on exit.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Generator
from contextlib import contextmanager

from control.ci.integration.chaos import process_chaos as _pc

logger = logging.getLogger(__name__)


def _iface(container_name: str) -> str:
    """Return the primary network interface inside a container."""
    code, out = _pc._exec(
        container_name,
        "ip route | awk '/default/ {print $5; exit}'",
    )
    iface = out.strip()
    return iface if (code == 0 and iface) else "eth0"


def add_netem(
    container_name: str,
    iface: str | None = None,
    latency_ms: int = 0,
    loss_pct: float = 0.0,
    duplicate_pct: float = 0.0,
    corrupt_pct: float = 0.0,
) -> None:
    """Apply tc-netem impairments to the container's egress traffic."""
    if iface is None:
        iface = _iface(container_name)
    parts = ["tc", "qdisc", "add", "dev", iface, "root", "netem"]
    if latency_ms:
        parts += ["delay", f"{latency_ms}ms"]
    if loss_pct:
        parts += ["loss", f"{loss_pct}%"]
    if duplicate_pct:
        parts += ["duplicate", f"{duplicate_pct}%"]
    if corrupt_pct:
        parts += ["corrupt", f"{corrupt_pct}%"]
    cmd = " ".join(parts)
    code, out = _pc._exec(container_name, cmd)
    if code != 0:
        raise RuntimeError(f"netem add failed: {out}")
    logger.info(f"netem applied to {container_name}/{iface}: {parts[7:]}")


def remove_netem(container_name: str, iface: str | None = None) -> None:
    """Remove tc-netem from the container's egress traffic (best-effort)."""
    if iface is None:
        iface = _iface(container_name)
    with contextlib.suppress(Exception):
        _pc._exec(container_name, f"tc qdisc del dev {iface} root")
    logger.info(f"netem removed from {container_name}/{iface}")


@contextmanager
def latency(
    container_name: str,
    latency_ms: int,
    iface: str | None = None,
) -> Generator[None]:
    """Add constant latency to outbound traffic for the block duration."""
    iface = iface or _iface(container_name)
    add_netem(container_name, iface=iface, latency_ms=latency_ms)
    try:
        yield
    finally:
        remove_netem(container_name, iface=iface)


@contextmanager
def packet_loss(
    container_name: str,
    loss_pct: float,
    iface: str | None = None,
) -> Generator[None]:
    """Add packet loss to outbound traffic for the block duration."""
    iface = iface or _iface(container_name)
    add_netem(container_name, iface=iface, loss_pct=loss_pct)
    try:
        yield
    finally:
        remove_netem(container_name, iface=iface)


class NetemContext:
    """Stateful netem manager for use in pytest fixtures (multiple applies)."""

    def __init__(self) -> None:
        self._applied: list[tuple[str, str]] = []  # (container, iface) pairs

    def add(
        self,
        container_name: str,
        iface: str | None = None,
        latency_ms: int = 0,
        loss_pct: float = 0.0,
    ) -> None:
        if iface is None:
            iface = _iface(container_name)
        add_netem(container_name, iface=iface, latency_ms=latency_ms, loss_pct=loss_pct)
        self._applied.append((container_name, iface))

    def restore_all(self) -> None:
        for container_name, iface in self._applied:
            with contextlib.suppress(Exception):
                remove_netem(container_name, iface=iface)
        self._applied.clear()
