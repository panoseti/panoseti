"""
orchestrator/lifecycle.py — Shared start / wait_healthy / tear_down helpers.

These functions operate on lists of PsetiContainer objects so the Fleet can
use them generically without coupling the orchestrator to specific container
types.
"""

from __future__ import annotations

import contextlib
import shutil
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ci.software_only.containers.base import PsetiContainer


def start_all(containers: list[PsetiContainer]) -> None:
    """Start each container in sequence.  Raises on the first failure."""
    for c in containers:
        c.start()


def wait_all_healthy(
    containers: list[PsetiContainer],
    *,
    timeout: float = 90.0,
) -> None:
    """Wait until every container's gRPC port is READY.

    Each container is given the full remaining time budget, so the total
    wall-clock time is at most ``timeout`` seconds (not N x timeout).
    """
    deadline = time.monotonic() + timeout
    for c in containers:
        remaining = max(1.0, deadline - time.monotonic())
        c.wait_grpc_ready(timeout=remaining)


def tear_down_all(
    containers: list[PsetiContainer],
    *,
    temp_dirs: list[str] | None = None,
) -> None:
    """Stop every container and clean up any host-side temp directories."""
    for c in containers:
        with contextlib.suppress(Exception):
            c.stop()

    for d in (temp_dirs or []):
        with contextlib.suppress(Exception):
            shutil.rmtree(d, ignore_errors=True)
