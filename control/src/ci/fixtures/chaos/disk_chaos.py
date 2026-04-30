"""
chaos/disk_chaos.py

Filesystem fault injection: fill a volume to force ENOSPC conditions.

All operations use Docker exec to run dd/fallocate inside containers.
Context managers guarantee teardown (fill file removal) even if tests raise.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Generator
from contextlib import contextmanager

from ci.fixtures.chaos import process_chaos as _pc

logger = logging.getLogger(__name__)


def fill_volume(
    container_name: str,
    mount_path: str,
    fill_pct: int = 99,
) -> str:
    """Fill mount_path to fill_pct% full using a sparse dd file.

    Returns the path of the fill file so the caller can remove it.
    Raises RuntimeError if dd fails.
    """
    fill_file = f"{mount_path}/.chaos_fill"
    # Get available space in bytes
    code, out = _pc._exec(
        container_name,
        f"df -B1 --output=avail {mount_path} | tail -1",
    )
    if code != 0:
        raise RuntimeError(f"df failed on {mount_path}: {out}")
    avail_bytes = int(out.strip())
    fill_bytes = int(avail_bytes * fill_pct / 100)
    fill_mb = max(1, fill_bytes // (1024 * 1024))

    code, out = _pc._exec(
        container_name,
        f"dd if=/dev/zero of={fill_file} bs=1M count={fill_mb} 2>&1",
    )
    if code != 0:
        raise RuntimeError(f"dd fill failed: {out}")
    logger.info(f"Filled {mount_path} to ~{fill_pct}% ({fill_mb} MB)")
    return fill_file


def release_fill(container_name: str, fill_file: str) -> None:
    """Remove the fill file, releasing disk space."""
    with contextlib.suppress(Exception):
        _pc._exec(container_name, f"rm -f {fill_file}")
    logger.info(f"Released fill file {fill_file}")


@contextmanager
def full_disk(
    container_name: str,
    mount_path: str,
    fill_pct: int = 99,
) -> Generator[None]:
    """Context manager: fill mount_path to fill_pct% for the test block.

    Guarantees teardown even if the test raises.
    """
    fill_file = fill_volume(container_name, mount_path, fill_pct)
    try:
        yield
    finally:
        release_fill(container_name, fill_file)
