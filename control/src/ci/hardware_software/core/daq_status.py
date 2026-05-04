"""
DAQ node status helpers.

Uses the gRPC DaqControl service to check disk usage on active runs, enabling
tests to assert that data is actually being written.
"""

from __future__ import annotations

import asyncio
import time


async def _get_disk_used_async(host: str, port: int, run_name: str) -> int:
    """Return bytes used by the given run on the DAQ node, or 0 on error."""
    try:
        from panoseti_grpc.daq_control.client import DaqControlClient
        async with DaqControlClient(host, port) as client:
            status = await client.StatusDaq()
            return getattr(status, "disk_used_bytes", 0) or 0
    except Exception:
        return 0


def disk_used_bytes(host: str, port: int, run_name: str) -> int:
    """Synchronous wrapper for disk usage query."""
    return asyncio.run(_get_disk_used_async(host, port, run_name))


def assert_disk_growing(host: str, port: int, run_name: str,
                        min_bytes: int = 500_000, window_s: float = 10.0) -> None:
    """Assert that data is being written to the DAQ node.

    Measures disk usage twice (window_s apart) and asserts the delta meets
    the minimum expected byte growth, indicating Hashpipe is writing data.

    Args:
        host: DAQ node host.
        port: gRPC port.
        run_name: Current run name (for logging clarity).
        min_bytes: Minimum expected byte growth over window_s.
        window_s: Observation window in seconds.

    Raises:
        AssertionError: if data is not growing fast enough.
    """
    before = disk_used_bytes(host, port, run_name)
    time.sleep(window_s)
    after = disk_used_bytes(host, port, run_name)
    delta = after - before
    assert delta >= min_bytes, (
        f"DAQ node {host} not writing fast enough: only {delta} bytes in {window_s:.0f}s "
        f"(expected ≥ {min_bytes}). Is Hashpipe running?"
    )
