"""
DaqData gRPC stream helpers.

Collects PanoImage frames from the daq_data gRPC server during an active run.
Requires the run to have been started with --init-snapshot (or a live
snapshot service on the DAQ node).
"""

from __future__ import annotations

import asyncio
import time


async def _collect_async(host: str, port: int, run_dir: str, module_id: int, n: int, timeout: float) -> list:
    from panoseti_grpc.daq_data.client import DaqDataClient
    frames: list = []
    deadline = time.monotonic() + timeout
    async with DaqDataClient(host, port) as client:
        await client.init_hp_io(run_dir, module_id)
        async for image in client.stream_images():
            frames.append(image)
            if len(frames) >= n or time.monotonic() > deadline:
                break
    return frames


def collect_n_frames(host: str, port: int, run_dir: str, module_id: int,
                     n: int = 10, timeout: float = 20.0) -> list:
    """Synchronous wrapper around the async DaqData stream.

    Args:
        host: DAQ node host (real_host from topology).
        port: gRPC port (typically 50051).
        run_dir: Run directory path on the DAQ node.
        module_id: Module ID to stream from.
        n: Minimum number of frames to collect.
        timeout: Maximum seconds to wait for n frames.

    Returns:
        List of PanoImage objects.

    Raises:
        AssertionError: if fewer than n frames are collected within timeout.
    """
    frames = asyncio.run(_collect_async(host, port, run_dir, module_id, n, timeout))
    assert len(frames) >= n, (
        f"DaqData stream yielded only {len(frames)} frames in {timeout:.0f}s "
        f"(expected ≥ {n}) from {host}:{port} module={module_id}"
    )
    return frames
