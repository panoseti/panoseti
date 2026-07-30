"""
DaqData gRPC stream helpers.

Collects PanoImage frames from the daq_data gRPC server during an active run.
"""

from __future__ import annotations

import asyncio
import time

from panoseti_grpc.daq_data.client import AioDaqDataClient


async def _collect_async(host: str, port: int, n: int, timeout_s: float) -> list:
    frames: list = []
    deadline = time.monotonic() + timeout_s

    async with AioDaqDataClient(host, port) as client:
        async for image in client.stream_images(
            stream_movie_data=True,
            stream_pulse_height_data=True,
            update_interval_seconds=0.5,
        ):
            frames.append(image)
            if len(frames) >= n or time.monotonic() > deadline:
                break
    return frames


def collect_n_frames(host: str, port: int, run_dir: str, module_id: int,
                     n: int = 10, timeout: float = 20.0) -> list:
    """Synchronous wrapper around the async DaqData stream.

    Args:
        host: DAQ node host (or gateway host).
        port: gRPC port (typically 50051).
        run_dir: Run directory path (unused; kept for call-site compatibility).
        module_id: Module ID to stream from (unused; kept for call-site compatibility).
        n: Minimum number of frames to collect.
        timeout: Maximum seconds to wait for n frames.

    Returns:
        List of PanoImage objects.

    Raises:
        AssertionError: if fewer than n frames are collected within timeout.
    """
    frames = asyncio.run(_collect_async(host, port, n, timeout))
    assert len(frames) >= n, (
        f"DaqData stream yielded only {len(frames)} frames in {timeout:.0f}s "
        f"(expected ≥ {n}) from {host}:{port} module={module_id}"
    )
    return frames
