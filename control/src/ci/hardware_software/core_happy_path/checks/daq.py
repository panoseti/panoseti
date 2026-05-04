"""DAQ node assertions for happy-path tests."""

from __future__ import annotations

from ci.hardware_software.core import daq_status as _daq


def disk_growing(host: str, port: int, run_name: str,
                 min_bytes: int = 500_000, window_s: float = 10.0) -> None:
    """Assert the DAQ node is actively writing data."""
    _daq.assert_disk_growing(host, port, run_name, min_bytes=min_bytes, window_s=window_s)
