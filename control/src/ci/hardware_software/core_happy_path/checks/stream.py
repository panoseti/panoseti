"""DaqData stream assertions for happy-path tests."""

from __future__ import annotations

import pytest

from ci.hardware_software.core import stream as _stream


def yields_frames(host: str, port: int, run_dir: str, module_id: int,
                  n: int = 10, timeout: float = 20.0) -> list:
    """Assert ≥ n frames are received from the gRPC stream.

    Skips (rather than fails) if the daq_data service is unavailable, since
    the DaqData snapshot server requires --init-snapshot at start time and
    is optional for basic observing tests.

    Returns the collected frames for further assertions if desired.
    """
    try:
        return _stream.collect_n_frames(host, port, run_dir, module_id, n=n, timeout=timeout)
    except Exception as exc:
        if "unavailable" in str(exc).lower() or "connection" in str(exc).lower():
            pytest.skip(f"DaqData gRPC unavailable: {exc}")
        raise
