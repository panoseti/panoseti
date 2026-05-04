"""Transfer queue assertions for happy-path tests."""

from __future__ import annotations

from ci.hardware_software.core import queue as _queue


def only_in_bucket(run_name: str, expected_bucket: str, timeout: float = 0.0) -> None:
    """Assert run_name is in expected_bucket and absent from all others."""
    _queue.assert_only_in_bucket(run_name, expected_bucket, timeout=timeout)
