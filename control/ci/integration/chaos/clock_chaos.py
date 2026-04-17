"""
chaos/clock_chaos.py

Clock skew injection for timing tests.

Two modes:
  1. OS-level skew: `date -s` inside a container (requires SYS_TIME capability).
  2. In-process skew: monkey-patch time.time() for a function under test.
"""

from __future__ import annotations

import contextlib
import time as _time_module
import unittest.mock as _mock
from contextlib import contextmanager
from typing import Callable, Generator

from . import process_chaos as _pc


@contextmanager
def skew_container_clock(
    container_name: str,
    skew_seconds: float,
) -> Generator[None, None, None]:
    """
    Skew the system clock inside a container by skew_seconds.

    Requires SYS_TIME capability on the container.
    Restores the original time on exit.
    """
    # Capture current time before skewing
    _pc._exec(container_name, "date '+%s' > /tmp/clock_orig")
    _pc._exec(
        container_name,
        f"date -s \"$(date -d '+{skew_seconds} seconds' '+%Y-%m-%d %H:%M:%S')\"",
    )
    try:
        yield
    finally:
        with contextlib.suppress(Exception):
            _pc._exec(
                container_name,
                "date -s \"$(cat /tmp/clock_orig | xargs -I{} date -d @{} '+%Y-%m-%d %H:%M:%S')\"",
            )


@contextmanager
def skew_process_time(skew_seconds: float) -> Generator[None, None, None]:
    """Monkey-patch time.time() to return current_time + skew_seconds.

    Only affects code in the current process (test runner side).
    Does NOT affect subprocess or gRPC server clocks.
    """
    real_time = _time_module.time
    with _mock.patch("time.time", side_effect=lambda: real_time() + skew_seconds):
        yield


@contextmanager
def skew_tv_sec(skew_seconds: int) -> Generator[None, None, None]:
    """Return skewed tv_sec value — patches time.time() for PFF timestamp tests."""
    with skew_process_time(float(skew_seconds)):
        yield
