"""DAQ node assertions for happy-path tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ci.hardware_software.core import daq_status as _daq

if TYPE_CHECKING:
    from control.utils.pydantic_config_models import DaqConfig, DaqNode


def disk_growing(
    node: DaqNode,
    daq_config: DaqConfig,
    run_name: str,
    min_bytes: int = 500_000,
    window_s: float = 10.0,
) -> None:
    """Assert the DAQ node is actively writing data."""
    _daq.assert_disk_growing(node, daq_config, run_name, min_bytes=min_bytes, window_s=window_s)


def hashpipe_healthy(
    node: DaqNode,
    daq_config: DaqConfig,
    run_name: str,
) -> None:
    """Assert Hashpipe is running AND past its stuck-at-init window (thread count)."""
    _daq.assert_hashpipe_healthy(node, daq_config, run_name)
