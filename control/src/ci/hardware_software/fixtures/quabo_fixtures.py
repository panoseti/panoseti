"""
Quabo hardware fixtures for HITL tests.
Provides session-scoped QUABO objects connected to real hardware,
derived from the active topology (no hardcoded IPs).
"""

from __future__ import annotations

import pytest
from typing import Generator

from control.driver.quabo_driver import QUABO



@pytest.fixture(scope="session")
def topology(obs_config, daq_config, network_config):
    """Session-scoped HwTopology wrapping the active configs."""
    from ci.hardware_software.hw_utils.topology import HwTopology
    return HwTopology()


@pytest.fixture(scope="session")
def quabo(topology) -> Generator[QUABO]:
    """
    Return a QUABO object for the first (Q0) quabo of the first module.
    Tests that need all quabos should iterate topology.quabo_ips() directly.
    """
    addrs = topology.quabo_ips()
    if not addrs:
        pytest.skip("No quabos in active topology")
    first = next(a for a in addrs if a.quadrant == 0)
    q = QUABO(first.real_ip, port=first.cmd_port)
    yield q
    q.close()


@pytest.fixture(scope="session")
def all_quabos(topology) -> Generator[list[QUABO]]:
    """Return QUABO objects for every quabo in the active topology."""
    qs = [QUABO(a.real_ip, port=a.cmd_port) for a in topology.quabo_ips()]
    yield qs
    for q in qs:
        q.close()


@pytest.fixture(scope="session")
def maroc_config() -> dict[str, str]:
    """
    Return a known-good MAROC config dict with canonical string values.
    Required because make_maroc_cmd expects comma-joined strings, not ints.
    """
    # Just a few representative fields; the driver loads the rest from file if needed.
    # But for round-trip tests, we need a dict that doesn't crash make_maroc_cmd.
    cfg = {
        "GAIN_CHANNEL_0": "0,0,0,0",
        "GAIN_CHANNEL_1": "0,0,0,0",
        "OTABG_ON": "1",
        "DAC_ON": "1",
        "SMALL_DAC": "0",
    }
    # For a full valid dict, we should ideally load quabo_config.txt 
    # and ensure all values are strings.
    from control.driver.quabo_driver import parse_quabo_config_file
    from control.utils.paths import PanoPaths
    
    config_path = PanoPaths.config_dir() / "quabo_config.txt"
    if config_path.exists():
        return parse_quabo_config_file(str(config_path))
    return cfg
