"""
Quabo hardware fixtures for HITL tests.
Provides session-scoped QUABO objects connected to real hardware,
derived from the active topology (no hardcoded IPs).
"""

from __future__ import annotations

import pytest

from control.driver.quabo_driver import QUABO


@pytest.fixture(scope="session")
def topology(obs_config, daq_config, network_config):
    """Session-scoped HwTopology wrapping the active configs."""
    from ci.hardware_software.hw_utils.topology import HwTopology
    return HwTopology()


@pytest.fixture(scope="session")
def quabo(topology) -> QUABO:
    """
    Return a QUABO object for the first (Q0) quabo of the first module.
    Tests that need all quabos should iterate topology.quabo_ips() directly.
    """
    addrs = topology.quabo_ips()
    if not addrs:
        pytest.skip("No quabos in active topology")
    first = next(a for a in addrs if a.quadrant == 0)
    return QUABO(first.ip)


@pytest.fixture(scope="session")
def all_quabos(topology) -> list[QUABO]:
    """Return QUABO objects for every quabo in the active topology."""
    return [QUABO(a.ip) for a in topology.quabo_ips()]
