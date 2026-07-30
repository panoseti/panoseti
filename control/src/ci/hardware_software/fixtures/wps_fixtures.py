"""
WPS (Web Power Switch) fixtures for HITL tests.
"""

from __future__ import annotations

import pytest

from control.power import quabo_power


@pytest.fixture(scope="session")
def wps_outlet(topology):
    """Return the first WPS outlet from the active topology, or skip if none defined."""
    outlets = topology.wps_outlets()
    if not outlets:
        pytest.skip("No WPS outlets configured in obs_config")
    return outlets[0]


@pytest.fixture
def wps_power_on(wps_outlet):
    """Turn Quabo power on; yield; power off in teardown."""
    quabo_power({"url": wps_outlet.url, "quabo_socket": wps_outlet.quabo_socket}, on=True)
    yield wps_outlet
    quabo_power({"url": wps_outlet.url, "quabo_socket": wps_outlet.quabo_socket}, on=False)
