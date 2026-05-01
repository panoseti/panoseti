"""
Shared conftest for all HITL suites.
Re-exports fixtures from the fixtures/ package so pytest discovers them
without requiring explicit imports in each test file.
"""

from ci.hardware_software.fixtures.packet_capture import fake_socket, real_udp_capture  # noqa: F401
from ci.hardware_software.fixtures.quabo_fixtures import all_quabos, maroc_config, quabo, topology  # noqa: F401
from ci.hardware_software.fixtures.telemetry_fixtures import (  # noqa: F401
    hk_socket,
    influx_client,
    redis_client,
)
from ci.hardware_software.fixtures.wps_fixtures import wps_outlet, wps_power_on  # noqa: F401
