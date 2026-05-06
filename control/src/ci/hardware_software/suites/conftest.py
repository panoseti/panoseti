"""
Shared conftest for all HITL suites.
Re-exports fixtures from the fixtures/ package so pytest discovers them
without requiring explicit imports in each test file.
"""

from ci.fixtures.topology_fixtures import topology
