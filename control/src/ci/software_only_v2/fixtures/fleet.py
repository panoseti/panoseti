"""
fixtures/fleet.py — session_fleet fixture for v2 tier-3+ tests.

Manages a Fleet (headnode + N sim daqnodes) scoped to the test session.
Uses the pseti_workspace_session fixture for config materialization.

Usage::

    def test_something(session_fleet):
        client = session_fleet.daq_control_client(0)
        ...

Parametric (different topology)::

    @pytest.mark.parametrize(
        "pseti_workspace_session",
        [FleetSpec.two_node_ci()],
        indirect=True,
    )
    def test_two_node(session_fleet):
        assert session_fleet.n_nodes == 2
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

from ci.software_only_v2.infra.workspace import Workspace

if TYPE_CHECKING:
    from ci.software_only_v2.orchestrator.fleet import Fleet


@pytest.fixture(scope="session")
def session_fleet(pseti_workspace_session: Workspace) -> Iterator[Fleet]:
    """Session-scoped Fleet: headnode + sim daqnodes, healthy before first test.

    The Fleet context manager starts all containers in start(), calls
    wait_healthy() to block until gRPC is READY, then tears everything down
    after the last test in the session exits.
    """
    # Import deferred so testcontainers is not imported at module level;
    # tier1/2 tests can load this conftest plugin without Docker being present.
    from ci.software_only_v2.orchestrator.fleet import Fleet

    topology = pseti_workspace_session.topology
    workspace = pseti_workspace_session

    fleet = Fleet.from_topology(topology, workspace)
    with fleet:
        fleet.wait_healthy()
        yield fleet
