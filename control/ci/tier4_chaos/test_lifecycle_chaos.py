"""
ci/tier4_chaos/test_lifecycle_chaos.py

Chaos tests for the run lifecycle (start/stop) requiring fault injection.
Verifies the 'Rollback Ladder' and post-mortem snapshot integrity.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from ci.fixtures.mocks import MockDaqNode
from ci.fixtures.state_probe import StateProbe
from control.start import start_run


def is_in_ci() -> bool:
    return os.path.exists("/.dockerenv")

pytestmark = pytest.mark.skipif(
    not is_in_ci(), reason="Chaos tests require the full Docker CI stack"
)

@pytest.mark.asyncio
async def test_when_one_node_fails_during_start_then_all_nodes_rolled_back(
    daq_config_factory,
    probe: StateProbe
) -> None:
    """
    Intent: Verify the 'Rollback Ladder' architectural invariant. 
           If StartDaq fails for one node, all previously started nodes 
           must be stopped, and a post-mortem snapshot captured.
    Scenario: 2-node fleet. Node 0 starts successfully; Node 1 fails.
    Assertion: Node 0 receives a StopDaq call, and a snapshot exists in state/snapshots/.
    """
    run_name = "rollback_chaos.pffd"
    daq_config = daq_config_factory(node_ips=["192.168.0.10", "192.168.0.20"])
    
    # 1. Setup mocks
    mock_daq0 = MockDaqNode("192.168.0.10")
    mock_daq1 = MockDaqNode("192.168.0.20")
    mock_daq1.client.StartDaq.side_effect = Exception("Node 1 Hardware Failure")
    
    # Map IPs to our mock clients
    def mock_client_factory(host, port):
        if host == "192.168.0.10":
            return mock_daq0.client
        if host == "192.168.0.20":
            return mock_daq1.client
        return MagicMock()

    with (
        patch("control.start.AsyncDaqControlClient", side_effect=mock_client_factory),
        patch("control.start.ph_baseline_file_ok", return_value=True),
        patch("control.start._check_daq_reachability")
    ):
        # 2. Execute start_run - this should raise due to Node 1 failure
        with pytest.raises(Exception, match="Node 1 Hardware Failure"):
            await start_run(daq_config, run_name=run_name, no_hv=True)
            
    # 3. Verify Rollback Assertion: Node 0 must have received a StopDaq call
    # even though its StartDaq succeeded.
    mock_daq0.client.StopDaq.assert_called_once()
    
    # 4. Verify Post-mortem Assertion: A snapshot must exist for this failed run
    assert probe.aborted_snapshot_exists(run_name), \
        "Rollback ladder failed to capture a post-mortem state snapshot"
    
    # 5. Verify Ledger Assertion: Status should reach a terminal error state or be cleared
    assert probe.ledger_status() in [None, "ABORTED", "STOPPED_WITH_ERRORS"]
