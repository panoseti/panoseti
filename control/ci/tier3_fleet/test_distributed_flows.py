"""
ci/tier3_fleet/test_distributed_flows.py

End-to-end distributed orchestration tests using dynamic fleets.
Verifies that start/stop commands propagate correctly across N nodes.
"""

from __future__ import annotations

import os
from ipaddress import IPv4Address
from unittest.mock import patch

import pytest

from control.start import start_run
from control.stop import stop_run
from control.utils import config_file
from ci.integration.fleet import make_fleet
from ci.fixtures.state_probe import StateProbe

def is_in_ci() -> bool:
    return os.path.exists("/.dockerenv")

pytestmark = pytest.mark.skipif(
    not is_in_ci(), reason="Fleet tests require the full Docker CI stack"
)

@pytest.mark.asyncio
async def test_when_distributed_run_started_then_all_nodes_recording(
    tmp_path,
    probe: StateProbe
) -> None:
    """
    Intent: Verify distributed gRPC orchestration for a multi-node run.
    Scenario: Dynamic 2-node fleet. 
    Assertion: Both nodes report 'hashpipe_running': True after start_run().
    """
    # 1. Spin up dynamic fleet
    fleet = make_fleet(n=2)
    fleet.start()
    try:
        # 2. Write dynamic configs for this fleet
        daq_config_path = tmp_path / "daq_config.json"
        head_node_ip = "10.0.1.1" # Mock head node IP in the shared network
        fleet.write_daq_config(daq_config_path, head_node_ip)
        
        # Load the generated config
        daq_cfg = config_file.get_daq_config(dir=str(tmp_path))
        
        # 3. Start the run
        run_name = "dist_test_run.pffd"
        await start_run(daq_cfg, run_name=run_name, no_hv=True)
        
        # 4. Verify Assertion: Both nodes must have hashpipe active
        for node in daq_cfg.daq_nodes:
            assert await probe.is_hashpipe_running(str(node.ip_addr))
            
    finally:
        fleet.tear_down()

@pytest.mark.asyncio
async def test_when_distributed_run_stopped_then_all_nodes_halted(
    tmp_path,
    probe: StateProbe
) -> None:
    """
    Intent: Verify clean teardown of a distributed observing run.
    Scenario: Dynamic 2-node fleet with an active run.
    Assertion: Hashpipe is stopped on all nodes and ledger reaches RECORDING_ENDED.
    """
    fleet = make_fleet(n=2)
    fleet.start()
    try:
        daq_config_path = tmp_path / "daq_config.json"
        fleet.write_daq_config(daq_config_path, "10.0.1.1")
        daq_cfg = config_file.get_daq_config(dir=str(tmp_path))
        
        # Pre-requisite: Run is active
        run_name = "stop_test_run.pffd"
        await start_run(daq_cfg, run_name=run_name, no_hv=True)
        
        # 1. Execute stop_run
        # (Need mock network/obs configs for full stop logic)
        net = config_file.get_network_config()
        uids = config_file.get_quabo_uids()
        await stop_run(daq_cfg, net, uids, run=run_name)
        
        # 2. Verify Assertion: Hashpipe halted
        for node in daq_cfg.daq_nodes:
             assert not await probe.is_hashpipe_running(str(node.ip_addr))
             
        # 3. Verify Ledger Assertion
        assert probe.ledger_status() == "RECORDING_ENDED"
            
    finally:
        fleet.tear_down()
