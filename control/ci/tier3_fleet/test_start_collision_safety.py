"""
test_start_collision_safety.py — Fleet tests for run collision and rollback safety.

Verifies:
1. A second start attempt with overlapping modules is rejected if a run is ACTIVE.
2. The active run is NOT stopped by the aborted second attempt (Rollback Ladder Safety).
3. Using --force-reset correctly stops the first run and its Hashpipe instances.
"""

from __future__ import annotations

import contextlib
import os
import unittest.mock
import uuid
from collections.abc import Iterator

import pytest

from ci.tier3_fleet.conftest import wait_hashpipe_stopped
from control.start import start_run
from control.utils import config_file
from control.utils.run_state import RunStateManager


@pytest.fixture
def ensure_clean_daq_state(daq_control_direct, daq_control_node2) -> Iterator[None]:
    """Ensure no Hashpipe instances are running before and after the test."""
    def _stop_all():
        for client in (daq_control_direct, daq_control_node2):
            with contextlib.suppress(Exception):
                # Use a long timeout to allow the server's 60s graceful wait to complete
                client.StopDaq({"data_dir": "/data", "run_dir": ""}, timeout=70.0)
            wait_hashpipe_stopped(client, "/data", timeout=10)
        from control.utils.run_state import RunStateManager
        RunStateManager().clear_state()

    _stop_all()
    yield
    _stop_all()


async def _check_hashpipe_on_nodes(daq_cfg, expected_running: bool):
    """Utility to check hashpipe status on all nodes via gRPC."""
    from panoseti_grpc.daq_control.client import DaqControlClient
    for node in daq_cfg.daq_nodes:
        if not node.module_ids:
            continue
        ip = str(node.port_forwarding.gw_ip) if node.port_forwarding else str(node.ip_addr)
        port = node.port_forwarding.grpc_port if node.port_forwarding else 50051
        client = DaqControlClient(host=ip, port=port)
        try:
            _, status = client.StatusDaq({
                "data_dir": node.data_dir,
                "check_hashpipe_running": True
            })
            assert status.get("hashpipe_running") == expected_running, \
                f"Node {ip} hashpipe_running should be {expected_running}"
        finally:
            client.close()


async def _prepare_daq_dirs_fleet(fleet, daq_cfg, run_name):
    """Prepare directories on testcontainers nodes via docker exec."""
    for i, node in enumerate(daq_cfg.daq_nodes):
        container = fleet.containers[i].get_wrapped_container()
        # Create root run dir
        container.exec_run(f"mkdir -p {node.data_dir}/{run_name}")
        # Create module run dirs and dummy files
        for mid in node.module_ids:
            mod_dir = f"{node.data_dir}/module_{mid}/{run_name}"
            container.exec_run(f"mkdir -p {mod_dir}")
            # Fix: Touch a .pff file to satisfy hashpipe output thread
            container.exec_run(f"touch {mod_dir}/dummy.pff")


@pytest.mark.asyncio
async def test_start_collision_does_not_stop_active_run(session_fleet, tmp_path, ensure_clean_daq_state):
    """
    Verify that an aborted start attempt does NOT kill a pre-existing active run.
    """
    fleet, daq_cfg_dict = session_fleet
    daq_cfg = config_file.DaqConfig.model_validate(daq_cfg_dict)
    config_dir = os.environ.get("PSETI_CONFIG")
    
    obs_cfg      = config_file.get_obs_config(dir=config_dir)
    quabo_uids   = config_file.get_quabo_uids()
    data_cfg     = config_file.get_data_config(dir=config_dir)
    network_cfg  = config_file.get_network_config(dir=config_dir)

    # 1. Start Run 1 (Success)
    run1_name = f"run1_{uuid.uuid4().hex[:8]}.pffd"
    await _prepare_daq_dirs_fleet(fleet, daq_cfg, run1_name)
    
    # Mock hardware interaction but keep gRPC and ledger logic real
    with unittest.mock.patch("control.start.ph_baseline_file_ok", return_value=True), \
            unittest.mock.patch("control.start._check_daq_reachability"), \
            unittest.mock.patch("control.start._check_quabo_reachability"), \
            unittest.mock.patch("control.start.start_data_flow"), \
            unittest.mock.patch("control.start.make_run_dirs"), \
            unittest.mock.patch("control.start.util.start_hk_recorder"), \
            unittest.mock.patch("control.start.util.write_run_name"):
        
        res1 = await start_run(
            obs_cfg, daq_cfg, quabo_uids, data_cfg, network_cfg,
            no_hv=True, no_redis=True, no_data=False,
            run_name=run1_name, no_check_daq=True
        )
        assert res1 == run1_name
        
    # Verify Run 1 is active
    ledger = RunStateManager().load_state()
    assert ledger.status == "ACTIVE"
    await _check_hashpipe_on_nodes(daq_cfg, expected_running=True)

    # 2. Attempt Run 2 (Should abort due to ACTIVE ledger)
    run2_name = f"run2_{uuid.uuid4().hex[:8]}.pffd"
    await _prepare_daq_dirs_fleet(fleet, daq_cfg, run2_name)
    
    with unittest.mock.patch("control.start.ph_baseline_file_ok", return_value=True), \
            unittest.mock.patch("control.start._check_daq_reachability"), \
            unittest.mock.patch("control.start._check_quabo_reachability"), \
            unittest.mock.patch("control.start.start_data_flow"), \
            unittest.mock.patch("control.start.make_run_dirs"), \
            unittest.mock.patch("control.start.util.start_hk_recorder"), \
            unittest.mock.patch("control.start.util.write_run_name"):
        
        res2 = await start_run(
            obs_cfg, daq_cfg, quabo_uids, data_cfg, network_cfg,
            no_hv=True, no_redis=True, no_data=False,
            run_name=run2_name, no_check_daq=True
        )
        # Should return None because it aborted
        assert res2 is None

    # 3. CRITICAL: Verify Run 1 is STILL ACTIVE and Hashpipe is STILL RUNNING
    # This confirms the second start attempt didn't erroneously roll back the first run's hardware.
    ledger = RunStateManager().load_state()
    assert ledger.run_name == run1_name
    assert ledger.status == "ACTIVE"
    await _check_hashpipe_on_nodes(daq_cfg, expected_running=True)


@pytest.mark.asyncio
async def test_start_with_force_reset_stops_previous_run(session_fleet, tmp_path, ensure_clean_daq_state):
    """
    Verify that start_run --force-reset correctly cleans up a previous active run.
    """
    fleet, daq_cfg_dict = session_fleet
    daq_cfg = config_file.DaqConfig.model_validate(daq_cfg_dict)
    config_dir = os.environ.get("PSETI_CONFIG")
    
    obs_cfg      = config_file.get_obs_config(dir=config_dir)
    quabo_uids   = config_file.get_quabo_uids()
    data_cfg     = config_file.get_data_config(dir=config_dir)
    network_cfg  = config_file.get_network_config(dir=config_dir)

    # 1. Start Run 1 (Success)
    run1_name = f"run1_{uuid.uuid4().hex[:8]}.pffd"
    await _prepare_daq_dirs_fleet(fleet, daq_cfg, run1_name)
    
    with unittest.mock.patch("control.start.ph_baseline_file_ok", return_value=True), \
            unittest.mock.patch("control.start._check_daq_reachability"), \
            unittest.mock.patch("control.start._check_quabo_reachability"), \
            unittest.mock.patch("control.start.start_data_flow"), \
            unittest.mock.patch("control.start.make_run_dirs"), \
            unittest.mock.patch("control.start.util.start_hk_recorder"), \
            unittest.mock.patch("control.start.util.write_run_name"):
        
        await start_run(
            obs_cfg, daq_cfg, quabo_uids, data_cfg, network_cfg,
            no_hv=True, no_redis=True, no_data=False,
            run_name=run1_name, no_check_daq=True
        )
    
    await _check_hashpipe_on_nodes(daq_cfg, expected_running=True)

    # 2. Start Run 2 with --force-reset
    run2_name = f"run2_{uuid.uuid4().hex[:8]}.pffd"
    await _prepare_daq_dirs_fleet(fleet, daq_cfg, run2_name)
    
    # We must patch hardware reachability but NOT _check_no_remote_hashpipe
    with unittest.mock.patch("control.start.ph_baseline_file_ok", return_value=True), \
            unittest.mock.patch("control.start._check_daq_reachability"), \
            unittest.mock.patch("control.start._check_quabo_reachability"), \
            unittest.mock.patch("control.start.start_data_flow"), \
            unittest.mock.patch("control.start.make_run_dirs"), \
            unittest.mock.patch("control.start.util.start_hk_recorder"), \
            unittest.mock.patch("control.start.util.write_run_name"):
        
        res2 = await start_run(
            obs_cfg, daq_cfg, quabo_uids, data_cfg, network_cfg,
            no_hv=True, no_redis=True, no_data=False,
            run_name=run2_name, no_check_daq=True,
            force_reset=True, force_restart=True # force_restart is needed to actually stop hashpipe
        )
        assert res2 == run2_name

    # 3. Verify Run 2 is ACTIVE and Run 1 was cleared
    ledger = RunStateManager().load_state()
    assert ledger.run_name == run2_name
    assert ledger.status == "ACTIVE"
    await _check_hashpipe_on_nodes(daq_cfg, expected_running=True)
