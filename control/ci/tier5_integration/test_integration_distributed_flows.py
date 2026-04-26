"""
test_integration_distributed_flows.py — Tier 5 Heavy Integration tests for multi-node runs.

Connects to the STATIC Docker Compose stack.
Verifies start/stop command propagation across nodes using real Hashpipe logic.
"""

from __future__ import annotations

import unittest.mock

import pytest

from control.start import start_run
from control.stop import stop_run
from control.utils import config_file


async def _node_hashpipe_running(node_ip: str) -> bool:
    """Query a single node via gRPC to check if hashpipe is running."""
    from panoseti_grpc.daq_control.client import DaqControlClient
    client = DaqControlClient(host=node_ip, port=50051)
    try:
        _, status = client.StatusDaq({
            "data_dir": "/data",
            "check_hashpipe_running": True,
            "check_disk_usage": False,
            "check_run_dirs": False
        })
        return status.get("hashpipe_running", False)
    finally:
        client.close()

@pytest.mark.asyncio
async def test_when_distributed_run_started_then_all_nodes_recording(
    tmp_path, daq_control_direct, daq_control_node2
) -> None:
    """Verify distributed gRPC orchestration for a multi-node run in heavy stack."""
    # Use the static configs from the environment
    obs_cfg      = config_file.get_obs_config()
    daq_cfg      = config_file.get_daq_config()
    quabo_uids   = config_file.get_quabo_uids()
    data_cfg     = config_file.get_data_config()
    network_cfg  = config_file.get_network_config()

    # Ensure head_node_data_dir exists for the test
    (tmp_path / "head_data").mkdir(parents=True, exist_ok=True)
    daq_cfg.head_node_data_dir = str(tmp_path / "head_data")

    run_name = "dist_test_run.pffd"
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
            run_name=run_name, no_check_daq=True,
        )

    # Each node must have hashpipe active
    for node in daq_cfg.daq_nodes:
        # Use mapped IPs for external client check
        ip = str(node.port_forwarding.gw_ip) if node.port_forwarding else str(node.ip_addr)
        assert await _node_hashpipe_running(ip), f"hashpipe not running on node {ip}"


@pytest.mark.asyncio
async def test_when_distributed_run_stopped_then_all_nodes_halted(
    tmp_path, daq_control_direct, daq_control_node2
) -> None:
    """Verify clean teardown of a distributed observing run in heavy stack."""
    daq_cfg     = config_file.get_daq_config()
    obs_cfg     = config_file.get_obs_config()
    quabo_uids  = config_file.get_quabo_uids()
    data_cfg    = config_file.get_data_config()
    network_cfg = config_file.get_network_config()

    (tmp_path / "head_data").mkdir(parents=True, exist_ok=True)
    daq_cfg.head_node_data_dir = str(tmp_path / "head_data")

    run_name = "stop_test_run.pffd"
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
            run_name=run_name, no_check_daq=True,
        )

    (tmp_path / "head_data" / run_name).mkdir(parents=True, exist_ok=True)

    await stop_run(
        daq_cfg, network_cfg, quabo_uids,
        run=run_name, no_collect=True, no_cleanup=True, no_transfer=True,
    )

    # All nodes must be halted
    for node in daq_cfg.daq_nodes:
        ip = str(node.port_forwarding.gw_ip) if node.port_forwarding else str(node.ip_addr)
        assert not await _node_hashpipe_running(ip), f"hashpipe still running on node {ip}"

    from control.utils.run_state import RunStateManager
    ledger = RunStateManager().load_state()
    assert ledger is not None and ledger.status == "RECORDING_ENDED"
