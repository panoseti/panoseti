"""
test_integration_distributed_flows.py — Tier 5 Heavy Integration tests for multi-node runs.

Connects to the STATIC Docker Compose stack.
Verifies start/stop command propagation across nodes using real Hashpipe logic.
"""

from __future__ import annotations

import os
import unittest.mock
import uuid

import pytest

from ci.tier3_fleet.conftest import wait_hashpipe_stopped
from control.start import start_run
from control.stop import stop_run
from control.utils import config_file
from control.utils.run_state import RunStateManager


def _prepare_daq_dirs(daq_cfg, run_name: str) -> None:
    """Prepare host-side directories mapped to the container's /data."""
    import pathlib
    host_data_root = os.environ.get("DAQ_DATA_DIR")
    if not host_data_root:
        return
    host_root = pathlib.Path(host_data_root)
    # Root run dir for validator
    main_dir = host_root / run_name
    main_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(main_dir, 0o777)
    for node in daq_cfg.daq_nodes:
        for mid in node.module_ids:
            mod_dir = host_root / f"module_{mid}" / run_name
            mod_dir.mkdir(parents=True, exist_ok=True)
            dummy_file = mod_dir / "dummy.pff"
            dummy_file.touch()
            os.chmod(mod_dir, 0o777)
            os.chmod(dummy_file, 0o777)


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
    tmp_path, daq_control_direct, daq_control_node2, ensure_clean_daq_state
) -> None:
    """Verify distributed gRPC orchestration for a multi-node run in heavy stack."""
    # Use the session-isolated configs from the environment
    print(f'{os.environ.get("PSETI_CONFIG", None)=}')
    config_dir = os.environ.get("PSETI_CONFIG", str(config_file.PanoPaths.config_dir()))
    obs_cfg      = config_file.get_obs_config(dir=config_dir)
    daq_cfg      = config_file.get_daq_config(dir=config_dir)
    # quabo_uids.json is in tmp_dir
    quabo_uids   = config_file.get_quabo_uids()
    data_cfg     = config_file.get_data_config(dir=config_dir)
    network_cfg  = config_file.get_network_config(dir=config_dir)

    from control.utils.config_file import validate_all
    validate_all(debug=True)

    # Ensure head_node_data_dir exists for the test
    (tmp_path / "head_data").mkdir(parents=True, exist_ok=True)
    daq_cfg.head_node_data_dir = str(tmp_path / "head_data")

    run_name = f"dist_test_run_{uuid.uuid4().hex[:8]}.pffd"
    _prepare_daq_dirs(daq_cfg, run_name)
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
    tmp_path, daq_control_direct, daq_control_node2, ensure_clean_daq_state
) -> None:
    """Verify clean teardown of a distributed observing run in heavy stack."""
    config_dir = os.environ.get("PSETI_CONFIG", str(config_file.PanoPaths.config_dir()))
    print(f"{config_dir=}")
    daq_cfg     = config_file.get_daq_config(dir=config_dir)
    obs_cfg     = config_file.get_obs_config(dir=config_dir)
    quabo_uids  = config_file.get_quabo_uids()
    data_cfg    = config_file.get_data_config(dir=config_dir)
    network_cfg = config_file.get_network_config(dir=config_dir)

    (tmp_path / "head_data").mkdir(parents=True, exist_ok=True)
    daq_cfg.head_node_data_dir = str(tmp_path / "head_data")

    run_name = f"stop_test_run_{uuid.uuid4().hex[:8]}.pffd"
    _prepare_daq_dirs(daq_cfg, run_name)
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
        # Use mapped IPs for external client check
        from panoseti_grpc.daq_control.client import DaqControlClient
        client = DaqControlClient(host=ip, port=50051)
        try:
            assert wait_hashpipe_stopped(client, "/data", timeout=15), f"hashpipe still running on node {ip}"
            # Explicit PID check
            _ok, status = client.StatusDaq({"data_dir": "/data", "check_hashpipe_running": True})
            assert status.get("hashpipe_pid") is None, f"Node {ip} still reporting hashpipe_pid={status.get('hashpipe_pid')}"
        finally:
            client.close()

    ledger = RunStateManager().load_state()
    assert ledger is not None and ledger.status == "RECORDING_ENDED"
