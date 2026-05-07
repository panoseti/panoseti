"""
tier5_integration/test_integration_distributed_flows.py — Multi-node distributed flows.

Tests start_run() / stop_run() orchestration across two real daqnodes using
configs materialized by the t5_workspace fixture (FleetSpec.two_node_ci()).
"""

from __future__ import annotations

import os
import pathlib
import unittest.mock
import uuid
from typing import Any

import pytest

from ci.software_only_v2.tier5_integration.conftest import (
    DAQ_DATA_DIR,
    GRPC_PORT,
    requires_compose_stack,
    wait_hashpipe_stopped,
)

pytestmark = [pytest.mark.tier5, requires_compose_stack]


def _prepare_daq_dirs(daq_cfg: Any, run_name: str) -> None:
    host_root = pathlib.Path(DAQ_DATA_DIR)
    (host_root / run_name).mkdir(parents=True, exist_ok=True)
    os.chmod(host_root / run_name, 0o777)
    for node in daq_cfg.daq_nodes:
        for mid in node.module_ids:
            mod_dir = host_root / f"module_{mid}" / run_name
            mod_dir.mkdir(parents=True, exist_ok=True)
            dummy = mod_dir / "dummy.pff"
            dummy.touch()
            os.chmod(mod_dir, 0o777)
            os.chmod(dummy, 0o777)


async def _node_hashpipe_running(node_ip: str) -> bool:
    from panoseti_grpc.daq_control.client import DaqControlClient
    client = DaqControlClient(host=node_ip, port=GRPC_PORT)
    try:
        _, status = client.StatusDaq({
            "data_dir": DAQ_DATA_DIR,
            "check_hashpipe_running": True,
            "check_disk_usage": False,
            "check_run_dirs": False,
        })
        return status.get("hashpipe_running", False)
    finally:
        client.close()


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_distributed_run_all_nodes_recording(
    tmp_path: pathlib.Path,
    t5_workspace: Any,
    daq_control_node1: Any,
    daq_control_node2: Any,
) -> None:
    """start_run() propagates to all daqnodes; each reports hashpipe running."""
    from control.start import start_run
    from control.utils import config_file

    daq_cfg = config_file.get_daq_config()
    obs_cfg = config_file.get_obs_config()
    quabo_uids = t5_workspace.topology.quabo_uids
    data_cfg = config_file.get_data_config()
    network_cfg = config_file.get_network_config()

    head_run = tmp_path / "head_data"
    head_run.mkdir(parents=True, exist_ok=True)
    daq_cfg.head_node_data_dir = str(head_run)

    run_name = f"dist_start_{uuid.uuid4().hex[:8]}.pffd"
    _prepare_daq_dirs(daq_cfg, run_name)

    with (
        unittest.mock.patch("control.start.ph_baseline_file_ok", return_value=True),
        unittest.mock.patch("control.start._check_daq_reachability"),
        unittest.mock.patch("control.start._check_quabo_reachability"),
        unittest.mock.patch("control.start.start_data_flow"),
        unittest.mock.patch("control.start.make_run_dirs"),
        unittest.mock.patch("control.start.util.start_hk_recorder"),
        unittest.mock.patch("control.start.util.write_run_name"),
    ):
        await start_run(
            obs_cfg, daq_cfg, quabo_uids, data_cfg, network_cfg,
            no_hv=True, no_redis=True, no_data=False,
            run_name=run_name, no_check_daq=True,
        )

    for node in daq_cfg.daq_nodes:
        ip = (
            str(node.port_forwarding.gw_ip)
            if node.port_forwarding
            else str(node.ip_addr)
        )
        assert await _node_hashpipe_running(ip), (
            f"hashpipe not running on node {ip}"
        )


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_distributed_run_all_nodes_halted_after_stop(
    tmp_path: pathlib.Path,
    t5_workspace: Any,
    daq_control_node1: Any,
    daq_control_node2: Any,
) -> None:
    """stop_run() halts hashpipe on all nodes and advances ledger to RECORDING_ENDED."""
    from panoseti_grpc.daq_control.client import DaqControlClient

    from control.start import start_run
    from control.stop import stop_run
    from control.utils import config_file
    from control.utils.run_state import RunStateManager, RunStatus

    daq_cfg = config_file.get_daq_config()
    obs_cfg = config_file.get_obs_config()
    quabo_uids = t5_workspace.topology.quabo_uids
    data_cfg = config_file.get_data_config()
    network_cfg = config_file.get_network_config()

    head_run = tmp_path / "head_data"
    head_run.mkdir(parents=True, exist_ok=True)
    daq_cfg.head_node_data_dir = str(head_run)

    run_name = f"dist_stop_{uuid.uuid4().hex[:8]}.pffd"
    _prepare_daq_dirs(daq_cfg, run_name)

    with (
        unittest.mock.patch("control.start.ph_baseline_file_ok", return_value=True),
        unittest.mock.patch("control.start._check_daq_reachability"),
        unittest.mock.patch("control.start._check_quabo_reachability"),
        unittest.mock.patch("control.start.start_data_flow"),
        unittest.mock.patch("control.start.make_run_dirs"),
        unittest.mock.patch("control.start.util.start_hk_recorder"),
        unittest.mock.patch("control.start.util.write_run_name"),
    ):
        await start_run(
            obs_cfg, daq_cfg, quabo_uids, data_cfg, network_cfg,
            no_hv=True, no_redis=True, no_data=False,
            run_name=run_name, no_check_daq=True,
        )

    (head_run / run_name).mkdir(parents=True, exist_ok=True)

    await stop_run(
        daq_cfg, network_cfg, quabo_uids,
        run=run_name, no_collect=True, no_cleanup=True, no_transfer=True,
    )

    for node in daq_cfg.daq_nodes:
        ip = (
            str(node.port_forwarding.gw_ip)
            if node.port_forwarding
            else str(node.ip_addr)
        )
        client = DaqControlClient(host=ip, port=GRPC_PORT)
        try:
            assert wait_hashpipe_stopped(client, DAQ_DATA_DIR, timeout=15), (
                f"hashpipe still running on node {ip} after stop_run()"
            )
        finally:
            client.close()

    ledger = RunStateManager().load_state()
    assert ledger is not None
    assert ledger.status == RunStatus.RECORDING_ENDED
