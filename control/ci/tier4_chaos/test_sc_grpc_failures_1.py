"""
scenarios/test_sc_grpc_failures_1.py

SC-001, SC-005, SC-006: gRPC failure isolation tests.
Part 1 of partitioned test suite.
"""

from __future__ import annotations

import contextlib
import unittest.mock
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from panoseti_grpc.daq_control.client import DaqControlClient

from ci.fixtures.state_probe import StateProbe
from ci.tier3_fleet.conftest import (
    DAQ_DATA_DIR,
    wait_hashpipe_running,
)
from ci.tier4_chaos.conftest import (
    StopPartialFailure,
)
from ci.tier4_chaos.conftest import (
    _stop as grpc_stop,
)

# ── SC-001: StartDaq timeout ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_SC001_startdaq_timeout_hangs_forever(
    daq_control_direct: DaqControlClient,
) -> None:
    """
    StartDaq with no timeout hangs forever when daqnode is unresponsive.

    Current bug (start.py): no deadline on DaqControlClient.StartDaq; hangs forever.
    Fix required: deadline/timeout on all StartDaq calls.
    """
    import asyncio

    import anyio

    import control.start as start
    from control.utils import config_file

    # Mock configs to avoid file I/O and point to loopback
    daq_config = config_file.get_daq_config()
    obs_config = config_file.get_obs_config()
    quabo_uids = config_file.get_quabo_uids()
    data_config = config_file.get_data_config()
    network_config = config_file.get_network_config()

    # Ensure daq_config handles all modules in the chaos quabo_uids to avoid validation errors
    mids = []
    for dome in quabo_uids.domes:
        for mod in dome.modules:
            mids.append(mod.id)
    daq_config.daq_nodes[0].module_ids = mids
    daq_config.head_node_container = True

    # Mock StartDaq to hang
    async def hanging_start_daq(*args: Any, **kwargs: Any) -> bool:
        await asyncio.sleep(3)  # Async hang
        return True

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.StartDaq = AsyncMock(side_effect=hanging_start_daq)

    with unittest.mock.patch("control.start.AsyncDaqControlClient", return_value=mock_client), \
         unittest.mock.patch("control.start.ph_baseline_file_ok", return_value=True), \
         unittest.mock.patch("control.start._check_daq_reachability"), \
         unittest.mock.patch("control.start.make_run_dirs"), \
         unittest.mock.patch("control.start.start_data_flow"), \
         unittest.mock.patch("control.start.util.is_hk_recorder_running", return_value=False), \
         unittest.mock.patch("control.start.util.kill_hk_recorder"), \
         unittest.mock.patch("control.start.util.kill_hv_updater"), \
         unittest.mock.patch("control.start.util.kill_module_temp_monitor"), \
         unittest.mock.patch("control.start.util.stop_data_flow"), \
         unittest.mock.patch("control.utils.util.local_ip", return_value=["127.0.0.1", str(daq_config.head_node_ip_addr)]):
        
        # We expect this to return False because it should timeout and trigger rollback
        # We use a timeout on the test itself to ensure we don't hang the runner if the fix is missing
        with anyio.fail_after(15):
            success = await start.start_run(
                obs_config, daq_config, quabo_uids, data_config, network_config,
                no_hv=True, no_redis=True, no_data=False
            )
            assert not success, "start_run should have failed due to StartDaq timeout"


# ── SC-005: Post-start liveness check ───────────────────────────────────────

@pytest.mark.asyncio
async def test_SC005_hashpipe_exits_immediately_not_detected(
    daq_control_direct: DaqControlClient,
) -> None:
    """
    StartDaq succeeds but hashpipe exits within 1s — control plane must detect this.

    Fix: Phase 5 Liveness Probe in start.py after heartbeat.
    """

    import control.start as start
    from control.utils import config_file

    async def fast_sleep(delay: float, result: Any = None) -> Any:
        import asyncio
        return await asyncio.sleep(0, result=result)

    daq_config = config_file.get_daq_config()
    obs_config = config_file.get_obs_config()
    quabo_uids = config_file.get_quabo_uids()
    data_config = config_file.get_data_config()
    network_config = config_file.get_network_config()

    # Ensure daq_config handles all modules in the chaos quabo_uids to avoid validation errors
    mids = []
    for dome in quabo_uids.domes:
        for mod in dome.modules:
            mids.append(mod.id)
    daq_config.daq_nodes[0].module_ids = mids
    daq_config.head_node_container = True

    from control.utils.run_state import RunStateManager
    RunStateManager().clear_state()

    # Mock StartDaq to succeed
    async def success_start_daq(*args: Any, **kwargs: Any) -> bool:
        return True

    # Mock StatusDaq: 
    # 1. First calls (heartbeat) return running=True
    # 2. Subsequent call (Phase 5 Liveness Probe) returns running=False
    status_responses = [
        (True, {"hashpipe_running": True, "hashpipe_pid": 1234}), # Heartbeat attempt 1
        (True, {"hashpipe_running": False, "hashpipe_pid": 0}),   # Phase 5 Liveness Probe
        (True, {"hashpipe_running": False, "hashpipe_pid": 0}),   # Safety
        (True, {"hashpipe_running": False, "hashpipe_pid": 0}),   # Safety
    ]

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.StartDaq = AsyncMock(side_effect=success_start_daq)
    mock_client.StatusDaq = AsyncMock(side_effect=status_responses)

    with unittest.mock.patch("control.start.AsyncDaqControlClient", return_value=mock_client), \
         unittest.mock.patch("asyncio.sleep", side_effect=fast_sleep), \
         unittest.mock.patch("control.start.ph_baseline_file_ok", return_value=True), \
         unittest.mock.patch("control.start._check_daq_reachability"), \
         unittest.mock.patch("control.start.make_run_dirs"), \
         unittest.mock.patch("control.start.start_data_flow"), \
         unittest.mock.patch("control.start.util.is_hk_recorder_running", return_value=False), \
         unittest.mock.patch("control.start.util.kill_hk_recorder"), \
         unittest.mock.patch("control.start.util.kill_hv_updater"), \
         unittest.mock.patch("control.start.util.kill_module_temp_monitor"), \
         unittest.mock.patch("control.start.util.stop_data_flow"), \
         unittest.mock.patch("control.utils.util.local_ip", return_value=["127.0.0.1", str(daq_config.head_node_ip_addr)]):
        
        success = await start.start_run(
            obs_config, daq_config, quabo_uids, data_config, network_config,
            no_hv=True, no_redis=True, no_data=False, force_reset=True, strict=False
        )
        assert not success, "start_run should have failed due to Phase 5 liveness probe failure"


# ── SC-006 (Exemplar C): StopDaq partial failure ─────────────────────────────

class TestSC006StopDaqPartialFailure:
    """
    SC-006: stop.py's stop_recording raises on the first failed StopDaq,
    skipping subsequent DAQ nodes.

    FAILS RED TODAY: stop.py::stop_recording has no per-node isolation;
    one failure causes all subsequent nodes to be skipped.
    """

    def test_SC006_stop_continues_after_per_node_timeout(
        self,
        daq_control_direct: DaqControlClient,
        daq_control_node2: DaqControlClient,
        run_params: dict[str, Any],
        state_probe: StateProbe,
        daqnode_container: Any,
    ) -> None:
        """
        With two DAQ nodes, a StopDaq timeout on node-0 must NOT prevent
        node-1 from being stopped.
        """
        # Start hashpipe on both nodes
        rp1 = dict(run_params)
        rp2 = dict(run_params, daq_ip_addr="192.168.0.20", module_id=[200])

        daq_control_direct.StartDaq(rp1)
        wait_hashpipe_running(daq_control_direct, DAQ_DATA_DIR, timeout=4)
        daq_control_node2.StartDaq(rp2)
        wait_hashpipe_running(daq_control_node2, DAQ_DATA_DIR, timeout=4)

        from ipaddress import IPv4Address

        import control.stop as stop_module
        from control.utils.pydantic_config_models import DaqConfig, DaqNode

        # Construct a dummy DaqConfig with both nodes
        daq_config = DaqConfig(
            head_node_ip_addr=IPv4Address("10.0.1.22"),
            head_node_data_dir="/data/head",
            daq_nodes=[
                DaqNode(ip_addr=IPv4Address(rp1["daq_ip_addr"]), data_dir=rp1["data_dir"], username="root", module_ids=rp1["module_id"]),
                DaqNode(ip_addr=IPv4Address(rp2["daq_ip_addr"]), data_dir=rp2["data_dir"], username="root", module_ids=rp2["module_id"])
            ]
        )

        import asyncio

        import grpc

        # Track calls
        stop_called_ips = set()

        def create_mock_client(host: str, port: int):
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)

            async def _stop(params, **kw):
                stop_called_ips.add(host)
                if host == rp1["daq_ip_addr"]:
                    exc = grpc.RpcError("RPC Timeout")
                    exc.code = lambda: grpc.StatusCode.DEADLINE_EXCEEDED
                    raise exc
                return True

            mock_client.StopDaq = AsyncMock(side_effect=_stop)
            return mock_client

        with unittest.mock.patch("control.stop.AsyncDaqControlClient", side_effect=create_mock_client), \
             unittest.mock.patch("subprocess.run", return_value=unittest.mock.MagicMock(returncode=0)):

            # Call actual stop_recording
            asyncio.run(stop_module.stop_recording(daq_config, rp1["run_dir"], verbose=False))

        # Node-1 must still have been attempted despite node-0 failure
        assert rp2["daq_ip_addr"] in stop_called_ips, (
            "Node-1 was never told to stop because node-0 raised first "
            "(SC-006 bug: stop_recording is not fault-isolated per node)"
        )
        # Cleanup node-0 which we skipped stopping due to the mock
        with contextlib.suppress(Exception):
            daq_control_direct.StopDaq({
                "data_dir": rp1["data_dir"], "run_dir": rp1["run_dir"]
            })
            daq_control_node2.StopDaq({
                "data_dir": rp2["data_dir"], "run_dir": rp2["run_dir"]
            })
        with contextlib.suppress(Exception):
            daq_control_direct.CleanupData({
                "data_dir": rp1["data_dir"],
                "run_dir": rp1["run_dir"],
                "module_id": rp1["module_id"],
            })
            daq_control_node2.CleanupData({
                "data_dir": rp2["data_dir"],
                "run_dir": rp2["run_dir"],
                "module_id": rp2["module_id"],
            })

def _call_stop_recording_for_two_nodes(
    client1: DaqControlClient,
    client2: DaqControlClient,
    params1: dict[str, Any],
    params2: dict[str, Any],
) -> None:
    """
    Simulate stop.py's sequential stop_recording loop for two nodes.
    Raises StopPartialFailure if either node fails; currently the loop
    stops at the first failure (the bug being tested).
    """
    errors = []
    for client, params in [(client1, params1), (client2, params2)]:
        ok, resp = grpc_stop(client, {
            "data_dir": params["data_dir"],
            "run_dir": params["run_dir"],
        })
        if not ok:
            errors.append(f"{params['daq_ip_addr']}: {resp}")
            # Current bug: production code raises here, never reaches next node
            raise StopPartialFailure(f"StopDaq failed: {errors}")
    if errors:
        raise StopPartialFailure("; ".join(errors))
