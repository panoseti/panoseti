"""
scenarios/test_sc_grpc_failures_2.py

SC-007, SC-009, SC-010, SC-013, SC-018, SC-003, SC-004: gRPC failure isolation tests.
Part 2 of partitioned test suite.
"""

from __future__ import annotations

import unittest.mock
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from panoseti_grpc.daq_control.client import DaqControlClient

from ci.fixtures.chaos import process_chaos
from ci.fixtures.state_probe import StateProbe
from ci.software_only.tier3_fleet.conftest import (
    DAQ_DATA_DIR,
    wait_hashpipe_running,
    wait_hashpipe_stopped,
)
from ci.software_only.tier4_chaos.conftest import (
    _cleanup as grpc_cleanup,
)
from ci.software_only.tier4_chaos.conftest import (
    _start as grpc_start,
)
from ci.software_only.tier4_chaos.conftest import (
    _stop as grpc_stop,
)
from ci.software_only.tier4_chaos.conftest import (
    any_pff_files_on_daqnode,
)

# ── SC-007: StopDaq on already-stopped service (contract test) ──────────────

def test_SC007_stop_on_already_stopped_returns_success(
    daq_control_direct: DaqControlClient,
    run_params: dict[str, Any],
    fresh_run_state: None,
) -> None:
    """
    StopDaq on a node with no running hashpipe MUST return success=True.

    Not TDD-forcing — pins the idempotent-stop contract.
    """
    ok, resp = grpc_stop(daq_control_direct, {
        "data_dir": run_params["data_dir"],
        "run_dir": run_params["run_dir"],
    })
    assert ok is True, f"StopDaq on stopped service returned ok=False: {resp}"


# ── SC-009: CleanupData while hashpipe is running ────────────────────────────

def test_SC009_cleanup_blocked_while_hashpipe_running(
    daq_control_direct: DaqControlClient,
    run_params: dict[str, Any],
    fresh_run_state: None,
) -> None:
    """
    CleanupData MUST refuse (not silently succeed) while hashpipe is running.

    This pins the data-safety contract: we never delete science data while
    recording is active.
    """
    daq_control_direct.StartDaq(run_params)
    assert wait_hashpipe_running(daq_control_direct, DAQ_DATA_DIR, timeout=4)
    try:
        ok, _resp = grpc_cleanup(daq_control_direct, {
            "data_dir":  run_params["data_dir"],
            "run_dir":   run_params["run_dir"],
            "module_id": run_params["module_id"],
        })
        assert not ok, (
            "CleanupData should refuse while hashpipe is running, "
            "but returned ok=True — data-safety contract violated"
        )
    finally:
        daq_control_direct.StopDaq({
            "data_dir": run_params["data_dir"],
            "run_dir": run_params["run_dir"],
        })
        wait_hashpipe_stopped(daq_control_direct, DAQ_DATA_DIR, timeout=4)


# ── SC-010 (Exemplar A): Orphaned hashpipe blocks CleanupData ─────────────

class TestSC010OrphanedHashpipe:
    """
    SC-010 / Exemplar A: when hashpipe is SIGKILLed (orphaned), the DAQ Control
    server's `hashpipe_pid > 0` gate blocks CleanupData forever.

    TDD-forcing test — FAILS RED on current master because:
      (a) CleanupData without force: returns ok=False (server thinks hashpipe is alive)
          but there is no CleanupRefusedPreserveData exception class
      (b) CleanupData with force=True: the `force` field doesn't exist in the proto
      (c) No Redis incident key is written

    Fix requires:
      1. Server liveness check: kill(pid, 0) / psutil.pid_exists
      2. Proto: add `force` bool to CleanupDataRequest
      3. Server: write Redis incident key on forced cleanup
      4. stop.py: add --force-cleanup flag
    """

    def test_SC010_cleanup_refused_without_force_when_hashpipe_orphaned(
        self,
        daq_control_direct: DaqControlClient,
        run_params: dict[str, Any],
        state_probe: StateProbe,
        daqnode_container: Any,
        fresh_run_state: None,
    ) -> None:
        """
        After hashpipe is SIGKILLed, CleanupData without force must:
        - Refuse to delete data (return ok=False / raise FAILED_PRECONDITION)
        - Leave .pff files intact on the DAQ node
        """
        daq_control_direct.StartDaq(run_params)
        assert wait_hashpipe_running(daq_control_direct, DAQ_DATA_DIR, timeout=4), \
            "hashpipe did not start"

        # Simulate crash: SIGKILL bypasses StopDaq
        process_chaos.kill_process(daqnode_container.name, "hashpipe", sig="KILL")
        assert process_chaos.wait_for_process_death(
            daqnode_container.name, "hashpipe", timeout=5
        ), "hashpipe did not die after SIGKILL"

        # The server still has hashpipe_pid > 0 in memory.
        # CleanupData MUST refuse — it should NOT silently delete science data.
        ok, _resp = grpc_cleanup(daq_control_direct, {
            "data_dir":  run_params["data_dir"],
            "run_dir":   run_params["run_dir"],
            "module_id": run_params["module_id"],
            "force":     False,
        })
        assert not ok, (
            "FAIL (SC-010): CleanupData succeeded after hashpipe was orphaned — "
            "this is a data-safety violation. The server must either detect the "
            "dead PID or require force=true to proceed."
        )

    def test_SC010_cleanup_with_force_removes_data_and_writes_incident_key(
        self,
        daq_control_direct: DaqControlClient,
        run_params: dict[str, Any],
        state_probe: StateProbe,
        daqnode_container: Any,
        fresh_run_state: None,
    ) -> None:
        """
        With force=True, CleanupData must:
        - Delete the run data
        - Write a Redis incident key

        FAILS TODAY: the `force` field is not in the proto and there is no
        incident-key logic in the server.
        """
        daq_control_direct.StartDaq(run_params)
        assert wait_hashpipe_running(daq_control_direct, DAQ_DATA_DIR, timeout=4)

        process_chaos.kill_process(daqnode_container.name, "hashpipe", sig="KILL")
        process_chaos.wait_for_process_death(daqnode_container.name, "hashpipe", timeout=5)

        # force=True override: allowed to delete orphaned run data
        ok, _resp = grpc_cleanup(daq_control_direct, {
            "data_dir":  run_params["data_dir"],
            "run_dir":   run_params["run_dir"],
            "module_id": run_params["module_id"],
            "force":     True,   # ← not in proto yet: test will fail here
        })
        assert ok, f"CleanupData(force=True) failed: {_resp}"

        # Data must be gone
        assert not any_pff_files_on_daqnode(run_params["run_dir"], run_params["module_id"]), \
            "PFF files still present after CleanupData(force=True)"

    def test_SC010b_force_on_live_hashpipe_is_refused(
        self,
        daq_control_direct: DaqControlClient,
        run_params: dict[str, Any],
        fresh_run_state: None,
    ) -> None:
        """
        SC-010b: force=True on a LIVE hashpipe must still be refused.
        force is only an escape hatch for the dead-PID path, not a kill switch.
        """
        daq_control_direct.StartDaq(run_params)
        assert wait_hashpipe_running(daq_control_direct, DAQ_DATA_DIR, timeout=4)
        try:
            ok, _resp = grpc_cleanup(daq_control_direct, {
                "data_dir":  run_params["data_dir"],
                "run_dir":   run_params["run_dir"],
                "module_id": run_params["module_id"],
                "force":     True,
            })
            assert not ok, (
                "CleanupData(force=True) on a LIVE hashpipe must still be refused — "
                "force is only for the dead-PID case"
            )
        finally:
            daq_control_direct.StopDaq({
                "data_dir": run_params["data_dir"],
                "run_dir": run_params["run_dir"],
            })
            wait_hashpipe_stopped(daq_control_direct, DAQ_DATA_DIR, timeout=4)


# ── SC-013: StatusDaq during StartDaq (contract test) ────────────────────────

def test_SC013_status_during_start_is_consistent(
    daq_control_direct: DaqControlClient,
    run_params: dict[str, Any],
    fresh_run_state: None,
) -> None:
    """
    StatusDaq called concurrently with StartDaq must not deadlock or crash.
    Pins the RPC ordering contract.
    """
    import threading

    daq_control_direct.StartDaq(run_params)
    results: list[Any] = []

    def _status() -> None:
        try:
            ok, resp = daq_control_direct.StatusDaq({
                "data_dir": run_params["data_dir"],
                "check_hashpipe_running": True,
                "check_disk_usage": False,
                "check_run_dirs": False,
            })
            results.append((ok, resp))
        except Exception as e:
            results.append(e)

    t = threading.Thread(target=_status)
    t.start()
    t.join(timeout=5)
    assert results, "StatusDaq call timed out"
    assert not isinstance(results[0], Exception), f"StatusDaq raised: {results[0]}"

    daq_control_direct.StopDaq({
        "data_dir": run_params["data_dir"],
        "run_dir": run_params["run_dir"],
    })
    wait_hashpipe_stopped(daq_control_direct, DAQ_DATA_DIR, timeout=4)


# ── SC-018: Concurrent StartDaq to same node (contract test) ─────────────────

def test_SC018_concurrent_start_same_node_only_one_wins(
    daq_control_direct: DaqControlClient,
    run_params: dict[str, Any],
    fresh_run_state: None,
) -> None:
    """
    Two concurrent StartDaq RPCs to the same node — at most one must succeed.
    The server enforces single-hashpipe-per-node.
    """
    import threading

    outcomes: list[Any] = []

    def _inner_start(run_dir_suffix: str) -> None:
        p = dict(run_params, run_dir=f"conctest_{run_dir_suffix}.pffd")
        ok, resp = grpc_start(daq_control_direct, p)
        outcomes.append((ok, resp, p["run_dir"]))

    t1 = threading.Thread(target=_inner_start, args=("a",))
    t2 = threading.Thread(target=_inner_start, args=("b",))
    t1.start()
    t2.start()
    t1.join(timeout=4)
    t2.join(timeout=4)

    winners = [o for o in outcomes if o[0]]
    assert len(winners) <= 1, (
        f"Both concurrent StartDaq calls succeeded: {outcomes}. "
        "Server must enforce single-hashpipe-per-node."
    )
    # Cleanup winner
    for ok, _resp, run_dir in outcomes:
        if ok:
            daq_control_direct.StopDaq({
                "data_dir": run_params["data_dir"],
                "run_dir": run_dir,
            })
            wait_hashpipe_stopped(daq_control_direct, DAQ_DATA_DIR, timeout=4)
            daq_control_direct.CleanupData({
                "data_dir":  run_params["data_dir"],
                "run_dir":   run_dir,
                "module_id": run_params["module_id"],
            })


# ── SC-003: StartDaq returns success=False (bad config) ──────────────────────

def test_SC003_startdaq_bad_run_dir_returns_failure(
    daq_control_direct: DaqControlClient,
    run_params: dict[str, Any],
    fresh_run_state: None,
) -> None:
    """
    SC-003: StartDaq with an invalid/empty run_dir must return ok=False with a
    descriptive error message, not crash the server.

    Pins the contract: server-side validation must surface config errors cleanly.
    """
    ok, resp = grpc_start(daq_control_direct, {
        **run_params,
        "run_dir": "",  # empty run_dir — invalid
    })
    assert not ok, (
        "StartDaq with empty run_dir must return ok=False. "
        "Server must validate required fields before launching hashpipe."
    )
    assert resp, "StartDaq failure must include a non-empty error message"


# ── SC-004: StartDaq transient UNAVAILABLE, succeeds on retry ────────────────

@pytest.mark.asyncio
async def test_SC004_startdaq_transient_unavailable_succeeds_on_retry(
    daq_control_direct: DaqControlClient,
) -> None:
    """
    SC-004: A transient UNAVAILABLE error on StartDaq must trigger a retry and
    eventually succeed.
    """

    import grpc

    import control.start as start
    from control.utils import config_file

    daq_config = config_file.get_daq_config()
    obs_config = config_file.get_obs_config()
    quabo_uids = config_file.get_quabo_uids()
    data_config = config_file.get_data_config()
    network_config = config_file.get_network_config()

    # Concentrate all modules on node 0 and silence remaining nodes so exactly
    # 2 StartDaq calls are expected: 1 UNAVAILABLE + 1 retry success.
    mids = []
    for dome in quabo_uids.domes:
        for mod in dome.modules:
            mids.append(mod.id)
    daq_config.daq_nodes[0].module_ids = mids
    for node in daq_config.daq_nodes[1:]:
        node.module_ids = []
    daq_config.head_node_container = True

    from control.utils.run_state import RunStateManager
    RunStateManager().clear_state()

    # Mock StartDaq:
    # 1. First call: raise grpc.RpcError with UNAVAILABLE
    # 2. Second call: return True (Success)
    call_count = 0
    async def retry_start_daq(*args: Any, **kwargs: Any) -> bool:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Create a mock RpcError
            exc = grpc.RpcError("Transiently unavailable")
            # Monkey-patch code() method which is what start.py checks
            exc.code = lambda: grpc.StatusCode.UNAVAILABLE
            exc.details = lambda: "Transiently unavailable"
            # start.py expects ConnectionError with __cause__ being the RpcError
            raise ConnectionError("gRPC failed: Transiently unavailable") from exc
        return True
    # We also need to mock StatusDaq for the heartbeat check
    async def success_status_daq(*args: Any, **kwargs: Any) -> tuple[bool, dict[str, Any]]:
        return True, {"hashpipe_running": True, "hashpipe_pid": 1234}

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.StartDaq = AsyncMock(side_effect=retry_start_daq)
    mock_client.StatusDaq = AsyncMock(side_effect=success_status_daq)

    with unittest.mock.patch("control.utils.config_file.get_quabo_uids", return_value=quabo_uids), \
         unittest.mock.patch("control.start.AsyncDaqControlClient", return_value=mock_client), \
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
        assert success, "start_run should have succeeded after transient retry"
        assert call_count == 2, f"Expected 2 StartDaq calls, but got {call_count}"
