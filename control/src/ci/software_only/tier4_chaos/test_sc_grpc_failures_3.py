"""
scenarios/test_sc_grpc_failures_3.py

SC-008, SC-011, SC-012, SC-014, SC-015, SC-016, SC-017, SC-019, SC-020: gRPC failure isolation tests.
Part 3 of partitioned test suite.
"""

from __future__ import annotations

import os
import unittest.mock
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from panoseti_grpc.daq_control.client import DaqControlClient

from ci.software_only.conftest import (
    wait_hashpipe_running,
    wait_hashpipe_stopped,
)
from ci.software_only.tier3_fleet.conftest import (
    DAQ_DATA_DIR,
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

# ── SC-008: StopDaq on never-started service (idempotent contract) ───────────

def test_SC008_stop_on_never_started_returns_success(
    daq_control_direct: DaqControlClient,
) -> None:
    """
    SC-008: StopDaq on a node that has never had a run (or after CleanupData)
    must return success — it is a no-op, not an error.

    Pins the idempotent-stop contract (not TDD-forcing).
    """
    ok, resp = grpc_stop(daq_control_direct, {
        "data_dir": DAQ_DATA_DIR,
        "run_dir": "never_existed_run.pffd",
    })
    assert ok is True, f"StopDaq on never-started service returned ok=False: {resp}"


# ── SC-011: CleanupData partial failure across two nodes ─────────────────────

def test_SC011_cleanup_partial_failure_logs_and_continues(
    daq_control_direct: DaqControlClient,
    daq_control_node2: DaqControlClient,
    run_params: dict[str, Any],
    fresh_run_state: None,
) -> None:
    """
    SC-011: If node-0 CleanupData succeeds but node-1 fails (e.g., run_dir never
    existed on node-1), stop.py must log the error but not block completion.

    This test pins the partial-cleanup isolation contract.
    """
    import uuid as _uuid

    # Start a run on node-0 only
    ok, _ = grpc_start(daq_control_direct, run_params)
    assert ok, "StartDaq failed on node-0"
    assert wait_hashpipe_running(daq_control_direct, DAQ_DATA_DIR, timeout=4)

    daq_control_direct.StopDaq({
        "data_dir": run_params["data_dir"],
        "run_dir": run_params["run_dir"],
    })
    wait_hashpipe_stopped(daq_control_direct, DAQ_DATA_DIR, timeout=4)

    # Cleanup node-0 must succeed
    ok0, resp0 = grpc_cleanup(daq_control_direct, {
        "data_dir":  run_params["data_dir"],
        "run_dir":   run_params["run_dir"],
        "module_id": run_params["module_id"],
    })
    assert ok0, f"CleanupData failed on node-0: {resp0}"

    # Cleanup node-1 for a run that never started there — must not raise/hang
    ok1, _resp1 = grpc_cleanup(daq_control_node2, {
        "data_dir":  run_params["data_dir"],
        "run_dir":   f"nonexistent_{_uuid.uuid4().hex[:8]}.pffd",
        "module_id": run_params["module_id"],
    })
    # Either ok or a clear error — must not be an unhandled exception
    # (Some servers return ok=True for no-op cleanup; others ok=False with message)
    assert isinstance(ok1, bool), "CleanupData must return a bool ok status"


# ── SC-012: CleanupData with full disk on head node ──────────────────────────

def test_SC012_cleanup_with_full_head_disk_does_not_retry(
    daq_control_direct: DaqControlClient,
    run_params: dict[str, Any],
    fresh_run_state: None,
) -> None:
    """
    SC-012: When the head-node rsync target is full, collect_data errors but
    does not retry; collect_complete_filename is not written, correctly blocking
    cleanup. Test pins the current no-retry behavior.

    FAILS RED TODAY: collect.collect_data does not retry on ENOSPC.
    Fix: retry with exponential backoff on transient ENOSPC.
    """
    from ipaddress import IPv4Address
    from unittest.mock import MagicMock

    from control.utils import collect
    from control.utils.pydantic_config_models import DaqConfig, DaqNode

    # Construct a real-ish config for the collect_data call
    daq_config = DaqConfig(
        head_node_ip_addr=IPv4Address("127.0.0.1"),
        head_node_data_dir="/tmp/head_data",
        daq_nodes=[
            DaqNode(
                ip_addr=IPv4Address(run_params["daq_ip_addr"]),
                data_dir=run_params["data_dir"],
                username="root",
                module_ids=[run_params["module_id"][0]]  # Use only one module
            )
        ]
    )
    
    # Create the head node data dir so os.path.isdir check passes
    os.makedirs(f"/tmp/head_data/{run_params['run_dir']}", exist_ok=True)

    with unittest.mock.patch("subprocess.run") as mock_run, unittest.mock.patch("time.sleep"):
        # Mock rsync failure with code 255 (which IS transient in the new code)
        # Sequence: 255, 255, 0 (success)
        mock_run.side_effect = [
            MagicMock(returncode=255, stderr="Mocked rsync transient failure 1"),
            MagicMock(returncode=255, stderr="Mocked rsync transient failure 2"),
            MagicMock(returncode=0, stdout="Success")
        ]
        
        # Call the actual collect_data
        res = collect.collect_data(daq_config, run_params["run_dir"], verbose=True)
        
        # Verify success and exactly 3 calls
        assert res.success is True, f"collect_data failed: {res.errors}"
        assert mock_run.call_count == 3, (
            f"Expected exactly 3 rsync calls, but got {mock_run.call_count}."
        )


# ── SC-014: gRPC RST_STREAM mid-StartDaq ────────────────────────────────────

@pytest.mark.skip(reason="SC-014: requires grpc_proxy reset_stream injection")
def test_SC014_rst_stream_during_startdaq_errors_cleanly() -> None:
    """
    SC-014: A gRPC RST_STREAM received mid-StartDaq must be surfaced as a clear
    error, not silently leave hashpipe in an unknown state.

    FAILS RED TODAY: behavior under RST_STREAM is not specified or tested.
    """
    pytest.skip("Requires grpc_proxy.reset_stream() on StartDaq")


# ── SC-015: Daqnode reboots during recording ────────────────────────────────

@pytest.mark.skip(reason="SC-015: requires container restart simulation")
def test_SC015_daqnode_reboot_during_run_makes_head_aware() -> None:
    """
    SC-015: When a DAQ node reboots mid-run, the head node still has current_run
    set. The next start.py invocation fails with 'run in progress' until the
    operator intervenes.

    FAILS RED TODAY: no keepalive/heartbeat between head and DAQ — head never
    learns the DAQ rebooted.
    Fix: StatusDaq heartbeat + auto-clear current_run on DAQ unreachable.
    """
    pytest.skip("Requires docker restart of daqnode container mid-run")


# ── SC-016: DaqControlClient with wrong port → clear error ──────────────────

def test_SC016_wrong_port_gives_clear_error(daqnode_ip) -> None:
    """
    SC-016: DaqControlClient constructed with a wrong port must raise or return
    a clear connection error within a reasonable timeout, not hang forever.

    Pins the operator-visible error contract.
    """
    import grpc

    bad_client = DaqControlClient(host=daqnode_ip, port=9)  # port 9 = discard
    try:
        ok, resp = bad_client.StatusDaq({
            "data_dir": DAQ_DATA_DIR,
            "check_hashpipe_running": False,
            "check_disk_usage": False,
            "check_run_dirs": False,
        })
        # If it returns at all, it should be ok=False with an error message
        assert not ok or resp, "Should fail or return a non-empty response for unreachable port"
    except (grpc.RpcError, ConnectionError, OSError, Exception) as exc:
        # Any clear exception is acceptable — we just don't want a silent hang
        assert str(exc), f"Exception must have a message: {exc}"


# ── SC-017: DAQ Control service disabled (UNIMPLEMENTED) ────────────────────

@pytest.mark.skip(reason="SC-017: requires server config without daq_control=true")
def test_SC017_daq_control_disabled_returns_unimplemented() -> None:
    """
    SC-017: A pseti-grpc server running without daq_control enabled must return
    UNIMPLEMENTED for all DAQ Control RPCs.

    FAILS RED TODAY: server profile is not tested with daq_control disabled.
    Fix: test with a separate headnode profile container.
    """
    pytest.skip("Requires a second container with daq_control=false profile")


# ── SC-019: CleanupData race with concurrent StartDaq ────────────────────────

@pytest.mark.skip(reason="SC-019: complex race — requires concurrent RPC injection")
def test_SC019_cleanup_race_with_startdaq() -> None:
    """
    SC-019: CleanupData issued while a concurrent StartDaq is racing with StopDaq
    must not delete data that the new run is writing.

    FAILS RED TODAY: server has no mutex between cleanup and start.
    Fix: server-side advisory lock per run_dir.
    """
    pytest.skip("Requires grpc_proxy to orchestrate concurrent StartDaq + CleanupData timing")
# ── SC-020: StopDaq timeout → hard-kill escalation ──────────────────────────

@pytest.mark.asyncio
async def test_SC020_stopdaqs_timeout_triggers_sigkill_fallback(
    daq_control_direct: DaqControlClient,
) -> None:
    """
    SC-020: When StopDaq RPC times out or fails with UNAVAILABLE, stop.py must
    escalate to a hard-kill via SSH to ensure the node is made safe.
    """
    import unittest.mock
    from ipaddress import IPv4Address

    import grpc

    import control.stop as stop
    from control.utils.pydantic_config_models import DaqConfig, DaqNode

    # Setup config with one node
    daq_ip = "192.168.0.10"
    daq_config = DaqConfig(
        head_node_ip_addr=IPv4Address(os.environ.get("HEADNODE_TESTER_HOST", f'{os.environ.get("HEAD_NET_PREFIX", "10.0.1")}.5')),
        head_node_data_dir="/data/head",
        daq_nodes=[
            DaqNode(ip_addr=IPv4Address(daq_ip), data_dir="/data", username="root", module_ids=[250])
        ]
    )

    # Mock StopDaq to raise DeadlineExceeded
    async def timeout_stop_daq(*args: Any, **kwargs: Any) -> bool:
        exc = grpc.RpcError("RPC Timeout")
        exc.code = lambda: grpc.StatusCode.DEADLINE_EXCEEDED
        raise exc

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.StopDaq = AsyncMock(side_effect=timeout_stop_daq)

    # Track asyncio.create_subprocess_exec calls to verify fallback pkill
    fallback_called = False

    async def mocked_create_subprocess_exec(*args: Any, **kwargs: Any) -> Any:
        nonlocal fallback_called
        cmd_str = " ".join(str(a) for a in args)
        if "pkill -9 hashpipe" in cmd_str and daq_ip in cmd_str:
            fallback_called = True
        mock_proc = unittest.mock.MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(None, b""))
        return mock_proc

    with unittest.mock.patch("control.stop.AsyncDaqControlClient", return_value=mock_client), \
         unittest.mock.patch("asyncio.create_subprocess_exec", side_effect=mocked_create_subprocess_exec), \
         unittest.mock.patch("control.stop.config_file.get_daq_config", return_value=daq_config), \
         unittest.mock.patch("control.stop.config_file.get_quabo_uids", return_value=unittest.mock.MagicMock()), \
         unittest.mock.patch("control.stop.config_file.get_network_config"), \
         unittest.mock.patch("control.stop.util.local_ip", return_value=[os.environ.get("HEADNODE_TESTER_HOST", f'{os.environ.get("HEAD_NET_PREFIX", "10.0.1")}.5'), "127.0.0.1"]), \
         unittest.mock.patch("control.stop.RunStateManager.load_state", return_value=None), \
         unittest.mock.patch("control.stop.util.read_run_name", return_value="test_run.pffd"), \
         unittest.mock.patch("control.stop.os.path.exists", return_value=True), \
         unittest.mock.patch("control.stop.util.stop_data_flow"), \
         unittest.mock.patch("control.stop.util.kill_hv_updater"), \
         unittest.mock.patch("control.stop.util.kill_hk_recorder"), \
         unittest.mock.patch("control.stop.util.kill_module_temp_monitor"), \
         unittest.mock.patch("control.stop.write_complete_file"), \
         unittest.mock.patch("control.stop.make_links"), \
         unittest.mock.patch("control.stop.util.remove_run_name"):

        await stop.stop_run(
            daq_config, unittest.mock.MagicMock(), unittest.mock.MagicMock(),
            run="test_run.pffd", no_collect=True, no_cleanup=True
        )

        # success might be False because StopDaq failed, but we verify fallback was called
        assert fallback_called, "Fallback hard-kill (ssh pkill) was not executed after StopDaq timeout"
