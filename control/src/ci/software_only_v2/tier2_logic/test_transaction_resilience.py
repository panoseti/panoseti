"""
test_transaction_resilience.py — Transactional integrity and rollback tests.

Ported from ci/software_only/tier2_logic/test_transaction_resilience.py.
"""

from __future__ import annotations

import errno
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from control.start import StartTransaction
from control.stop import stop_run
from control.utils import config_file
from control.utils.run_state import RunStateLedger, RunStateManager, RunStatus
from ci.software_only_v2.infra.workspace import Workspace

@pytest.mark.asyncio
async def test_start_rollback_continues_on_erofs_archival_failure(
    pseti_workspace: Workspace,
    caplog: pytest.LogCaptureFixture
) -> None:
    """Verify that StartTransaction rollback ladder completes even if /data is read-only."""
    # pseti_workspace isolates PSETI_STATE
    state_mgr = RunStateManager(base_dir=pseti_workspace.root / "state")
    daq_cfg = pseti_workspace.topology.daq
    uids = pseti_workspace.topology.quabo_uids
    net = pseti_workspace.topology.network
    
    # Mock os.makedirs to simulate Read-only file system for the archival step
    # We want Step 5 (archival) to fail, but Step 3 (stop_data_flow) to still run.
    with patch("os.makedirs", side_effect=OSError(errno.EROFS, "Read-only file system")), \
         patch("control.utils.util.kill_hk_recorder"), \
         patch("control.utils.util.stop_data_flow") as m_stop_flow:
         
        tx = StartTransaction(state_mgr, "run1", daq_cfg, uids, net)
        tx.data_flow_started = True # Force Step 3 to execute
        
        # Trigger rollback via __aexit__
        await tx.__aexit__(RuntimeError, RuntimeError("boom"), None)
        
        # 1. Check for the specific EROFS warning
        assert "Failed to archive partial artifacts (non-fatal)" in caplog.text
        # 2. Verify subsequent ladder steps (Step 3) were NOT skipped
        m_stop_flow.assert_called_once()


@pytest.mark.asyncio
async def test_cleanup_refused_on_uncertain_liveness_without_force() -> None:
    """Proves that CleanupData rejects deletion if PID status is uncertain (corrupted file)."""
    from panoseti_grpc.daq_control.server import DaqControlServicer
    from panoseti_grpc.generated import daq_control_pb2
    
    servicer = DaqControlServicer(grpc_enabled=False)
    # Inject a non-integer garbage PID (simulating corrupted pid file state)
    servicer.hashpipe_pid = "GARBAGE" # type: ignore
    
    req = daq_control_pb2.CleanupDataRequest(
        data_dir="/tmp",
        run_dir="test",
        module_id=[1],
        force=False
    )
    
    context = MagicMock()
    context.abort = AsyncMock()
    
    resp = await servicer.CleanupData(req, context)
    assert resp.success is False
    assert "status uncertain" in resp.message
    
    # Now try with force=True
    req.force = True
    resp = await servicer.CleanupData(req, context)
    # Passed the PID gate (fails later on missing dir, which is fine)
    assert "status uncertain" not in resp.message


@pytest.mark.asyncio
async def test_stop_run_bypasses_enqueue_on_fundamental_failure(
    pseti_workspace: Workspace,
) -> None:
    """Verify that StopTransaction does NOT enqueue a transfer job if a crash occurs in the ladder."""
    state_mgr = RunStateManager(base_dir=pseti_workspace.root / "state")
    
    daq_cfg = pseti_workspace.topology.daq
    run_name = "myrun.pffd"
    (pseti_workspace.root / "head_data" / run_name).mkdir(parents=True, exist_ok=True)
    
    # Pre-write an ACTIVE ledger
    ledger = RunStateLedger(
        run_name=run_name,
        status=RunStatus.ACTIVE,
        start_time="2026-01-01T00:00:00Z"
    )
    state_mgr.save_state(ledger)
    
    with patch("control.stop.RunStateManager", return_value=state_mgr), \
         patch("control.stop.TransferQueue") as m_tq_cls, \
         patch("control.utils.util.local_ip", side_effect=RuntimeError("Ladder Crash")):
         
         mock_tq = m_tq_cls.return_value
         
         # stop_run returns success status
         success = await stop_run(
             daq_cfg, pseti_workspace.topology.network, pseti_workspace.topology.quabo_uids,
             run=run_name
         )
         
         assert success is False
         # CRITICAL: Enqueue must NEVER be called if an exception hit the transaction
         mock_tq.enqueue.assert_not_called()
         
         # Ledger must reflect the failure
         updated_ledger = state_mgr.load_state()
         assert updated_ledger is not None
         assert updated_ledger.status == RunStatus.STOPPED_WITH_ERRORS
         assert "Ladder Crash" in (updated_ledger.last_transfer_error or "")
