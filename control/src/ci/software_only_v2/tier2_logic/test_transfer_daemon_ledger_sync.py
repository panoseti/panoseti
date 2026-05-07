"""
test_transfer_daemon_ledger_sync.py — Transfer daemon and ledger synchronization.

Ported from ci/software_only/tier2_logic/test_transfer_daemon_ledger_sync.py.
"""

from __future__ import annotations

import asyncio
import contextlib
import pathlib
from collections.abc import Callable
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from control.transfer.daemon import run_daemon
from control.transfer.lifecycle import MAX_ATTEMPTS
from control.transfer.models import TransferJob, TransferNodeSpec
from control.transfer.queue import TransferQueue
from control.utils.paths import PanoPaths
from control.utils.run_state import RunStateLedger, RunStatus, RunStateManager
from ci.software_only_v2.infra.workspace import Workspace


async def _run_daemon_until(
    done_pred: Callable[[], bool],
    *,
    timeout_iters: int = 200,
    poll_interval: float = 0.1,
) -> None:
    task = asyncio.create_task(run_daemon(poll_interval=poll_interval))
    for i in range(timeout_iters):
        await asyncio.sleep(poll_interval)
        if done_pred():
            break
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


@pytest.mark.asyncio
async def test_transfer_daemon_ledger_sync(
    pseti_workspace: Workspace,
) -> None:
    # pseti_workspace already sets up PSETI_STATE and creates runs/ dir via PanoPaths.ensure_dirs()
    state_mgr = RunStateManager(base_dir=pseti_workspace.root / "state")
    
    ledger_obj = RunStateLedger(
        run_name="r1",
        status=RunStatus.RECORDING_ENDED,
        start_time=datetime.now(UTC).isoformat(),
        nodes=[]
    )
    state_mgr.save_state(ledger_obj)

    tq = TransferQueue(queue_dir=PanoPaths.transfer_queue_dir())
    
    # Create a job
    job = TransferJob(
        schema_version=1,
        run_name="r1",
        head_data_dir=str(pseti_workspace.root / "head_data"),
        head_node_username="panoseti",
        created_at=datetime.now(UTC),
        daq_nodes=[],
    )
    tq.enqueue(job)

    with patch("control.transfer.daemon._process_job", new_callable=AsyncMock) as mock_process:
        mock_process.return_value = (False, "rsync_blackbox_error")
        # Shorten RETRY_DELAYS for test
        with patch("control.transfer.daemon.RETRY_DELAYS", [0.01, 0.01, 0.01]):
            await _run_daemon_until(lambda: (tq._queue / "failed" / "r1.job.toml").exists())

    # Assert the on-disk ledger has been updated
    updated_ledger = state_mgr.load_state()
    assert updated_ledger is not None
    assert updated_ledger.status == RunStatus.TRANSFER_FAILED
    assert updated_ledger.transfer_attempts == MAX_ATTEMPTS
    assert updated_ledger.last_transfer_error == "rsync_blackbox_error"
