"""Tier 2 (Logic): Transfer daemon -> ledger sync.

Verifies:
- Transfer daemon updates the run ledger with attempts and errors.
"""
from __future__ import annotations

import asyncio
import contextlib
import pathlib
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from control.transfer.daemon import run_daemon
from control.transfer.lifecycle import MAX_ATTEMPTS
from control.transfer.models import TransferJob
from control.transfer.queue import TransferQueue
from control.utils.paths import PanoPaths
from control.utils.run_state import RunStateLedger, RunStatus


async def _run_daemon_until(
    done_pred: object,
    *,
    timeout_iters: int = 200,
    poll_interval: float = 0.1,
) -> None:
    task = asyncio.create_task(run_daemon(poll_interval=poll_interval))
    for i in range(timeout_iters):
        await asyncio.sleep(poll_interval)
        if callable(done_pred) and done_pred():
            print(f"Done predicate met at iter {i}")
            break
    else:
        print("Timeout reached in _run_daemon_until")
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task

@pytest.mark.asyncio
async def test_transfer_daemon_ledger_sync(
    mock_workspace, 
    tmp_path: pathlib.Path,
    transfer_queue: TransferQueue,
    transfer_job_factory: Callable[..., TransferJob],
) -> None:
    # mock_workspace already sets up PSETI_STATE and creates runs/ dir via PanoPaths.ensure_dirs()
    state_dir = PanoPaths.state_dir()
    
    from control.utils.run_state import RunStateManager
    state_mgr = RunStateManager(base_dir=state_dir)
    
    ledger_obj = RunStateLedger(run_name="r1", status=RunStatus.RECORDING_ENDED, start_time=datetime.now(UTC).isoformat(), nodes=[])
    state_mgr.save_state(ledger_obj)

    tq = transfer_queue
    job = transfer_job_factory(run_name="r1", head_data_dir=tmp_path / "head")
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
