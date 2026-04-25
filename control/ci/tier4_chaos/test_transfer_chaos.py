"""
ci/tier4_chaos/test_transfer_chaos.py

Chaos tests for the transfer pipeline requiring full Docker stack and fault injection.
Verifies resilience against network loss and gRPC failures.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import pathlib
import time
from unittest.mock import MagicMock, patch

import pytest

from ci.fixtures.mocks import MockDaqNode
from control.transfer.daemon import run_daemon
from control.transfer.queue import TransferQueue

# ---------------------------------------------------------------------------
# CI guard
# ---------------------------------------------------------------------------

def is_in_ci() -> bool:
    return os.path.exists("/.dockerenv")

pytestmark = pytest.mark.skipif(
    not is_in_ci(), reason="Chaos tests require the full Docker CI stack"
)

@pytest.mark.asyncio
async def test_when_rsync_fails_then_retries_until_success(
    tmp_path: pathlib.Path,
    transfer_job_factory
) -> None:
    """
    Intent: Verify that the daemon correctly executes the retry ladder on rsync failure.
    Scenario: rsync fails once with exit code 1, then succeeds on the second attempt.
    Assertion: The job eventually reaches 'completed/' and subprocess.run is called twice.
    """
    # 1. Setup isolated state
    run_name = "retry_test.pffd"
    head_root = tmp_path / "head"
    job = transfer_job_factory(run_name=run_name, head_data_dir=head_root)
    
    tq = TransferQueue()
    tq.enqueue(job)
    
    # 2. Mock responses: Fail, then Success
    responses = [
        MagicMock(returncode=1, stderr="network loss"),
        MagicMock(returncode=0)
    ]
    
    mock_daq = MockDaqNode("192.168.0.10")
    
    with patch("panoseti_grpc.daq_control.client.AsyncDaqControlClient", return_value=mock_daq.client), \
         patch("control.transfer.daemon.subprocess.run") as mock_sub, \
         patch("control.transfer.daemon._RETRY_BACKOFF_SEC", [0.01, 0.01]):
        
        mock_sub.side_effect = responses
        
        # 3. Drive daemon loop
        daemon_task = asyncio.create_task(run_daemon(poll_interval=0.01))
        
        # 4. Wait for success
        start_t = time.time()
        completed = False
        while time.time() - start_t < 5:
            if (tq._queue / "completed" / f"{run_name}.job.toml").exists():
                completed = True
                break
            await asyncio.sleep(0.05)
            
        daemon_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await daemon_task
            
    assert completed is True
    assert mock_sub.call_count == 2

@pytest.mark.asyncio
async def test_when_cleanup_precondition_fails_then_pff_preserved(
    tmp_path: pathlib.Path,
    transfer_job_factory
) -> None:
    """
    Intent: Ensure that a gRPC 'FAILED_PRECONDITION' during cleanup (Stage 4) 
    stops the archiving process and preserves files.
    Scenario: DAQ server returns success: False for CleanupData.
    Assertion: _process_job returns False and 'run_complete' is NOT written.
    """
    from control.transfer.daemon import _process_job
    head_root = tmp_path / "head"
    run_name = "cleanup_chaos.pffd"
    job = transfer_job_factory(run_name=run_name, head_data_dir=head_root)
    
    mock_daq = MockDaqNode("192.168.0.10")
    mock_daq.set_cleanup_failure("FAILED_PRECONDITION: wrong manifest digest")
    
    with patch("panoseti_grpc.daq_control.client.AsyncDaqControlClient", return_value=mock_daq.client), \
         patch("control.transfer.daemon.subprocess.run", return_value=MagicMock(returncode=0)):
        
        success = await _process_job(job)
        
    assert success is False
    assert not (head_root / run_name / "run_complete").exists()
