"""
ci/tier2_logic/test_transfer.py

Logic tests for the transfer pipeline using isolated state and mocked gRPC.
"""

from __future__ import annotations

import pathlib
import uuid
from unittest.mock import MagicMock, patch

import pytest

from control.transfer.daemon import _process_job, _sweep_stranded_jobs
from control.transfer.queue import TransferQueue
from ci.fixtures.mocks import MockDaqNode

@pytest.mark.asyncio
async def test_when_transfer_job_processed_then_reaches_archived(
    tmp_path: pathlib.Path,
    transfer_job_factory,
    daq_fs_simulator
) -> None:
    """
    Intent: Verify the 5-stage transfer state machine completes successfully 
    when gRPC and rsync both succeed.
    Scenario: Standard transfer with one DAQ node.
    Assertion: _process_job returns True and run_complete marker is written.
    """
    # 1. Setup isolated mock state
    run_name = "tier2_test_run.pffd"
    head_root = tmp_path / "head"
    job = transfer_job_factory(run_name=run_name, head_data_dir=head_root)
    
    # Simulate DAQ filesystem (on what will be the 'remote' but is shared in Tier 2)
    daq_root = tmp_path / "daq_root"
    daq_fs_simulator(daq_root, run_name, module_ids=[201, 254])
    
    # 2. Mock gRPC using our standard Tier 2 mock
    mock_daq = MockDaqNode("192.168.0.10")
    
    # Inject mock into the daemon's gRPC client factory
    with patch("panoseti_grpc.daq_control.client.AsyncDaqControlClient", return_value=mock_daq.client):
        # 3. Mock rsync (subprocess.run)
        with patch("control.transfer.daemon.subprocess.run", return_value=MagicMock(returncode=0)):
            success = await _process_job(job)
            
    assert success is True
    
    # 4. Verify side effects
    head_run_dir = head_root / run_name
    assert (head_run_dir / "run_complete").exists()

@pytest.mark.asyncio
async def test_when_manifest_fails_then_transfer_aborts(
    tmp_path: pathlib.Path,
    transfer_job_factory
) -> None:
    """
    Intent: Ensure Stage 1 failure prevents Stage 2 (rsync) from executing.
    Scenario: DAQ node returns success: False for GenerateManifest.
    Assertion: _process_job returns False and subprocess.run is never called.
    """
    head_root = tmp_path / "head"
    job = transfer_job_factory(head_data_dir=head_root)
    mock_daq = MockDaqNode("192.168.0.10")
    mock_daq.set_manifest_failure("Disk full on DAQ")
    
    with patch("panoseti_grpc.daq_control.client.AsyncDaqControlClient", return_value=mock_daq.client):
        with patch("control.transfer.daemon.subprocess.run") as mock_rsync:
            success = await _process_job(job)
            
            assert success is False
            mock_rsync.assert_not_called()

@pytest.mark.asyncio
async def test_when_rsync_fails_then_job_returns_false(
    tmp_path: pathlib.Path,
    transfer_job_factory
) -> None:
    """
    Intent: Verify that rsync (Stage 2) failure triggers a False return for the job.
    Scenario: subprocess.run returns non-zero exit code during rsync.
    Assertion: _process_job returns False.
    """
    head_root = tmp_path / "head"
    job = transfer_job_factory(head_data_dir=head_root)
    
    mock_daq = MockDaqNode("192.168.0.10")
    
    with patch("panoseti_grpc.daq_control.client.AsyncDaqControlClient", return_value=mock_daq.client):
        with patch("control.transfer.daemon.subprocess.run", return_value=MagicMock(returncode=1, stderr="network loss")):
            success = await _process_job(job)
            
    assert success is False

@pytest.mark.asyncio
async def test_when_file_corrupted_then_verify_fails(
    tmp_path: pathlib.Path,
    transfer_job_factory
) -> None:
    """
    Intent: Ensure Stage 3 (VERIFYING) catches head-node data corruption.
    Scenario: A file on the head node is mutated after rsync but before verify_manifest.
    Assertion: _process_job returns False and Stage 4 (Cleanup) is bypassed.
    """
    import hashlib
    run_name = "corrupt_test.pffd"
    head_root = tmp_path / "head"
    run_dir = head_root / run_name
    run_dir.mkdir(parents=True)
    
    # 1. Write a real file and a matching manifest
    pff_file = run_dir / "data.pff"
    pff_file.write_bytes(b"original content")
    digest = hashlib.sha256(b"original content").hexdigest()
    manifest = run_dir / "manifest.sha256"
    manifest.write_text(f"{digest}  16  0  data.pff\n")
    
    # 2. Corrupt the file
    pff_file.write_bytes(b"corrupted!")
    
    job = transfer_job_factory(run_name=run_name, head_data_dir=head_root)
    mock_daq = MockDaqNode("192.168.0.10")
    
    with patch("panoseti_grpc.daq_control.client.AsyncDaqControlClient", return_value=mock_daq.client):
        with patch("control.transfer.daemon.subprocess.run", return_value=MagicMock(returncode=0)):
            success = await _process_job(job)
            
    assert success is False
    # Verify cleanup was bypassed (mock cleanup never called)
    mock_daq.client.CleanupData.assert_not_called()

def test_when_job_stranded_then_recovered_on_startup(
    tmp_path: pathlib.Path,
    transfer_job_factory
) -> None:
    """
    Intent: Verify that daemon startup correctly recovers jobs stranded in 'active/'.
    Scenario: A daemon crash leaves a job in active/ without completion or failure.
    Assertion: _sweep_stranded_jobs() moves the job back to 'pending/'.
    """
    tq = TransferQueue()
    job = transfer_job_factory(run_name="stranded_run")
    tq.enqueue(job)
    
    # Simulate crash: manually move to active/
    import shutil
    pending_path = tq._queue / "pending" / f"{job.run_name}.job.toml"
    active_path = tq._queue / "active" / f"{job.run_name}.job.toml"
    shutil.move(str(pending_path), str(active_path))
    
    _sweep_stranded_jobs(tq)
    
    assert pending_path.exists()
    assert not active_path.exists()

def test_when_double_enqueued_then_queue_is_idempotent():
    """
    Intent: Ensure concurrent stop operations don't create duplicate transfer jobs.
    Scenario: enqueue() is called twice for the same run name.
    Assertion: Second call returns False and exactly one job file exists.
    """
    from ci.fixtures.factories import make_transfer_job
    
    tq = TransferQueue()
    job = make_transfer_job(run_name="idempotent_test")
    
    res1 = tq.enqueue(job)
    res2 = tq.enqueue(job)
    
    assert res1 is True
    assert res2 is False
    
    pending = list((tq._queue / "pending").glob("*.toml"))
    assert len(pending) == 1
