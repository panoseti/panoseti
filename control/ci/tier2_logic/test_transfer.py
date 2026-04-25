"""
ci/tier2_logic/test_transfer.py

Logic tests for the transfer pipeline using isolated state and mocked gRPC.
"""

from __future__ import annotations

import pathlib
from unittest.mock import MagicMock, patch

import pytest

from control.transfer.daemon import _process_job
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
