"""
test_transfer_cleanup.py

Verifies the CleanupData behavior of the TransferDaemon on DAQ nodes after
a successful transfer and verification.
"""

import asyncio
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from ci.fixtures.rsync_fixtures import RsyncMock
from ci.software_only.tier3_fleet.transfer_testing_utils import (
    generate_mocked_run,
    get_mapped_client_factory,
    simulate_rsync_from_fleet,
)
from control.transfer.daemon import _process_job
from control.transfer.models import TransferJob
from control.utils import config_file
from control.utils.pydantic_config_models import RunStatus
from control.utils.run_state import RunStateManager


@pytest.mark.asyncio
async def test_transfer_selective_cleanup(
    session_fleet: Any,
    ensure_clean_daq_state: Any,
    mock_rsync_transfer: RsyncMock,
    isolated_transfer_env: tuple[Path, config_file.DaqConfig],
    transfer_job_factory: Callable[..., TransferJob],
) -> None:
    """
    Scenario: A successful run transfer occurs.
    Expectation: The TransferDaemon invokes CleanupData to delete .pff files
    on the DAQ nodes while preserving metadata like .json and .log files.
    """
    fleet, _ = session_fleet
    head_data_dir, daq_config = isolated_transfer_env
    run_name = f"cleanup_test_{uuid.uuid4().hex[:8]}.pffd"
    
    # 1. Generate real run data on the DAQ containers (.pff files and meta.json)
    await generate_mocked_run(fleet, daq_config, run_name)
    
    mgr = RunStateManager()
    job = transfer_job_factory(run_name=run_name, head_data_dir=head_data_dir, no_cleanup=False)

    def rsync_side_effect(*args, **kwargs):
        simulate_rsync_from_fleet(fleet, run_name, head_data_dir / run_name)
        return None

    mock_rsync_transfer.side_effect = rsync_side_effect

    with patch("panoseti_grpc.daq_control.client.AsyncDaqControlClient", side_effect=get_mapped_client_factory(daq_config)):
         
         # Allow some time for the full state machine to run including RPCs
         success, _err = await asyncio.wait_for(_process_job(job, asyncio.Event(), mgr), timeout=30.0)
         
         assert success is True, f"Job failed: {_err}"
         
         state = mgr.load_state()
         assert state is not None
         assert state.status == RunStatus.ARCHIVED

    # Verify that cleanup happened on the DAQ nodes
    for i, temp_dir in enumerate(fleet._temp_dirs):
        host_root = Path(temp_dir)
        spec = fleet.specs[i]
        
        # Check root run dir metadata
        meta_file = host_root / run_name / "meta.json"
        assert meta_file.exists(), f"Metadata file {meta_file} should have been preserved on DAQ node"
        
        # Check module subdirs for .pff deletion
        for mid in spec.module_ids:
            host_mod_run_dir = host_root / f"module_{mid}" / run_name
            if host_mod_run_dir.exists():
                pff_files = list(host_mod_run_dir.glob("*.pff"))
                assert len(pff_files) == 0, f"Expected 0 .pff files in {host_mod_run_dir} after cleanup, found {len(pff_files)}"


@pytest.mark.asyncio
async def test_transfer_cleanup_isolation(
    session_fleet: Any,
    ensure_clean_daq_state: Any,
    mock_rsync_transfer: RsyncMock,
    isolated_transfer_env: tuple[Path, config_file.DaqConfig],
    transfer_job_factory: Callable[..., TransferJob],
) -> None:
    """
    Scenario: Multiple runs exist on the DAQ node.
    Expectation: Cleanup for run A MUST NOT affect run B.
    """
    fleet, _ = session_fleet
    head_data_dir, daq_config = isolated_transfer_env
    
    run_to_clean = f"clean_me_{uuid.uuid4().hex[:8]}.pffd"
    run_to_keep = f"keep_me_{uuid.uuid4().hex[:8]}.pffd"
    
    # 1. Generate data for both runs
    await generate_mocked_run(fleet, daq_config, run_to_clean)
    await generate_mocked_run(fleet, daq_config, run_to_keep)
    
    mgr = RunStateManager()
    job = transfer_job_factory(run_name=run_to_clean, head_data_dir=head_data_dir, no_cleanup=False)

    def rsync_side_effect(*args, **kwargs):
        simulate_rsync_from_fleet(fleet, run_to_clean, head_data_dir / run_to_clean)
        return None

    mock_rsync_transfer.side_effect = rsync_side_effect

    with patch("panoseti_grpc.daq_control.client.AsyncDaqControlClient", side_effect=get_mapped_client_factory(daq_config)):
         
         success, _ = await asyncio.wait_for(_process_job(job, asyncio.Event(), mgr), timeout=30.0)
         assert success is True

    # Verify Isolation
    for i, temp_dir in enumerate(fleet._temp_dirs):
        host_root = Path(temp_dir)
        spec = fleet.specs[i]
        
        for mid in spec.module_ids:
            # Run A should be cleaned
            clean_dir = host_root / f"module_{mid}" / run_to_clean
            if clean_dir.exists():
                assert len(list(clean_dir.glob("*.pff"))) == 0
            
            # Run B should be UNTOUCHED
            keep_dir = host_root / f"module_{mid}" / run_to_keep
            assert keep_dir.exists()
            assert len(list(keep_dir.glob("*.pff"))) > 0, f"Run {run_to_keep} was accidentally cleaned!"


@pytest.mark.asyncio
async def test_transfer_no_cleanup_on_verification_failure(
    session_fleet: Any,
    ensure_clean_daq_state: Any,
    mock_rsync_transfer: RsyncMock,
    isolated_transfer_env: tuple[Path, config_file.DaqConfig],
    transfer_job_factory: Callable[..., TransferJob],
) -> None:
    """
    Scenario: Verification fails (data corruption).
    Expectation: Cleanup MUST NOT be performed; data preserved on DAQ for recovery.
    """
    fleet, _ = session_fleet
    head_data_dir, daq_config = isolated_transfer_env
    run_name = f"fail_cleanup_{uuid.uuid4().hex[:8]}.pffd"
    
    await generate_mocked_run(fleet, daq_config, run_name)
    
    mgr = RunStateManager()
    job = transfer_job_factory(run_name=run_name, head_data_dir=head_data_dir, no_cleanup=False)

    def rsync_side_effect(*args, **kwargs):
        simulate_rsync_from_fleet(fleet, run_name, head_data_dir / run_name)
        # Corrupt head-node copy to trigger verification failure
        dest_run = head_data_dir / run_name
        pff = next(dest_run.glob("*.pff"))
        pff.write_bytes(b"CORRUPTED DATA")
        return None

    mock_rsync_transfer.side_effect = rsync_side_effect

    with patch("panoseti_grpc.daq_control.client.AsyncDaqControlClient", side_effect=get_mapped_client_factory(daq_config)):
         
         success, _ = await asyncio.wait_for(_process_job(job, asyncio.Event(), mgr), timeout=30.0)
         assert success is False
         assert mgr.load_state().status == RunStatus.VERIFY_FAILED

    # Verify NO cleanup happened on DAQ
    for i, temp_dir in enumerate(fleet._temp_dirs):
        host_root = Path(temp_dir)
        spec = fleet.specs[i]
        for mid in spec.module_ids:
            host_mod_run_dir = host_root / f"module_{mid}" / run_name
            if host_mod_run_dir.exists():
                assert len(list(host_mod_run_dir.glob("*.pff"))) > 0, "Data was cleaned despite verification failure!"
