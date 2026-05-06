"""
test_transfer_manifest_edge_cases.py

'Expect-to-fail' scenarios designed to test the strictness of the manifest
verification and selective cleanup contracts.
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
from control.utils.pydantic_config_models import RunStatus
from control.utils.run_state import RunStateManager


@pytest.mark.asyncio
# @pytest.mark.xfail(strict=True, reason="Strict VERIFYING stage should catch this data corruption")
async def test_manifest_corruption_aborts_cleanup(
    session_fleet: Any,
    ensure_clean_daq_state: Any,
    mock_rsync_transfer: RsyncMock,
    isolated_transfer_env: tuple[Path, config_file.DaqConfig],
    transfer_job_factory: Callable[..., TransferJob],
) -> None:
    """
    Scenario: A bit flip occurs in a .pff file during/after transfer.
    Expectation: The VERIFYING stage MUST detect the digest mismatch, abort the
    job, transition to VERIFY_FAILED, and NEVER call CleanupData.
    """
    fleet, _ = session_fleet
    head_data_dir, daq_config = isolated_transfer_env
    run_name = f"xfail_corrupt_{uuid.uuid4().hex[:8]}.pffd"
    
    # 1. Generate real run data on the DAQ containers
    await generate_mocked_run(fleet, daq_config, run_name)
    
    mgr = RunStateManager()
    job = transfer_job_factory(run_name=run_name, head_data_dir=head_data_dir)

    def rsync_side_effect(*args, **kwargs):
        # Do the real mock copy
        simulate_rsync_from_fleet(fleet, run_name, head_data_dir / run_name)
        
        # Now artificially corrupt one of the .pff files on the head node
        dest_run = head_data_dir / run_name
        pff_files = list(dest_run.glob("**/*.pff"))
        non_empty_pff_files = [fname for fname in pff_files if fname.stat().st_size > 0]
        assert len(non_empty_pff_files) > 0
        if non_empty_pff_files:
            # Flip a byte
            data = bytearray(non_empty_pff_files[0].read_bytes())
            data[0] ^= 0xFF
            non_empty_pff_files[0].write_bytes(data)
        return None

    mock_rsync_transfer.side_effect = rsync_side_effect

    # Use a spy to track if CleanupData is called
    cleanup_called = False
    
    def wrapped_client_factory(*args, **kwargs):
        client = get_mapped_client_factory(daq_config)(*args, **kwargs)
        
        # Wrap the context manager return value
        original_aenter = client.__aenter__
        async def mocked_aenter():
            mock_stub = await original_aenter()
            
            original_cleanup = mock_stub.CleanupData
            async def spy_cleanup(*c_args, **c_kwargs):
                nonlocal cleanup_called
                cleanup_called = True
                return await original_cleanup(*c_args, **c_kwargs)
            
            mock_stub.CleanupData = spy_cleanup
            return mock_stub
            
        client.__aenter__ = mocked_aenter
        return client

    with patch("panoseti_grpc.daq_control.client.AsyncDaqControlClient", side_effect=wrapped_client_factory):
         
         success, _err = await _process_job(job, asyncio.Event(), mgr)
         
         # The test is strictly designed: it MUST fail the job.
         assert success is False, "Job should have failed due to data corruption"
         assert not cleanup_called, "CleanupData MUST NOT be called if verification fails"
         
         state = mgr.load_state()
         assert state is not None
         assert state.status == RunStatus.VERIFY_FAILED


@pytest.mark.asyncio
@pytest.mark.xfail(strict=True, reason="Strict verify requires all digests to match, including the manifest itself")
async def test_transfer_cleanup_rejected_on_digest_mismatch(
    session_fleet: Any,
    ensure_clean_daq_state: Any,
    mock_rsync_transfer: RsyncMock,
    isolated_transfer_env: tuple[Path, config_file.DaqConfig],
    transfer_job_factory: Callable[..., TransferJob],
) -> None:
    """
    Scenario: The TransferDaemon passes a manifest_digest to CleanupData, but
    the DAQ node's local manifest has a different digest (simulated by mocking
    the DAQ node's rejection).
    Expectation: The cleanup is rejected with FAILED_PRECONDITION, job fails,
    and no files are deleted.
    """
    fleet, _ = session_fleet
    head_data_dir, daq_config = isolated_transfer_env
    run_name = f"xfail_precond_{uuid.uuid4().hex[:8]}.pffd"
    
    await generate_mocked_run(fleet, daq_config, run_name)
    
    mgr = RunStateManager()
    job = transfer_job_factory(run_name=run_name, head_data_dir=head_data_dir)

    def rsync_side_effect(*args, **kwargs):
        simulate_rsync_from_fleet(fleet, run_name, head_data_dir / run_name)
        return None

    mock_rsync_transfer.side_effect = rsync_side_effect

    def wrapped_client_factory(*args, **kwargs):
        client = get_mapped_client_factory(daq_config)(*args, **kwargs)
        original_aenter = client.__aenter__
        async def mocked_aenter():
            mock_stub = await original_aenter()
            async def reject_cleanup(*c_args, **c_kwargs):
                # Simulate the DAQ node rejecting the manifest_digest precondition
                return {"success": False, "message": "FAILED_PRECONDITION: manifest_digest mismatch"}
            mock_stub.CleanupData = reject_cleanup
            return mock_stub
        client.__aenter__ = mocked_aenter
        return client

    with patch("panoseti_grpc.daq_control.client.AsyncDaqControlClient", side_effect=wrapped_client_factory):
         
         success, err = await _process_job(job, asyncio.Event(), mgr)
         
         assert success is False, "Job should fail if cleanup is rejected"
         assert "FAILED_PRECONDITION" in (err or "")
         
         state = mgr.load_state()
         assert state is not None
         assert state.status == RunStatus.VERIFY_FAILED
