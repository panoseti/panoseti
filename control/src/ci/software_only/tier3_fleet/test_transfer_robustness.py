"""
test_transfer_robustness.py

Integration test for TransferDaemon robustness.
Verifies that the daemon can survive multiple interruptions and low bandwidth
while transferring several MB of data, eventually converging to success.
"""

import asyncio
import logging
import os
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ci.software_only.tier3_fleet.transfer_testing_utils import (
    generate_mocked_run,
    get_mapped_client_factory,
    setup_isolated_transfer_env,
    simulate_rsync_from_fleet,
)
from control.transfer.daemon import _process_job
from control.transfer.queue import TransferQueue
from control.utils.pydantic_config_models import RunStatus
from control.utils.run_state import RunStateManager

logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_transfer_robustness_low_bandwidth_interrupted(
    session_fleet: Any,
    tmp_path: Path,
    ensure_clean_daq_state: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Scenario: Low bandwidth transfer with 4 random interruptions.
    Expectation: The TransferQueue and Daemon robustly resume and complete the job.
    """
    fleet, daq_cfg_dict = session_fleet
    head_data_dir, daq_config = setup_isolated_transfer_env(tmp_path, monkeypatch, daq_cfg_dict)
    run_name = f"robust_transfer_{uuid.uuid4().hex[:8]}.pffd"
    
    # 1. Update data_config.json with bwlimit before run
    data_config_path = Path(os.environ["PSETI_CONFIG"]) / "data_config.json"
    import json
    dc_data = json.loads(data_config_path.read_text())
    dc_data["xfr_bwlimit"] = 1024
    data_config_path.write_text(json.dumps(dc_data))
    
    # Invalidate config cache
    import importlib

    from control.utils import config_file
    importlib.reload(config_file)

    # 2. Generate real-ish run data on the DAQ containers (approx 2MB total)
    # file_size_kb=512, 2 nodes, 1 module each, 2 files per module = 2MB
    # This calls pseti stop, which will now enqueue the job with bwlimit=1024
    await generate_mocked_run(fleet, daq_config, run_name, file_size_kb=512)
    
    mgr = RunStateManager()
    tq = TransferQueue()
    
    # 3. Robustness Simulation
    interruptions_remaining = 4
    
    async def throttled_interrupted_rsync(*args, **kwargs):
        nonlocal interruptions_remaining
        # Validate that bwlimit was passed in the arguments
        assert any("--bwlimit=1024" in str(arg) for arg in args), f"bwlimit missing in args: {args}"
        
        # Simulate transfer time
        await asyncio.sleep(0.2)
        
        if interruptions_remaining > 0:
            interruptions_remaining -= 1
            logger.info("ROBUSTNESS: Triggering simulated daemon interruption (%d remaining)", interruptions_remaining)
            raise RuntimeError("Simulated Daemon Crash/Interruption")
            
        # If we didn't crash, perform the mock transfer
        simulate_rsync_from_fleet(fleet, run_name, head_data_dir / run_name)
        
        proc = MagicMock(returncode=0)
        proc.wait = AsyncMock(return_value=0)
        proc.communicate = AsyncMock(return_value=(b"", b""))
        proc.stdout.readline = AsyncMock(return_value=b"")
        proc.stderr.read = AsyncMock(return_value=b"")
        return proc

    # 4. Convergence Loop
    max_attempts = 20
    for attempt in range(max_attempts):
        current_job = tq.claim()
        if not current_job:
            # Check if it was already completed
            if (tq._queue / "completed" / f"{run_name}.job.toml").exists():
                break
            await asyncio.sleep(0.1)
            continue
            
        # Ensure bwlimit is set on the claimed job
        assert current_job.bwlimit == 1024

        with patch("control.transfer.daemon.asyncio.create_subprocess_exec", side_effect=throttled_interrupted_rsync), \
             patch("panoseti_grpc.daq_control.client.AsyncDaqControlClient", side_effect=get_mapped_client_factory(daq_config)):
             
             shutdown_event = asyncio.Event()
             success, err = await _process_job(current_job, shutdown_event, mgr)
             
             if success:
                 tq.complete(current_job.run_name)
                 break
             else:
                 # In a real daemon, the job would stay in active/ and be moved back to pending/ 
                 # after a timeout. For the test, we move it back immediately to speed up the loop.
                 active_path = tq._queue / "active" / f"{run_name}.job.toml"
                 if active_path.exists():
                     os.rename(active_path, tq._queue / "pending" / f"{run_name}.job.toml")

    # 5. Final Assertions
    assert interruptions_remaining == 0, f"Expected 4 interruptions, but only {4 - interruptions_remaining} occurred."
    assert (tq._queue / "completed" / f"{run_name}.job.toml").exists()
    assert mgr.load_state().status == RunStatus.ARCHIVED
    
    # Verify files arrived on head node
    dest_run = head_data_dir / run_name
    pff_files = list(dest_run.glob("*.pff"))
    assert len(pff_files) > 0
    # Check that our synthetic large files arrived (ignoring empty ones from real hashpipe)
    large_files = [f for f in pff_files if f.stat().st_size == 512 * 1024]
    assert len(large_files) >= 4, f"Expected 4+ large files, found {len(large_files)}. Total .pff: {len(pff_files)}"
