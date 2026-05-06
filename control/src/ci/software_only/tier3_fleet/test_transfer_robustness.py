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
from unittest.mock import patch

import pytest

from ci.fixtures.rsync_fixtures import RsyncMock
from ci.software_only.tier3_fleet.transfer_testing_utils import (
    generate_mocked_run,
    get_mapped_client_factory,
    simulate_rsync_from_fleet,
)
from control.transfer.daemon import _process_job
from control.transfer.queue import TransferQueue
from control.utils import config_file
from control.utils.paths import PanoPaths
from control.utils.pydantic_config_models import RunStatus
from control.utils.run_state import RunStateManager

logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_transfer_robustness_low_bandwidth_interrupted(
    session_fleet: Any,
    ensure_clean_daq_state: Any,
    mock_rsync_transfer: RsyncMock,
    isolated_transfer_env: tuple[Path, config_file.DaqConfig],
    transfer_queue: TransferQueue,
) -> None:
    """
    Scenario: Low bandwidth transfer with 4 random interruptions.
    Expectation: The TransferQueue and Daemon robustly resume and complete the job.
    """
    fleet, _ = session_fleet
    head_data_dir, daq_config = isolated_transfer_env
    run_name = f"robust_transfer_{uuid.uuid4().hex[:8]}.pffd"

    # 1. Update data_config.json with bwlimit before run
    data_config_path = PanoPaths.config_dir() / "data_config.json"
    import json
    dc_data = json.loads(data_config_path.read_text())
    dc_data["xfr_bwlimit"] = 1024
    data_config_path.write_text(json.dumps(dc_data))

    # Invalidate config cache
    import importlib

    from control.utils import config_file
    importlib.reload(config_file)

    # 2. Generate real-ish run data on the DAQ containers (approx 2MB total)
    await generate_mocked_run(fleet, daq_config, run_name, file_size_kb=512)

    mgr = RunStateManager()
    tq = transfer_queue

    # 3. Robustness Simulation
    interruptions_remaining = 4

    def rsync_side_effect(*args, **kwargs):
        nonlocal interruptions_remaining
        # Validate that bwlimit was passed in the arguments
        assert any("--bwlimit=1024" in str(arg) for arg in args), f"bwlimit missing in args: {args}"

        if interruptions_remaining > 0:
            interruptions_remaining -= 1
            logger.info("ROBUSTNESS: Triggering simulated daemon interruption (%d remaining)", interruptions_remaining)
            raise RuntimeError("Simulated Daemon Crash/Interruption")

        # If we didn't crash, perform the mock transfer
        simulate_rsync_from_fleet(fleet, run_name, head_data_dir / run_name)
        return None

    mock_rsync_transfer.side_effect = rsync_side_effect
    mock_rsync_transfer.delay = 0.2

    # 4. Convergence Loop
    max_attempts = 20
    for _attempt in range(max_attempts):
        current_job = tq.claim()
        if not current_job:
            # Check if it was already completed
            if (tq._queue / "completed" / f"{run_name}.job.toml").exists():
                break
            await asyncio.sleep(0.1)
            continue

        # Ensure bwlimit is set on the claimed job
        assert current_job.bwlimit == 1024

        with patch("panoseti_grpc.daq_control.client.AsyncDaqControlClient", side_effect=get_mapped_client_factory(daq_config)):

             shutdown_event = asyncio.Event()
             success, _err = await _process_job(current_job, shutdown_event, mgr)

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
    run_state = mgr.load_state()
    assert run_state is not None
    assert run_state.status == RunStatus.ARCHIVED

    # Verify files arrived on head node
    dest_run = head_data_dir / run_name
    pff_files = list(dest_run.glob("*.pff"))
    assert len(pff_files) > 0
    # Check that our synthetic large files arrived (ignoring empty ones from real hashpipe)
    large_files = [f for f in pff_files if f.stat().st_size == 512 * 1024]
    assert len(large_files) >= 4, f"Expected 4+ large files, found {len(large_files)}. Total .pff: {len(pff_files)}"

