"""
tier3_fleet/test_transfer_robustness.py

Integration test for TransferDaemon robustness.
Verifies the daemon survives multiple rsync interruptions and eventually
converges to success.

Ported from software_only/tier3_fleet/test_transfer_robustness.py.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest

from ci.fixtures.rsync_fixtures import RsyncMock
from ci.software_only.orchestrator.fleet import Fleet
from ci.software_only.tier3_fleet.conftest import requires_docker
from ci.software_only.tier3_fleet.transfer_testing_utils import (
    generate_mocked_run,
    get_mapped_client_factory,
    simulate_rsync_from_fleet,
)
from control.transfer.daemon import _process_job
from control.transfer.models import TransferJob, TransferStatus
from control.transfer.queue import TransferQueue
from control.utils.pydantic_config_models import RunStatus
from control.utils.run_state import RunStateManager

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.tier3


@requires_docker
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_when_rsync_interrupted_four_times_then_transfer_converges(
    session_fleet: Fleet,
    mock_rsync_transfer: RsyncMock,
    transfer_queue: TransferQueue,
    transfer_job_factory: Callable[..., TransferJob],
) -> None:
    """TransferQueue and daemon converge after 4 simulated rsync interruptions.

    Scenario: rsync raises RuntimeError on the first 4 calls; the test loop
    retries by moving the job back to pending/ on each failure.
    Expectation: after exactly 4 interruptions the transfer completes and
    the run reaches ARCHIVED.
    """
    fleet = session_fleet
    daq_config = fleet.live_daq_config
    head_data_dir = Path(daq_config.head_node_data_dir)
    run_name = f"robust_{uuid.uuid4().hex[:8]}.pffd"

    # 1. Valid start and stop condition
    await generate_mocked_run(fleet, daq_config, run_name, file_size_kb=512)

    mgr = RunStateManager()
    tq = transfer_queue

    # stop_run auto-enqueued a job. We remove it so we can enqueue our
    # custom one with bwlimit.
    pending_path = tq._job_path(TransferStatus.PENDING, run_name)
    if pending_path.exists():
        pending_path.unlink()

    # Create the job with an explicit bwlimit so the assertion below is meaningful.
    job_initial = transfer_job_factory(
        run_name=run_name,
        head_data_dir=head_data_dir,
        bwlimit=1024,
        daq_config=daq_config,
    )
    assert tq.enqueue(job_initial) is True

    interruptions_remaining = 4
    def rsync_side_effect(*args: object, **kwargs: object) -> None:
        nonlocal interruptions_remaining
        # Verify the bwlimit flag is present in the rsync command.
        assert any("--bwlimit=1024" in str(a) for a in args), (
            f"bwlimit flag missing: {args}"
        )
        if interruptions_remaining > 0:
            interruptions_remaining -= 1
            logger.info(
                "ROBUSTNESS: interruption triggered (%d remaining)",
                interruptions_remaining,
            )
            raise RuntimeError("Simulated rsync interruption")
        simulate_rsync_from_fleet(fleet, run_name, head_data_dir / run_name)

    mock_rsync_transfer.side_effect = rsync_side_effect

    max_attempts = 20
    with patch(
        "panoseti_grpc.daq_control.client.AsyncDaqControlClient",
        side_effect=get_mapped_client_factory(daq_config),
    ):
        for _attempt in range(max_attempts):
            current_job = tq.claim()
            if current_job is None:
                if (tq._queue / "completed" / f"{run_name}.job.toml").exists():
                    break
                await asyncio.sleep(0.1)
                continue

            assert current_job.bwlimit == 1024

            shutdown_event = asyncio.Event()
            success, _err = await _process_job(current_job, shutdown_event, mgr)

            if success:
                tq.complete(current_job.run_name)
                break
            else:
                active_path = tq._queue / "active" / f"{run_name}.job.toml"
                if active_path.exists():
                    os.rename(
                        active_path,
                        tq._queue / "pending" / f"{run_name}.job.toml",
                    )

    assert interruptions_remaining == 0, (
        f"Expected all 4 interruptions to fire; "
        f"only {4 - interruptions_remaining} occurred"
    )
    assert (tq._queue / "completed" / f"{run_name}.job.toml").exists()
    assert mgr.load_state() is not None
    assert mgr.load_state().status == RunStatus.ARCHIVED  # type: ignore[union-attr]

    dest_run = head_data_dir / run_name
    large_files = [
        f for f in dest_run.glob("*.pff") if f.stat().st_size == 512 * 1024
    ]
    assert len(large_files) >= 2, (
        f"Expected synthetic 512 KB files; found {len(large_files)}"
    )
