"""
test_integration_transfer_advanced.py — Advanced Tier 5 Integration scenarios for transfer queue.
Includes queue draining and continuous watching tests.
"""

from __future__ import annotations

import asyncio
import contextlib
import pathlib
import time
from typing import Any
from unittest.mock import patch

import pytest

from ci.tier5_integration.transfer_integration_utils import (
    generate_integration_run,
    mocked_build_rsync_cmd,
    setup_isolated_integration_transfer_env,
    verify_integration_transfer_accuracy,
)
from control.transfer.daemon import run_daemon
from control.utils.paths import PanoPaths
from control.utils.run_state import RunStateManager


@pytest.mark.parametrize("num_runs", [2])
@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_integration_transfer_queue_drain(
    num_runs: int,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    daqnode_container: Any,
) -> None:
    """Verify multiple runs can be queued and then drained sequentially by the daemon."""
    head_data_dir, daq_config = setup_isolated_integration_transfer_env(tmp_path, monkeypatch)
    mgr = RunStateManager()
    tq_dir = PanoPaths.transfer_queue_dir()
    
    run_names = []
    
    # 1. Generate multiple runs while daemon is stopped
    for i in range(num_runs):
        run_name = f"int_drain_{i}_{int(time.time())}.pffd"
        run_names.append(run_name)
        await generate_integration_run(run_name, daq_config, daqnode_container)
        
        # Verify ledger state (singleton ledger matches last stopped run)
        ledger = mgr.load_state()
        assert ledger.run_name == run_name
        assert ledger.status == "RECORDING_ENDED"
        
    # Verify queue depth
    pending_jobs = list((tq_dir / "pending").glob("*.job.toml"))
    assert len(pending_jobs) == num_runs
    
    # 2. Start Transfer Daemon Loop
    # We wait a bit to ensure all stops have fully enqueued and settled
    await asyncio.sleep(5.0)
    
    with patch("control.transfer.daemon.build_rsync_cmd", side_effect=mocked_build_rsync_cmd):
        daemon_task = asyncio.create_task(run_daemon(poll_interval=1.0))
        
        # Wait for daemon heartbeat
        hb_path = tq_dir.parent / "daemon.heartbeat"
        start_wait = time.time()
        while time.time() - start_wait < 10:
            if hb_path.exists():
                break
            await asyncio.sleep(0.5)
        
        try:
            # 3. Poll until all runs are processed (jobs moved from pending)
            timeout = 180.0 * num_runs
            start_time = time.time()
            while time.time() - start_time < timeout:
                remaining = list((tq_dir / "pending").glob("*.job.toml"))
                if not remaining:
                    # Final check that last run in ledger is ARCHIVED
                    ledger = mgr.load_state()
                    if ledger and ledger.status == "ARCHIVED":
                        break
                await asyncio.sleep(2.0)
            else:
                pytest.fail(f"Timed out waiting for runs to archive. Pending jobs: {len(list((tq_dir / 'pending').glob('*.job.toml')))}")
                
            # 4. Final Validation for all runs
            for rn in run_names:
                verify_integration_transfer_accuracy(head_data_dir, rn, daq_config)
                
        finally:
            daemon_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await daemon_task


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_integration_transfer_queue_active_daemon(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    daqnode_container: Any,
) -> None:
    """Verify daemon automatically picks up a run enqueued while it's already active."""
    head_data_dir, daq_config = setup_isolated_integration_transfer_env(tmp_path, monkeypatch)
    mgr = RunStateManager()
    tq_dir = PanoPaths.transfer_queue_dir()
    
    # 1. Start Transfer Daemon first
    with patch("control.transfer.daemon.build_rsync_cmd", side_effect=mocked_build_rsync_cmd):
        daemon_task = asyncio.create_task(run_daemon(poll_interval=1.0))
        
        # Wait for heartbeat
        hb_path = tq_dir.parent / "daemon.heartbeat"
        start_wait = time.time()
        while time.time() - start_wait < 10:
            if hb_path.exists():
                break
            await asyncio.sleep(0.5)
            
        try:
            # 2. Generate a run
            run_name = f"int_active_{int(time.time())}.pffd"
            await generate_integration_run(run_name, daq_config, daqnode_container)
            
            # 3. Poll ledger - should transition to ARCHIVED automatically
            timeout = 180.0
            start_time = time.time()
            while time.time() - start_time < timeout:
                ledger = mgr.load_state()
                if ledger and ledger.run_name == run_name and ledger.status == "ARCHIVED":
                    break
                await asyncio.sleep(2.0)
            else:
                ledger = mgr.load_state()
                pytest.fail(f"Run did not archive automatically. Current ledger: {ledger.run_name if ledger else 'None'} ({ledger.status if ledger else 'None'})")
                
            # 4. Final Validation
            verify_integration_transfer_accuracy(head_data_dir, run_name, daq_config)
            
        finally:
            daemon_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await daemon_task
