"""
test_integration_transfer_queue_validity.py — Tier 5 Heavy Integration test for transfer queue.

Validates the full lifecycle:
1. Start run (real Hashpipe + tcpreplay).
2. Wait for real data generation.
3. Stop run (enqueues transfer job).
4. Run real Transfer Daemon in background.
5. Poll ledger until ARCHIVED.
6. Verify 100% byte accuracy and selective cleanup on DAQ nodes.
"""

from __future__ import annotations

import asyncio
import contextlib
import pathlib
import time
from typing import Any
from unittest.mock import patch

import pytest

from ci.shared.transfer_helpers import (
    generate_integration_run,
    mocked_build_rsync_cmd,
    setup_isolated_integration_transfer_env,
    verify_integration_transfer_accuracy,
)
from control.transfer.daemon import run_daemon
from control.utils.paths import PanoPaths
from control.utils.run_state import RunStateManager, RunStatus


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_integration_transfer_queue_lifecycle(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    daqnode_container: Any,
) -> None:
    """
    Test the full transfer queue lifecycle in a Tier 5 integration environment.
    """
    head_data_dir, daq_config = setup_isolated_integration_transfer_env(tmp_path, monkeypatch)
    mgr = RunStateManager()
    run_name = f"int_transfer_lifecycle_{int(time.time())}.pffd"

    # --- Step 1: Generate Run (Start/Stop) ---
    await generate_integration_run(run_name, daq_config, daqnode_container)

    # --- Step 2: Run Transfer Daemon Loop as a task ---
    with patch("control.transfer.daemon.build_rsync_cmd", side_effect=mocked_build_rsync_cmd):
        daemon_task = asyncio.create_task(run_daemon(poll_interval=1.0))

        # Wait for daemon heartbeat
        tq_dir = PanoPaths.transfer_queue_dir()
        hb_path = tq_dir.parent / "daemon.heartbeat"
        start_wait = time.time()
        while time.time() - start_wait < 10:
            if hb_path.exists():
                break
            await asyncio.sleep(0.5)

        try:
            # --- Step 3: Poll Ledger until ARCHIVED ---
            timeout = 180.0
            start_time = time.time()
            while time.time() - start_time < timeout:
                ledger = mgr.load_state()
                if ledger and ledger.run_name == run_name and ledger.status == RunStatus.ARCHIVED:
                    break
                await asyncio.sleep(2.0)
            else:
                last_ledger = mgr.load_state()
                last_status = last_ledger.status if last_ledger else "None"
                pytest.fail(f"Timed out waiting for ARCHIVED status. Current status: {last_status}")

            # --- Step 4: Final Validation ---
            verify_integration_transfer_accuracy(head_data_dir, run_name, daq_config)

        finally:
            daemon_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await daemon_task
