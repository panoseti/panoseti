"""
tier5_integration/test_integration_transfer_queue_validity.py — Transfer lifecycle.

Full end-to-end transfer lifecycle against the real compose stack:
  1. Start run (real hashpipe + tcpreplay).
  2. Stop run (enqueues transfer job).
  3. Run Transfer Daemon.
  4. Poll ledger to ARCHIVED.
  5. Verify transfer accuracy and selective cleanup.
"""

from __future__ import annotations

import asyncio
import contextlib
import pathlib
import time
from typing import Any
from unittest.mock import patch

import pytest

from ci.software_only_v2.tier5_integration.conftest import (
    DAQ_DATA_DIR,
    requires_compose_stack,
)
from ci.shared.transfer_helpers import (
    generate_integration_run,
    mocked_build_rsync_cmd,
    setup_isolated_integration_transfer_env,
    verify_integration_transfer_accuracy,
)
from control.transfer.daemon import run_daemon
from control.utils.paths import PanoPaths
from control.utils.run_state import RunStateManager, RunStatus

pytestmark = [pytest.mark.tier5, requires_compose_stack]


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_integration_transfer_queue_lifecycle(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    daqnode_docker_container: Any,
    t5_workspace: Any,
) -> None:
    """Full transfer lifecycle: Start → tcpreplay → Stop → Daemon → ARCHIVED."""
    head_data_dir, daq_config = setup_isolated_integration_transfer_env(
        tmp_path, monkeypatch
    )
    mgr = RunStateManager()
    run_name = f"t5_transfer_{int(time.time())}.pffd"

    # Step 1: Generate run (start + tcpreplay + stop)
    await generate_integration_run(run_name, daq_config, daqnode_docker_container)

    # Step 2: Run Transfer Daemon
    with patch("control.transfer.daemon.build_rsync_cmd", side_effect=mocked_build_rsync_cmd):
        daemon_task = asyncio.create_task(run_daemon(poll_interval=1.0))

        tq_dir = PanoPaths.transfer_queue_dir()
        hb_path = tq_dir.parent / "daemon.heartbeat"
        start_wait = time.time()
        while time.time() - start_wait < 10:
            if hb_path.exists():
                break
            await asyncio.sleep(0.5)

        try:
            # Step 3: Poll ledger until ARCHIVED
            deadline = time.time() + 180.0
            while time.time() < deadline:
                ledger = mgr.load_state()
                if (
                    ledger
                    and ledger.run_name == run_name
                    and ledger.status == RunStatus.ARCHIVED
                ):
                    break
                await asyncio.sleep(2.0)
            else:
                last = mgr.load_state()
                last_status = last.status if last else "None"
                pytest.fail(
                    f"Timed out waiting for ARCHIVED. Current status: {last_status}"
                )

            # Step 4: Verify transfer accuracy
            verify_integration_transfer_accuracy(head_data_dir, run_name, daq_config)

        finally:
            daemon_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await daemon_task


@pytest.mark.asyncio
@pytest.mark.timeout(600)
@pytest.mark.parametrize("num_runs", [2])
async def test_integration_transfer_queue_drain(
    num_runs: int,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    daqnode_docker_container: Any,
) -> None:
    """Multiple runs queued while daemon is paused; daemon drains all to ARCHIVED."""
    head_data_dir, daq_config = setup_isolated_integration_transfer_env(
        tmp_path, monkeypatch
    )
    mgr = RunStateManager()
    tq_dir = PanoPaths.transfer_queue_dir()
    run_names = []

    for i in range(num_runs):
        run_name = f"t5_drain_{i}_{int(time.time())}.pffd"
        run_names.append(run_name)
        await generate_integration_run(run_name, daq_config, daqnode_docker_container)

        ledger = mgr.load_state()
        assert ledger is not None
        assert ledger.run_name == run_name
        assert ledger.status == RunStatus.RECORDING_ENDED

    pending = list((tq_dir / "pending").glob("*.job.toml"))
    assert len(pending) == num_runs

    await asyncio.sleep(5.0)

    with patch("control.transfer.daemon.build_rsync_cmd", side_effect=mocked_build_rsync_cmd):
        daemon_task = asyncio.create_task(run_daemon(poll_interval=1.0))

        hb_path = tq_dir.parent / "daemon.heartbeat"
        start_wait = time.time()
        while time.time() - start_wait < 10:
            if hb_path.exists():
                break
            await asyncio.sleep(0.5)

        try:
            timeout = 180.0 * num_runs
            deadline = time.time() + timeout
            while time.time() < deadline:
                remaining = list((tq_dir / "pending").glob("*.job.toml"))
                if not remaining:
                    break
                await asyncio.sleep(2.0)
            else:
                remaining = list((tq_dir / "pending").glob("*.job.toml"))
                pytest.fail(
                    f"Queue not drained: {len(remaining)} jobs still pending after {timeout}s"
                )
        finally:
            daemon_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await daemon_task

    # Every run must have been archived
    completed = list((tq_dir / "completed").glob("*.job.toml"))
    completed_names = {j.stem.replace(".job", "") for j in completed}
    for rn in run_names:
        stem = rn.replace(".pffd", "")
        assert stem in completed_names or any(rn in n for n in completed_names), (
            f"Run {rn} not found in completed queue after drain"
        )
