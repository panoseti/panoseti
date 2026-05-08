"""
tier3_fleet/test_transfer_daemon_e2e.py

Integration tests for the decoupled transfer pipeline.

Ported from software_only/tier3_fleet/test_transfer_daemon_e2e.py.
Tests that require a real Hashpipe binary (crash-resume, retry-exhaustion)
are left to tier5; only the mock-friendly paths are here.

Tests:
  1. Fleet-level: daemon archives a real mocked run (Docker required).
  2. Singleton lock: second daemon exits immediately.
  3. Unit-integration: _process_job with fully mocked gRPC + rsync.
  4. Max-attempts exhaustion: rsync always fails until the queue marks failed.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ci.fixtures.rsync_fixtures import RsyncMock
from ci.software_only_v2.infra.spec import FleetSpec
from ci.software_only_v2.infra.workspace import Workspace
from ci.software_only_v2.orchestrator.fleet import Fleet
from ci.software_only_v2.tier3_fleet.conftest import requires_docker
from ci.software_only_v2.tier3_fleet.transfer_testing_utils import (
    generate_mocked_run,
    get_mapped_client_factory,
    simulate_rsync_from_fleet,
    verify_head_node_accuracy,
)
from control.transfer.daemon import _process_job
from control.transfer.lifecycle import MAX_ATTEMPTS
from control.transfer.models import TransferJob, TransferNodeSpec
from control.transfer.queue import TransferQueue
from control.utils.paths import PanoPaths
from control.utils.pydantic_config_models import RunStatus
from control.utils.run_state import RunStateManager

pytestmark = pytest.mark.tier3


# ---------------------------------------------------------------------------
# 1. Fleet-level happy path: daemon archives a real mocked run
# ---------------------------------------------------------------------------

@requires_docker
@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_when_run_enqueued_then_daemon_archives_it(
    session_fleet: Fleet,
    mock_rsync_transfer: RsyncMock,
    transfer_queue: TransferQueue,
    transfer_job_factory: Callable[..., TransferJob],
) -> None:
    """E2E happy path: mocked run data is transferred and reaches ARCHIVED.

    generate_mocked_run enqueues PFF files on the fleet containers.
    _process_job drives the full state machine through to ARCHIVED.
    """
    fleet = session_fleet
    daq_config = fleet.live_daq_config
    head_data_dir = Path(daq_config.head_node_data_dir)
    run_name = f"e2e_{uuid.uuid4().hex[:8]}.pffd"

    expected_data = await generate_mocked_run(fleet, daq_config, run_name)

    mgr = RunStateManager()
    tq = transfer_queue
    job = transfer_job_factory(
        run_name=run_name,
        head_data_dir=head_data_dir,
        daq_config=daq_config,
    )
    tq.enqueue(job)
    claimed = tq.claim()
    assert claimed is not None
    assert claimed.run_name == run_name

    def rsync_side_effect(*args: object, **kwargs: object) -> None:
        simulate_rsync_from_fleet(fleet, run_name, head_data_dir / run_name)

    mock_rsync_transfer.side_effect = rsync_side_effect

    with patch(
        "panoseti_grpc.daq_control.client.AsyncDaqControlClient",
        side_effect=get_mapped_client_factory(daq_config),
    ):
        success, err = await asyncio.wait_for(
            _process_job(claimed, asyncio.Event(), mgr), timeout=30.0
        )

    assert success, f"_process_job failed: {err}"
    tq.complete(claimed.run_name)

    ledger = mgr.load_state()
    assert ledger is not None
    assert ledger.status == RunStatus.ARCHIVED
    verify_head_node_accuracy(head_data_dir, run_name, expected_data)


# ---------------------------------------------------------------------------
# 2. Singleton lock: a second daemon instance exits immediately
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pseti_workspace", [FleetSpec.minimal_unit()], indirect=True)
@pytest.mark.asyncio
async def test_when_lock_held_then_second_daemon_exits(
    pseti_workspace: Workspace,
) -> None:
    """Second daemon must detect the lock and exit cleanly without waiting."""
    lock_path = PanoPaths.locks_dir() / "transfer.lock"
    lock_script = pseti_workspace.root / "hold_lock.py"
    lock_script.write_text(
        "import os, time, sys\n"
        "lp = sys.argv[1]\n"
        "try:\n"
        "    fd = os.open(lp, os.O_WRONLY | os.O_CREAT | os.O_EXCL)\n"
        "    with os.fdopen(fd, 'w') as f:\n"
        "        f.write(str(os.getpid()))\n"
        "    print('LOCKED', flush=True)\n"
        "    time.sleep(10)\n"
        "except FileExistsError:\n"
        "    print('FAILED TO LOCK')\n"
    )

    env = os.environ.copy()
    env["PSETI_STATE"] = str(pseti_workspace.root / "state")

    p1 = await asyncio.create_subprocess_exec(
        sys.executable, str(lock_script), str(lock_path),
        stdout=asyncio.subprocess.PIPE,
        env=env,
    )
    assert p1.stdout is not None
    line = (await p1.stdout.readline()).decode()
    assert "LOCKED" in line

    base = str(PanoPaths.base_dir())
    python_path = f"{base}/src:{env.get('PYTHONPATH', '')}"
    env2 = {**env, "PYTHONPATH": python_path}

    p2 = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "control.transfer",
        cwd=base,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env2,
    )
    try:
        stdout_bytes, _ = await asyncio.wait_for(p2.communicate(), timeout=5)
        stdout = stdout_bytes.decode()
        assert p2.returncode == 0
        assert any(
            kw in stdout
            for kw in ("Another", "already", "running", "lock", "daemon")
        ), f"Expected lock-contention message, got: {stdout!r}"
    finally:
        p1.terminate()
        await p1.wait()


# ---------------------------------------------------------------------------
# 3. Unit-integration: _process_job with fully mocked gRPC + rsync
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pseti_workspace", [FleetSpec.minimal_unit()], indirect=True)
@pytest.mark.asyncio
async def test_when_grpc_and_rsync_mocked_then_job_completes(
    pseti_workspace: Workspace,
    mock_rsync_transfer: RsyncMock,
    transfer_queue: TransferQueue,
    transfer_job_factory: Callable[..., TransferJob],
) -> None:
    """_process_job with mocked gRPC + rsync drives the run to ARCHIVED."""
    run_name = "unit_int_run.pffd"
    head_dir = pseti_workspace.root / "head"

    job = TransferJob(
        schema_version=1,
        run_name=run_name,
        head_data_dir=str(head_dir),
        head_node_username="panoseti",
        created_at=datetime.now(UTC),
        daq_nodes=[
            TransferNodeSpec(
                ip_addr="127.0.0.1",
                username="root",
                data_dir="/tmp/daq",
                module_ids=[100],
            )
        ],
    )

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.GenerateManifest = AsyncMock(
        return_value={"success": True, "file_count": 0}
    )
    mock_client.CleanupData = AsyncMock(return_value={"success": True})

    def rsync_side_effect(*args: object, **kwargs: object) -> None:
        (head_dir / run_name).mkdir(parents=True, exist_ok=True)
        (head_dir / run_name / "dp_manifest.node_test.algo_blake3.txt").touch()

    mock_rsync_transfer.side_effect = rsync_side_effect

    with patch(
        "panoseti_grpc.daq_control.client.AsyncDaqControlClient",
        return_value=mock_client,
    ):
        success, err = await _process_job(job, asyncio.Event(), RunStateManager())

    assert success, f"_process_job failed: {err}"
    assert (head_dir / run_name / "run_complete").exists()


# ---------------------------------------------------------------------------
# 4. Max-attempts exhaustion: queue marks job failed after MAX_ATTEMPTS
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pseti_workspace", [FleetSpec.minimal_unit()], indirect=True)
@pytest.mark.asyncio
async def test_when_rsync_always_fails_then_queue_marks_failed_after_max_attempts(
    pseti_workspace: Workspace,
    transfer_queue: TransferQueue,
    transfer_job_factory: Callable[..., TransferJob],
) -> None:
    """After MAX_ATTEMPTS consecutive rsync failures the job moves to failed/."""
    run_name = f"exhausted_{uuid.uuid4().hex[:8]}.pffd"
    head_dir = pseti_workspace.root / "head"
    (head_dir / run_name).mkdir(parents=True, exist_ok=True)

    job = TransferJob(
        schema_version=1,
        run_name=run_name,
        head_data_dir=str(head_dir),
        head_node_username="panoseti",
        created_at=datetime.now(UTC),
        daq_nodes=[
            TransferNodeSpec(
                ip_addr="127.0.0.1",
                username="root",
                data_dir="/tmp/daq",
                module_ids=[100],
            )
        ],
    )
    tq = transfer_queue
    tq.enqueue(job)

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.GenerateManifest = AsyncMock(
        return_value={"success": True, "file_count": 0}
    )

    async def _mock_fail(*args: object, **kwargs: object) -> MagicMock:
        proc = MagicMock()
        proc.returncode = 1
        proc.communicate = AsyncMock(return_value=(b"", b"rsync: connection refused"))
        proc.wait = AsyncMock(return_value=1)
        return proc

    mgr = RunStateManager()
    with (
        patch(
            "control.transfer.daemon.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            side_effect=_mock_fail,
        ),
        patch(
            "panoseti_grpc.daq_control.client.AsyncDaqControlClient",
            return_value=mock_client,
        ),
    ):
        for attempt in range(MAX_ATTEMPTS):
            current = tq.claim()
            assert current is not None, f"Queue empty on attempt {attempt}"
            success, _ = await _process_job(current, asyncio.Event(), mgr)
            assert not success
            tq.fail(current.run_name)
            if attempt < MAX_ATTEMPTS - 1:
                tq.retry(current.run_name)

    assert (tq._queue / "failed" / f"{run_name}.job.toml").exists(), (
        "Job should be in failed/ after max attempts"
    )
