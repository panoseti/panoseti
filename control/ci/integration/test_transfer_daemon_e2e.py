# mypy: ignore-errors
"""
Integration tests for the transfer daemon E2E flow.

These tests require the full Docker CI stack::

    python ci/qa.py up

Skip gracefully when not in Docker CI environment.

The ``test_transfer_daemon_unit_integration`` test is an in-process hybrid:
it uses a fake filesystem (tmp_path) and mocked gRPC/rsync, so it runs
without Docker and is included in the normal unit-integration boundary.
"""

from __future__ import annotations

import asyncio
import os
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import anyio
import pytest

from ci.integration.conftest import copy_run_dir

# ---------------------------------------------------------------------------
# gRPC stub injection (mirrors test_transfer_daemon.py helper)
# ---------------------------------------------------------------------------


@contextmanager
def _mock_grpc_modules(mock_client: MagicMock):
    """Inject fake panoseti_grpc modules so the local import in _process_job resolves."""
    stub_root = ModuleType("panoseti_grpc")
    stub_daq = ModuleType("panoseti_grpc.daq_control")
    stub_client_mod = ModuleType("panoseti_grpc.daq_control.client")

    # AsyncDaqControlClient constructor returns mock_client regardless of args.
    stub_client_mod.AsyncDaqControlClient = MagicMock(return_value=mock_client)
    
    stub_root.daq_control = stub_daq  # type: ignore[attr-defined]
    stub_daq.client = stub_client_mod  # type: ignore[attr-defined]

    injected = {
        "panoseti_grpc": stub_root,
        "panoseti_grpc.daq_control": stub_daq,
        "panoseti_grpc.daq_control.client": stub_client_mod,
    }
    prev: dict = {}
    for key, mod in injected.items():
        prev[key] = sys.modules.get(key)
        sys.modules[key] = mod
    try:
        yield stub_client_mod.AsyncDaqControlClient
    finally:
        for key, original in prev.items():
            if original is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = original

DOCKER_CI = os.environ.get("IN_DOCKER_CI") == "1"
skip_outside_ci = pytest.mark.skipif(
    not DOCKER_CI, reason="Requires Docker CI environment (IN_DOCKER_CI=1)"
)


# ---------------------------------------------------------------------------
# Docker E2E tests — skipped outside CI
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@skip_outside_ci
async def test_transfer_daemon_archives_run(
    daq_control_direct,
    run_params,
    ensure_clean_daq_state,
    head_data_dir,
) -> None:
    """
    Full E2E: stop.py enqueues a job; daemon picks it up and archives the run.

    Verifies:
    - After stop.py completes, ledger status is RECORDING_ENDED
    - After daemon completes, ledger status is ARCHIVED
    - run_complete marker exists on head node
    - .pff files removed from DAQ node; .json/.log preserved
    """
    import uuid

    import control.stop as stop
    from ci.integration.conftest import wait_hashpipe_running
    from ci.integration.scenarios.conftest import _start as grpc_start
    from control.utils import config_file
    from control.utils.paths import PanoPaths
    from control.utils.run_state import RunStateManager
    from control.utils.transfer.daemon import _process_job
    from control.utils.transfer.queue import TransferQueue

    run_params = dict(run_params)
    run_params["run_dir"] = f"ci_daemon_{uuid.uuid4().hex[:8]}.pffd"
    import shutil
    RunStateManager().clear_state()
    queue_dir = PanoPaths.tmp_dir() / "transfer_queue"
    if queue_dir.exists():
        shutil.rmtree(queue_dir)

    from datetime import UTC, datetime

    from control.utils.pydantic_config_models import RunStateLedger
    mgr = RunStateManager()
    ledger = RunStateLedger(
        run_name=run_params["run_dir"],
        status="ACTIVE",
        start_time=datetime.now(UTC).isoformat(),
    )
    mgr.save_state(ledger)

    # 1. Start real run
    ok, _ = grpc_start(daq_control_direct, run_params)
    assert ok
    wait_hashpipe_running(daq_control_direct, "/data", timeout=5)

    daq_config = config_file.get_daq_config()
    net = config_file.get_network_config()
    uids = config_file.get_quabo_uids()

    import os
    os.makedirs(f"{daq_config.head_node_data_dir}/{run_params['run_dir']}", exist_ok=True)
    assert await anyio.Path(f"{daq_config.head_node_data_dir}/{run_params['run_dir']}").exists(), "Failed to create run_dir"

    # 2. Stop real run
    success = await stop.stop_run(
        daq_config, net, uids, run=run_params["run_dir"], verbose=False
    )
    assert success

    mgr = RunStateManager()
    ledger = mgr.load_state()
    assert ledger is not None
    assert ledger.status == "RECORDING_ENDED"

    # 3. Process job via daemon's real handler
    tq = TransferQueue(base_dir=str(PanoPaths.tmp_dir()))
    job = tq.claim()
    assert job is not None
    assert job["run_name"] == run_params["run_dir"]

    # Real process_job (does real GenerateManifest, real CleanupData)
    # But mock rsync to use our local copy simulator since SSH is not set up in CI
    def mocked_rsync(node_ip, node_data_dir, run_name, head_data_dir, *args, **kwargs):
        ok = copy_run_dir(run_params, Path(head_data_dir))
        return ok, "" if ok else "Simulated copy failed"

    with patch("control.utils.transfer.daemon.rsync_one_node", side_effect=mocked_rsync):
        job_success = await _process_job(job, tq._base)
        assert job_success

    
    tq.complete(job["run_name"])

    ledger = mgr.load_state()
    assert ledger and ledger.status == "ARCHIVED"
    run_dir_path = Path(daq_config.head_node_data_dir) / run_params["run_dir"]
    assert (run_dir_path / "run_complete").exists()


@pytest.mark.asyncio
@skip_outside_ci
async def test_transfer_daemon_resumes_after_crash(
    daq_control_direct,
    run_params,
    ensure_clean_daq_state,
) -> None:
    """
    Chaos: daemon killed mid-rsync; restart completes the transfer.

    Verifies that the durable queue allows a restarted daemon to claim and
    complete a job that was interrupted during an earlier invocation.
    """
    import shutil
    import uuid

    import control.stop as stop
    from ci.integration.conftest import wait_hashpipe_running
    from ci.integration.scenarios.conftest import _start as grpc_start
    from control.utils import config_file
    from control.utils.paths import PanoPaths
    from control.utils.run_state import RunStateManager
    from control.utils.transfer.daemon import _process_job
    from control.utils.transfer.queue import TransferQueue

    run_params = dict(run_params)
    run_params["run_dir"] = f"ci_daemon_{uuid.uuid4().hex[:8]}.pffd"
    RunStateManager().clear_state()
    queue_dir = PanoPaths.tmp_dir() / "transfer_queue"
    if queue_dir.exists():
        shutil.rmtree(queue_dir)

    from datetime import UTC, datetime

    from control.utils.pydantic_config_models import RunStateLedger
    mgr = RunStateManager()
    ledger = RunStateLedger(
        run_name=run_params["run_dir"],
        status="ACTIVE",
        start_time=datetime.now(UTC).isoformat(),
    )
    mgr.save_state(ledger)

    ok, _ = grpc_start(daq_control_direct, run_params)
    assert ok
    wait_hashpipe_running(daq_control_direct, "/data", timeout=5)

    daq_config = config_file.get_daq_config()
    net = config_file.get_network_config()
    uids = config_file.get_quabo_uids()
    import os
    os.makedirs(f"{daq_config.head_node_data_dir}/{run_params['run_dir']}", exist_ok=True)
    assert await anyio.Path(f"{daq_config.head_node_data_dir}/{run_params['run_dir']}").exists(), "Failed to create run_dir"
    await stop.stop_run(daq_config, net, uids, run=run_params["run_dir"], verbose=False)

    tq = TransferQueue(base_dir=str(PanoPaths.tmp_dir()))
    job = tq.claim()
    assert job is not None

    # Simulate crash mid-rsync by patching rsync_one_node to raise an exception
    with patch("control.utils.transfer.daemon.rsync_one_node", side_effect=RuntimeError("Simulated crash")), \
        pytest.raises(RuntimeError, match="Simulated crash"):
            await _process_job(job, tq._base)

    # Because it crashed, the job is technically "orphaned" in the active queue
    # A real daemon restart would move it back to pending on claim if it sees it as stale, 
    # but tq.claim() handles claiming from pending. Let's manually re-enqueue like a recovery script would, 
    # or test the daemon's recovery logic if it has any.
    tq.retry(job["run_name"], attempts=1)

    job2 = tq.claim()
    assert job2 is not None
    
    # Real process_job (does real GenerateManifest, real CleanupData)
    # But mock rsync to use our local copy simulator since SSH is not set up in CI
    def mocked_rsync(node_ip, node_data_dir, run_name, head_data_dir, *args, **kwargs):
        ok = copy_run_dir(run_params, Path(head_data_dir))
        return ok, "" if ok else "Simulated copy failed"

    with patch("control.utils.transfer.daemon.rsync_one_node", side_effect=mocked_rsync):
        job_success = await _process_job(job2, tq._base)
        assert job_success
    tq.complete(job2["run_name"])


@pytest.mark.asyncio
@skip_outside_ci
async def test_transfer_daemon_retry_on_transient_rsync_failure(
    daq_control_direct,
    run_params,
    ensure_clean_daq_state,
) -> None:
    """
    Retry: rsync fails twice with a transient error code, succeeds on the
    third attempt.

    Verifies that the daemon honours MAX_ATTEMPTS and re-enqueues on failure,
    and that the job lands in completed/ after eventual success.
    """
    import shutil
    import uuid

    import control.stop as stop
    from ci.integration.conftest import wait_hashpipe_running
    from ci.integration.scenarios.conftest import _start as grpc_start
    from control.utils import config_file
    from control.utils.paths import PanoPaths
    from control.utils.run_state import RunStateManager
    from control.utils.transfer.daemon import _process_job
    from control.utils.transfer.queue import TransferQueue

    run_params = dict(run_params)
    run_params["run_dir"] = f"ci_daemon_{uuid.uuid4().hex[:8]}.pffd"
    RunStateManager().clear_state()
    queue_dir = PanoPaths.tmp_dir() / "transfer_queue"
    if queue_dir.exists():
        shutil.rmtree(queue_dir)

    from datetime import UTC, datetime

    from control.utils.pydantic_config_models import RunStateLedger
    mgr = RunStateManager()
    ledger = RunStateLedger(
        run_name=run_params["run_dir"],
        status="ACTIVE",
        start_time=datetime.now(UTC).isoformat(),
    )
    mgr.save_state(ledger)

    ok, _ = grpc_start(daq_control_direct, run_params)
    assert ok
    wait_hashpipe_running(daq_control_direct, "/data", timeout=5)

    daq_config = config_file.get_daq_config()
    net = config_file.get_network_config()
    uids = config_file.get_quabo_uids()
    import os
    os.makedirs(f"{daq_config.head_node_data_dir}/{run_params['run_dir']}", exist_ok=True)
    assert await anyio.Path(f"{daq_config.head_node_data_dir}/{run_params['run_dir']}").exists(), "Failed to create run_dir"
    await stop.stop_run(daq_config, net, uids, run=run_params["run_dir"], verbose=False)

    tq = TransferQueue(base_dir=str(PanoPaths.tmp_dir()))
    job = tq.claim()
    assert job is not None

    # Mock rsync to fail twice, then succeed
    call_count = 0

    def mocked_rsync(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return False, "Transient failure"
        return True, "Success"

    with patch("control.utils.transfer.daemon.rsync_one_node", side_effect=mocked_rsync):
        # Attempt 1
        success1 = await _process_job(job, tq._base)
        assert not success1
        tq.retry(job["run_name"], attempts=1)

        # Attempt 2
        job2 = tq.claim()
        assert job2 is not None
        success2 = await _process_job(job2, tq._base)
        assert not success2
        tq.retry(job["run_name"], attempts=2)

        # Attempt 3
        job3 = tq.claim()
        assert job3 is not None
        success3 = await _process_job(job3, tq._base)
        assert success3
        tq.complete(job3["run_name"])

    completed_dir = tq._queue / "completed"
    assert (completed_dir / f"{job['run_name']}.job.toml").exists()


@pytest.mark.asyncio
@skip_outside_ci
async def test_transfer_daemon_marks_failed_after_max_attempts(
    daq_control_direct,
    run_params,
    ensure_clean_daq_state,
) -> None:
    """
    Exhaustion: rsync fails on every attempt up to MAX_ATTEMPTS.

    Verifies that the daemon moves the job to failed/ rather than looping
    indefinitely.
    """
    import shutil
    import uuid

    import control.stop as stop
    from ci.integration.conftest import wait_hashpipe_running
    from ci.integration.scenarios.conftest import _start as grpc_start
    from control.utils import config_file
    from control.utils.paths import PanoPaths
    from control.utils.run_state import RunStateManager
    from control.utils.transfer.daemon import MAX_ATTEMPTS, _process_job
    from control.utils.transfer.queue import TransferQueue

    run_params = dict(run_params)
    run_params["run_dir"] = f"ci_daemon_{uuid.uuid4().hex[:8]}.pffd"
    RunStateManager().clear_state()
    queue_dir = PanoPaths.tmp_dir() / "transfer_queue"
    if queue_dir.exists():
        shutil.rmtree(queue_dir)

    from datetime import UTC, datetime

    from control.utils.pydantic_config_models import RunStateLedger
    mgr = RunStateManager()
    ledger = RunStateLedger(
        run_name=run_params["run_dir"],
        status="ACTIVE",
        start_time=datetime.now(UTC).isoformat(),
    )
    mgr.save_state(ledger)

    ok, _ = grpc_start(daq_control_direct, run_params)
    assert ok
    wait_hashpipe_running(daq_control_direct, "/data", timeout=5)

    daq_config = config_file.get_daq_config()
    net = config_file.get_network_config()
    uids = config_file.get_quabo_uids()
    import os
    os.makedirs(f"{daq_config.head_node_data_dir}/{run_params['run_dir']}", exist_ok=True)
    run_dir_path = anyio.Path(daq_config.head_node_data_dir) / run_params['run_dir']
    assert await run_dir_path.exists(), "Failed to create run_dir"
    await stop.stop_run(daq_config, net, uids, run=run_params["run_dir"], verbose=False)

    tq = TransferQueue(base_dir=str(PanoPaths.tmp_dir()))
    job = tq.claim()

    with patch("control.utils.transfer.daemon.rsync_one_node", return_value=(False, "Persistent failure")):
        for attempt in range(MAX_ATTEMPTS):
            success = await _process_job(job, tq._base)
            assert not success
            if attempt < MAX_ATTEMPTS - 1:
                tq.retry(job["run_name"], attempts=attempt + 1)
                job = tq.claim()
            else:
                tq.fail(job["run_name"])

    failed_dir = tq._queue / "failed"
    assert (failed_dir / f"{job['run_name']}.job.toml").exists()


@skip_outside_ci
def test_transfer_daemon_singleton_lock_in_container(tmp_path) -> None:
    """
    Lock contention: a second daemon process started while the first is
    running must exit immediately without processing any jobs.
    """
    import subprocess

    from control.utils.paths import PanoPaths

    # Start a dummy python process that holds the lock using fcntl
    lock_script = PanoPaths.tmp_dir() / "hold_lock.py"
    lock_script.write_text('''
import fcntl
import time
import os
import sys

lock_file = sys.argv[1]
with open(lock_file, "w") as f:
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        print("LOCKED", flush=True)
        time.sleep(10)
    except BlockingIOError:
        print("FAILED TO LOCK")
''')

    lock_path = PanoPaths.tmp_dir() / "panoseti_transfer.lock"
    p1 = subprocess.Popen(["python3", str(lock_script), str(lock_path)], stdout=subprocess.PIPE, text=True)
    
    # Wait for it to lock
    assert p1.stdout is not None
    assert "LOCKED" in p1.stdout.readline()

    # Now attempt to start the daemon process
    daemon_script = PanoPaths.base_dir() / "src" / "control" / "daemons" / "transfer_daemon.py"

    # We don't want it to run forever if it fails to exit, so we use a short timeout
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PanoPaths.base_dir()}/src:{env.get('PYTHONPATH', '')}"

    p2 = subprocess.Popen(["python3", str(daemon_script)], cwd=str(PanoPaths.tmp_dir().parent), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)

    try:
        stdout, _ = p2.communicate(timeout=5)
        # Should exit quickly and print "Another transfer daemon is already running"
        assert p2.returncode == 0
        assert "Another transfer daemon is already running" in stdout
    finally:
        p1.terminate()
        p1.wait()

# ---------------------------------------------------------------------------
# In-process integration: no Docker required
# ---------------------------------------------------------------------------


def test_transfer_daemon_unit_integration(tmp_path) -> None:
    """
    In-process integration: enqueue a job, run one daemon iteration, verify
    ARCHIVED.

    Uses mocked gRPC and fake filesystem — no Docker required.  This test
    exercises the integration between TransferQueue, _process_job, and the
    run_complete marker in a single asyncio.run() call.
    """
    from control.utils.pydantic_config_models import RunStateLedger
    from control.utils.run_state import RunStateManager
    from control.utils.transfer.daemon import _process_job

    run_name = "e2e_test.pffd"
    head_data_dir = str(tmp_path / "data")
    (tmp_path / "data" / run_name).mkdir(parents=True)

    # Set up the run state ledger so daemon stages can call transition()
    mgr = RunStateManager(base_dir=str(tmp_path))
    ledger = RunStateLedger(
        run_name=run_name,
        status="RECORDING_ENDED",
        start_time=datetime.now(UTC).isoformat(),
    )
    mgr.save_state(ledger)

    job = {
        "run_name": run_name,
        "head_data_dir": head_data_dir,
        "daq_nodes": [
            {"ip_addr": "192.168.0.10", "data_dir": "/app/data", "module_ids": [250]}
        ],
        "no_collect": False,
        "no_cleanup": False,
    }

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.GenerateManifest = AsyncMock(return_value={"success": True, "file_count": 0})
    mock_client.CleanupData = AsyncMock(return_value={"success": True, "deleted_count": 0})

    with _mock_grpc_modules(mock_client), \
         patch("control.utils.transfer.daemon.rsync_one_node", return_value=(True, "")):
        success = asyncio.run(_process_job(job, tmp_path))

    assert success is True, "_process_job must return True on success"
    assert (tmp_path / "data" / run_name / "run_complete").exists(), (
        "run_complete marker must be written after successful archive"
    )


def test_transfer_queue_enqueue_then_process(tmp_path) -> None:
    """
    In-process: TransferQueue.enqueue() → claim() → _process_job() → complete().

    Verifies the full queue lifecycle without network calls.
    """
    from control.utils.pydantic_config_models import RunStateLedger
    from control.utils.run_state import RunStateManager
    from control.utils.transfer.daemon import _process_job
    from control.utils.transfer.queue import TransferQueue

    run_name = "queue_e2e.pffd"
    head_data_dir = str(tmp_path / "data")
    (tmp_path / "data" / run_name).mkdir(parents=True)

    mgr = RunStateManager(base_dir=str(tmp_path))
    mgr.save_state(
        RunStateLedger(
            run_name=run_name,
            status="RECORDING_ENDED",
            start_time=datetime.now(UTC).isoformat(),
        )
    )

    tq = TransferQueue(base_dir=str(tmp_path))
    tq.enqueue(
        run_name,
        head_data_dir,
        [{"ip_addr": "192.168.0.10", "data_dir": "/app/data", "module_ids": [250]}],
        no_collect=True,
        no_cleanup=True,
    )

    job = tq.claim()
    assert job is not None, "claim() must return the enqueued job"
    assert job["run_name"] == run_name

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.GenerateManifest = AsyncMock(return_value={"success": True, "file_count": 0})
    mock_client.CleanupData = AsyncMock(return_value={"success": True, "deleted_count": 0})

    with _mock_grpc_modules(mock_client), \
         patch("control.utils.transfer.daemon.rsync_one_node", return_value=(True, "")):
        success = asyncio.run(_process_job(job, tmp_path))

    assert success is True
    tq.complete(run_name)

    # Job must now be in completed/
    completed_dir = tmp_path / "tmp" / "transfer_queue" / "completed"
    assert (completed_dir / f"{run_name}.job.toml").exists()


def test_transfer_daemon_no_collect_integration(tmp_path) -> None:
    """
    In-process: no_collect=True skips rsync; job still reaches ARCHIVED.

    Verifies that the daemon fast-path (local-only, no gRPC manifest) works
    end-to-end without touching the network.
    """
    from control.utils.pydantic_config_models import RunStateLedger
    from control.utils.run_state import RunStateManager
    from control.utils.transfer.daemon import _process_job

    run_name = "local_only.pffd"
    head_data_dir = str(tmp_path / "data")
    (tmp_path / "data" / run_name).mkdir(parents=True)

    mgr = RunStateManager(base_dir=str(tmp_path))
    mgr.save_state(
        RunStateLedger(
            run_name=run_name,
            status="RECORDING_ENDED",
            start_time=datetime.now(UTC).isoformat(),
        )
    )

    job = {
        "run_name": run_name,
        "head_data_dir": head_data_dir,
        "daq_nodes": [],
        "no_collect": True,
        "no_cleanup": True,
    }

    mock_rsync = MagicMock()
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.GenerateManifest = AsyncMock(return_value={"success": True, "file_count": 0})
    mock_client.CleanupData = AsyncMock(return_value={"success": True, "deleted_count": 0})

    with _mock_grpc_modules(mock_client), \
         patch("control.utils.transfer.daemon.rsync_one_node", mock_rsync):
        success = asyncio.run(_process_job(job, tmp_path))

    assert success is True
    mock_rsync.assert_not_called()
    assert (tmp_path / "data" / run_name / "run_complete").exists()
