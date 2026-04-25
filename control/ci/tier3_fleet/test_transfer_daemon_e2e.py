"""test_transfer_daemon_e2e.py

Integration tests for the decoupled transfer pipeline.

These tests require the full Docker stack to verify the interaction between:
1.  pseti stop (enqueue)
2.  TransferQueue (durable storage)
3.  transfer_daemon (state machine)
4.  daq_control_server (manifest/cleanup RPCs)
"""

import os
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from control.transfer.daemon import _process_job
from control.transfer.models import TransferJob, TransferNodeSpec
from control.transfer.queue import TransferQueue
from control.utils import config_file
from control.utils.paths import PanoPaths
from control.utils.run_state import RunStateManager


# Helpers for skipping tests when not in Docker CI
def is_in_ci() -> bool:
    return os.path.exists("/.dockerenv")

skip_outside_ci = pytest.mark.skipif(
    not is_in_ci(), reason="This test requires the Docker CI environment"
)


def copy_run_dir(run_params: dict, head_data_dir: Path) -> bool:
    """Mock rsync by manually copying files from /data to /data/head."""
    run_dir = run_params["run_dir"]
    daq_data = Path(run_params["data_dir"])
    
    source_root = daq_data
    dest_run = head_data_dir / run_dir
    dest_run.mkdir(parents=True, exist_ok=True)
    
    success = False
    for mid in run_params["module_id"]:
        src = source_root / f"module_{mid}" / run_dir
        if src.exists():
            import shutil
            dest_mod = dest_run / f"module_{mid}"
            if dest_mod.exists():
                shutil.rmtree(dest_mod)
            shutil.copytree(src, dest_mod)
            success = True
    return success


# ---------------------------------------------------------------------------
# Docker-based Integration Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@skip_outside_ci
async def test_transfer_daemon_archives_run(
    daq_control_direct: Any,
    run_params: dict[str, Any],
    ensure_clean_daq_state: Any,
    tmp_path: Path,
) -> None:
    """
    E2E happy path: enqueue job → daemon processes all 5 stages → job lands
    in completed/ → ledger ARCHIVED.
    """
    import control.stop as stop
    from ci.tier3_fleet.conftest import wait_hashpipe_running
    from ci.tier4_chaos.conftest import _start as grpc_start

    # Isolate state
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("PSETI_STATE", str(tmp_path))
    PanoPaths.ensure_state_dirs()

    run_params = dict(run_params)
    run_params["run_dir"] = f"ci_daemon_{uuid.uuid4().hex[:8]}.pffd"
    RunStateManager().clear_state()

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

    # Ensure head dir exists for stop_run
    os.makedirs(f"{daq_config.head_node_data_dir}/{run_params['run_dir']}", exist_ok=True)

    # Ensure daqnode dirs exist for manifest generation (shared volume)
    for node in daq_config.daq_nodes:
        for mid in node.module_ids:
            module_dir = Path(run_params["data_dir"]) / f"module_{mid}" / run_params["run_dir"]
            module_dir.mkdir(parents=True, exist_ok=True)
            (module_dir / "data.pff").write_bytes(b"synthetic data")
    
    # Also create the root run dir because the GenerateManifestModel validator expects it!
    root_run_dir = Path(run_params["data_dir"]) / run_params["run_dir"]
    root_run_dir.mkdir(parents=True, exist_ok=True)

    # 2. Stop real run (enqueues job)
    success = await stop.stop_run(
        daq_config, net, uids, run=run_params["run_dir"], verbose=False
    )
    assert success

    ledger = mgr.load_state()
    assert ledger is not None
    assert ledger.status == "RECORDING_ENDED"

    # 3. Process job
    tq = TransferQueue()
    job = tq.claim()
    assert job is not None

    def mocked_rsync(*args, **kwargs):
        ok = copy_run_dir(run_params, Path(job.head_data_dir))
        return MagicMock(returncode=0 if ok else 1, stderr="Simulated copy failed")

    with patch("control.transfer.daemon.subprocess.run", side_effect=mocked_rsync):
        job_success = await _process_job(job)
        assert job_success

    tq.complete(job.run_name)

    ledger = mgr.load_state()
    assert ledger and ledger.status == "ARCHIVED"
    run_dir_path = Path(daq_config.head_node_data_dir) / run_params["run_dir"]
    assert (run_dir_path / "run_complete").exists()
    monkeypatch.undo()


@pytest.mark.asyncio
@skip_outside_ci
async def test_transfer_daemon_resumes_after_crash(
    daq_control_direct: Any,
    run_params: dict[str, Any],
    ensure_clean_daq_state: Any,
    tmp_path: Path,
) -> None:
    """
    Chaos: daemon killed mid-rsync; restart completes the transfer.
    """
    import control.stop as stop
    from ci.tier3_fleet.conftest import wait_hashpipe_running
    from ci.tier4_chaos.conftest import _start as grpc_start

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("PSETI_STATE", str(tmp_path))
    PanoPaths.ensure_state_dirs()

    run_params = dict(run_params)
    run_params["run_dir"] = f"ci_daemon_{uuid.uuid4().hex[:8]}.pffd"
    RunStateManager().clear_state()

    RunStateManager()
    
    ok, _ = grpc_start(daq_control_direct, run_params)
    assert ok
    wait_hashpipe_running(daq_control_direct, "/data", timeout=5)

    daq_config = config_file.get_daq_config()
    net = config_file.get_network_config()
    uids = config_file.get_quabo_uids()

    os.makedirs(f"{daq_config.head_node_data_dir}/{run_params['run_dir']}", exist_ok=True)
    for node in daq_config.daq_nodes:
        for mid in node.module_ids:
            module_dir = Path(run_params["data_dir"]) / f"module_{mid}" / run_params["run_dir"]
            module_dir.mkdir(parents=True, exist_ok=True)
            (module_dir / "data.pff").write_bytes(b"synthetic data")
    
    # Also create the root run dir because the GenerateManifestModel validator expects it!
    root_run_dir = Path(run_params["data_dir"]) / run_params["run_dir"]
    root_run_dir.mkdir(parents=True, exist_ok=True)

    await stop.stop_run(daq_config, net, uids, run=run_params["run_dir"], verbose=False)

    tq = TransferQueue()
    job = tq.claim()
    assert job is not None

    # Simulate crash mid-rsync (Stage 2)
    # We must ensure Stage 1 (Manifest) passes first!
    with patch("control.transfer.daemon.subprocess.run", side_effect=RuntimeError("Simulated crash")), \
         pytest.raises(RuntimeError, match="Simulated crash"):
        await _process_job(job)
    # Orphaned in active/. Move back to pending/
    os.rename(tq._queue / "active" / f"{job.run_name}.job.toml", tq._queue / "pending" / f"{job.run_name}.job.toml")

    job2 = tq.claim()
    assert job2 is not None

    with patch("control.transfer.daemon.subprocess.run", return_value=MagicMock(returncode=0)):
        success2 = await _process_job(job2)
        assert success2
        tq.complete(job2.run_name)

    assert (tq._queue / "completed" / f"{run_params['run_dir']}.job.toml").exists()
    monkeypatch.undo()


@pytest.mark.asyncio
@skip_outside_ci
async def test_transfer_daemon_retry_on_transient_rsync_failure(
    daq_control_direct: Any,
    run_params: dict[str, Any],
    ensure_clean_daq_state: Any,
    tmp_path: Path,
) -> None:
    """
    Retry: rsync fails twice, succeeds on third attempt.
    """
    import control.stop as stop

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("PSETI_STATE", str(tmp_path))
    PanoPaths.ensure_state_dirs()

    run_params = dict(run_params)
    run_params["run_dir"] = f"ci_daemon_{uuid.uuid4().hex[:8]}.pffd"
    RunStateManager().clear_state()

    daq_config = config_file.get_daq_config()
    net = config_file.get_network_config()
    uids = config_file.get_quabo_uids()

    os.makedirs(f"{daq_config.head_node_data_dir}/{run_params['run_dir']}", exist_ok=True)
    for node in daq_config.daq_nodes:
        for mid in node.module_ids:
            module_dir = Path(run_params["data_dir"]) / f"module_{mid}" / run_params["run_dir"]
            module_dir.mkdir(parents=True, exist_ok=True)
            (module_dir / "data.pff").write_bytes(b"synthetic data")

    # Also create the root run dir because the GenerateManifestModel validator expects it!
    root_run_dir = Path(run_params["data_dir"]) / run_params["run_dir"]
    root_run_dir.mkdir(parents=True, exist_ok=True)

    await stop.stop_run(daq_config, net, uids, run=run_params["run_dir"], verbose=False)

    tq = TransferQueue()

    # Attempt 1
    job1 = tq.claim()
    assert job1 is not None
    with patch("control.transfer.daemon.subprocess.run", return_value=MagicMock(returncode=1, stderr="Transient error")):
        success1 = await _process_job(job1)
        assert not success1
        tq.fail(job1.run_name)

    tq.retry(job1.run_name)

    # Attempt 2
    job2 = tq.claim()
    assert job2 is not None
    with patch("control.transfer.daemon.subprocess.run", return_value=MagicMock(returncode=1, stderr="Transient error")):
        success2 = await _process_job(job2)
        assert not success2
        tq.fail(job2.run_name)

    tq.retry(job2.run_name)

    # Attempt 3
    job3 = tq.claim()
    assert job3 is not None
    with patch("control.transfer.daemon.subprocess.run", return_value=MagicMock(returncode=0)):
        success3 = await _process_job(job3)
        assert success3
        tq.complete(job3.run_name)

    assert (tq._queue / "completed" / f"{run_params['run_dir']}.job.toml").exists()
    monkeypatch.undo()


@pytest.mark.asyncio
@skip_outside_ci
async def test_transfer_daemon_marks_failed_after_max_attempts(
    daq_control_direct: Any,
    run_params: dict[str, Any],
    ensure_clean_daq_state: Any,
    tmp_path: Path,
) -> None:
    """
    Exhaustion: rsync fails up to MAX_ATTEMPTS.
    """
    import control.stop as stop
    from control.transfer.daemon import MAX_ATTEMPTS

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("PSETI_STATE", str(tmp_path))
    PanoPaths.ensure_state_dirs()

    run_params = dict(run_params)
    run_params["run_dir"] = f"ci_daemon_{uuid.uuid4().hex[:8]}.pffd"
    RunStateManager().clear_state()

    daq_config = config_file.get_daq_config()
    net = config_file.get_network_config()
    uids = config_file.get_quabo_uids()

    os.makedirs(f"{daq_config.head_node_data_dir}/{run_params['run_dir']}", exist_ok=True)
    for node in daq_config.daq_nodes:
        for mid in node.module_ids:
            module_dir = Path(run_params["data_dir"]) / f"module_{mid}" / run_params["run_dir"]
            module_dir.mkdir(parents=True, exist_ok=True)
            (module_dir / "data.pff").write_bytes(b"synthetic data")

    # Also create the root run dir because the GenerateManifestModel validator expects it!
    root_run_dir = Path(run_params["data_dir"]) / run_params["run_dir"]
    root_run_dir.mkdir(parents=True, exist_ok=True)

    await stop.stop_run(daq_config, net, uids, run=run_params["run_dir"], verbose=False)

    tq = TransferQueue()
    job = tq.claim()
    assert job is not None

    with patch("control.transfer.daemon.subprocess.run", return_value=MagicMock(returncode=1, stderr="Persistent failure")):
        for attempt in range(MAX_ATTEMPTS):
            success = await _process_job(job)
            assert not success
            tq.fail(job.run_name)
            if attempt < MAX_ATTEMPTS - 1:
                tq.retry(job.run_name)
                job = tq.claim()
                assert job is not None

    assert (tq._queue / "failed" / f"{run_params['run_dir']}.job.toml").exists()
    monkeypatch.undo()


@skip_outside_ci
def test_transfer_daemon_singleton_lock_in_container(tmp_path: Path) -> None:
    """
    Lock contention: second daemon must exit.
    """
    from control.utils.paths import PanoPaths

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("PSETI_STATE", str(tmp_path))
    PanoPaths.ensure_state_dirs()

    lock_script = tmp_path / "hold_lock.py"
    lock_script.write_text('''
import os
import time
import sys
lock_file = sys.argv[1]
try:
    fd = os.open(lock_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(fd, "w") as f:
        f.write(str(os.getpid()))
    print("LOCKED", flush=True)
    time.sleep(10)
except FileExistsError:
    print("FAILED TO LOCK")
''')

    lock_path = PanoPaths.locks_dir() / "transfer.lock"
    env = os.environ.copy()
    env["PSETI_STATE"] = str(tmp_path)
    env["PYTHONPATH"] = f"{PanoPaths.base_dir()}/src:{env.get('PYTHONPATH', '')}"

    p1 = subprocess.Popen(["python3", str(lock_script), str(lock_path)], stdout=subprocess.PIPE, text=True, env=env)
    
    assert p1.stdout is not None
    line = p1.stdout.readline()
    assert "LOCKED" in line

    p2 = subprocess.Popen(["python3", "-m", "control.transfer"], cwd=str(PanoPaths.base_dir()), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    
    try:
        stdout, _ = p2.communicate(timeout=5)
        assert p2.returncode == 0
        assert "Another transfer daemon is already running" in stdout
    finally:
        p1.terminate()
        p1.wait()
    monkeypatch.undo()


# ---------------------------------------------------------------------------
# In-process integration: no Docker required
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_transfer_daemon_unit_integration(tmp_path: Path) -> None:
    """
    Test _process_job with mocked gRPC and rsync.
    """
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("PSETI_STATE", str(tmp_path))
    PanoPaths.ensure_state_dirs()

    run_name = "unit_int_run.pffd"
    job = TransferJob(
        schema_version=1,
        run_name=run_name,
        head_data_dir=str(tmp_path / "head"),
        head_node_username="panoseti",
        created_at=datetime.now(UTC),
        daq_nodes=[
            TransferNodeSpec(
                ip_addr="127.0.0.1",
                username="root",
                data_dir="/tmp/daq",
                module_ids=[100]
            )
        ]
    )

    # Mock gRPC client module
    import sys
    from types import ModuleType
    
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.GenerateManifest = AsyncMock(return_value={"success": True})
    mock_client.CleanupData = AsyncMock(return_value={"success": True})

    stub_mod = ModuleType("panoseti_grpc.daq_control.client")
    stub_mod.AsyncDaqControlClient = MagicMock(return_value=mock_client) # type: ignore
    
    orig_mod = sys.modules.get("panoseti_grpc.daq_control.client")
    sys.modules["panoseti_grpc.daq_control.client"] = stub_mod

    try:
        with patch("control.transfer.daemon.subprocess.run", return_value=MagicMock(returncode=0)):
            success = await _process_job(job)
            assert success
    finally:
        if orig_mod:
            sys.modules["panoseti_grpc.daq_control.client"] = orig_mod
        else:
            del sys.modules["panoseti_grpc.daq_control.client"]
    
    assert (Path(job.head_data_dir) / run_name / "run_complete").exists()
    monkeypatch.undo()
