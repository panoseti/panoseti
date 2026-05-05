"""test_transfer_daemon_e2e.py

Integration tests for the decoupled transfer pipeline.

These tests require the full Docker stack to verify the interaction between:
1.  pseti stop (enqueue)
2.  TransferQueue (durable storage)
3.  transfer_daemon (state machine)
4.  daq_control_server (manifest/cleanup RPCs)
"""

import asyncio
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Docker-based Integration Tests
# ---------------------------------------------------------------------------
from ci.fixtures.fleet import Fleet
from control.transfer.daemon import _process_job
from control.transfer.models import TransferJob, TransferNodeSpec
from control.transfer.queue import TransferQueue
from control.utils import config_file
from control.utils.paths import PanoPaths
from control.utils.run_state import RunStateManager


def _prepare_container_dirs(fleet: Fleet, run_dir: str) -> None:
    """Create data directories in the ephemeral temp dirs used by containers."""
    for i, temp_dir in enumerate(fleet._temp_dirs):
        host_root = Path(temp_dir)
        spec = fleet.specs[i]

        # Root run dir (e.g. /data/ci_run_xxx.pffd/)
        main_run_dir = host_root / run_dir
        main_run_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(main_run_dir, 0o777)
        except OSError:
            pass

        # Touch metadata logs to satisfy verification. Name must include node name.
        (main_run_dir / f"hp_stdout_{spec.name}.log").touch()
        (main_run_dir / "meta.json").write_text('{"test": true}')

        # Module subdirs (e.g. /data/module_250/ci_run_xxx.pffd/)
        for mid in spec.module_ids:
            mod_root = host_root / f"module_{mid}"
            mod_root.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(mod_root, 0o777)
            except OSError:
                pass

            mod_run_dir = mod_root / run_dir
            mod_run_dir.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(mod_run_dir, 0o777)
            except OSError:
                pass

            # Dummy data - name must match what GenerateManifest picks up
            f_path = mod_run_dir / f"data.module_{mid}.pff"
            f_path.write_bytes(b"synthetic data")
            try:
                os.chmod(f_path, 0o666)
            except OSError:
                pass


def copy_run_dir_from_fleet(fleet: Fleet, run_dir: str, head_data_dir: Path) -> bool:
    """Mock rsync by copying from all isolated container volumes to head node.
    Simulates the inclusive rsync which pulls from both root and module directories.
    """
    dest_run = head_data_dir / run_dir
    dest_run.mkdir(parents=True, exist_ok=True)
    # Create a dummy manifest so the VERIFYING stage doesn't fail
    (dest_run / "dp_manifest.node_mock.algo_blake3.txt").write_text("")

    success = False

    import shutil
    for i, temp_dir in enumerate(fleet._temp_dirs):
        host_root = Path(temp_dir)
        spec = fleet.specs[i]
        # 1. Simulate root run dir transfer
        src_root = host_root / run_dir
        if src_root.is_dir():
            # Copy contents of root run dir into dest_run
            for item in src_root.iterdir():
                dest_path = dest_run / item.name
                if item.is_dir():
                    if dest_path.exists():
                        shutil.rmtree(dest_path)
                    shutil.copytree(item, dest_path)
                else:
                    shutil.copy2(item, dest_path)
            success = True

        # 2. Simulate module run dir transfer (flattened)
        for mid in spec.module_ids:
            src_mod = host_root / f"module_{mid}" / run_dir
            if src_mod.is_dir():
                for item in src_mod.iterdir():
                    if item.is_file():
                        shutil.copy2(item, dest_run / item.name)
                success = True
    return success

@pytest.mark.asyncio
async def test_transfer_daemon_archives_run(
    daq_control_direct: Any,
    run_params: dict[str, Any],
    ensure_clean_daq_state: Any,
    tmp_path: Path,
    session_fleet: Any,
) -> None:
    """
    E2E happy path: enqueue job → daemon processes all 5 stages → job lands
    in completed/ → ledger ARCHIVED.
    """
    import control.stop as stop
    from ci.software_only.conftest import wait_hashpipe_running
    from ci.software_only.tier4_chaos.conftest import _start as grpc_start

    # Isolate state and data
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("PSETI_STATE", str(tmp_path))
    head_data_tmp = tmp_path / "head_data"
    head_data_tmp.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HEAD_DATA_DIR", str(head_data_tmp))
    PanoPaths.ensure_state_dirs()
    
    run_params = dict(run_params)
    run_params["run_dir"] = f"ci_daemon_{uuid.uuid4().hex[:8]}.pffd"
    run_params["data_dir"] = "/data" # Container perspective
    
    RunStateManager().clear_state()

    from control.utils.pydantic_config_models import RunStateLedger, RunStatus
    mgr = RunStateManager()
    ledger = RunStateLedger(
        run_name=run_params["run_dir"],
        status=RunStatus.ACTIVE,
        start_time=datetime.now(UTC).isoformat(),
    )
    mgr.save_state(ledger)

    # Prepare host directories FIRST
    fleet, _ = session_fleet
    _prepare_container_dirs(fleet, run_params["run_dir"])

    ok, _ = grpc_start(daq_control_direct, run_params)
    assert ok
    wait_hashpipe_running(daq_control_direct, "/data", timeout=5)
    daq_config = config_file.get_daq_config()
    daq_config.head_node_data_dir = str(head_data_tmp)

    net = config_file.get_network_config()
    uids = config_file.get_quabo_uids()

    # Ensure head dir exists for stop_run
    os.makedirs(f"{daq_config.head_node_data_dir}/{run_params['run_dir']}", exist_ok=True)

    # 2. Stop real run (enqueues job)
    success = await stop.stop_run(
        daq_config, net, uids, run=run_params["run_dir"], verbose=False
    )
    assert success

    ledger = mgr.load_state()
    assert ledger is not None
    assert ledger.status == RunStatus.RECORDING_ENDED

    # 3. Process job
    tq = TransferQueue()
    job = tq.claim()
    assert job is not None

    async def mocked_rsync(*args, **kwargs):
        ok = copy_run_dir_from_fleet(fleet, run_params["run_dir"], Path(job.head_data_dir))
        proc = MagicMock()
        proc.returncode = 0 if ok else 1
        proc.wait = AsyncMock(return_value=proc.returncode)
        proc.communicate = AsyncMock(return_value=(b'', b'Simulated copy failed' if not ok else b''))
        proc.stdout.readline = AsyncMock(return_value=b'')
        proc.stderr.read = AsyncMock(return_value=b'Simulated copy failed' if not ok else b'')
        return proc
    # Use fully qualified path for patching
    from panoseti_grpc.daq_control.client import AsyncDaqControlClient
    
    def _get_mapped_client(host, port=50051):
        for node in daq_config.daq_nodes:
            if str(node.ip_addr) == host:
                return AsyncDaqControlClient(host=node.port_forwarding.gw_ip, port=node.port_forwarding.grpc_port)
        return AsyncDaqControlClient(host=host, port=port)

    with patch("control.transfer.daemon.asyncio.create_subprocess_exec", side_effect=mocked_rsync), \
         patch("panoseti_grpc.daq_control.client.AsyncDaqControlClient", side_effect=_get_mapped_client):
        # Allow up to 30s for the entire multi-node manifest/transfer/cleanup state machine
        job_success = await asyncio.wait_for(_process_job(job, asyncio.Event(), RunStateManager()), timeout=30.0)
        assert job_success

    tq.complete(job.run_name)

    ledger = mgr.load_state()
    assert ledger and ledger.status == RunStatus.ARCHIVED
    run_dir_path = Path(daq_config.head_node_data_dir) / run_params["run_dir"]
    assert (run_dir_path / "run_complete").exists()

    # Strengthened checks: Ensure manifests and logs were transferred
    manifests = list(run_dir_path.glob("dp_manifest.node_*.txt"))
    assert len(manifests) > 0, "No manifest files found on head node after transfer"

    # Check for unique log files
    logs = list(run_dir_path.glob("hp_stdout_*.log"))
    assert len(logs) > 0, "No node-specific log files found on head node after transfer"

    monkeypatch.undo()


@pytest.mark.asyncio
async def test_transfer_daemon_resumes_after_crash(
    daq_control_direct: Any,
    run_params: dict[str, Any],
    ensure_clean_daq_state: Any,
    tmp_path: Path,
    session_fleet: Any,
) -> None:
    """
    Chaos: daemon killed mid-rsync; restart completes the transfer.
    """
    import control.stop as stop
    from ci.software_only.conftest import wait_hashpipe_running
    from ci.software_only.tier4_chaos.conftest import _start as grpc_start

    # Isolate state and data
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("PSETI_STATE", str(tmp_path))
    head_data_tmp = tmp_path / "head_data"
    head_data_tmp.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HEAD_DATA_DIR", str(head_data_tmp))
    PanoPaths.ensure_state_dirs()
    
    run_params = dict(run_params)
    run_params["run_dir"] = f"ci_daemon_{uuid.uuid4().hex[:8]}.pffd"
    run_params["data_dir"] = "/data"
    
    RunStateManager().clear_state()

    os.makedirs(f"{head_data_tmp}/{run_params['run_dir']}", exist_ok=True)
    fleet, _ = session_fleet
    _prepare_container_dirs(fleet, run_params["run_dir"])

    ok, _ = grpc_start(daq_control_direct, run_params)
    assert ok
    wait_hashpipe_running(daq_control_direct, "/data", timeout=5)
    daq_config = config_file.get_daq_config()
    daq_config.head_node_data_dir = str(head_data_tmp)

    net = config_file.get_network_config()
    uids = config_file.get_quabo_uids()
    await stop.stop_run(daq_config, net, uids, run=run_params["run_dir"], verbose=False)

    tq = TransferQueue()
    job = tq.claim()
    assert job is not None

    from panoseti_grpc.daq_control.client import AsyncDaqControlClient
    def _get_mapped_client(host, port=50051):
        for node in daq_config.daq_nodes:
            if str(node.ip_addr) == host:
                return AsyncDaqControlClient(host=node.port_forwarding.gw_ip, port=node.port_forwarding.grpc_port)
        return AsyncDaqControlClient(host=host, port=port)

    # Simulate crash mid-rsync (Stage 2)
    # We must ensure Stage 1 (Manifest) passes first!
    with patch("control.transfer.daemon.asyncio.create_subprocess_exec", side_effect=RuntimeError("Simulated crash")), \
         patch("panoseti_grpc.daq_control.client.AsyncDaqControlClient", side_effect=_get_mapped_client):
        success, err = await _process_job(job, asyncio.Event(), RunStateManager())
        assert not success
        assert "Simulated crash" in err
                
    # Orphaned in active/. Move back to pending/
    os.rename(tq._queue / "active" / f"{job.run_name}.job.toml", tq._queue / "pending" / f"{job.run_name}.job.toml")

    job2 = tq.claim()
    assert job2 is not None

    with patch("control.transfer.daemon.asyncio.create_subprocess_exec", side_effect=_mock_subprocess_ok), \
         patch("panoseti_grpc.daq_control.client.AsyncDaqControlClient", side_effect=_get_mapped_client):
        success2 = await asyncio.wait_for(_process_job(job2, asyncio.Event(), RunStateManager()), timeout=30.0)
        assert success2
        tq.complete(job2.run_name)


    assert (tq._queue / "completed" / f"{run_params['run_dir']}.job.toml").exists()
    monkeypatch.undo()


@pytest.mark.asyncio
async def test_transfer_daemon_retry_on_transient_rsync_failure(
    daq_control_direct: Any,
    run_params: dict[str, Any],
    ensure_clean_daq_state: Any,
    tmp_path: Path,
    session_fleet: Any,
) -> None:
    """
    Retry: rsync fails twice, succeeds on third attempt.
    """
    import control.stop as stop

    # Isolate state and data
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("PSETI_STATE", str(tmp_path))
    head_data_tmp = tmp_path / "head_data"
    head_data_tmp.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HEAD_DATA_DIR", str(head_data_tmp))
    PanoPaths.ensure_state_dirs()
    
    run_params = dict(run_params)
    run_params["run_dir"] = f"ci_daemon_{uuid.uuid4().hex[:8]}.pffd"
    run_params["data_dir"] = "/data"
    
    RunStateManager().clear_state()

    os.makedirs(f"{head_data_tmp}/{run_params['run_dir']}", exist_ok=True)
    fleet, _ = session_fleet
    _prepare_container_dirs(fleet, run_params["run_dir"])

    daq_config = config_file.get_daq_config()
    daq_config.head_node_data_dir = str(head_data_tmp)
    net = config_file.get_network_config()
    uids = config_file.get_quabo_uids()
    await stop.stop_run(daq_config, net, uids, run=run_params["run_dir"], verbose=False)

    tq = TransferQueue()

    from panoseti_grpc.daq_control.client import AsyncDaqControlClient
    def _get_mapped_client(host, port=50051):
        for node in daq_config.daq_nodes:
            if str(node.ip_addr) == host:
                return AsyncDaqControlClient(host=node.port_forwarding.gw_ip, port=node.port_forwarding.grpc_port)
        return AsyncDaqControlClient(host=host, port=port)

    # Attempt 1
    job1 = tq.claim()
    assert job1 is not None
    with patch("control.transfer.daemon.asyncio.create_subprocess_exec", side_effect=_mock_subprocess_fail), \
         patch("panoseti_grpc.daq_control.client.AsyncDaqControlClient", side_effect=_get_mapped_client):
        success1 = (await _process_job(job1, asyncio.Event(), RunStateManager()))[0]
        assert not success1
        tq.fail(job1.run_name)

    tq.retry(job1.run_name)

    # Attempt 2
    job2 = tq.claim()
    assert job2 is not None
    with patch("control.transfer.daemon.asyncio.create_subprocess_exec", side_effect=_mock_subprocess_fail), \
         patch("panoseti_grpc.daq_control.client.AsyncDaqControlClient", side_effect=_get_mapped_client):
        success2 = (await _process_job(job2, asyncio.Event(), RunStateManager()))[0]
        assert not success2
        tq.fail(job2.run_name)

    tq.retry(job2.run_name)

    # Attempt 3
    job3 = tq.claim()
    assert job3 is not None
    with patch("control.transfer.daemon.asyncio.create_subprocess_exec", side_effect=_mock_subprocess_ok), \
         patch("panoseti_grpc.daq_control.client.AsyncDaqControlClient", side_effect=_get_mapped_client):
        success3 = await asyncio.wait_for(_process_job(job3, asyncio.Event(), RunStateManager()), timeout=30.0)
        assert success3
        tq.complete(job3.run_name)


    assert (tq._queue / "completed" / f"{run_params['run_dir']}.job.toml").exists()
    monkeypatch.undo()


@pytest.mark.asyncio
async def test_transfer_daemon_marks_failed_after_max_attempts(
    daq_control_direct: Any,
    run_params: dict[str, Any],
    ensure_clean_daq_state: Any,
    tmp_path: Path,
    session_fleet: Any,
) -> None:
    """
    Exhaustion: rsync fails up to MAX_ATTEMPTS.
    """
    import control.stop as stop
    from control.transfer.daemon import MAX_ATTEMPTS

    # Isolate state and data
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("PSETI_STATE", str(tmp_path))
    head_data_tmp = tmp_path / "head_data"
    head_data_tmp.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HEAD_DATA_DIR", str(head_data_tmp))
    PanoPaths.ensure_state_dirs()
    
    run_params = dict(run_params)
    run_params["run_dir"] = f"ci_daemon_{uuid.uuid4().hex[:8]}.pffd"
    run_params["data_dir"] = "/data"
    
    RunStateManager().clear_state()

    os.makedirs(f"{head_data_tmp}/{run_params['run_dir']}", exist_ok=True)
    fleet, _ = session_fleet
    _prepare_container_dirs(fleet, run_params["run_dir"])

    daq_config = config_file.get_daq_config()
    daq_config.head_node_data_dir = str(head_data_tmp)
    net = config_file.get_network_config()
    uids = config_file.get_quabo_uids()
    await stop.stop_run(daq_config, net, uids, run=run_params["run_dir"], verbose=False)

    tq = TransferQueue()
    job = tq.claim()
    assert job is not None

    from panoseti_grpc.daq_control.client import AsyncDaqControlClient
    def _get_mapped_client(host, port=50051):
        for node in daq_config.daq_nodes:
            if str(node.ip_addr) == host:
                return AsyncDaqControlClient(host=node.port_forwarding.gw_ip, port=node.port_forwarding.grpc_port)
        return AsyncDaqControlClient(host=host, port=port)

    with patch("control.transfer.daemon.asyncio.create_subprocess_exec", side_effect=_mock_subprocess_fail), \
         patch("panoseti_grpc.daq_control.client.AsyncDaqControlClient", side_effect=_get_mapped_client):
        for attempt in range(MAX_ATTEMPTS):
            success = (await _process_job(job, asyncio.Event(), RunStateManager()))[0]
            assert not success
            tq.fail(job.run_name)
            if attempt < MAX_ATTEMPTS - 1:
                tq.retry(job.run_name)
                job = tq.claim()
                assert job is not None

    assert (tq._queue / "failed" / f"{run_params['run_dir']}.job.toml").exists()
    monkeypatch.undo()


async def test_transfer_daemon_singleton_lock_in_container(tmp_path: Path) -> None:
    """
    Lock contention: second daemon must exit.
    """
    from control.utils.paths import PanoPaths

    # Isolate state and data
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("PSETI_STATE", str(tmp_path))
    head_data_tmp = tmp_path / "head_data"
    head_data_tmp.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HEAD_DATA_DIR", str(head_data_tmp))
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

    p1 = await asyncio.create_subprocess_exec(
        "python3", str(lock_script), str(lock_path),
        stdout=asyncio.subprocess.PIPE, env=env
    )
    
    assert p1.stdout is not None
    line = (await p1.stdout.readline()).decode()
    assert "LOCKED" in line

    p2 = await asyncio.create_subprocess_exec(
        "python3", "-m", "control.transfer",
        cwd=str(PanoPaths.base_dir()),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env
    )
    
    try:
        stdout_bytes, _ = await asyncio.wait_for(p2.communicate(), timeout=5)
        stdout = stdout_bytes.decode()
        assert p2.returncode == 0
        assert all(x in stdout for x in ["Another", "transfer", "daemon", "already", "running"])
    finally:
        p1.terminate()
        await p1.wait()
    monkeypatch.undo()


# ---------------------------------------------------------------------------
# In-process integration: no Docker required
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_transfer_daemon_unit_integration(tmp_path: Path) -> None:
    """
    Test _process_job with mocked gRPC and rsync.
    """
    # Isolate state and data
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("PSETI_STATE", str(tmp_path))
    head_data_tmp = tmp_path / "head_data"
    head_data_tmp.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HEAD_DATA_DIR", str(head_data_tmp))
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
        async def custom_mock_rsync(*args, **kwargs):
            (Path(job.head_data_dir) / run_name).mkdir(parents=True, exist_ok=True)
            (Path(job.head_data_dir) / run_name / "dp_manifest.node_test.algo_blake3.txt").touch()
            return await _mock_subprocess_ok()

        with patch("control.transfer.daemon.asyncio.create_subprocess_exec", side_effect=custom_mock_rsync):
            success = (await _process_job(job, asyncio.Event(), RunStateManager()))[0]
            assert success
    finally:
        if orig_mod:
            sys.modules["panoseti_grpc.daq_control.client"] = orig_mod
        else:
            del sys.modules["panoseti_grpc.daq_control.client"]
    
    assert (Path(job.head_data_dir) / run_name / "run_complete").exists()
    monkeypatch.undo()

async def _mock_subprocess_ok(*args, **kwargs):
    proc = MagicMock()
    proc.returncode = 0
    proc.wait = AsyncMock(return_value=0)
    proc.communicate = AsyncMock(return_value=(b"", b""))
    proc.stdout.readline = AsyncMock(return_value=b"")
    proc.stderr.read = AsyncMock(return_value=b"")
    return proc

async def _mock_subprocess_fail(*args, **kwargs):
    proc = MagicMock()
    proc.returncode = 1
    proc.wait = AsyncMock(return_value=1)
    proc.communicate = AsyncMock(return_value=(b"", b"error"))
    proc.stdout.readline = AsyncMock(return_value=b"")
    proc.stderr.read = AsyncMock(return_value=b"error")
    return proc
