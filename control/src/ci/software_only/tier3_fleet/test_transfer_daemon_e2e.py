"""test_transfer_daemon_e2e.py

Integration tests for the decoupled transfer pipeline.

These tests require the full Docker stack to verify the interaction between:
1.  pseti stop (enqueue)
2.  TransferQueue (durable storage)
3.  transfer_daemon (state machine)
4.  daq_control_server (manifest/cleanup RPCs)
"""

import asyncio
import contextlib
import os
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Docker-based Integration Tests
# ---------------------------------------------------------------------------
from ci.fixtures.fleet import Fleet
from ci.fixtures.rsync_fixtures import RsyncMock
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
        with contextlib.suppress(OSError):
            os.chmod(main_run_dir, 0o777)

        # Touch metadata logs to satisfy verification. Name must include node name.
        (main_run_dir / f"hp_stdout_{spec.name}.log").touch()
        (main_run_dir / "meta.json").write_text('{"test": true}')

        # Module subdirs (e.g. /data/module_250/ci_run_xxx.pffd/)
        for mid in spec.module_ids:
            mod_root = host_root / f"module_{mid}"
            mod_root.mkdir(parents=True, exist_ok=True)
            with contextlib.suppress(OSError):
                os.chmod(mod_root, 0o777)

            mod_run_dir = mod_root / run_dir
            mod_run_dir.mkdir(parents=True, exist_ok=True)
            with contextlib.suppress(OSError):
                os.chmod(mod_run_dir, 0o777)

            # Dummy data - name must match what GenerateManifest picks up
            f_path = mod_run_dir / f"data.module_{mid}.pff"
            f_path.write_bytes(b"synthetic data")
            with contextlib.suppress(OSError):
                os.chmod(f_path, 0o666)


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
    daq_client: DaqControlClient,
    ensure_clean_daq_state: Any,
    session_fleet: Any,
    mock_rsync_transfer: RsyncMock,
    isolated_transfer_env: tuple[Path, config_file.DaqConfig],
    transfer_queue: TransferQueue,
) -> None:
    """
    E2E happy path: Start Run → Stop Run (enqueue) → daemon processes job → ARCHIVED.
    """
    fleet, _ = session_fleet
    head_data_dir, daq_config = isolated_transfer_env
    run_name = f"ci_daemon_{uuid.uuid4().hex[:8]}.pffd"
    
    # 1. Start and Stop run via generate_mocked_run helper
    # This automatically calls pseti start/stop logic and enqueues a TransferJob.
    expected_data = await generate_mocked_run(fleet, daq_config, run_name)

    mgr = RunStateManager()
    tq = transfer_queue
    job = tq.claim()
    assert job is not None
    assert job.run_name == run_name

    def rsync_side_effect(*args, **kwargs):
        simulate_rsync_from_fleet(fleet, run_name, Path(job.head_data_dir) / run_name)
        return None

    mock_rsync_transfer.side_effect = rsync_side_effect

    with patch("panoseti_grpc.daq_control.client.AsyncDaqControlClient", side_effect=get_mapped_client_factory(daq_config)):
        # Execute the state machine
        job_success = await asyncio.wait_for(_process_job(job, asyncio.Event(), mgr), timeout=30.0)
        assert job_success

    tq.complete(job.run_name)

    # 2. Verify Final State
    ledger = mgr.load_state()
    assert ledger and ledger.status == RunStatus.ARCHIVED
    verify_head_node_accuracy(head_data_dir, run_name, expected_data)


@pytest.mark.asyncio
async def test_transfer_daemon_resumes_after_crash(
    daq_control_direct: Any,
    ensure_clean_daq_state: Any,
    session_fleet: Any,
    mock_rsync_transfer: RsyncMock,
    isolated_transfer_env: tuple[Path, config_file.DaqConfig],
    transfer_queue: TransferQueue,
) -> None:
    """
    Chaos: daemon killed mid-rsync; restart completes the transfer.
    """
    import control.stop as stop
    from ci.software_only.conftest import wait_hashpipe_running
    from ci.software_only.tier4_chaos.conftest import _start as grpc_start

    head_data_dir, daq_config = isolated_transfer_env
    run_name = f"ci_daemon_{uuid.uuid4().hex[:8]}.pffd"
    
    RunStateManager().clear_state()

    # Use run_params from direct fixture but override run_dir
    run_params = {
        "data_dir": "/data",
        "run_dir": run_name,
        "module_id": [232, 236],
        "daq_ip_addr": session_fleet[0].node_ip(0),
        "bindhost": "lo"
    }

    fleet, _ = session_fleet
    _prepare_container_dirs(fleet, run_name)

    ok, _ = grpc_start(daq_control_direct, run_params)
    assert ok
    wait_hashpipe_running(daq_control_direct, "/data", timeout=5)
    
    net = config_file.get_network_config()
    uids = config_file.get_quabo_uids()
    await stop.stop_run(daq_config, net, uids, run=run_name, verbose=False)

    tq = transfer_queue
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
    mock_rsync_transfer.side_effect = RuntimeError("Simulated crash")
    
    with patch("panoseti_grpc.daq_control.client.AsyncDaqControlClient", side_effect=_get_mapped_client):
        success, err = await _process_job(job, asyncio.Event(), RunStateManager())
        assert not success
        assert "Simulated crash" in err
                
    # Orphaned in active/. Move back to pending/
    os.rename(tq._queue / "active" / f"{job.run_name}.job.toml", tq._queue / "pending" / f"{job.run_name}.job.toml")

    job2 = tq.claim()
    assert job2 is not None

    # Restart with WORKING rsync
    mock_rsync_transfer.side_effect = None

    with patch("panoseti_grpc.daq_control.client.AsyncDaqControlClient", side_effect=_get_mapped_client):
        success2 = await asyncio.wait_for(_process_job(job2, asyncio.Event(), RunStateManager()), timeout=30.0)
        assert success2
        tq.complete(job2.run_name)


    assert (tq._queue / "completed" / f"{run_name}.job.toml").exists()


@pytest.mark.asyncio
async def test_transfer_daemon_retry_on_transient_rsync_failure(
    mock_workspace,
    daq_control_direct: Any,
    run_params: dict[str, Any],
    ensure_clean_daq_state: Any,
    tmp_path: Path,
    session_fleet: Any,
    mock_rsync_transfer: RsyncMock,
) -> None:
    """
    Retry: rsync fails twice, succeeds on third attempt.
    """
    import control.stop as stop

    # mock_workspace already isolates PSETI_STATE and PSETI_CONFIG
    head_data_tmp = tmp_path / "head_data"
    head_data_tmp.mkdir(parents=True, exist_ok=True)
    
    run_params = dict(run_params)
    run_params["run_dir"] = f"ci_daemon_{uuid.uuid4().hex[:8]}.pffd"
    run_params["data_dir"] = "/data"
    
    RunStateManager().clear_state()

    (head_data_tmp / run_params["run_dir"]).mkdir(parents=True, exist_ok=True)
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
                grpc_port = node.port_forwarding.grpc_port if node.port_forwarding else 50051
                host = node.port_forwarding.gw_ip if node.port_forwarding else "localhost"
                return AsyncDaqControlClient(host=host, port=grpc_port)
        return AsyncDaqControlClient(host=host, port=port)

    # Attempt 1
    job1 = tq.claim()
    assert job1 is not None
    mock_rsync_transfer.side_effect = RuntimeError("rsync failed")

    with patch("panoseti_grpc.daq_control.client.AsyncDaqControlClient", side_effect=_get_mapped_client):
        success1 = (await _process_job(job1, asyncio.Event(), RunStateManager()))[0]
        assert not success1
        tq.fail(job1.run_name)

    tq.retry(job1.run_name)

    # Attempt 2
    job2 = tq.claim()
    assert job2 is not None
    # side_effect persists
    with patch("panoseti_grpc.daq_control.client.AsyncDaqControlClient", side_effect=_get_mapped_client):
        success2 = (await _process_job(job2, asyncio.Event(), RunStateManager()))[0]
        assert not success2
        tq.fail(job2.run_name)

    tq.retry(job2.run_name)

    # Attempt 3
    job3 = tq.claim()
    assert job3 is not None
    mock_rsync_transfer.side_effect = None # use default success mock
    
    with patch("panoseti_grpc.daq_control.client.AsyncDaqControlClient", side_effect=_get_mapped_client):
        success3 = await asyncio.wait_for(_process_job(job3, asyncio.Event(), RunStateManager()), timeout=30.0)
        assert success3
        tq.complete(job3.run_name)


    assert (tq._queue / "completed" / f"{run_params['run_dir']}.job.toml").exists()


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


async def test_transfer_daemon_singleton_lock_in_container(mock_workspace, tmp_path: Path) -> None:
    """
    Lock contention: second daemon must exit.
    """
    from control.utils.paths import PanoPaths

    # mock_workspace isolates PSETI_STATE
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


# ---------------------------------------------------------------------------
# In-process integration: no Docker required
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_transfer_daemon_unit_integration(
    mock_workspace,
    tmp_path: Path,
    mock_rsync_transfer: RsyncMock,
    transfer_job_factory: Callable[..., TransferJob],
    transfer_queue: TransferQueue,
) -> None:
    """
    Test _process_job with mocked gRPC and rsync.
    """
    run_name = "unit_int_run.pffd"
    job = transfer_job_factory(
        run_name=run_name,
        head_data_dir=tmp_path / "head",
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
        def rsync_side_effect(*args, **kwargs):
            (Path(job.head_data_dir) / run_name).mkdir(parents=True, exist_ok=True)
            (Path(job.head_data_dir) / run_name / "dp_manifest.node_test.algo_blake3.txt").touch()
            return None

        mock_rsync_transfer.side_effect = rsync_side_effect
        success = (await _process_job(job, asyncio.Event(), RunStateManager()))[0]
        assert success
    finally:
        if orig_mod:
            sys.modules["panoseti_grpc.daq_control.client"] = orig_mod
        else:
            del sys.modules["panoseti_grpc.daq_control.client"]
    
    assert (Path(job.head_data_dir) / run_name / "run_complete").exists()
