"""
test_transfer_advanced.py — Advanced scenarios for the transfer queue.
Covers chaos recovery, scale, queue depth, and lifecycle resilience.
"""

import asyncio
import contextlib
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ci.tier3_fleet.transfer_testing_utils import (
    generate_mocked_run,
    get_mapped_client_factory,
    setup_isolated_transfer_env,
    simulate_rsync_from_fleet,
    verify_head_node_accuracy,
)
from control.transfer.daemon import _process_job, run_daemon
from control.transfer.queue import TransferQueue
from control.utils.paths import PanoPaths
from control.utils.run_state import RunStateManager

# ---------------------------------------------------------------------------
# 1. Chaos: Partial Transfer Recovery
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_transfer_queue_chaos_partial_transfer_recovery(
    session_fleet: Any,
    tmp_path: Path,
    ensure_clean_daq_state: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the queue recovers from an rsync error and succeeds on retry."""
    fleet, daq_cfg_dict = session_fleet
    head_data_dir, daq_config = setup_isolated_transfer_env(tmp_path, monkeypatch, daq_cfg_dict)
    run_name = f"chaos_recovery_{uuid.uuid4().hex[:8]}.pffd"
    
    expected_data = await generate_mocked_run(fleet, daq_config, run_name)
    
    mgr = RunStateManager()
    tq = TransferQueue()
    job = tq.claim()
    
    call_count = 0
    async def failing_rsync(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Simulate partial transfer failure
            raise RuntimeError("Network timeout mid-rsync")
        
        # Second attempt succeeds
        simulate_rsync_from_fleet(fleet, run_name, head_data_dir / run_name)
        proc = MagicMock()

        proc.returncode = 0
        proc.wait = AsyncMock(return_value=0)
        proc.communicate = AsyncMock(return_value=(b'', b''))
        proc.stdout.readline = AsyncMock(return_value=b'')
        proc.stderr.read = AsyncMock(return_value=b'')
        return proc


    from datetime import UTC, datetime
    job.head_node_username = "panoseti"
    job.created_at = datetime.now(UTC)

    with patch("control.transfer.daemon.asyncio.create_subprocess_exec", side_effect=failing_rsync), \
         patch("panoseti_grpc.daq_control.client.AsyncDaqControlClient", side_effect=get_mapped_client_factory(daq_config)):

        # 1st attempt: fails fast
        success1, err1 = await _process_job(job, asyncio.Event(), mgr)
        assert not success1
        assert "Network timeout" in err1

        # Simulate the daemon's retry cycle: fail -> retry (reset) -> claim
        tq.fail(job.run_name)
        tq.retry(job.run_name)
        job2 = tq.claim()
        job2.head_node_username = "panoseti"
        job2.created_at = datetime.now(UTC)

        # 2nd attempt: succeeds
        success2, err2 = await _process_job(job2, asyncio.Event(), mgr)
        assert success2, f"Job failed even after retry: {err2}"
        tq.complete(job.run_name)

    verify_head_node_accuracy(head_data_dir, run_name, expected_data)
    # Call count: 1 (fail) + 2 (success across 2 nodes) = 3
    assert call_count == len(daq_config.daq_nodes) + 1
# ---------------------------------------------------------------------------
# 2. Scale: Parameterized Mock Fleet (2-8 nodes)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_nodes", [2, 4, 8])
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_transfer_queue_parameterized_scale(
    num_nodes: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the transfer queue handles scaling to 8 concurrent nodes over gRPC."""
    # We use a mock configuration with N nodes and mock clients to test concurrency logic
    # without the overhead of 8 actual Docker containers.
    monkeypatch.setenv("PSETI_STATE", str(tmp_path))
    head_data_dir = tmp_path / "head_data"
    head_data_dir.mkdir(parents=True)
    monkeypatch.setenv("HEAD_DATA_DIR", str(head_data_dir))
    PanoPaths.ensure_state_dirs()
    
    from control.utils.config_file import DaqNode
    nodes = [
        DaqNode(
            ip_addr=f"192.168.100.{10+i}",
            username="panoseti",
            data_dir="/data",
            module_ids=[i+100]
        )
        for i in range(num_nodes)
    ]
    # daq_config = DaqConfig(
    #     daq_nodes=nodes, 
    #     head_node_data_dir=str(head_data_dir),
    #     head_node_ip_addr="127.0.0.1"
    # )
    
    run_name = "scale_test.pffd"
    tq = TransferQueue()
    from datetime import UTC, datetime

    from control.transfer.models import TransferJob, TransferNodeSpec
    job = TransferJob(
        run_name=run_name,
        head_data_dir=str(head_data_dir),
        head_node_username="panoseti",
        created_at=datetime.now(UTC),
        daq_nodes=[
            TransferNodeSpec(
                ip_addr=n.ip_addr,
                username="panoseti",
                data_dir=n.data_dir,
                module_ids=n.module_ids
            )
            for n in nodes
        ]
    )
    tq.enqueue(job)
    active_job = tq.claim()
    
    # Mock Manifest and Rsync
    async def mock_gen_manifest(*args, **kwargs):
        return {"success": True, "manifest_path": "/data/manifest.blake3", "file_count": 1}

    async def mock_rsync(*args, **kwargs):
        # Just create the manifest file on head node so verification passes
        (head_data_dir / run_name).mkdir(parents=True, exist_ok=True)
        
        proc = MagicMock()
        proc.returncode = 0
        proc.wait = AsyncMock(return_value=0)
        proc.communicate = AsyncMock(return_value=(b'', b''))
        proc.stdout.readline = AsyncMock(return_value=b'')
        proc.stderr.read = AsyncMock(return_value=b'')
        return proc

    # Mock verifying to always pass
    with patch("panoseti_grpc.daq_control.client.AsyncDaqControlClient") as mock_client_cls, \
         patch("control.transfer.daemon.asyncio.create_subprocess_exec", side_effect=mock_rsync), \
         patch("control.transfer.daemon.verify_manifest", return_value=(True, [])):
        
        mock_client = AsyncMock()
        mock_client.GenerateManifest.side_effect = mock_gen_manifest
        mock_client.CleanupData.return_value = {"success": True}
        mock_client.__aenter__.return_value = mock_client
        mock_client_cls.return_value = mock_client

        mgr = RunStateManager()
        from datetime import UTC, datetime
        active_job.head_node_username = "panoseti"
        active_job.created_at = datetime.now(UTC)
        success, err = await asyncio.wait_for(_process_job(active_job, asyncio.Event(), mgr), timeout=10.0)
        assert success, err
        assert mock_client.GenerateManifest.call_count == num_nodes


# ---------------------------------------------------------------------------
# 3. Queue Depth: Drain 6 Deep
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_transfer_queue_drain_6_deep(
    session_fleet: Any,
    tmp_path: Path,
    ensure_clean_daq_state: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the daemon correctly drains a pending queue 6 transfers deep."""
    fleet, daq_cfg_dict = session_fleet
    head_data_dir, daq_config = setup_isolated_transfer_env(tmp_path, monkeypatch, daq_cfg_dict)
    
    runs = []
    for i in range(6):
        run_name = f"drain_{i}_{uuid.uuid4().hex[:4]}.pffd"
        expected_data = await generate_mocked_run(fleet, daq_config, run_name)
        runs.append((run_name, expected_data))
    
    tq = TransferQueue()
    assert len(list((tq._queue / "pending").glob("*.job.toml"))) == 6
    
    # mgr = RunStateManager()
    
    async def mocked_rsync(*args, **kwargs):
        # Extract run name from args (the destination path contains it)
        head_run_path = None
        for arg in args:
            if str(head_data_dir) in str(arg):
                head_run_path = Path(arg)
                break
        
        run_name = head_run_path.name
        simulate_rsync_from_fleet(fleet, run_name, head_run_path)
        proc = MagicMock()
        proc.returncode = 0
        proc.wait = AsyncMock(return_value=0)
        proc.communicate = AsyncMock(return_value=(b'', b''))
        proc.stdout.readline = AsyncMock(return_value=b'')
        proc.stderr.read = AsyncMock(return_value=b'')
        return proc

    with patch("control.transfer.daemon.asyncio.create_subprocess_exec", side_effect=mocked_rsync), \
         patch("panoseti_grpc.daq_control.client.AsyncDaqControlClient", side_effect=get_mapped_client_factory(daq_config)):
        
        # Start daemon loop as a task
        daemon_task = asyncio.create_task(run_daemon(poll_interval=0.1))
        
        # Poll until all are archived
        timeout = 60.0
        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < timeout:
            completed = list((tq._queue / "completed").glob("*.job.toml"))
            if len(completed) == 6:
                break
            await asyncio.sleep(1.0)
        else:
            pytest.fail("Timed out waiting for 6 jobs to complete")
        
        daemon_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await daemon_task

    for run_name, expected_data in runs:
        verify_head_node_accuracy(head_data_dir, run_name, expected_data)


# ---------------------------------------------------------------------------
# 4. Resilience: Stop/Start Daemon During Transfer
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_transfer_queue_stop_start_during_transfer(
    session_fleet: Any,
    tmp_path: Path,
    ensure_clean_daq_state: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify daemon resume from crash mid-transfer without data loss."""
    fleet, daq_cfg_dict = session_fleet
    head_data_dir, daq_config = setup_isolated_transfer_env(tmp_path, monkeypatch, daq_cfg_dict)
    run_name = f"stop_start_{uuid.uuid4().hex[:8]}.pffd"
    
    expected_data = await generate_mocked_run(fleet, daq_config, run_name)
    
    mgr = RunStateManager()
    tq = TransferQueue()
    
    # 1. Start daemon, intercept rsync to "stall"
    sync_event = asyncio.Event()
    daemon_cancelled = asyncio.Event()

    async def stalling_rsync(*args, **kwargs):
        sync_event.set()
        await daemon_cancelled.wait()
        # This will be interrupted by task cancellation
        return MagicMock(returncode=0)

    with patch("control.transfer.daemon.asyncio.create_subprocess_exec", side_effect=stalling_rsync), \
         patch("panoseti_grpc.daq_control.client.AsyncDaqControlClient", side_effect=get_mapped_client_factory(daq_config)):
        
        daemon_task = asyncio.create_task(run_daemon(poll_interval=0.1))
        
        # Wait for daemon to reach TRANSFERRING
        await asyncio.wait_for(sync_event.wait(), timeout=10.0)
        assert mgr.load_state().status == "TRANSFERRING"
        
        # 2. "Crash" the daemon
        daemon_task.cancel()
        daemon_cancelled.set()
        with contextlib.suppress(asyncio.CancelledError):
            await daemon_task

    # Job should be stranded in active/
    assert (tq._queue / "active" / f"{run_name}.job.toml").exists()
    
    # 3. Restart daemon with working rsync
    async def mocked_rsync(*args, **kwargs):
        simulate_rsync_from_fleet(fleet, run_name, head_data_dir / run_name)
        proc = MagicMock()
        proc.returncode = 0
        proc.wait = AsyncMock(return_value=0)
        proc.communicate = AsyncMock(return_value=(b'', b''))
        proc.stdout.readline = AsyncMock(return_value=b'')
        proc.stderr.read = AsyncMock(return_value=b'')
        return proc

    with patch("control.transfer.daemon.asyncio.create_subprocess_exec", side_effect=mocked_rsync), \
         patch("panoseti_grpc.daq_control.client.AsyncDaqControlClient", side_effect=get_mapped_client_factory(daq_config)):
        
        daemon_task2 = asyncio.create_task(run_daemon(poll_interval=0.1))
        
        # Wait for ARCHIVED
        timeout = 20.0
        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < timeout:
            if mgr.load_state().status == "ARCHIVED":
                break
            await asyncio.sleep(0.5)
        else:
            pytest.fail("Job did not finish after daemon restart")
            
        daemon_task2.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await daemon_task2

    verify_head_node_accuracy(head_data_dir, run_name, expected_data)
    assert (tq._queue / "completed" / f"{run_name}.job.toml").exists()
