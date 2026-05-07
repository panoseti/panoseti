# mypy: ignore-errors
"""
test_transfer.py — Logic tests for the transfer pipeline.

Ported from ci/software_only/tier2_logic/test_transfer.py.
"""

from __future__ import annotations

import asyncio
import pathlib
import shutil
import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ci.fixtures.mocks import MockDaqNode
from control.transfer.daemon import _process_job, _sweep_stranded_jobs
from control.transfer.queue import TransferQueue
from control.transfer.models import TransferJob, TransferNodeSpec
from control.utils.run_state import RunStateManager
from ci.software_only_v2.infra.workspace import Workspace

async def _mock_subprocess_ok(*args, **kwargs):
    dest = None
    if args:
        for arg in reversed(args):
            if isinstance(arg, (str, pathlib.Path)) and str(arg).endswith(".pffd"):
                dest = pathlib.Path(arg)
                break
    
    if dest:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "dp_manifest.node_test.algo_blake3.txt").touch()

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

def _daq_fs_simulator(dest_dir: pathlib.Path, run_name: str, module_ids: list[int]):
    # 1. Root run dir
    run_root = dest_dir / run_name
    run_root.mkdir(parents=True, exist_ok=True)
    import json
    (run_root / "meta.json").write_text(json.dumps({"test_run": True}))
    
    # 2. Module-specific data
    for mid in module_ids:
        mod_run = dest_dir / f"module_{mid}" / run_name
        mod_run.mkdir(parents=True, exist_ok=True)
        (mod_run / f"data.module_{mid}.pff").write_bytes(b"pff data")


class TestTransferLogic:
    """Tests for the transfer state machine logic."""

    @pytest.mark.asyncio
    async def test_when_transfer_job_processed_then_reaches_archived(
        self, pseti_workspace: Workspace
    ) -> None:
        run_name = "tier2_test_run.pffd"
        head_root = pseti_workspace.root / "head"
        head_root.mkdir(parents=True, exist_ok=True)
        
        job = TransferJob(
            schema_version=1,
            run_name=run_name,
            head_data_dir=str(head_root),
            head_node_username="panoseti",
            created_at=asyncio.get_event_loop().time(),
            daq_nodes=[
                TransferNodeSpec(
                    ip_addr="192.168.0.10",
                    username="u",
                    data_dir=str(pseti_workspace.root / "daq_root"),
                    module_ids=[201, 254]
                )
            ]
        )
        
        daq_root = pseti_workspace.root / "daq_root"
        _daq_fs_simulator(daq_root, run_name, module_ids=[201, 254])
        
        mock_daq = MockDaqNode("192.168.0.10")
        
        with (
            patch("panoseti_grpc.daq_control.client.AsyncDaqControlClient", return_value=mock_daq.client),
            patch("control.transfer.daemon.asyncio.create_subprocess_exec", side_effect=_mock_subprocess_ok)
        ):
            success = await _process_job(job, asyncio.Event(), RunStateManager(base_dir=pseti_workspace.root / "state"))
                
        assert success[0] is True
        assert (head_root / run_name / "run_complete").exists()

    @pytest.mark.asyncio
    async def test_when_manifest_fails_then_transfer_aborts(
        self, pseti_workspace: Workspace
    ) -> None:
        head_root = pseti_workspace.root / "head"
        head_root.mkdir(parents=True, exist_ok=True)
        job = TransferJob(
            schema_version=1,
            run_name="fail_run",
            head_data_dir=str(head_root),
            head_node_username="u",
            created_at=0,
            daq_nodes=[TransferNodeSpec(ip_addr="192.168.0.10", username="u", data_dir="/d", module_ids=[1])]
        )
        mock_daq = MockDaqNode("192.168.0.10")
        mock_daq.set_manifest_failure("Disk full on DAQ")
        
        with (
            patch("panoseti_grpc.daq_control.client.AsyncDaqControlClient", return_value=mock_daq.client),
            patch("control.transfer.daemon.asyncio.create_subprocess_exec", side_effect=_mock_subprocess_ok) as mock_rsync
        ):
            success = await _process_job(job, asyncio.Event(), RunStateManager(base_dir=pseti_workspace.root / "state"))
                
            assert success[0] is False
            mock_rsync.assert_not_called()

    @pytest.mark.asyncio
    async def test_when_rsync_fails_then_job_returns_false(
        self, pseti_workspace: Workspace
    ) -> None:
        head_root = pseti_workspace.root / "head"
        head_root.mkdir(parents=True, exist_ok=True)
        job = TransferJob(
            schema_version=1,
            run_name="rsync_fail",
            head_data_dir=str(head_root),
            head_node_username="u",
            created_at=0,
            daq_nodes=[TransferNodeSpec(ip_addr="192.168.0.10", username="u", data_dir="/d", module_ids=[1])]
        )
        mock_daq = MockDaqNode("192.168.0.10")
        
        with (
            patch("panoseti_grpc.daq_control.client.AsyncDaqControlClient", return_value=mock_daq.client),
            patch("control.transfer.daemon.asyncio.create_subprocess_exec", side_effect=_mock_subprocess_fail)
        ):
            success = await _process_job(job, asyncio.Event(), RunStateManager(base_dir=pseti_workspace.root / "state"))
                
        assert success[0] is False

    @pytest.mark.asyncio
    async def test_when_file_corrupted_then_verify_fails(
        self, pseti_workspace: Workspace
    ) -> None:
        run_name = "corrupt_test.pffd"
        head_root = pseti_workspace.root / "head"
        run_dir = head_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        
        pff_file = run_dir / "data.pff"
        pff_file.write_bytes(b"original content")
        digest = hashlib.sha256(b"original content").hexdigest()
        manifest = run_dir / "manifest.sha256"
        manifest.write_text(f"{digest}  16  0  data.pff\n")
        
        pff_file.write_bytes(b"corrupted!")
        
        job = TransferJob(
            schema_version=1,
            run_name=run_name,
            head_data_dir=str(head_root),
            head_node_username="u",
            created_at=0,
            daq_nodes=[TransferNodeSpec(ip_addr="192.168.0.10", username="u", data_dir="/d", module_ids=[1])]
        )
        mock_daq = MockDaqNode("192.168.0.10")
        
        with (
            patch("panoseti_grpc.daq_control.client.AsyncDaqControlClient", return_value=mock_daq.client),
            patch("control.transfer.daemon.asyncio.create_subprocess_exec", side_effect=_mock_subprocess_ok)
        ):
            success = await _process_job(job, asyncio.Event(), RunStateManager(base_dir=pseti_workspace.root / "state"))
                
        assert success[0] is False
        mock_daq.client.CleanupData.assert_not_called()

    def test_when_job_stranded_then_recovered_on_startup(
        self, pseti_workspace: Workspace
    ) -> None:
        from control.utils.paths import PanoPaths
        tq = TransferQueue(queue_dir=PanoPaths.transfer_queue_dir())
        job = TransferJob(
            schema_version=1,
            run_name="stranded_run",
            head_data_dir=str(pseti_workspace.root),
            head_node_username="u",
            created_at=0,
            daq_nodes=[]
        )
        tq.enqueue(job)
        
        pending_path = tq._queue / "pending" / f"{job.run_name}.job.toml"
        active_path = tq._queue / "active" / f"{job.run_name}.job.toml"
        shutil.move(str(pending_path), str(active_path))
        
        _sweep_stranded_jobs(tq)
        
        assert pending_path.exists()
        assert not active_path.exists()

    def test_when_double_enqueued_then_queue_is_idempotent(
        self, pseti_workspace: Workspace
    ) -> None:
        from control.utils.paths import PanoPaths
        tq = TransferQueue(queue_dir=PanoPaths.transfer_queue_dir())
        job = TransferJob(
            schema_version=1,
            run_name="idempotent_test",
            head_data_dir=str(pseti_workspace.root),
            head_node_username="u",
            created_at=0,
            daq_nodes=[]
        )
        
        res1 = tq.enqueue(job)
        res2 = tq.enqueue(job)
        
        assert res1 is True
        assert res2 is False
        
        pending = list((tq._queue / "pending").glob("*.toml"))
        test_jobs = [p for p in pending if "idempotent_test" in p.name]
        assert len(test_jobs) == 1
