"""
test_transfer_basic.py — Basic transfer pipeline tests.

Ported from ci/software_only/tier3_fleet/test_transfer_basic.py.
"""

from __future__ import annotations

import asyncio
import hashlib
import pathlib
import tomllib
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ci.fixtures.rsync_fixtures import RsyncMock
from ci.software_only_v2.infra.workspace import Workspace
from control.transfer.daemon import _process_job
from control.transfer.models import TransferJob, TransferNodeSpec
from control.transfer.queue import TransferQueue
from control.transfer.verify import verify_manifest
from control.utils.pydantic_config_models import RunStateLedger
from control.utils.run_state import RunStateManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grpc_client_ok() -> MagicMock:
    c = MagicMock()
    c.__aenter__ = AsyncMock(return_value=c)
    c.__aexit__ = AsyncMock(return_value=None)
    c.GenerateManifest = AsyncMock(return_value={"success": True, "file_count": 2})
    c.CleanupData = AsyncMock(return_value={"success": True, "deleted_count": 1})
    return c


@pytest.fixture
def run_name() -> str:
    return f"ci_transfer_basic_{uuid.uuid4().hex[:8]}.pffd"


@pytest.fixture
def run_dir(run_name: str, pseti_workspace: Workspace, dummy_data_generator) -> pathlib.Path:
    """Create a head-node run dir with synthetic PFF and manifest files."""
    head_data_dir = pseti_workspace.root / "head_data"
    head_data_dir.mkdir(parents=True, exist_ok=True)
    d = head_data_dir / run_name
    d.mkdir(parents=True, exist_ok=True)

    # Use dummy_data_generator to generate files, then move them to d
    # dummy_data_generator(dest_dir, run_name, module_ids)
    # creates dest_dir/module_mid/run_name/*.pff
    tmp_dest = pseti_workspace.root / "tmp_gen"
    dummy_data_generator(tmp_dest, run_name, [200])
    
    mod_dir = tmp_dest / "module_200" / run_name
    for pff in mod_dir.glob("*.pff"):
        pff.rename(d / pff.name)

    # Write a sha256 manifest
    manifest = d / "manifest.sha256"
    lines = []
    for f in sorted(d.iterdir()):
        if f.suffix == ".pff":
            digest = hashlib.sha256(f.read_bytes()).hexdigest()
            lines.append(f"{digest}  {f.stat().st_size}  0  {f.name}\n")
    manifest.write_text("".join(lines))
    return d


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTransferBasicHappyPath:
    """Standard single-node transfer → ARCHIVED."""

    @pytest.mark.asyncio
    async def test_process_job_returns_true(
        self,
        pseti_workspace: Workspace,
        run_name: str,
        run_dir: pathlib.Path,
        mock_rsync_transfer: RsyncMock,
        transfer_job_factory,
    ) -> None:
        """_process_job returns True and writes run_complete."""
        job = transfer_job_factory(
            run_name=run_name,
            head_data_dir=run_dir.parent,
            no_collect=True,
            no_cleanup=True,
            daq_nodes=[TransferNodeSpec(ip_addr="1.1.1.1", username="u", data_dir="/d", module_ids=[1])]
        )
        
        client = _grpc_client_ok()
        state_mgr = RunStateManager(base_dir=pseti_workspace.root / "state")
        state_mgr.save_state(RunStateLedger(run_name=run_name, status="RECORDING_ENDED", start_time=""))

        with patch("panoseti_grpc.daq_control.client.AsyncDaqControlClient", return_value=client):
            result, _ = await _process_job(job, asyncio.Event(), state_mgr)
            
        assert result is True
        assert (run_dir / "run_complete").exists()

    def test_queue_job_toml_is_valid(
        self,
        pseti_workspace: Workspace,
        run_name: str,
        transfer_job_factory,
    ) -> None:
        from control.utils.paths import PanoPaths
        tq = TransferQueue(queue_dir=PanoPaths.transfer_queue_dir())
        job = transfer_job_factory(
            run_name=run_name,
            daq_nodes=[TransferNodeSpec(ip_addr="1.1.1.1", username="u", data_dir="/d", module_ids=[1])]
        )
        tq.enqueue(job)
        pending = tq._queue / "pending" / f"{run_name}.job.toml"
        assert pending.exists()
        data = tomllib.loads(pending.read_text())
        reloaded = TransferJob.model_validate(data)
        assert reloaded.run_name == run_name


class TestTransferBasicVerify:
    """verify_manifest() produces correct results."""

    def test_valid_manifest_ok(self, run_dir: pathlib.Path) -> None:
        manifest = run_dir / "manifest.sha256"
        ok, errs = verify_manifest(manifest, run_dir)
        assert ok is True
        assert errs == []

    def test_corrupt_manifest_fails(self, run_dir: pathlib.Path) -> None:
        manifest = run_dir / "manifest.sha256"
        pff = next(run_dir.glob("*.pff"))
        original = pff.read_bytes()
        pff.write_bytes(bytes([original[0] ^ 0xFF]) + original[1:])
        ok, errs = verify_manifest(manifest, run_dir)
        assert ok is False
        assert any(pff.name in e for e in errs)

    def test_missing_file_fails(self, run_dir: pathlib.Path) -> None:
        manifest = run_dir / "manifest.sha256"
        pff = next(run_dir.glob("*.pff"))
        pff.unlink()
        ok, _errs = verify_manifest(manifest, run_dir)
        assert ok is False


class TestTransferBasicQueue:
    """TransferQueue idempotency and bucket transitions."""

    def test_double_enqueue_is_idempotent(
        self,
        pseti_workspace: Workspace,
        run_name: str,
        transfer_job_factory,
    ) -> None:
        from control.utils.paths import PanoPaths
        tq = TransferQueue(queue_dir=PanoPaths.transfer_queue_dir())
        job = transfer_job_factory(run_name=run_name, daq_nodes=[])
        first = tq.enqueue(job)
        second = tq.enqueue(job)
        assert first is True
        assert second is False

    def test_claim_moves_to_active(
        self,
        pseti_workspace: Workspace,
        run_name: str,
        transfer_job_factory,
    ) -> None:
        from control.utils.paths import PanoPaths
        tq = TransferQueue(queue_dir=PanoPaths.transfer_queue_dir())
        job = transfer_job_factory(run_name=run_name, daq_nodes=[])
        tq.enqueue(job)
        claimed = tq.claim()
        assert claimed.run_name == run_name
        assert (tq._queue / "active" / f"{run_name}.job.toml").exists()
        assert not (tq._queue / "pending" / f"{run_name}.job.toml").exists()
