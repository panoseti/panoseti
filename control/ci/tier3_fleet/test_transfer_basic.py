# mypy: ignore-errors
"""
test_transfer_basic.py

Phase 3 integration tests: standard transfer, no port-forwarding.

Requires the Docker CI stack (IN_DOCKER_CI=1):
    pseti test sw integration -k transfer_basic

Topology:
    test-runner  ─ direct ─→  daqnode (192.168.0.10)
                              head data dir: /data/head

Each test exercises the full TransferQueue → _process_job() state machine
against real synthetic PFF files placed on the shared Docker volume.
gRPC (GenerateManifest, CleanupData) and subprocess (rsync) are mocked so
the suite runs without real hashpipe output or real SSH keys.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import pathlib
import sys
import tomllib
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from control.transfer.daemon import _process_job
from control.transfer.models import TransferJob, TransferNodeSpec
from control.transfer.queue import TransferQueue
from control.transfer.verify import verify_manifest
from control.utils.pydantic_config_models import RunStateLedger
from control.utils.run_state import RunStateManager

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DAQNODE_IP = os.environ.get("DAQNODE_DIRECT_HOST", "192.168.0.10")
HEAD_DATA_DIR = pathlib.Path(os.environ.get("HEAD_DATA_DIR", "/data/head"))
DAQ_DATA_DIR = pathlib.Path(os.environ.get("DAQ_DATA_DIR", "/data"))


# ---------------------------------------------------------------------------
# gRPC stub injection (mirrors test_transfer_daemon.py)
# ---------------------------------------------------------------------------

@contextmanager
def _mock_grpc(mock_client: MagicMock):
    stub_root = ModuleType("panoseti_grpc")
    stub_daq = ModuleType("panoseti_grpc.daq_control")
    stub_cm = ModuleType("panoseti_grpc.daq_control.client")
    stub_cm.AsyncDaqControlClient = MagicMock(return_value=mock_client)
    stub_root.daq_control = stub_daq
    stub_daq.client = stub_cm
    injected = {
        "panoseti_grpc": stub_root,
        "panoseti_grpc.daq_control": stub_daq,
        "panoseti_grpc.daq_control.client": stub_cm,
    }
    prev = {k: sys.modules.get(k) for k in injected}
    sys.modules.update(injected)
    try:
        yield stub_cm.AsyncDaqControlClient
    finally:
        for k, orig in prev.items():
            if orig is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = orig


def _grpc_client_ok() -> MagicMock:
    c = MagicMock()
    c.__aenter__ = AsyncMock(return_value=c)
    c.__aexit__ = AsyncMock(return_value=None)
    c.GenerateManifest = AsyncMock(return_value={"success": True, "file_count": 2})
    c.CleanupData = AsyncMock(return_value={"success": True, "deleted_count": 1})
    return c


# ---------------------------------------------------------------------------
# Synthetic run fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def run_name() -> str:
    return f"ci_transfer_basic_{uuid.uuid4().hex[:8]}.pffd"


@pytest.fixture
def run_dir(run_name: str, head_data_dir: pathlib.Path) -> pathlib.Path:
    """Create a head-node run dir with synthetic PFF and manifest files."""
    d = head_data_dir / run_name
    d.mkdir(parents=True, exist_ok=True)

    # Write two synthetic files
    for i in range(2):
        fname = f"start_2024.dp_ph256.module_200.seqno_{i}.pff"
        data = os.urandom(512)
        (d / fname).write_bytes(data)
    # Write a sha256 manifest
    manifest = d / "manifest.sha256"
    lines = []
    for f in sorted(d.iterdir()):
        if f.suffix == ".pff":
            digest = hashlib.sha256(f.read_bytes()).hexdigest()
            lines.append(f"{digest}  {f.stat().st_size}  0  {f.name}\n")
    manifest.write_text("".join(lines))
    yield d
    # Cleanup
    import shutil
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def state_mgr(tmp_path: pathlib.Path, run_name: str) -> RunStateManager:
    mgr = RunStateManager(base_dir=str(tmp_path))
    mgr.save_state(RunStateLedger(
        run_name=run_name,
        status="RECORDING_ENDED",
        start_time=datetime.now(UTC).isoformat(),
    ))
    return mgr


@pytest.fixture
def transfer_job(run_name: str, run_dir: pathlib.Path, head_data_dir: pathlib.Path) -> TransferJob:
    return TransferJob(schema_version=1, 
        run_name=run_name,
        head_data_dir=str(head_data_dir),
        head_node_username="panoseti",
        created_at=datetime.now(UTC),
        no_collect=True,   # skip rsync in basic suite; filesystem already has files
        no_cleanup=True,
        daq_nodes=[
            TransferNodeSpec(
                ip_addr=DAQNODE_IP,
                username="panoseti",
                data_dir=str(DAQ_DATA_DIR),
                module_ids=[200],
                port_forwarding=None,
            )
        ],
    )


# ---------------------------------------------------------------------------
# 1. Happy path: job reaches ARCHIVED
# ---------------------------------------------------------------------------

class TestTransferBasicHappyPath:
    """Standard single-node transfer → ARCHIVED."""

    async def test_process_job_returns_true(
        self,
        tmp_path: pathlib.Path,
        run_name: str,
        run_dir: pathlib.Path,
        transfer_job: TransferJob,
    ) -> None:
        """_process_job returns True and writes run_complete."""
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setenv("PSETI_CONTROL", str(tmp_path))
        client = _grpc_client_ok()
        with _mock_grpc(client), \
             patch("control.transfer.daemon.asyncio.create_subprocess_exec", side_effect=_mock_subprocess_ok) as mock_sub:
            
            result, _ = await _process_job(transfer_job, asyncio.Event(), RunStateManager())
        assert result is True
        assert (run_dir / "run_complete").exists(), "run_complete must be written on ARCHIVED"
        monkeypatch.undo()

    async def test_queue_job_toml_is_valid(
        self,
        tmp_path: pathlib.Path,
        run_name: str,
        transfer_job: TransferJob,
    ) -> None:
        """Enqueued TOML is parseable and round-trips TransferJob exactly."""
        queue_dir = tmp_path / "state" / "transfer" / "queue"
        tq = TransferQueue(queue_dir=queue_dir)
        tq.enqueue(transfer_job)
        pending = queue_dir / "pending" / f"{run_name}.job.toml"
        assert pending.exists(), "pending job file must be written"
        data = tomllib.loads(pending.read_text())
        reloaded = TransferJob.model_validate(data)
        assert reloaded.run_name == run_name
        assert reloaded.daq_nodes[0].port_forwarding is None

    async def test_run_complete_idempotent(
        self,
        tmp_path: pathlib.Path,
        run_name: str,
        run_dir: pathlib.Path,
        transfer_job: TransferJob,
    ) -> None:
        """If run_complete already exists, _process_job does not overwrite it."""
        sentinel = "original"
        (run_dir / "run_complete").write_text(sentinel)
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setenv("PSETI_CONTROL", str(tmp_path))
        client = _grpc_client_ok()
        with _mock_grpc(client), \
             patch("control.transfer.daemon.asyncio.create_subprocess_exec", side_effect=_mock_subprocess_ok) as mock_sub:
            
            _, _ = await _process_job(transfer_job, asyncio.Event(), RunStateManager())
        assert (run_dir / "run_complete").read_text() == sentinel
        monkeypatch.undo()


# ---------------------------------------------------------------------------
# 2. Manifest verification
# ---------------------------------------------------------------------------

class TestTransferBasicVerify:
    """verify_manifest() produces correct results for good and bad manifests."""

    def test_valid_manifest_ok(self, run_dir: pathlib.Path) -> None:
        manifest = run_dir / "manifest.sha256"
        ok, errs = verify_manifest(manifest, run_dir)
        assert ok is True
        assert errs == []

    def test_corrupt_manifest_fails(self, run_dir: pathlib.Path) -> None:
        """Flipping one byte of a listed file causes verify_manifest to fail."""
        manifest = run_dir / "manifest.sha256"
        pff = next(run_dir.glob("*.pff"))
        original = pff.read_bytes()
        pff.write_bytes(bytes([original[0] ^ 0xFF]) + original[1:])
        ok, errs = verify_manifest(manifest, run_dir)
        assert ok is False
        assert any(pff.name in e for e in errs)
        pff.write_bytes(original)  # restore

    def test_missing_file_fails(self, run_dir: pathlib.Path) -> None:
        manifest = run_dir / "manifest.sha256"
        pff = next(run_dir.glob("*.pff"))
        pff.unlink()
        ok, _errs = verify_manifest(manifest, run_dir)
        assert ok is False


# ---------------------------------------------------------------------------
# 3. Queue idempotency
# ---------------------------------------------------------------------------

class TestTransferBasicQueue:
    """TransferQueue idempotency and bucket transitions."""

    def test_double_enqueue_is_idempotent(
        self,
        tmp_path: pathlib.Path,
        run_name: str,
        transfer_job: TransferJob,
    ) -> None:
        queue_dir = tmp_path / "state" / "transfer" / "queue"
        tq = TransferQueue(queue_dir=queue_dir)
        first = tq.enqueue(transfer_job)
        second = tq.enqueue(transfer_job)
        assert first is True
        assert second is False

    def test_claim_moves_to_active(
        self,
        tmp_path: pathlib.Path,
        run_name: str,
        transfer_job: TransferJob,
    ) -> None:
        queue_dir = tmp_path / "state" / "transfer" / "queue"
        tq = TransferQueue(queue_dir=queue_dir)
        tq.enqueue(transfer_job)
        job = tq.claim()
        assert job is not None
        assert job.run_name == run_name
        assert (queue_dir / "active" / f"{run_name}.job.toml").exists()
        assert not (queue_dir / "pending" / f"{run_name}.job.toml").exists()

    def test_complete_moves_to_completed(
        self,
        tmp_path: pathlib.Path,
        run_name: str,
        transfer_job: TransferJob,
    ) -> None:
        queue_dir = tmp_path / "state" / "transfer" / "queue"
        tq = TransferQueue(queue_dir=queue_dir)
        tq.enqueue(transfer_job)
        tq.claim()
        tq.complete(run_name)
        assert (queue_dir / "completed" / f"{run_name}.job.toml").exists()

    def test_fail_and_retry(
        self,
        tmp_path: pathlib.Path,
        run_name: str,
        transfer_job: TransferJob,
    ) -> None:
        queue_dir = tmp_path / "state" / "transfer" / "queue"
        tq = TransferQueue(queue_dir=queue_dir)
        tq.enqueue(transfer_job)
        tq.claim()
        tq.fail(run_name)
        assert (queue_dir / "failed" / f"{run_name}.job.toml").exists()
        ok = tq.retry(run_name)
        assert ok is True
        assert (queue_dir / "pending" / f"{run_name}.job.toml").exists()

async def _mock_subprocess_ok(*args, **kwargs):
    proc = MagicMock()
    proc.returncode = 0
    proc.wait = AsyncMock(return_value=0)
    proc.communicate = AsyncMock(return_value=(b"", b""))
    proc.stdout.readline = AsyncMock(return_value=b"")
    proc.stderr.read = AsyncMock(return_value=b"")
    return proc
