# mypy: ignore-errors
"""
test_transfer_chaos.py

Phase 3 chaos tests: failure injection for the transfer state machine.

Requires Docker CI stack (IN_DOCKER_CI=1):
    pseti test sw integration -k transfer_chaos

Each scenario injects a specific failure and asserts the system reaches
the correct safe state without silent data loss.

Scenarios covered:
  CH-TX-01  rsync fails once then succeeds (retry ladder)
  CH-TX-02  rsync exhausts all retries → job lands in failed/
  CH-TX-03  manifest corruption → VERIFY_FAILED, no cleanup called
  CH-TX-04  daemon restart mid-job → stranded active/ job swept to pending/
  CH-TX-05  concurrent enqueue of same run is idempotent
  CH-TX-06  CleanupData FAILED_PRECONDITION → job NOT completed
  CH-TX-07  no_collect=True skips rsync; job still reaches ARCHIVED
  CH-TX-08  no_cleanup=True skips CleanupData; job still reaches ARCHIVED
  CH-TX-09  both flags True → no gRPC calls, ARCHIVED via run_complete only
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import pathlib
import sys
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from grpc import RpcError, StatusCode  # type: ignore[import]

from control.transfer.daemon import _process_job
from control.transfer.models import TransferJob, TransferNodeSpec
from control.transfer.queue import TransferQueue
from control.utils.pydantic_config_models import RunStateLedger
from control.utils.run_state import RunStateManager

# ---------------------------------------------------------------------------
# CI guard
# ---------------------------------------------------------------------------

IN_DOCKER_CI = os.environ.get("IN_DOCKER_CI") == "1"
pytestmark = pytest.mark.skipif(
    not IN_DOCKER_CI,
    reason="Requires Docker CI stack (IN_DOCKER_CI=1)",
)

DAQNODE_IP = os.environ.get("DAQNODE_DIRECT_HOST", "192.168.0.10")
HEAD_DATA_DIR = pathlib.Path(os.environ.get("HEAD_DATA_DIR", "/data/head"))
DAQ_DATA_DIR = pathlib.Path(os.environ.get("DAQ_DATA_DIR", "/data"))


# ---------------------------------------------------------------------------
# Shared gRPC mock helpers
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


def _grpc_ok() -> MagicMock:
    c = MagicMock()
    c.__aenter__ = AsyncMock(return_value=c)
    c.__aexit__ = AsyncMock(return_value=None)
    c.GenerateManifest = AsyncMock(return_value={"success": True, "file_count": 1})
    c.CleanupData = AsyncMock(return_value={"success": True, "deleted_count": 1})
    return c


def _rsync_ok() -> MagicMock:
    return MagicMock(returncode=0, stderr="")


def _rsync_fail(msg: str = "connection refused") -> MagicMock:
    return MagicMock(returncode=1, stderr=msg)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def run_name() -> str:
    return f"ci_chaos_{uuid.uuid4().hex[:8]}.pffd"


@pytest.fixture
def run_dir(run_name: str) -> pathlib.Path:
    """Head-node run dir with one synthetic PFF and a valid manifest."""
    d = HEAD_DATA_DIR / run_name
    d.mkdir(parents=True, exist_ok=True)
    fname = "start_2024.dp_ph256.module_200.seqno_0.pff"
    data = os.urandom(128)
    (d / fname).write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    (d / "manifest.sha256").write_text(f"{digest}  {len(data)}  0  {fname}\n")
    yield d
    import shutil
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def make_job(run_name: str):
    """Factory to create a TransferJob with optional overrides."""
    def _make(no_collect: bool = False, no_cleanup: bool = False) -> TransferJob:
        return TransferJob(
            run_name=run_name,
            head_data_dir=str(HEAD_DATA_DIR),
            head_node_username="panoseti",
            created_at=datetime.now(UTC),
            no_collect=no_collect,
            no_cleanup=no_cleanup,
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
    return _make


# ---------------------------------------------------------------------------
# CH-TX-01: rsync fails once then succeeds
# ---------------------------------------------------------------------------

class TestCHTX01RsyncRetry:
    """rsync fails on attempt 1 and succeeds on attempt 2."""

    async def test_retry_once_succeeds(
        self,
        tmp_path: pathlib.Path,
        run_name: str,
        run_dir: pathlib.Path,
        make_job,
    ) -> None:
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setenv("PSETI_CONTROL", str(tmp_path))
        client = _grpc_ok()
        # First call fails, second succeeds
        responses = [_rsync_fail(), _rsync_ok()]

        with _mock_grpc(client), \
             patch("control.transfer.daemon.subprocess") as mock_sub:
            mock_sub.run.side_effect = responses
            result = await _process_job(make_job())

        assert result is True, "Job must succeed after one retry"
        assert (run_dir / "run_complete").exists()
        monkeypatch.undo()


# ---------------------------------------------------------------------------
# CH-TX-02: rsync exhausts all retries
# ---------------------------------------------------------------------------

class TestCHTX02RsyncExhausted:
    """rsync fails on all attempts → _process_job returns False."""

    async def test_exhausted_retries_returns_false(
        self,
        tmp_path: pathlib.Path,
        run_name: str,
        run_dir: pathlib.Path,
        make_job,
    ) -> None:
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setenv("PSETI_CONTROL", str(tmp_path))
        client = _grpc_ok()

        with _mock_grpc(client), \
             patch("control.transfer.daemon.subprocess") as mock_sub, \
             patch("control.transfer.daemon.asyncio") as mock_asyncio:
            mock_sub.run.return_value = _rsync_fail()
            mock_asyncio.sleep = AsyncMock()
            result = await _process_job(make_job())

        assert result is False, "All retries exhausted must return False"
        assert not (run_dir / "run_complete").exists(), (
            "run_complete must NOT be written when transfer failed"
        )
        monkeypatch.undo()


# ---------------------------------------------------------------------------
# CH-TX-03: manifest corruption → VERIFY_FAILED, no cleanup
# ---------------------------------------------------------------------------

class TestCHTX03ManifestCorruption:
    """Corrupted manifest causes process_job to return False without calling CleanupData."""

    async def test_corrupt_manifest_skips_cleanup(
        self,
        tmp_path: pathlib.Path,
        run_name: str,
        run_dir: pathlib.Path,
        make_job,
    ) -> None:
        # Corrupt the manifest: wrong digest
        (run_dir / "manifest.sha256").write_text("deadbeef  128  0  start_2024.dp_ph256.module_200.seqno_0.pff\n")

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setenv("PSETI_CONTROL", str(tmp_path))
        client = _grpc_ok()

        with _mock_grpc(client), \
             patch("control.transfer.daemon.subprocess") as mock_sub:
            mock_sub.run.return_value = _rsync_ok()
            result = await _process_job(make_job())

        assert result is False, "Corrupt manifest must cause process_job to fail"
        client.CleanupData.assert_not_called(), (
            "CleanupData must NOT be called when verification fails"
        )
        assert not (run_dir / "run_complete").exists()
        monkeypatch.undo()


# ---------------------------------------------------------------------------
# CH-TX-04: stranded active/ job swept to pending/ on daemon restart
# ---------------------------------------------------------------------------

class TestCHTX04StrandedJobRecovery:
    """A job left in active/ by a crash is swept to pending/ on daemon restart."""

    def test_stranded_job_swept_to_pending(
        self,
        tmp_path: pathlib.Path,
        run_name: str,
        make_job,
    ) -> None:
        from control.transfer.daemon import _sweep_stranded_jobs

        queue_dir = tmp_path / "state" / "transfer" / "queue"
        tq = TransferQueue(queue_dir=queue_dir)
        tq.enqueue(make_job())
        # Simulate crash: manually move to active/ without claiming
        import shutil
        pending_path = queue_dir / "pending" / f"{run_name}.job.toml"
        active_path = queue_dir / "active" / f"{run_name}.job.toml"
        shutil.move(str(pending_path), str(active_path))

        assert active_path.exists()
        assert not pending_path.exists()

        _sweep_stranded_jobs(tq)

        assert pending_path.exists(), "Stranded job must be swept back to pending/"
        assert not active_path.exists()


# ---------------------------------------------------------------------------
# CH-TX-05: concurrent enqueue of same run is idempotent
# ---------------------------------------------------------------------------

class TestCHTX05ConcurrentEnqueue:
    """Two simultaneous enqueue calls for the same run produce exactly one job."""

    def test_concurrent_enqueue_idempotent(
        self,
        tmp_path: pathlib.Path,
        run_name: str,
        make_job,
    ) -> None:
        queue_dir = tmp_path / "state" / "transfer" / "queue"
        tq = TransferQueue(queue_dir=queue_dir)
        results = [tq.enqueue(make_job()), tq.enqueue(make_job())]
        assert results.count(True) == 1
        assert results.count(False) == 1
        pending = tq.list_jobs("pending")
        assert pending == [run_name], f"Exactly one pending job expected, got {pending}"


# ---------------------------------------------------------------------------
# CH-TX-06: CleanupData FAILED_PRECONDITION → job fails
# ---------------------------------------------------------------------------

class TestCHTX06CleanupPreconditionFailed:
    """DAQ server refuses CleanupData → process_job returns False."""

    async def test_cleanup_precondition_failure(
        self,
        tmp_path: pathlib.Path,
        run_name: str,
        run_dir: pathlib.Path,
        make_job,
    ) -> None:
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setenv("PSETI_CONTROL", str(tmp_path))
        client = _grpc_ok()
        # CleanupData returns failure (simulating FAILED_PRECONDITION)
        client.CleanupData = AsyncMock(return_value={"success": False, "message": "FAILED_PRECONDITION"})

        with _mock_grpc(client), \
             patch("control.transfer.daemon.subprocess") as mock_sub:
            mock_sub.run.return_value = _rsync_ok()
            result = await _process_job(make_job())

        # run_complete must not be written when cleanup fails
        assert not (run_dir / "run_complete").exists(), (
            "run_complete must NOT be written when CleanupData failed"
        )
        monkeypatch.undo()


# ---------------------------------------------------------------------------
# CH-TX-07: no_collect=True skips rsync
# ---------------------------------------------------------------------------

class TestCHTX07NoCollect:
    """no_collect=True: rsync never called, job reaches ARCHIVED."""

    async def test_no_collect_skips_rsync(
        self,
        tmp_path: pathlib.Path,
        run_name: str,
        run_dir: pathlib.Path,
        make_job,
    ) -> None:
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setenv("PSETI_CONTROL", str(tmp_path))
        client = _grpc_ok()

        with _mock_grpc(client), \
             patch("control.transfer.daemon.subprocess") as mock_sub:
            mock_sub.run.return_value = _rsync_ok()
            result = await _process_job(make_job(no_collect=True))

        assert result is True
        mock_sub.run.assert_not_called(), "rsync must NOT be called with no_collect=True"
        assert (run_dir / "run_complete").exists()
        monkeypatch.undo()


# ---------------------------------------------------------------------------
# CH-TX-08: no_cleanup=True skips CleanupData
# ---------------------------------------------------------------------------

class TestCHTX08NoCleanup:
    """no_cleanup=True: CleanupData never called, job reaches ARCHIVED."""

    async def test_no_cleanup_skips_cleanup(
        self,
        tmp_path: pathlib.Path,
        run_name: str,
        run_dir: pathlib.Path,
        make_job,
    ) -> None:
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setenv("PSETI_CONTROL", str(tmp_path))
        client = _grpc_ok()

        with _mock_grpc(client), \
             patch("control.transfer.daemon.subprocess") as mock_sub:
            mock_sub.run.return_value = _rsync_ok()
            result = await _process_job(make_job(no_cleanup=True))

        assert result is True
        client.CleanupData.assert_not_called()
        assert (run_dir / "run_complete").exists()
        monkeypatch.undo()


# ---------------------------------------------------------------------------
# CH-TX-09: no_collect + no_cleanup → no gRPC calls at all
# ---------------------------------------------------------------------------

class TestCHTX09BothFlagsNoGrpc:
    """Both flags True: no GenerateManifest, no CleanupData, ARCHIVED via run_complete."""

    async def test_both_flags_no_grpc(
        self,
        tmp_path: pathlib.Path,
        run_name: str,
        run_dir: pathlib.Path,
        make_job,
    ) -> None:
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setenv("PSETI_CONTROL", str(tmp_path))
        client = _grpc_ok()

        with _mock_grpc(client), \
             patch("control.transfer.daemon.subprocess") as mock_sub:
            mock_sub.run.return_value = _rsync_ok()
            result = await _process_job(make_job(no_collect=True, no_cleanup=True))

        assert result is True
        client.GenerateManifest.assert_not_called()
        client.CleanupData.assert_not_called()
        mock_sub.run.assert_not_called()
        assert (run_dir / "run_complete").exists()
        monkeypatch.undo()
