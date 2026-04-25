# mypy: ignore-errors
"""
test_transfer_daemon.py

Unit tests for the transfer daemon state machine, lock helpers, and
verify_manifest utility.

All tests are hardware-agnostic: gRPC and subprocess are mocked; the
filesystem is isolated via tmp_path and env-var overrides.
"""

from __future__ import annotations

import hashlib
import pathlib
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

from control.transfer.daemon import (
    _acquire_transfer_lock,
    _process_job,
    _release_transfer_lock,
)
from control.transfer.models import TransferJob, TransferNodeSpec
from control.transfer.verify import verify_manifest
from control.utils.pydantic_config_models import RunStateLedger
from control.utils.run_state import RunStateManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_ledger(
    tmp_path: pathlib.Path,
    run_name: str = "myrun.pffd",
    status: str = "RECORDING_ENDED",
) -> RunStateManager:
    """Create a RunStateManager with a minimal ledger written to tmp_path."""
    mgr = RunStateManager(base_dir=str(tmp_path))
    ledger = RunStateLedger(
        run_name=run_name,
        status=status,  # type: ignore[arg-type]
        start_time=datetime.now(UTC).isoformat(),
    )
    mgr.save_state(ledger)
    return mgr


def _make_job(
    tmp_path: pathlib.Path,
    run_name: str = "myrun.pffd",
    no_collect: bool = False,
    no_cleanup: bool = False,
) -> TransferJob:
    """Return a minimal valid TransferJob for testing."""
    return TransferJob(
        run_name=run_name,
        head_data_dir=str(tmp_path / "data"),
        head_node_username="panoseti",
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
        no_collect=no_collect,
        no_cleanup=no_cleanup,
        daq_nodes=[
            TransferNodeSpec(
                ip_addr="192.168.0.10",
                username="panoseti",
                data_dir="/app/data",
                module_ids=[250],
            )
        ],
    )


def _make_run_dir(tmp_path: pathlib.Path, run_name: str = "myrun.pffd") -> pathlib.Path:
    """Create the head-node run directory expected by _process_job()."""
    run_dir = tmp_path / "data" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _mock_grpc_client() -> MagicMock:
    """Return a MagicMock that mimics AsyncDaqControlClient."""
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.GenerateManifest = AsyncMock(return_value={"success": True, "file_count": 0})
    client.CleanupData = AsyncMock(return_value={"success": True, "deleted_count": 0})
    return client


@contextmanager
def _mock_grpc_modules(mock_client: MagicMock):
    """Inject fake panoseti_grpc modules into sys.modules so that the
    local import inside _process_job() resolves to our mock.
    """
    stub_root = ModuleType("panoseti_grpc")
    stub_daq = ModuleType("panoseti_grpc.daq_control")
    stub_client_mod = ModuleType("panoseti_grpc.daq_control.client")
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


def _mock_rsync_ok() -> MagicMock:
    """Return a mock subprocess.CompletedProcess representing rsync success."""
    result = MagicMock()
    result.returncode = 0
    result.stderr = ""
    return result


def _mock_rsync_fail(msg: str = "rsync: connection timeout") -> MagicMock:
    """Return a mock subprocess.CompletedProcess representing rsync failure."""
    result = MagicMock()
    result.returncode = 1
    result.stderr = msg
    return result


# ---------------------------------------------------------------------------
# 1. Happy path: full state machine → ARCHIVED
# ---------------------------------------------------------------------------


async def test_process_job_happy_path(tmp_path, monkeypatch):
    """_process_job() drives all stages and returns True on success."""
    monkeypatch.setenv("PSETI_CONTROL", str(tmp_path))
    run_name = "myrun.pffd"
    _make_test_ledger(tmp_path, run_name)
    _make_run_dir(tmp_path, run_name)
    job = _make_job(tmp_path, run_name)

    mock_client = _mock_grpc_client()

    with _mock_grpc_modules(mock_client), \
         patch("control.transfer.daemon.subprocess") as mock_sub:
        mock_sub.run.return_value = _mock_rsync_ok()
        result = await _process_job(job)

    assert result is True
    run_complete = tmp_path / "data" / run_name / "run_complete"
    assert run_complete.exists(), "run_complete marker must be written on success"


# ---------------------------------------------------------------------------
# 2. rsync failure → returns False
# ---------------------------------------------------------------------------


async def test_process_job_rsync_failure(tmp_path, monkeypatch):
    """_process_job() returns False when rsync fails."""
    monkeypatch.setenv("PSETI_CONTROL", str(tmp_path))
    run_name = "myrun.pffd"
    _make_test_ledger(tmp_path, run_name)
    _make_run_dir(tmp_path, run_name)
    job = _make_job(tmp_path, run_name)

    mock_client = _mock_grpc_client()

    with _mock_grpc_modules(mock_client), \
         patch("control.transfer.daemon.subprocess") as mock_sub:
        mock_sub.run.return_value = _mock_rsync_fail()
        result = await _process_job(job)

    assert result is False
    run_complete = tmp_path / "data" / run_name / "run_complete"
    assert not run_complete.exists()


# ---------------------------------------------------------------------------
# 3. no_collect=True skips rsync
# ---------------------------------------------------------------------------


async def test_process_job_no_collect_skips_rsync(tmp_path, monkeypatch):
    """With no_collect=True, rsync is not called and job reaches ARCHIVED."""
    monkeypatch.setenv("PSETI_CONTROL", str(tmp_path))
    run_name = "myrun.pffd"
    _make_test_ledger(tmp_path, run_name)
    _make_run_dir(tmp_path, run_name)
    job = _make_job(tmp_path, run_name, no_collect=True)

    mock_client = _mock_grpc_client()

    with _mock_grpc_modules(mock_client), \
         patch("control.transfer.daemon.subprocess") as mock_sub:
        mock_sub.run.return_value = _mock_rsync_ok()
        result = await _process_job(job)

    assert result is True
    mock_sub.run.assert_not_called()
    run_complete = tmp_path / "data" / run_name / "run_complete"
    assert run_complete.exists()


# ---------------------------------------------------------------------------
# 4. no_cleanup=True skips CleanupData
# ---------------------------------------------------------------------------


async def test_process_job_no_cleanup_skips_cleanup(tmp_path, monkeypatch):
    """With no_cleanup=True, CleanupData is not called on the gRPC client."""
    monkeypatch.setenv("PSETI_CONTROL", str(tmp_path))
    run_name = "myrun.pffd"
    _make_test_ledger(tmp_path, run_name)
    _make_run_dir(tmp_path, run_name)
    job = _make_job(tmp_path, run_name, no_cleanup=True)

    mock_client = _mock_grpc_client()

    with _mock_grpc_modules(mock_client), \
         patch("control.transfer.daemon.subprocess") as mock_sub:
        mock_sub.run.return_value = _mock_rsync_ok()
        result = await _process_job(job)

    assert result is True
    mock_client.CleanupData.assert_not_called()


# ---------------------------------------------------------------------------
# 5. run_complete is idempotent (already exists)
# ---------------------------------------------------------------------------


async def test_process_job_run_complete_idempotent(tmp_path, monkeypatch):
    """If run_complete already exists, _process_job() must not overwrite it."""
    monkeypatch.setenv("PSETI_CONTROL", str(tmp_path))
    run_name = "myrun.pffd"
    _make_test_ledger(tmp_path, run_name)
    run_dir = _make_run_dir(tmp_path, run_name)
    sentinel = "original content"
    (run_dir / "run_complete").write_text(sentinel)

    job = _make_job(tmp_path, run_name, no_collect=True, no_cleanup=True)
    mock_client = _mock_grpc_client()

    with _mock_grpc_modules(mock_client), \
         patch("control.transfer.daemon.subprocess") as mock_sub:
        mock_sub.run.return_value = _mock_rsync_ok()
        result = await _process_job(job)

    assert result is True
    assert (run_dir / "run_complete").read_text() == sentinel


# ---------------------------------------------------------------------------
# 6. no_collect + no_cleanup: no gRPC calls at all
# ---------------------------------------------------------------------------


async def test_process_job_no_collect_no_cleanup_no_grpc(tmp_path, monkeypatch):
    """With both flags True, no DaqControlClient methods are called."""
    monkeypatch.setenv("PSETI_CONTROL", str(tmp_path))
    run_name = "myrun.pffd"
    _make_test_ledger(tmp_path, run_name)
    _make_run_dir(tmp_path, run_name)
    job = _make_job(tmp_path, run_name, no_collect=True, no_cleanup=True)

    mock_client = _mock_grpc_client()

    with _mock_grpc_modules(mock_client), \
         patch("control.transfer.daemon.subprocess") as mock_sub:
        mock_sub.run.return_value = _mock_rsync_ok()
        result = await _process_job(job)

    assert result is True
    mock_client.GenerateManifest.assert_not_called()
    mock_client.CleanupData.assert_not_called()


# ---------------------------------------------------------------------------
# 7. Multiple DAQ nodes: subprocess.run called once per node
# ---------------------------------------------------------------------------


async def test_process_job_multiple_nodes(tmp_path, monkeypatch):
    """subprocess.run (rsync) is called once per DAQ node."""
    monkeypatch.setenv("PSETI_CONTROL", str(tmp_path))
    run_name = "myrun.pffd"
    _make_test_ledger(tmp_path, run_name)
    _make_run_dir(tmp_path, run_name)

    job = TransferJob(
        run_name=run_name,
        head_data_dir=str(tmp_path / "data"),
        head_node_username="panoseti",
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
        no_cleanup=True,
        daq_nodes=[
            TransferNodeSpec(ip_addr="192.168.0.10", username="panoseti", data_dir="/data", module_ids=[250]),
            TransferNodeSpec(ip_addr="192.168.0.20", username="panoseti", data_dir="/data", module_ids=[251]),
        ],
    )

    mock_client = _mock_grpc_client()

    with _mock_grpc_modules(mock_client), \
         patch("control.transfer.daemon.subprocess") as mock_sub:
        mock_sub.run.return_value = _mock_rsync_ok()
        result = await _process_job(job)

    assert result is True
    assert mock_sub.run.call_count == 2


# ---------------------------------------------------------------------------
# 8-10. Lock helpers
# ---------------------------------------------------------------------------


class TestDaemonSingletonLock:
    """Tests for _acquire_transfer_lock / _release_transfer_lock."""

    def test_first_acquire_succeeds(self, tmp_path, monkeypatch) -> None:
        """_acquire_transfer_lock must return a non-None file handle."""
        monkeypatch.setenv("PSETI_LOCKS_DIR", str(tmp_path / "locks"))
        fh = _acquire_transfer_lock()
        assert fh is not None
        _release_transfer_lock(fh)

    def test_second_acquire_fails_while_held(self, tmp_path, monkeypatch) -> None:
        """A second acquire attempt while first holds lock returns None."""
        monkeypatch.setenv("PSETI_LOCKS_DIR", str(tmp_path / "locks"))
        fh1 = _acquire_transfer_lock()
        assert fh1 is not None
        try:
            fh2 = _acquire_transfer_lock()
            assert fh2 is None, "Second lock attempt must fail while first is held"
        finally:
            _release_transfer_lock(fh1)

    def test_release_none_is_noop(self, tmp_path, monkeypatch) -> None:
        """_release_transfer_lock(None) must not raise."""
        monkeypatch.setenv("PSETI_LOCKS_DIR", str(tmp_path / "locks"))
        _release_transfer_lock(None)  # must not raise


# ---------------------------------------------------------------------------
# 11. verify_manifest helper
# ---------------------------------------------------------------------------


class TestVerifyManifest:
    """Tests for the verify_manifest() utility function."""

    def test_sha256_manifest_ok(self, tmp_path) -> None:
        """verify_manifest returns (True, []) for a valid SHA-256 manifest."""
        data = b"hello panoseti"
        data_file = tmp_path / "frame_0.pff"
        data_file.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        size = len(data)
        manifest = tmp_path / "manifest.sha256"
        manifest.write_text(f"{digest}  {size}  0  frame_0.pff\n")

        ok, errs = verify_manifest(manifest, tmp_path)
        assert ok is True
        assert errs == []

    def test_sha256_manifest_corrupt(self, tmp_path) -> None:
        """verify_manifest returns (False, [...]) when a digest is wrong."""
        data_file = tmp_path / "frame_0.pff"
        data_file.write_bytes(b"original")
        # Write a manifest with intentionally wrong digest
        manifest = tmp_path / "manifest.sha256"
        manifest.write_text("deadbeef  8  0  frame_0.pff\n")

        ok, errs = verify_manifest(manifest, tmp_path)
        assert ok is False
        assert len(errs) > 0

    def test_missing_file_in_manifest(self, tmp_path) -> None:
        """verify_manifest fails when a file listed in the manifest is absent."""
        manifest = tmp_path / "manifest.sha256"
        manifest.write_text("abcd1234  0  0  missing_file.pff\n")

        ok, errs = verify_manifest(manifest, tmp_path)
        assert ok is False
