# mypy: ignore-errors
"""
test_transfer_daemon.py

Phase 3 unit tests for the transfer daemon state machine, lock helpers, and
verify_manifest utility.

All tests are hardware-agnostic: gRPC and rsync are mocked; the filesystem is
provided by pytest's tmp_path fixture.
"""

from __future__ import annotations

import asyncio
import hashlib
import pathlib
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from utils.pydantic_config_models import RunStateLedger
from utils.run_state import RunStateManager
from utils.transfer.daemon import (
    _acquire_transfer_lock,
    _process_job,
    _release_transfer_lock,
)
from utils.transfer.verify import verify_manifest


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
) -> dict:
    """Return a minimal job dict for _process_job()."""
    return {
        "run_name": run_name,
        "head_data_dir": str(tmp_path / "data"),
        "daq_nodes": [
            {
                "ip_addr": "192.168.0.10",
                "data_dir": "/app/data",
                "module_ids": [250],
            }
        ],
        "no_collect": no_collect,
        "no_cleanup": no_cleanup,
    }


def _make_run_dir(tmp_path: pathlib.Path, run_name: str = "myrun.pffd") -> pathlib.Path:
    """Create the head-node run directory expected by _process_job()."""
    run_dir = tmp_path / "data" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _mock_grpc_client() -> MagicMock:
    """Return a MagicMock that mimics DaqControlClient."""
    client = MagicMock()
    client.GenerateManifest.return_value = {"success": True, "file_count": 0}
    client.CleanupData.return_value = {"success": True, "deleted_count": 0}
    return client


@contextmanager
def _mock_grpc_modules(mock_client: MagicMock):
    """Inject fake panoseti_grpc modules into sys.modules so that the
    local import inside _process_job() resolves to our mock.

    The daemon does:
        from panoseti_grpc.daq_control.client import DaqControlClient
    inside the function body, which bypasses normal module-level patching.
    We must pre-populate sys.modules with stub module objects.
    """
    stub_root = ModuleType("panoseti_grpc")
    stub_daq = ModuleType("panoseti_grpc.daq_control")
    stub_client_mod = ModuleType("panoseti_grpc.daq_control.client")
    stub_client_mod.DaqControlClient = type(
        "DaqControlClient", (), {"__init__": lambda self, **kw: None}
    )

    # DaqControlClient constructor returns mock_client regardless of args.
    stub_client_mod.DaqControlClient = MagicMock(return_value=mock_client)

    stub_root.daq_control = stub_daq  # type: ignore[attr-defined]
    stub_daq.client = stub_client_mod  # type: ignore[attr-defined]

    injected = {
        "panoseti_grpc": stub_root,
        "panoseti_grpc.daq_control": stub_daq,
        "panoseti_grpc.daq_control.client": stub_client_mod,
    }
    # Only inject keys that are not already in sys.modules (avoid overwriting
    # a real installation).
    prev: dict = {}
    for key, mod in injected.items():
        prev[key] = sys.modules.get(key)
        sys.modules[key] = mod
    try:
        yield stub_client_mod.DaqControlClient
    finally:
        for key, original in prev.items():
            if original is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = original


# ---------------------------------------------------------------------------
# 1. Happy path: full state machine → ARCHIVED
# ---------------------------------------------------------------------------


async def test_process_job_happy_path(tmp_path):
    """_process_job() drives all stages and returns True on success."""
    run_name = "myrun.pffd"
    _make_test_ledger(tmp_path, run_name)
    _make_run_dir(tmp_path, run_name)
    job = _make_job(tmp_path, run_name)

    mock_client = _mock_grpc_client()

    with _mock_grpc_modules(mock_client), \
         patch("utils.transfer.daemon.rsync_one_node", return_value=(True, "")):
        result = await _process_job(job, tmp_path)

    assert result is True
    run_complete = tmp_path / "data" / run_name / "run_complete"
    assert run_complete.exists(), "run_complete marker must be written on success"


# ---------------------------------------------------------------------------
# 2. rsync failure → returns False
# ---------------------------------------------------------------------------


async def test_process_job_rsync_failure(tmp_path):
    """_process_job() returns False when rsync fails."""
    run_name = "myrun.pffd"
    _make_test_ledger(tmp_path, run_name)
    _make_run_dir(tmp_path, run_name)
    job = _make_job(tmp_path, run_name)

    mock_client = _mock_grpc_client()

    with _mock_grpc_modules(mock_client), \
         patch("utils.transfer.daemon.rsync_one_node", return_value=(False, "rsync: connection timeout")):
        result = await _process_job(job, tmp_path)

    assert result is False
    # run_complete must NOT be written on failure
    run_complete = tmp_path / "data" / run_name / "run_complete"
    assert not run_complete.exists()


# ---------------------------------------------------------------------------
# 3. no_collect=True skips rsync
# ---------------------------------------------------------------------------


async def test_process_job_no_collect_skips_rsync(tmp_path):
    """With no_collect=True, rsync is not called and job reaches ARCHIVED."""
    run_name = "myrun.pffd"
    _make_test_ledger(tmp_path, run_name)
    _make_run_dir(tmp_path, run_name)
    job = _make_job(tmp_path, run_name, no_collect=True)

    mock_client = _mock_grpc_client()
    mock_rsync = MagicMock(return_value=(True, ""))

    with _mock_grpc_modules(mock_client), \
         patch("utils.transfer.daemon.rsync_one_node", mock_rsync):
        result = await _process_job(job, tmp_path)

    assert result is True
    mock_rsync.assert_not_called()
    run_complete = tmp_path / "data" / run_name / "run_complete"
    assert run_complete.exists()


# ---------------------------------------------------------------------------
# 4. no_cleanup=True skips CleanupData
# ---------------------------------------------------------------------------


async def test_process_job_no_cleanup_skips_cleanup(tmp_path):
    """With no_cleanup=True, CleanupData is not called on the gRPC client."""
    run_name = "myrun.pffd"
    _make_test_ledger(tmp_path, run_name)
    _make_run_dir(tmp_path, run_name)
    job = _make_job(tmp_path, run_name, no_cleanup=True)

    mock_client = _mock_grpc_client()

    with _mock_grpc_modules(mock_client), \
         patch("utils.transfer.daemon.rsync_one_node", return_value=(True, "")):
        result = await _process_job(job, tmp_path)

    assert result is True
    mock_client.CleanupData.assert_not_called()


# ---------------------------------------------------------------------------
# 5. run_complete is idempotent (already exists)
# ---------------------------------------------------------------------------


async def test_process_job_run_complete_idempotent(tmp_path):
    """If run_complete already exists, _process_job() must not overwrite it."""
    run_name = "myrun.pffd"
    _make_test_ledger(tmp_path, run_name)
    run_dir = _make_run_dir(tmp_path, run_name)
    sentinel = "original content"
    (run_dir / "run_complete").write_text(sentinel)

    job = _make_job(tmp_path, run_name, no_collect=True, no_cleanup=True)
    mock_client = _mock_grpc_client()

    with _mock_grpc_modules(mock_client), \
         patch("utils.transfer.daemon.rsync_one_node", return_value=(True, "")):
        result = await _process_job(job, tmp_path)

    assert result is True
    assert (run_dir / "run_complete").read_text() == sentinel


# ---------------------------------------------------------------------------
# 6. no_collect + no_cleanup: no gRPC calls at all
# ---------------------------------------------------------------------------


async def test_process_job_no_collect_no_cleanup_no_grpc(tmp_path):
    """With both flags True, no DaqControlClient methods are called."""
    run_name = "myrun.pffd"
    _make_test_ledger(tmp_path, run_name)
    _make_run_dir(tmp_path, run_name)
    job = _make_job(tmp_path, run_name, no_collect=True, no_cleanup=True)

    mock_client = _mock_grpc_client()

    with _mock_grpc_modules(mock_client), \
         patch("utils.transfer.daemon.rsync_one_node", return_value=(True, "")):
        result = await _process_job(job, tmp_path)

    assert result is True
    mock_client.GenerateManifest.assert_not_called()
    mock_client.CleanupData.assert_not_called()


# ---------------------------------------------------------------------------
# 7. Multiple DAQ nodes: rsync called once per node
# ---------------------------------------------------------------------------


async def test_process_job_multiple_nodes(tmp_path):
    """rsync_one_node is called once per DAQ node in daq_nodes list."""
    run_name = "myrun.pffd"
    _make_test_ledger(tmp_path, run_name)
    _make_run_dir(tmp_path, run_name)

    job = {
        "run_name": run_name,
        "head_data_dir": str(tmp_path / "data"),
        "daq_nodes": [
            {"ip_addr": "192.168.0.10", "data_dir": "/app/data", "module_ids": [250]},
            {"ip_addr": "192.168.0.20", "data_dir": "/app/data", "module_ids": [251]},
        ],
        "no_collect": False,
        "no_cleanup": True,
    }

    mock_client = _mock_grpc_client()
    mock_rsync = MagicMock(return_value=(True, ""))

    with _mock_grpc_modules(mock_client), \
         patch("utils.transfer.daemon.rsync_one_node", mock_rsync):
        result = await _process_job(job, tmp_path)

    assert result is True
    assert mock_rsync.call_count == 2


# ---------------------------------------------------------------------------
# 8–10. Lock helpers
# ---------------------------------------------------------------------------


class TestDaemonSingletonLock:
    """Tests for _acquire_transfer_lock / _release_transfer_lock."""

    def test_first_acquire_succeeds(self, tmp_path):
        """_acquire_transfer_lock must return a non-None file handle."""
        fh = _acquire_transfer_lock(tmp_path)
        assert fh is not None
        _release_transfer_lock(fh)

    def test_second_acquire_fails_while_held(self, tmp_path):
        """A second acquire attempt while first holds lock returns None."""
        fh1 = _acquire_transfer_lock(tmp_path)
        assert fh1 is not None
        try:
            fh2 = _acquire_transfer_lock(tmp_path)
            assert fh2 is None, "Second lock attempt must fail while first is held"
        finally:
            _release_transfer_lock(fh1)

    def test_acquire_succeeds_after_release(self, tmp_path):
        """After releasing the first lock, a third acquire must succeed."""
        fh1 = _acquire_transfer_lock(tmp_path)
        _release_transfer_lock(fh1)

        fh3 = _acquire_transfer_lock(tmp_path)
        assert fh3 is not None
        _release_transfer_lock(fh3)

    def test_release_none_is_noop(self):
        """_release_transfer_lock(None) must not raise."""
        _release_transfer_lock(None)  # should be silently ignored

    def test_lock_file_created_in_tmp(self, tmp_path):
        """The lock file is written to tmp/panoseti_transfer.lock under base_dir."""
        fh = _acquire_transfer_lock(tmp_path)
        assert fh is not None
        try:
            lock_path = tmp_path / "tmp" / "panoseti_transfer.lock"
            assert lock_path.exists()
        finally:
            _release_transfer_lock(fh)


# ---------------------------------------------------------------------------
# 11–13. verify_manifest
# ---------------------------------------------------------------------------


class TestVerifyManifest:

    def _write_manifest(
        self,
        manifest_path: pathlib.Path,
        entries: list[tuple[str, str]],
    ) -> None:
        """Write a manifest file with ``<digest>  <size>  <relpath>`` lines."""
        lines = [f"{digest}  {size}  {relpath}" for digest, size, relpath in entries]
        manifest_path.write_text("\n".join(lines))

    def test_verify_manifest_success(self, tmp_path):
        """verify_manifest returns (True, []) when all digests match."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        content = b"hello pff world"
        (data_dir / "test.pff").write_bytes(content)

        digest = hashlib.sha256(content).hexdigest()
        manifest = tmp_path / "manifest.txt"
        self._write_manifest(manifest, [(digest, str(len(content)), "test.pff")])

        ok, errors = verify_manifest(manifest, data_dir)
        assert ok is True
        assert errors == []

    def test_verify_manifest_missing_file(self, tmp_path):
        """verify_manifest returns (False, [error]) when a file is missing."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        manifest = tmp_path / "manifest.txt"
        self._write_manifest(
            manifest,
            [("a" * 64, "0", "ghost.pff")],
        )

        ok, errors = verify_manifest(manifest, data_dir)
        assert ok is False
        assert len(errors) == 1
        assert "Missing" in errors[0] or "ghost.pff" in errors[0]

    def test_verify_manifest_digest_mismatch(self, tmp_path):
        """verify_manifest returns (False, [error]) on digest mismatch."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "real.pff").write_bytes(b"real content")

        wrong_digest = "b" * 64  # valid length but wrong value
        manifest = tmp_path / "manifest.txt"
        self._write_manifest(manifest, [(wrong_digest, "12", "real.pff")])

        ok, errors = verify_manifest(manifest, data_dir)
        assert ok is False
        assert len(errors) == 1
        assert "mismatch" in errors[0].lower() or "real.pff" in errors[0]

    def test_verify_manifest_not_found(self, tmp_path):
        """verify_manifest returns (False, [error]) when manifest file is absent."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        missing_manifest = tmp_path / "nonexistent.txt"

        ok, errors = verify_manifest(missing_manifest, data_dir)
        assert ok is False
        assert len(errors) == 1

    def test_verify_manifest_empty_file(self, tmp_path):
        """verify_manifest returns (True, []) for an empty manifest (zero entries)."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        manifest = tmp_path / "manifest.txt"
        manifest.write_text("")

        ok, errors = verify_manifest(manifest, data_dir)
        assert ok is True
        assert errors == []

    def test_verify_manifest_multiple_files(self, tmp_path):
        """verify_manifest validates all files; partial mismatch → False."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        good = b"good data"
        bad = b"bad data"
        (data_dir / "good.pff").write_bytes(good)
        (data_dir / "bad.pff").write_bytes(bad)

        good_digest = hashlib.sha256(good).hexdigest()
        wrong_digest = "c" * 64

        manifest = tmp_path / "manifest.txt"
        self._write_manifest(
            manifest,
            [
                (good_digest, str(len(good)), "good.pff"),
                (wrong_digest, str(len(bad)), "bad.pff"),
            ],
        )

        ok, errors = verify_manifest(manifest, data_dir)
        assert ok is False
        assert len(errors) == 1
        assert "bad.pff" in errors[0]

    def test_verify_manifest_4col_format(self, tmp_path):
        """verify_manifest handles 4-column format with mtime_ns field."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        content = b"four column test"
        (data_dir / "img.pff").write_bytes(content)

        digest = hashlib.sha256(content).hexdigest()
        manifest = tmp_path / "manifest.txt"
        manifest.write_text(f"{digest}  {len(content)}  1234567890  img.pff\n")

        ok, errors = verify_manifest(manifest, data_dir)
        assert ok is True
        assert errors == []
