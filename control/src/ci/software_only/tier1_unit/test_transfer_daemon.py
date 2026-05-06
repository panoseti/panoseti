# mypy: ignore-errors
"""
test_transfer_daemon.py

Unit tests for the transfer daemon state machine, lock helpers, and
verify_manifest utility.
"""

from __future__ import annotations

import hashlib
import pathlib
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock

from ci.fixtures.rsync_fixtures import RsyncMock
from control.transfer.daemon import (
    _acquire_transfer_lock,
    _process_job,
    _release_transfer_lock,
)
from control.transfer.models import TransferNodeSpec
from control.transfer.verify import verify_manifest
from control.utils.pydantic_config_models import RunStateLedger, RunStatus
from control.utils.run_state import RunStateManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    """Inject fake panoseti_grpc modules into sys.modules."""
    stub_root = ModuleType("panoseti_grpc")
    stub_daq = ModuleType("panoseti_grpc.daq_control")
    stub_client_mod = ModuleType("panoseti_grpc.daq_control.client")
    stub_client_mod.AsyncDaqControlClient = MagicMock(return_value=mock_client)
    stub_root.daq_control = stub_daq
    stub_daq.client = stub_client_mod

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


# ---------------------------------------------------------------------------
# 1. Happy path: full state machine → ARCHIVED
# ---------------------------------------------------------------------------

async def test_process_job_happy_path(mock_workspace, transfer_job_factory, mock_rsync_transfer: RsyncMock):
    """_process_job() drives all stages and returns True on success."""
    run_name = "myrun.pffd"
    state_mgr = RunStateManager()
    state_mgr.save_state(RunStateLedger(
        run_name=run_name, status=RunStatus.RECORDING_ENDED, start_time=datetime.now(UTC).isoformat()
    ))
    
    # Create the head-node run directory and a dummy manifest
    from control.utils import config_file
    head_data_dir = pathlib.Path(config_file.get_daq_config().head_node_data_dir)
    run_dir = head_data_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "dp_manifest.node_test.algo_blake3.txt").touch()

    job = transfer_job_factory(run_name=run_name)
    mock_client = _mock_grpc_client()

    with _mock_grpc_modules(mock_client):
        import asyncio as _asyncio
        result = await _process_job(job, _asyncio.Event(), state_mgr)

    ok, _ = result
    assert ok is True
    assert (run_dir / "run_complete").exists()


# ---------------------------------------------------------------------------
# 2. rsync failure → returns False
# ---------------------------------------------------------------------------

async def test_process_job_rsync_failure(mock_workspace, transfer_job_factory, mock_rsync_transfer: RsyncMock):
    """_process_job() returns False when rsync fails."""
    run_name = "myrun.pffd"
    state_mgr = RunStateManager()
    state_mgr.save_state(RunStateLedger(
        run_name=run_name, status=RunStatus.RECORDING_ENDED, start_time=datetime.now(UTC).isoformat()
    ))
    
    from control.utils import config_file
    head_data_dir = pathlib.Path(config_file.get_daq_config().head_node_data_dir)
    run_dir = head_data_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "dp_manifest.node_test.algo_blake3.txt").touch()

    job = transfer_job_factory(run_name=run_name)
    mock_rsync_transfer.side_effect = mock_rsync_transfer.mock_process_fail
    
    mock_client = _mock_grpc_client()

    with _mock_grpc_modules(mock_client):
        import asyncio as _asyncio
        result = await _process_job(job, _asyncio.Event(), state_mgr)

    ok, _ = result
    assert ok is False
    assert not (run_dir / "run_complete").exists()


# ---------------------------------------------------------------------------
# 3. no_collect=True skips rsync
# ---------------------------------------------------------------------------

async def test_process_job_no_collect_skips_rsync(mock_workspace, transfer_job_factory, mock_rsync_transfer: RsyncMock):
    """With no_collect=True, rsync is not called and job reaches ARCHIVED."""
    run_name = "myrun.pffd"
    state_mgr = RunStateManager()
    state_mgr.save_state(RunStateLedger(
        run_name=run_name, status=RunStatus.RECORDING_ENDED, start_time=datetime.now(UTC).isoformat()
    ))
    
    from control.utils import config_file
    head_data_dir = pathlib.Path(config_file.get_daq_config().head_node_data_dir)
    run_dir = head_data_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "dp_manifest.node_test.algo_blake3.txt").touch()

    job = transfer_job_factory(run_name=run_name, no_collect=True)
    mock_client = _mock_grpc_client()

    with _mock_grpc_modules(mock_client):
        import asyncio as _asyncio
        result = await _process_job(job, _asyncio.Event(), state_mgr)

    ok, _ = result
    assert ok is True
    assert mock_rsync_transfer.call_count == 0
    assert (run_dir / "run_complete").exists()


# ---------------------------------------------------------------------------
# 4. no_cleanup=True skips CleanupData
# ---------------------------------------------------------------------------

async def test_process_job_no_cleanup_skips_cleanup(mock_workspace, transfer_job_factory, mock_rsync_transfer: RsyncMock):
    """With no_cleanup=True, CleanupData is not called on the gRPC client."""
    run_name = "myrun.pffd"
    state_mgr = RunStateManager()
    state_mgr.save_state(RunStateLedger(
        run_name=run_name, status=RunStatus.RECORDING_ENDED, start_time=datetime.now(UTC).isoformat()
    ))
    
    from control.utils import config_file
    head_data_dir = pathlib.Path(config_file.get_daq_config().head_node_data_dir)
    run_dir = head_data_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "dp_manifest.node_test.algo_blake3.txt").touch()

    job = transfer_job_factory(run_name=run_name, no_cleanup=True)
    mock_client = _mock_grpc_client()

    with _mock_grpc_modules(mock_client):
        import asyncio as _asyncio
        result = await _process_job(job, _asyncio.Event(), state_mgr)

    ok, _ = result
    assert ok is True
    mock_client.CleanupData.assert_not_called()


# ---------------------------------------------------------------------------
# 5. run_complete is idempotent (already exists)
# ---------------------------------------------------------------------------

async def test_process_job_run_complete_idempotent(mock_workspace, transfer_job_factory, mock_rsync_transfer: RsyncMock):
    """If run_complete already exists, _process_job() must not overwrite it."""
    run_name = "myrun.pffd"
    state_mgr = RunStateManager()
    state_mgr.save_state(RunStateLedger(
        run_name=run_name, status=RunStatus.RECORDING_ENDED, start_time=datetime.now(UTC).isoformat()
    ))
    
    from control.utils import config_file
    head_data_dir = pathlib.Path(config_file.get_daq_config().head_node_data_dir)
    run_dir = head_data_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "dp_manifest.node_test.algo_blake3.txt").touch()
    sentinel = "original content"
    (run_dir / "run_complete").write_text(sentinel)

    job = transfer_job_factory(run_name=run_name, no_collect=True, no_cleanup=True)
    mock_client = _mock_grpc_client()

    with _mock_grpc_modules(mock_client):
        import asyncio as _asyncio
        result = await _process_job(job, _asyncio.Event(), state_mgr)

    ok, _ = result
    assert ok is True
    assert (run_dir / "run_complete").read_text() == sentinel


# ---------------------------------------------------------------------------
# 7. Multiple DAQ nodes: subprocess.run called once per node
# ---------------------------------------------------------------------------

async def test_process_job_multiple_nodes(mock_workspace, transfer_job_factory, mock_rsync_transfer: RsyncMock):
    """rsync is called once per DAQ node."""
    run_name = "myrun.pffd"
    state_mgr = RunStateManager()
    state_mgr.save_state(RunStateLedger(
        run_name=run_name, status=RunStatus.RECORDING_ENDED, start_time=datetime.now(UTC).isoformat()
    ))
    
    from control.utils import config_file
    head_data_dir = pathlib.Path(config_file.get_daq_config().head_node_data_dir)
    run_dir = head_data_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "dp_manifest.node_test.algo_blake3.txt").touch()

    job = transfer_job_factory(
        run_name=run_name,
        no_cleanup=True,
        daq_nodes=[
            TransferNodeSpec(ip_addr="192.168.0.10", username="panoseti", data_dir="/data", module_ids=[250]),
            TransferNodeSpec(ip_addr="192.168.0.20", username="panoseti", data_dir="/data", module_ids=[251]),
        ],
    )

    mock_client = _mock_grpc_client()

    with _mock_grpc_modules(mock_client):
        import asyncio as _asyncio
        result = await _process_job(job, _asyncio.Event(), state_mgr)

    ok, _ = result
    assert ok is True
    assert mock_rsync_transfer.call_count == 2


# ---------------------------------------------------------------------------
# 8-10. Lock helpers
# ---------------------------------------------------------------------------

class TestDaemonSingletonLock:
    """Tests for _acquire_transfer_lock / _release_transfer_lock."""

    def test_first_acquire_succeeds(self, mock_workspace) -> None:
        """_acquire_transfer_lock must return a non-None file handle."""
        fh = _acquire_transfer_lock()
        assert fh is not None
        _release_transfer_lock(fh)

    def test_second_acquire_fails_while_held(self, mock_workspace) -> None:
        """A second acquire attempt while first holds lock returns None."""
        fh1 = _acquire_transfer_lock()
        assert fh1 is not None
        try:
            fh2 = _acquire_transfer_lock()
            assert fh2 is None, "Second lock attempt must fail while first is held"
        finally:
            _release_transfer_lock(fh1)

    def test_release_none_is_noop(self, mock_workspace) -> None:
        """_release_transfer_lock(None) must not raise."""
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

        ok, _errs = verify_manifest(manifest, tmp_path)
        assert ok is False
