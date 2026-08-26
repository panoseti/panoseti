"""
test_stop_fast_path.py

Phase 2 RED tests for the decoupled stop() fast-path.

Target design (Phase 2):
- stop_run() completes quickly without blocking rsync
- Ledger ends in RECORDING_ENDED (not COMPLETED)
- run_complete marker is NOT written (for the right reason: Phase 2 doesn't write it)
- A pending TransferQueue job IS created

All test_stop_* tests must FAIL RED on the current codebase:
  - test_stop_completes_quickly    → fails because ledger status is COMPLETED (not RECORDING_ENDED)
  - test_stop_enqueues_transfer_job → fails because no transfer job is created
  - test_stop_ledger_recording_ended → fails because status is not RECORDING_ENDED
  - test_stop_does_not_write_run_complete → fails because run_complete IS written in current code

test_stop_does_not_write_run_complete will currently PASS for the wrong reason (run_dir not
found path) — but Phase 2 must ensure it passes for the right reason (fast-path doesn't write it
even when the run_dir exists).  We keep it as a RED assertion marker.
"""

from __future__ import annotations

import pathlib
import time
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

import control.stop as stop
from control.utils.pydantic_config_models import (
    DaqConfig,
    NetworkConfig,
    QuaboUids,
    RunStateLedger,
    RunStatus,
)
from control.utils.run_state import RunStateManager

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RUN_NAME = "start_2024-01-01T00:00:00Z.sci"
FAST_PATH_TIMEOUT_SECS = 5.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def daq_config(tmp_path, monkeypatch) -> DaqConfig:
    """
    Minimal DaqConfig whose data_dir points to tmp_path so the run_dir
    computed inside StopTransaction matches the one we create on disk.
    Also isolates PSETI_STATE to tmp_path for TransferQueue.
    """
    monkeypatch.setenv("PSETI_STATE", str(tmp_path))
    monkeypatch.setenv("PSETI_TQ_DIR", str(tmp_path / "state" / "transfer" / "queue"))

    return DaqConfig(

        head_node_data_dir=str(tmp_path),
        head_node_ip_addr="127.0.0.1",
        head_node_container=True,
        daq_nodes=[
            {
                "username": "panoseti",
                "data_dir": str(tmp_path),
                "ip_addr": "127.0.0.2",
                "module_ids": [254],
                "bindhost": "lo",
            }
        ],
    )


@pytest.fixture
def daq_config_with_grpc_port(tmp_path, monkeypatch) -> DaqConfig:
    """Same as daq_config, but the DAQ node carries an explicit grpc_port override."""
    monkeypatch.setenv("PSETI_STATE", str(tmp_path))
    monkeypatch.setenv("PSETI_TQ_DIR", str(tmp_path / "state" / "transfer" / "queue"))

    return DaqConfig(
        head_node_data_dir=str(tmp_path),
        head_node_ip_addr="127.0.0.1",
        head_node_container=True,
        daq_nodes=[
            {
                "username": "panoseti",
                "data_dir": str(tmp_path),
                "ip_addr": "127.0.0.2",
                "module_ids": [254],
                "bindhost": "lo",
                "grpc_port": 50099,
            }
        ],
    )


@pytest.fixture
def network_config() -> NetworkConfig:
    return NetworkConfig()


@pytest.fixture
def quabo_uids() -> QuaboUids:
    return QuaboUids(domes=[])


@pytest.fixture
def run_dir(tmp_path) -> pathlib.Path:
    """Create a fake run directory so StopTransaction doesn't hit the 'not found' branch."""
    d = tmp_path / RUN_NAME
    d.mkdir(parents=True)
    return d


@pytest.fixture
def state_mgr(tmp_path, run_dir) -> RunStateManager:
    """
    RunStateManager pre-loaded with an ACTIVE ledger.
    run_dir fixture is included to ensure the run directory exists before the test.
    """
    mgr = RunStateManager(base_dir=str(tmp_path))
    ledger = RunStateLedger(
        run_name=RUN_NAME,
        status=RunStatus.ACTIVE,
        start_time="2024-01-01T00:00:00Z"
    )

    mgr.save_state(ledger)
    return mgr


# ---------------------------------------------------------------------------
# Context manager that applies all the standard mocks for stop_run()
# ---------------------------------------------------------------------------

def _stop_patches(state_mgr: RunStateManager):
    """
    Return a context manager (ExitStack-compatible) that stubs out all
    network/hardware calls inside stop.py, redirecting state management
    to the provided RunStateManager.
    """
    stack = ExitStack()
    stack.enter_context(patch("control.stop.util.local_ip", return_value=["127.0.0.1"]))
    stack.enter_context(patch("control.stop.util.stop_data_flow", return_value=None))
    stack.enter_context(patch("control.stop.util.remove_run_name", return_value=None))
    stack.enter_context(patch("control.stop.util.read_run_name", return_value=RUN_NAME))
    stack.enter_context(patch("control.stop.config_file.get_data_config", return_value=MagicMock()))

    # Redirect RunStateManager to our tmp_path-based instance
    stack.enter_context(patch("control.stop.RunStateManager", return_value=state_mgr))
    return stack


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStopFastPath:
    """
    Phase 2 fast-path: stop_run() must finish quickly, leave the ledger in
    RECORDING_ENDED, NOT write run_complete, and enqueue a transfer job.
    """

    async def test_stop_completes_quickly(
        self, tmp_path, run_dir, state_mgr, daq_config, network_config, quabo_uids
    ):
        """stop_run() must complete in under 5 s."""
        from ci.fixtures.adapters.fake_adapters import (
            FakeFileSystemManager,
            FakeNetworkClient,
            FakeProcessManager,
        )
        with _stop_patches(state_mgr):
            t0 = time.monotonic()
            await stop.stop_run(
                daq_config,
                network_config,
                quabo_uids,
                FakeProcessManager(),
                FakeNetworkClient(),
                FakeFileSystemManager(),
                run=RUN_NAME,
                no_collect=True,
                no_cleanup=True,
            )
            elapsed = time.monotonic() - t0

        assert elapsed < FAST_PATH_TIMEOUT_SECS, (
            f"stop_run() took {elapsed:.2f}s — too slow for fast path"
        )

    async def test_stop_does_not_write_run_complete(
        self, tmp_path, run_dir, state_mgr, daq_config, network_config, quabo_uids
    ):
        """run_complete marker must NOT be written during the fast stop path."""
        from ci.fixtures.adapters.fake_adapters import (
            FakeFileSystemManager,
            FakeNetworkClient,
            FakeProcessManager,
        )
        with _stop_patches(state_mgr):
            await stop.stop_run(
                daq_config,
                network_config,
                quabo_uids,
                FakeProcessManager(),
                FakeNetworkClient(),
                FakeFileSystemManager(),
                run=RUN_NAME,
                no_collect=True,
                no_cleanup=True,
            )

        run_complete = run_dir / "run_complete"
        assert not run_complete.exists(), (
            "run_complete marker was written — Phase 2 fast-path must NOT write it. "
            "Transfer ownership moves to the background TransferWorker."
        )

    async def test_stop_enqueues_transfer_job(
        self, tmp_path, run_dir, state_mgr, daq_config, network_config, quabo_uids
    ):
        """After stop_run(), a pending transfer job must exist in TransferQueue."""
        from ci.fixtures.adapters.fake_adapters import (
            FakeFileSystemManager,
            FakeNetworkClient,
            FakeProcessManager,
        )
        with _stop_patches(state_mgr):
            await stop.stop_run(
                daq_config,
                network_config,
                quabo_uids,
                FakeProcessManager(),
                FakeNetworkClient(),
                FakeFileSystemManager(),
                run=RUN_NAME,
                no_collect=True,
                no_cleanup=True,
            )

        pending_job = tmp_path / "state" / "transfer" / "queue" / "pending" / f"{RUN_NAME}.job.toml"
        assert pending_job.exists(), (
            f"Expected pending transfer job at {pending_job}.\n"
            "stop_run() must enqueue the run for background transfer after hardware teardown."
        )

    async def test_stop_enqueues_transfer_job_with_grpc_port(
        self, tmp_path, run_dir, state_mgr, daq_config_with_grpc_port, network_config, quabo_uids
    ):
        """A node's daq_config.json grpc_port override must be snapshotted into the job TOML."""
        import tomllib

        from ci.fixtures.adapters.fake_adapters import (
            FakeFileSystemManager,
            FakeNetworkClient,
            FakeProcessManager,
        )
        with _stop_patches(state_mgr):
            await stop.stop_run(
                daq_config_with_grpc_port,
                network_config,
                quabo_uids,
                FakeProcessManager(),
                FakeNetworkClient(),
                FakeFileSystemManager(),
                run=RUN_NAME,
                no_collect=True,
                no_cleanup=True,
            )

        pending_job = tmp_path / "state" / "transfer" / "queue" / "pending" / f"{RUN_NAME}.job.toml"
        data = tomllib.loads(pending_job.read_text())

        assert data["daq_nodes"][0]["grpc_port"] == 50099, (
            "stop_run() must carry the DAQ node's daq_config.json grpc_port override "
            "into the TransferNodeSpec written to the job TOML."
        )

    async def test_stop_ledger_recording_ended(
        self, tmp_path, run_dir, state_mgr, daq_config, network_config, quabo_uids
    ):
        """Ledger status must be RECORDING_ENDED after the fast stop path."""
        from ci.fixtures.adapters.fake_adapters import (
            FakeFileSystemManager,
            FakeNetworkClient,
            FakeProcessManager,
        )
        with _stop_patches(state_mgr):
            await stop.stop_run(
                daq_config,
                network_config,
                quabo_uids,
                FakeProcessManager(),
                FakeNetworkClient(),
                FakeFileSystemManager(),
                run=RUN_NAME,
                no_collect=True,
                no_cleanup=True,
            )

        loaded = state_mgr.load_state()
        assert loaded is not None, "Ledger must exist after stop_run()"
        assert loaded.status == RunStatus.RECORDING_ENDED, (
            f"Expected RECORDING_ENDED but got {loaded.status!r}.\n"
            "The current codebase sets COMPLETED — Phase 2 must change this."
        )
