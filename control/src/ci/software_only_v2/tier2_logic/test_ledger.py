# mypy: ignore-errors
"""
test_ledger.py — RunStateManager locking and ledger transition tests.

Ported from ci/software_only/tier2_logic/test_ledger.py.

Each test receives an isolated pseti_workspace so that PSETI_STATE points
to a fresh temporary directory. RunStateManager() (no base_dir argument)
resolves its lock and ledger paths through PanoPaths, which in turn reads
the monkeypatched PSETI_STATE — giving each test complete isolation without
any manual base_dir wiring.
"""

from __future__ import annotations

import pathlib
import socket
from datetime import UTC, datetime

import pytest

from control.utils.pydantic_config_models import RunStateLedger, RunStatus
from control.utils.run_state import LockError, RunStateManager

from ci.software_only_v2.infra.workspace import Workspace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manager(workspace: Workspace) -> RunStateManager:
    """Return a RunStateManager rooted in the workspace's isolated state dir."""
    return RunStateManager(base_dir=workspace.root / "state")


def _ledger(status: RunStatus, pid: int = 1234, run_name: str = "test_run") -> RunStateLedger:
    return RunStateLedger(
        run_name=run_name,
        status=status,
        start_time=datetime.now(UTC).isoformat(),
        pid=pid,
        host=socket.gethostname(),
    )


# ---------------------------------------------------------------------------
# Lock acquisition
# ---------------------------------------------------------------------------

class TestLockAcquisition:
    """RunStateManager advisory locking — mutual exclusion and stale-PID healing."""

    def test_stale_lock_is_self_healed(self, pseti_workspace: Workspace) -> None:
        mgr = _make_manager(pseti_workspace)
        dead_pid = 2 ** 22
        mgr.lock_path.parent.mkdir(parents=True, exist_ok=True)
        mgr.lock_path.write_text(str(dead_pid))

        acquired = mgr.acquire_lock()
        assert acquired, "RunStateManager must self-heal a stale lock (dead PID)"
        mgr.release_lock()

    def test_live_lock_prevents_concurrent_acquisition(
        self, pseti_workspace: Workspace
    ) -> None:
        mgr1 = _make_manager(pseti_workspace)
        mgr2 = _make_manager(pseti_workspace)

        assert mgr1.acquire_lock() is True
        with pytest.raises(LockError):
            mgr2.acquire_lock()
        mgr1.release_lock()

        # After release, mgr2 can now acquire
        assert mgr2.acquire_lock() is True
        mgr2.release_lock()

    def test_lock_file_written_with_current_pid(self, pseti_workspace: Workspace) -> None:
        import os
        mgr = _make_manager(pseti_workspace)
        mgr.acquire_lock()
        try:
            written_pid = int(mgr.lock_path.read_text().strip())
            assert written_pid == os.getpid()
        finally:
            mgr.release_lock()

    def test_lock_file_removed_on_release(self, pseti_workspace: Workspace) -> None:
        mgr = _make_manager(pseti_workspace)
        mgr.acquire_lock()
        assert mgr.lock_path.exists()
        mgr.release_lock()
        assert not mgr.lock_path.exists()


# ---------------------------------------------------------------------------
# Ledger status transitions
# ---------------------------------------------------------------------------

class TestLedgerTransitions:
    """RunStateManager ledger read/write and status transitions."""

    def test_load_state_returns_none_when_no_ledger(
        self, pseti_workspace: Workspace
    ) -> None:
        mgr = _make_manager(pseti_workspace)
        assert mgr.load_state() is None

    def test_save_and_load_roundtrip(self, pseti_workspace: Workspace) -> None:
        mgr = _make_manager(pseti_workspace)
        original = _ledger(RunStatus.STARTING)
        mgr.save_state(original)
        loaded = mgr.load_state()
        assert loaded is not None
        assert loaded.status == RunStatus.STARTING
        assert loaded.run_name == original.run_name

    def test_transition_to_aborted(self, pseti_workspace: Workspace) -> None:
        mgr = _make_manager(pseti_workspace)
        mgr.save_state(_ledger(RunStatus.STARTING, run_name="abort_run"))
        mgr.transition(RunStatus.ABORTED)
        updated = mgr.load_state()
        assert updated is not None
        assert updated.status == RunStatus.ABORTED

    def test_transition_to_active(self, pseti_workspace: Workspace) -> None:
        mgr = _make_manager(pseti_workspace)
        mgr.save_state(_ledger(RunStatus.STARTING))
        mgr.transition(RunStatus.ACTIVE)
        assert mgr.load_state().status == RunStatus.ACTIVE

    def test_full_lifecycle_transition_sequence(self, pseti_workspace: Workspace) -> None:
        mgr = _make_manager(pseti_workspace)
        mgr.save_state(_ledger(RunStatus.STARTING))

        for status in (
            RunStatus.ACTIVE,
            RunStatus.STOPPING,
            RunStatus.RECORDING_ENDED,
            RunStatus.TRANSFERRING,
            RunStatus.ARCHIVED,
        ):
            mgr.transition(status)
            assert mgr.load_state().status == status


# ---------------------------------------------------------------------------
# Stale ledger detection
# ---------------------------------------------------------------------------

class TestStaleLedgerDetection:
    """Verify that a ledger with a dead PID is detectable (mimics start.py logic)."""

    def test_stale_active_ledger_has_dead_pid(self, pseti_workspace: Workspace) -> None:
        mgr = _make_manager(pseti_workspace)
        dead_pid = 2 ** 22
        stale = _ledger(RunStatus.ACTIVE, pid=dead_pid, run_name="stale_run")
        mgr.save_state(stale)

        current = mgr.load_state()
        assert current is not None
        assert current.status == RunStatus.ACTIVE

        # Mimics start.py's is-process-alive check
        try:
            import os
            os.kill(dead_pid, 0)
            is_alive = True
        except (ProcessLookupError, PermissionError):
            is_alive = False
        assert not is_alive, "The dead_pid must not correspond to a live process"

    def test_two_workspaces_have_isolated_ledgers(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ledger isolation: two different base_dirs must not share state."""
        state_a = tmp_path / "state_a"
        state_b = tmp_path / "state_b"
        for d in (state_a / "runs", state_a / "locks",
                  state_b / "runs", state_b / "locks"):
            d.mkdir(parents=True)

        mgr_a = RunStateManager(base_dir=state_a)
        mgr_b = RunStateManager(base_dir=state_b)

        mgr_a.save_state(_ledger(RunStatus.ACTIVE, run_name="run_a"))

        # mgr_b has no state
        assert mgr_b.load_state() is None
        # mgr_a state is unchanged
        assert mgr_a.load_state().run_name == "run_a"
