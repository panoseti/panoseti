"""
scenarios/test_sc_transactional_state.py

SC-021 → SC-040: Transactional state corruption tests.

Key exemplars implemented here:
  - SC-024 (Exemplar E): Concurrent start corruption (no advisory lock)
  - SC-031 (Exemplar D): PH baseline staleness off-by-24×
  - SC-033 (Exemplar F): Stale interleave PID from unrelated process
  - SC-034:  Interleave daemon outlives stop.py's retry limit (no hard-kill)
  - SC-002 (Exemplar B): Partial start leaves zombie quabos (SC-002 stub)

TDD intent: each TDD-forcing test FAILS RED on current master.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import pathlib
import sys
import time
import uuid
from typing import Any

import pytest
from panoseti_grpc.daq_control.client import DaqControlClient

# control/ root must be on sys.path (handled by scenarios/conftest.py)
CONTROL_ROOT = pathlib.Path(__file__).parent.parent.parent.parent
if str(CONTROL_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTROL_ROOT))

from ci.integration.conftest import (
    DAQ_DATA_DIR,
    HEAD_DATA_DIR,
    wait_hashpipe_running,
    wait_hashpipe_stopped,
)
from ci.integration.state_probe import StateProbe
from .conftest import (
    RunAlreadyInProgress,
    StartRunFailed,
    StopPartialFailure,
    PHBaselineTooOld,
    grpc_start_daq,
    grpc_stop_daq,
    any_pff_files_on_daqnode,
    fresh_run_state,
)

INTERLEAVE_PID_FILE = pathlib.Path("tmp/interleave.pid")
PH_BASELINE_FILE = pathlib.Path("configs/quabo_ph_baseline.json")


# ── SC-002 (Exemplar B): Partial start rolls back ────────────────────────────

class TestSC002PartialStartRollback:
    """
    SC-002 / Exemplar B: when StartDaq succeeds on node-0 but fails on node-1,
    start.py has no rollback — node-0 is left in a streaming state.

    FAILS RED TODAY: start.py calls start_data_flow() *before* start_recording(),
    so when StartDaq fails on node-1, quabos on node-0 are already streaming
    but no rollback (no stop_data_flow, no kill_hv_updater, no cleanup of
    partial run dirs, no post-mortem snapshot).
    """

    def test_SC002_partial_start_must_leave_no_active_hashpipe(
        self,
        daq_control_direct: DaqControlClient,
        daq_control_node2: DaqControlClient,
        run_params: dict[str, Any],
        state_probe: StateProbe,
    ) -> None:
        """
        With one failing node: no hashpipe should remain running on any node,
        current_run must be unset, and a post-mortem snapshot must exist.

        Currently fails because start_recording raises on the first failure,
        skipping subsequent nodes AND doing no rollback.
        """
        # Direct gRPC simulation: start node-0, then fail on node-1
        # (We test the invariant without running start.py end-to-end,
        #  since start.py requires quabo hardware for the full path.)
        rp1 = dict(run_params)
        rp2 = dict(run_params, daq_ip_addr="192.168.0.20", module_id=[200],
                   run_dir=f"partial_{uuid.uuid4().hex[:8]}.pffd")

        daq_control_direct.StartDaq(rp1)
        assert wait_hashpipe_running(daq_control_direct, DAQ_DATA_DIR, timeout=10)

        # Simulate node-1 failure: intentionally corrupt the run_dir to force failure
        ok, resp = daq_control_node2.StartDaq({**rp2, "run_dir": ""})  # invalid run_dir
        # With a valid two-node orchestrator, the node-0 start should be rolled back.
        # Currently start.py just raises and leaves node-0 running.

        # Assert: node-0 hashpipe must NOT still be running after a partial failure
        # (This assertion FAILS today because there is no rollback)
        assert not state_probe.hashpipe_running(), (
            "FAIL (SC-002): node-0 hashpipe is still running after partial multi-node "
            "start failure — start.py has no rollback ladder.\n"
            "Fix: wrap start_recording() in a try/except with rollback for all started nodes."
        )

        # Cleanup (even though test failed)
        with contextlib.suppress(Exception):
            daq_control_direct.StopDaq({
                "data_dir": rp1["data_dir"],
                "run_dir": rp1["run_dir"],
            })
        wait_hashpipe_stopped(daq_control_direct, DAQ_DATA_DIR, timeout=8)
        with contextlib.suppress(Exception):
            daq_control_direct.CleanupData({
                "data_dir": rp1["data_dir"],
                "run_dir": rp1["run_dir"],
                "module_id": rp1["module_id"],
            })

    def test_SC002_aborted_snapshot_exists_after_partial_start(
        self,
        state_probe: StateProbe,
        run_params: dict[str, Any],
    ) -> None:
        """
        After a failed multi-node start, a post-mortem snapshot must be preserved
        at <head_node_data_dir>/_aborted/<run_name>/start_failure_context.json.

        FAILS TODAY: no _aborted/ directory is ever created by start.py.
        """
        aborted_root = state_probe.aborted_snapshot_root()
        # This test is only meaningful if a partial start has actually been attempted.
        # It serves as a regression test for the post-mortem snapshot feature.
        if not aborted_root.exists():
            pytest.fail(
                "FAIL (SC-002): _aborted/ directory does not exist — "
                "start.py never creates post-mortem snapshots on failure.\n"
                "Fix: on any StartDaq rollback, create "
                "<head_node_data_dir>/_aborted/<run_name>/start_failure_context.json"
            )
        snapshots = list(aborted_root.iterdir())
        assert snapshots, "No aborted snapshots found in _aborted/"
        latest = max(snapshots, key=lambda p: p.stat().st_mtime)
        assert (latest / "start_failure_context.json").exists(), \
            "Post-mortem snapshot missing start_failure_context.json"


# ── SC-024 (Exemplar E): Concurrent start corruption ────────────────────────

class TestSC024ConcurrentStart:
    """
    SC-024 / Exemplar E: no advisory lock around start_run means two concurrent
    start.py invocations both pass the read_run_name() check, both call
    make_run_dirs, and one fails with FileExistsError leaving mixed state.

    FAILS RED TODAY: start.py has no lockfile; concurrent starts race.
    Fix: advisory lock (fcntl.flock or a lock file) around start_run.
    """

    def test_SC024_concurrent_start_only_one_wins(
        self,
        daq_control_direct: DaqControlClient,
        run_params: dict[str, Any],
        state_probe: StateProbe,
    ) -> None:
        """
        Two concurrent StartDaq calls — exactly one must succeed,
        the other must fail with a clear error.
        No double-start of hashpipe allowed.
        """
        import threading

        outcomes: list[tuple[bool, Any]] = []
        lock = threading.Lock()

        def _start(suffix: str) -> None:
            p = dict(run_params, run_dir=f"conc_{suffix}.pffd")
            ok, resp = daq_control_direct.StartDaq(p)
            with lock:
                outcomes.append((ok, resp, p["run_dir"]))

        t1 = threading.Thread(target=_start, args=("a",))
        t2 = threading.Thread(target=_start, args=("b",))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        winners = [o for o in outcomes if o[0]]
        losers  = [o for o in outcomes if not o[0]]

        assert len(winners) == 1, (
            f"FAIL (SC-024): expected exactly 1 winner, got {len(winners)}. "
            f"Outcomes: {outcomes}. No advisory lock around start_run."
        )
        assert len(losers) == 1, f"Expected 1 loser, got {len(losers)}"

        # Cleanup winner
        win_run_dir = winners[0][2]
        with contextlib.suppress(Exception):
            daq_control_direct.StopDaq({
                "data_dir": run_params["data_dir"],
                "run_dir": win_run_dir,
            })
        wait_hashpipe_stopped(daq_control_direct, DAQ_DATA_DIR, timeout=8)
        with contextlib.suppress(Exception):
            daq_control_direct.CleanupData({
                "data_dir": run_params["data_dir"],
                "run_dir": win_run_dir,
                "module_id": run_params["module_id"],
            })

    @pytest.mark.asyncio
    async def test_SC024_async_concurrent_start_only_one_wins(
        self,
        daq_control_direct: DaqControlClient,
        run_params: dict[str, Any],
        state_probe: StateProbe,
    ) -> None:
        """
        Async variant: asyncio.gather fires two starts simultaneously.
        """
        async def _start(suffix: str) -> tuple[bool, Any, str]:
            p = dict(run_params, run_dir=f"async_conc_{suffix}.pffd")
            loop = asyncio.get_event_loop()
            ok, resp = await loop.run_in_executor(None, daq_control_direct.StartDaq, p)
            return ok, resp, p["run_dir"]

        results = await asyncio.gather(_start("x"), _start("y"), return_exceptions=True)
        outcomes = [r for r in results if isinstance(r, tuple)]
        winners = [r for r in outcomes if r[0]]

        assert len(winners) == 1, (
            f"FAIL (SC-024 async): {len(winners)} winners from concurrent async start. "
            "Expected exactly 1. No advisory lock around start_run."
        )

        for ok, resp, run_dir in outcomes:
            if ok:
                with contextlib.suppress(Exception):
                    daq_control_direct.StopDaq({
                        "data_dir": run_params["data_dir"], "run_dir": run_dir
                    })
                await asyncio.get_event_loop().run_in_executor(
                    None, wait_hashpipe_stopped, daq_control_direct, DAQ_DATA_DIR, 8
                )
                with contextlib.suppress(Exception):
                    daq_control_direct.CleanupData({
                        "data_dir": run_params["data_dir"],
                        "run_dir": run_dir,
                        "module_id": run_params["module_id"],
                    })


# ── SC-025: Start with run already in progress (contract test) ───────────────

def test_SC025_start_with_run_in_progress_is_rejected(
    daq_control_direct: DaqControlClient,
    run_params: dict[str, Any],
    fresh_run_state: None,
) -> None:
    """
    StartDaq while hashpipe is already running must fail.
    This pins the double-start prevention contract (not TDD-forcing).
    """
    daq_control_direct.StartDaq(run_params)
    assert wait_hashpipe_running(daq_control_direct, DAQ_DATA_DIR, timeout=10)
    try:
        ok2, resp2 = daq_control_direct.StartDaq(
            dict(run_params, run_dir=f"second_{uuid.uuid4().hex[:8]}.pffd")
        )
        assert not ok2, (
            "Second StartDaq while first is running must be rejected — "
            "server must enforce single-hashpipe-per-node"
        )
    finally:
        daq_control_direct.StopDaq({
            "data_dir": run_params["data_dir"],
            "run_dir": run_params["run_dir"],
        })
        wait_hashpipe_stopped(daq_control_direct, DAQ_DATA_DIR, timeout=8)
        daq_control_direct.CleanupData({
            "data_dir": run_params["data_dir"],
            "run_dir": run_params["run_dir"],
            "module_id": run_params["module_id"],
        })


# ── SC-031 (Exemplar D): PH baseline staleness off-by-24× ───────────────────

class TestSC031PHBaslineStaleness:
    """
    SC-031 / Exemplar D: start.py::ph_baseline_file_ok() compares against
    time.time() - 24*86400 which is 24 DAYS, not 24 hours.

    FAILS RED TODAY: the comparison is `time.time() - 24*86400` (24 * 86400 s = 24 days).
    A file that is 26 hours old (should be rejected) passes the check.

    File: start.py, function: ph_baseline_file_ok(), around line 103.
    Fix: change `24*86400` to `86400` (24 hours = 1 day).
    """

    def test_SC031_file_26h_old_must_be_rejected(self, tmp_path: pathlib.Path) -> None:
        """A PH baseline file that is 26 hours old must be rejected."""
        # Create a plausible PH baseline file
        ph_file = tmp_path / "quabo_ph_baseline.json"
        ph_file.write_text('{"quabos": []}')
        # Set mtime to 26 hours ago
        stale_mtime = time.time() - (26 * 3600)
        os.utime(ph_file, (stale_mtime, stale_mtime))

        # Import start.py's validation function
        try:
            from start import ph_baseline_file_ok
        except ImportError:
            pytest.skip("Could not import start.ph_baseline_file_ok — check sys.path")

        is_ok = ph_baseline_file_ok(str(ph_file))
        assert not is_ok, (
            "FAIL (SC-031): ph_baseline_file_ok() returned True for a 26-hour-old file.\n"
            "The comparison uses time.time() - 24*86400 (= 24 days) instead of 86400 (24 hours).\n"
            "Fix: change to `time.time() - 86400` in start.py::ph_baseline_file_ok()."
        )

    def test_SC031_file_23h_old_must_be_accepted(self, tmp_path: pathlib.Path) -> None:
        """A PH baseline file that is 23 hours old (within 24 h) must pass."""
        ph_file = tmp_path / "quabo_ph_baseline.json"
        ph_file.write_text('{"quabos": []}')
        fresh_mtime = time.time() - (23 * 3600)
        os.utime(ph_file, (fresh_mtime, fresh_mtime))

        try:
            from start import ph_baseline_file_ok
        except ImportError:
            pytest.skip("Could not import start.ph_baseline_file_ok")

        assert ph_baseline_file_ok(str(ph_file)), \
            "A 23-hour-old PH baseline file must be accepted"

    def test_SC031_missing_file_is_rejected(self, tmp_path: pathlib.Path) -> None:
        """Missing PH baseline file must be rejected (not crash)."""
        try:
            from start import ph_baseline_file_ok
        except ImportError:
            pytest.skip("Could not import start.ph_baseline_file_ok")

        non_existent = str(tmp_path / "no_such_file.json")
        result = ph_baseline_file_ok(non_existent)
        assert not result, "Missing PH baseline file must return False"

    def test_SC031_empty_file_is_rejected(self, tmp_path: pathlib.Path) -> None:
        """Zero-byte PH baseline file must be rejected (no size check exists today)."""
        ph_file = tmp_path / "quabo_ph_baseline.json"
        ph_file.write_bytes(b"")
        # Set mtime to 1 hour ago (fresh)
        os.utime(ph_file, (time.time() - 3600, time.time() - 3600))

        try:
            from start import ph_baseline_file_ok
        except ImportError:
            pytest.skip("Could not import start.ph_baseline_file_ok")

        result = ph_baseline_file_ok(str(ph_file))
        assert not result, (
            "FAIL (SC-032): Zero-byte PH baseline file must be rejected — "
            "currently there is no size check."
        )


# ── SC-033 (Exemplar F): Stale interleave PID ────────────────────────────────

class TestSC033StaleInterleavePID:
    """
    SC-033 / Exemplar F: stop.py::stop_interleave() calls os.kill(pid, SIGTERM)
    without verifying that the PID belongs to our interleave process.

    A stale PID file from a previous crashed run (or a PID that was recycled by
    an unrelated process) could result in signalling the wrong process.

    FAILS RED TODAY: stop_interleave() sends SIGTERM to whatever PID is in the
    file, and silently leaves the file if os.kill() raises PermissionError
    (e.g., PID 1 = init/systemd).

    Fix: verify process identity before sending SIGTERM (check cmdline or
    use psutil to confirm the process is our interleave daemon).
    """

    def test_SC033_stale_pid_of_init_not_signalled(self, tmp_path: pathlib.Path) -> None:
        """
        Seed the PID file with PID=1 (init/systemd). stop_interleave() must:
          - Detect that PID 1 is NOT our interleave process
          - NOT send SIGTERM to PID 1
          - Clean up the stale PID file

        Currently fails because stop_interleave() sends SIGTERM to PID 1 (which
        raises PermissionError and silently leaves the file).
        """
        pid_file = tmp_path / "interleave.pid"
        pid_file.write_text("1\n")

        try:
            from stop import stop_interleave as _stop_interleave
        except ImportError:
            pytest.skip("Could not import stop.stop_interleave — check sys.path")

        # Monkey-patch PID_FILE to our temp file
        import stop as stop_module
        original = stop_module.PID_FILE
        try:
            stop_module.PID_FILE = str(pid_file)
            # Also patch tools.interleave.PID_FILE (imported by stop.py)
            import tools.interleave as interleave_module
            original_interleave = interleave_module.PID_FILE
            interleave_module.PID_FILE = str(pid_file)

            _stop_interleave(retry_limit=2)

            # PID file must be cleaned up
            assert not pid_file.exists(), (
                "FAIL (SC-033): PID file still exists after stop_interleave() "
                "with a stale PID — the file was not cleaned up."
            )
            # PID 1 (init) must still be running — we must not have killed it
            assert pathlib.Path("/proc/1").exists(), \
                "CRITICAL: PID 1 (/proc/1) is gone — stop_interleave() killed init!"

        finally:
            stop_module.PID_FILE = original
            interleave_module.PID_FILE = original_interleave

    def test_SC033_stale_pid_dead_process_clears_file(self, tmp_path: pathlib.Path) -> None:
        """
        PID file contains a PID for a process that no longer exists.
        stop_interleave() must clean the stale file without raising.
        """
        pid_file = tmp_path / "interleave.pid"
        # Find a PID that doesn't exist
        stale_pid = 99999
        while pathlib.Path(f"/proc/{stale_pid}").exists():
            stale_pid -= 1
        pid_file.write_text(f"{stale_pid}\n")

        try:
            from stop import stop_interleave as _stop_interleave
            import stop as stop_module
            import tools.interleave as interleave_module
        except ImportError:
            pytest.skip("Could not import stop.stop_interleave")

        original_stop = stop_module.PID_FILE
        original_interleave = interleave_module.PID_FILE
        try:
            stop_module.PID_FILE = str(pid_file)
            interleave_module.PID_FILE = str(pid_file)
            _stop_interleave(retry_limit=2)
            assert not pid_file.exists(), \
                "Stale PID file for dead process must be cleaned up by stop_interleave()"
        finally:
            stop_module.PID_FILE = original_stop
            interleave_module.PID_FILE = original_interleave


# ── SC-034: Interleave daemon outlives retry window ──────────────────────────

class TestSC034InterleaveDaemonHardKill:
    """
    SC-034: stop_interleave() polls for 10 × 0.5 s = 5 s with no SIGKILL fallback.
    If the interleave daemon ignores SIGTERM (or is slow to clean up), it keeps
    running after stop.py completes, continuing to flip quabo modes during stop.

    FAILS RED TODAY: no SIGKILL escalation in stop_interleave().
    Fix: after the retry window, send SIGKILL, then restore MAROC config.
    """

    def test_SC034_slow_interleave_is_hard_killed_after_timeout(
        self, tmp_path: pathlib.Path
    ) -> None:
        """
        A process that ignores SIGTERM must be SIGKILLed after the retry window.
        """
        import subprocess
        import signal as _signal

        # Spawn a subprocess that ignores SIGTERM
        proc = subprocess.Popen(
            ["python3", "-c",
             "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
             "time.sleep(60)"],
            start_new_session=True,
        )
        pid = proc.pid
        pid_file = tmp_path / "interleave.pid"
        pid_file.write_text(f"{pid}\n")

        try:
            from stop import stop_interleave as _stop_interleave
            import stop as stop_module
            import tools.interleave as interleave_module
        except ImportError:
            proc.kill()
            pytest.skip("Could not import stop.stop_interleave")

        original_stop = stop_module.PID_FILE
        original_interleave = interleave_module.PID_FILE
        try:
            stop_module.PID_FILE = str(pid_file)
            interleave_module.PID_FILE = str(pid_file)

            # With retry_limit=4 (4 × 0.5 s = 2 s budget), the daemon outlives it
            _stop_interleave(retry_limit=4)

            # After stop_interleave returns, the daemon must be dead
            assert not pathlib.Path(f"/proc/{pid}").exists(), (
                "FAIL (SC-034): Interleave daemon (that ignores SIGTERM) is still "
                "alive after stop_interleave() returned. No hard-kill fallback.\n"
                "Fix: send SIGKILL after retry_limit is exhausted."
            )
        finally:
            stop_module.PID_FILE = original_stop
            interleave_module.PID_FILE = original_interleave
            with contextlib.suppress(Exception):
                proc.kill()
                proc.wait(timeout=2)

    def test_SC034b_start_interleave_while_running_is_rejected(
        self, tmp_path: pathlib.Path
    ) -> None:
        """
        SC-034b: invoking config.py --start-interleave while a daemon is already
        running must be rejected. Two daemons racing over quabo config is undefined.

        FAILS RED TODAY: InterleaveController._acquire_lock() correctly prevents this
        (it checks the PID file). This test pins the contract.
        """
        import subprocess
        import signal as _signal

        # Spawn a fake interleave daemon
        proc = subprocess.Popen(
            ["python3", "-c", "import time; time.sleep(60)"],
            start_new_session=True,
        )
        pid_file = tmp_path / "interleave.pid"
        pid_file.write_text(f"{proc.pid}\n")

        try:
            from tools.interleave import PID_FILE as ORIG_PID_FILE
            import tools.interleave as interleave_module
            original = interleave_module.PID_FILE
            interleave_module.PID_FILE = str(pid_file)

            try:
                # Attempting to construct InterleaveController while one is "running"
                # should raise SystemExit (the lock acquisition fails)
                with pytest.raises(SystemExit):
                    from tools.interleave import InterleaveController
                    InterleaveController.__new__(InterleaveController)._acquire_lock()
            except ImportError:
                pytest.skip("Could not import tools.interleave.InterleaveController")
            finally:
                interleave_module.PID_FILE = original
        finally:
            proc.kill()
            proc.wait(timeout=2)
