"""
scenarios/test_sc_transactional_state_2.py

SC-033, SC-034: Transactional state corruption tests.
Part 2 of partitioned test suite.
"""

from __future__ import annotations

import contextlib
import pathlib

import pytest

from control.utils.paths import PanoPaths

PH_BASELINE_FILE = PanoPaths.config_dir() / "quabo_ph_baseline.json"


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

    def test_SC033_stale_pid_of_init_not_signalled(self, mock_workspace, tmp_path: pathlib.Path) -> None:
        """
        Seed the PID file with PID=1 (init/systemd). stop_interleave() must:
          - Detect that PID 1 is NOT our interleave process
          - NOT send SIGTERM to PID 1
          - Clean up the stale PID file
        """
        # mock_workspace isolates PSETI_TMP
        pid_file = PanoPaths.tmp_dir() / "interleave.lock"
        pid_file.write_text("1\n")

        try:
            from control.stop import stop_interleave as _stop_interleave
        except ImportError:
            pytest.skip("Could not import stop.stop_interleave")

        # Monkey-patch PID_FILE to our temp file
        import control.stop as stop_module
        import control.tools.interleave as interleave_module
        original = stop_module.INTERLEAVE_LOCK_PATH
        try:
            stop_module.INTERLEAVE_LOCK_PATH = str(pid_file)
            # Also patch tools.interleave.PID_FILE (imported by stop.py)
            original_interleave = interleave_module.INTERLEAVE_LOCK_PATH
            interleave_module.INTERLEAVE_LOCK_PATH = str(pid_file)

            _stop_interleave(retry_limit=2)

            # PID file must be cleaned up
            assert not pid_file.exists(), (
                "FAIL (SC-033): PID file still exists after stop_interleave() "
                "with a stale PID — the file was not cleaned up."
            )
            # PID 1 (init) must still be running — we must not have killed it
            # /proc/1 check only works on Linux
            if pathlib.Path("/proc").exists():
                assert pathlib.Path("/proc/1").exists(), \
                    "CRITICAL: PID 1 (/proc/1) is gone — stop_interleave() killed init!"

        finally:
            stop_module.INTERLEAVE_LOCK_PATH = original
            interleave_module.INTERLEAVE_LOCK_PATH = original_interleave

    def test_SC033_stale_pid_dead_process_clears_file(self, tmp_path: pathlib.Path) -> None:
        """
        PID file contains a PID for a process that no longer exists.
        stop_interleave() must clean the stale file without raising.
        """
        pid_file = tmp_path / "interleave.lock"
        # Find a PID that doesn't exist
        stale_pid = 99999
        while pathlib.Path(f"/proc/{stale_pid}").exists():
            stale_pid -= 1
        pid_file.write_text(f"{stale_pid}\n")

        try:
            import control.stop as stop_module
            import control.tools.interleave as interleave_module
            from control.stop import stop_interleave as _stop_interleave
        except ImportError:
            pytest.skip("Could not import stop.stop_interleave")

        original_stop = stop_module.INTERLEAVE_LOCK_PATH
        original_interleave = interleave_module.INTERLEAVE_LOCK_PATH
        try:
            stop_module.INTERLEAVE_LOCK_PATH = str(pid_file)
            interleave_module.INTERLEAVE_LOCK_PATH = str(pid_file)
            _stop_interleave(retry_limit=2)
            assert not pid_file.exists(), \
                "Stale PID file for dead process must be cleaned up by stop_interleave()"
        finally:
            stop_module.INTERLEAVE_LOCK_PATH = original_stop
            interleave_module.INTERLEAVE_LOCK_PATH = original_interleave


# ── SC-034: Interleave daemon outlives retry window ──────────────────────────

class TestSC034InterleaveDaemonHardKill:
    """
    SC-034: stop_interleave() polls for 10 x 0.5 s = 5 s with no SIGKILL fallback.
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

        # Spawn a subprocess that ignores SIGTERM
        proc = subprocess.Popen(
            ["python3", "-c",
             "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
             "time.sleep(60)"],
            start_new_session=True,
        )
        pid = proc.pid
        pid_file = tmp_path / "interleave.lock"
        pid_file.write_text(f"{pid}\n")

        try:
            import control.stop as stop_module
            import control.tools.interleave as interleave_module
            from control.stop import stop_interleave as _stop_interleave
        except ImportError:
            proc.kill()
            pytest.skip("Could not import stop.stop_interleave")

        original_stop = stop_module.INTERLEAVE_LOCK_PATH
        original_interleave = interleave_module.INTERLEAVE_LOCK_PATH
        try:
            stop_module.INTERLEAVE_LOCK_PATH = str(pid_file)
            interleave_module.INTERLEAVE_LOCK_PATH = str(pid_file)

            # With retry_limit=4 (4 x 0.5 s = 2 s budget), the daemon outlives it
            _stop_interleave(retry_limit=4)

            # After stop_interleave returns, the daemon must be dead
            assert not pathlib.Path(f"/proc/{pid}").exists(), (
                "FAIL (SC-034): Interleave daemon (that ignores SIGTERM) is still "
                "alive after stop_interleave() returned. No hard-kill fallback.\n"
                "Fix: send SIGKILL after retry_limit is exhausted."
            )
        finally:
            stop_module.INTERLEAVE_LOCK_PATH = original_stop
            interleave_module.INTERLEAVE_LOCK_PATH = original_interleave
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

        # Spawn a fake interleave daemon
        proc = subprocess.Popen(
            ["python3", "interleave.py", "-c", "import time; time.sleep(60)"],
            start_new_session=True,
        )
        pid_file = tmp_path / "interleave.lock"
        pid_file.write_text(f"{proc.pid}\n")

        try:
            import control.tools.interleave as interleave_module
            original = interleave_module.INTERLEAVE_LOCK_PATH
            interleave_module.INTERLEAVE_LOCK_PATH = str(pid_file)

            try:
                # Attempting to construct InterleaveController while one is "running"
                # should raise SystemExit (the lock acquisition fails)
                with pytest.raises(SystemExit):
                    from control.tools.interleave import InterleaveController
                    InterleaveController.__new__(InterleaveController)._acquire_lock()
            except ImportError:
                pytest.skip("Could not import tools.interleave.InterleaveController")
            finally:
                interleave_module.INTERLEAVE_LOCK_PATH = original
        finally:
            proc.kill()
            proc.wait(timeout=2)


# ── SC-021 → SC-023: start.py interrupted at various stages ──────────────────

# [REMOVED redundant mock_daq_config_for_headnode; use chaos_headnode_workspace fixture]
