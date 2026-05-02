"""
scenarios/test_sc_transactional_state_2.py

SC-033, SC-034: Transactional state corruption tests.
Part 2 of partitioned test suite.
"""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
import unittest.mock
from typing import Any

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

    def test_SC033_stale_pid_of_init_not_signalled(self, tmp_path: pathlib.Path) -> None:
        """
        Seed the PID file with PID=1 (init/systemd). stop_interleave() must:
          - Detect that PID 1 is NOT our interleave process
          - NOT send SIGTERM to PID 1
          - Clean up the stale PID file

        Currently fails because stop_interleave() sends SIGTERM to PID 1 (which
        raises PermissionError and silently leaves the file).
        """
        pid_file = tmp_path / "interleave.lock"
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
            ["python3", "-c", "import time; time.sleep(60)"],
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

@contextlib.contextmanager
def mock_daq_config_for_headnode():
    """Temporarily patch daq_config.json to point to localhost (CI headnode)."""

    from control.utils import config_file
    from control.utils.paths import PanoPaths
    path = PanoPaths.config_dir() / "daq_config.json"
    backup = str(path) + ".bak"
    # Ensure tmp/ and configs/ exist (should already, but let's be safe)
    PanoPaths.ensure_dirs()
    PanoPaths.config_dir().mkdir(parents=True, exist_ok=True)
    
    # Create a dummy PH baseline if missing
    ph_baseline = PanoPaths.tmp_dir() / "quabo_ph_baseline.json"
    if not os.path.exists(ph_baseline):
        with open(ph_baseline, "w") as f:
            json.dump({"quabos": []}, f)

    if os.path.exists(path):
        import shutil
        shutil.copyfile(path, backup)
    
    with open(path) as f:
        cfg = json.load(f)
    
    import tempfile
    # Prefer the isolated HEAD_DATA_DIR set by auto_isolate so the subprocess
    # env and the daq_config.json written here share the same path.  Fall back
    # to a fresh tempdir only when running outside the pytest harness.
    tmp_data_dir = os.environ.get("HEAD_DATA_DIR") or tempfile.mkdtemp()
    os.makedirs(tmp_data_dir, exist_ok=True)

    tester_ip = f'{os.environ.get("HEAD_NET_PREFIX", "10.0.1")}.5'
    cfg["head_node_ip_addr"] = tester_ip
    cfg["head_node_data_dir"] = tmp_data_dir
    cfg["head_node_container"] = True
    
    # Coherence Fix: Ensure the DAQ node is handling ALL modules defined in the 
    # current obs_config.json to prevent "no DAQ node is handling module X" errors.
    mids = []
    obs = config_file.get_obs_config()
    for dome in obs.domes:
        for module in dome.modules:
            mids.append(config_file.ip_addr_to_module_id(str(module.ip_addr)))

    # Assign ALL modules to the single available CI node
    # Use a DAQ IP that is on the same /24 subnet as the modules (192.168.3.x)
    # to pass strict Tier-2 Subnet Coherence validation.
    daqnode_ip = "192.168.3.30"
    cfg["daq_nodes"] = [
        {
            "ip_addr": daqnode_ip,
            "data_dir": "/data",
            "username": "root",
            "module_ids": mids,
            "bindhost": "lo"
        }
    ]
    
    with open(path, "w") as f:
        json.dump(cfg, f, indent=4)
    
    # Write matching quabo_uids.json to tmp/ so associate() in subprocess passes
    uids_path = PanoPaths.tmp_dir() / "quabo_uids.json"
    uids_path.parent.mkdir(parents=True, exist_ok=True)
    from control.utils.pydantic_config_models import QuaboUids
    uids_dict: dict[str, Any] = {"domes": [{"num": 0, "modules": []}]}
    for mid in mids:
        uids_dict["domes"][0]["modules"].append({
            "id": mid,
            "ip_addr": f"192.168.3.{mid}",
            "quabos": [{"uid": f"q{mid}_{j}"} if j==0 else {"uid": ""} for j in range(4)]
        })
    with open(uids_path, "w") as f:
        json.dump(uids_dict, f, indent=4)

    with unittest.mock.patch("control.utils.config_file.get_quabo_uids", return_value=QuaboUids(**uids_dict)):
        try:
            yield
        finally:
            if os.path.exists(backup):
                import shutil
                shutil.move(backup, path)
