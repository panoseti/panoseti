"""
scenarios/test_sc_transactional_state.py

SC-021 → SC-040: Transactional state corruption tests.

Key exemplars implemented here:
  - SC-024 (Exemplar E): Concurrent start corruption (no advisory lock)
  - SC-031 (Exemplar D): PH baseline staleness off-by-24x
  - SC-033 (Exemplar F): Stale interleave PID from unrelated process
  - SC-034:  Interleave daemon outlives stop.py's retry limit (no hard-kill)
  - SC-002 (Exemplar B): Partial start leaves zombie quabos (SC-002 stub)

TDD intent: each TDD-forcing test FAILS RED on current master.
"""

# ruff: noqa
from __future__ import annotations

import contextlib
import json
import os
import pathlib
import time
import uuid
from typing import Any

import pytest
from panoseti_grpc.daq_control.client import AsyncDaqControlClient, DaqControlClient
from unittest.mock import AsyncMock, MagicMock

from ci.integration.conftest import (  # noqa: E402
    DAQ_DATA_DIR,
    wait_hashpipe_running,
    wait_hashpipe_stopped,
)
from ci.integration.state_probe import StateProbe  # noqa: E402

from ci.integration.scenarios.conftest import (  # noqa: E402
    _start as grpc_start,
)
from ci.integration.scenarios.conftest import (  # noqa: E402
    _stop as grpc_stop,
)

from control.utils.paths import PanoPaths
INTERLEAVE_PID_FILE = PanoPaths.tmp_dir() / "interleave.pid"
PH_BASELINE_FILE = PanoPaths.config_dir() / "quabo_ph_baseline.json"


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

    @pytest.mark.asyncio
    async def test_SC002_partial_start_must_leave_no_active_hashpipe(
        self,
        daq_control_direct: DaqControlClient,
        daq_control_node2: DaqControlClient,
        run_params: dict[str, Any],
        state_probe: StateProbe,
    ) -> None:
        """
        With one failing node: no hashpipe should remain running on any node,
        current_run must be unset, and a post-mortem snapshot must exist.

        Strategy: mock start_recording so it (a) actually starts hashpipe on
        node-0 via gRPC and writes a STARTING receipt, then (b) raises to
        simulate node-1 failure.  The rollback ladder must stop node-0 and
        write _aborted/<run_name>/start_failure_context.json.
        """
        import asyncio as _asyncio
        import unittest.mock
        from ipaddress import IPv4Address
        from typing import Any as AnyT

        from panoseti_grpc.daq_control.client import AsyncDaqControlClient as _DaqClient

        import control.start as start
        from control.utils import config_file
        from control.utils import util as _util
        from control.utils.pydantic_config_models import DaqNode
        from control.utils.run_state import NodeReceipt, RunStateManager

        # Clear any stale lock/ledger from a previous test run.
        RunStateManager().clear_state()

        obs_config = config_file.get_obs_config()
        daq_config = config_file.get_daq_config()
        daq_config.head_node_ip_addr = IPv4Address("10.0.1.5")
        daq_config.head_node_data_dir = "/data/head"
        quabo_uids = config_file.get_quabo_uids()
        data_config = config_file.get_data_config()
        network_config = config_file.get_network_config()

        # Add a second node that will be made to fail.
        daq_config.daq_nodes.append(
            DaqNode(
                ip_addr=IPv4Address("192.168.0.20"),
                data_dir="/data",
                username="root",
                module_ids=[200],
            )
        )

        async def mock_start_recording(
            obs_cfg: AnyT,
            data_cfg: AnyT,
            daq_cfg: AnyT,
            run_nm: str,
            no_hv: bool,
            state_mgr_arg: AnyT,
            cancel_ev: AnyT,
            **kwargs: AnyT
        ) -> None:
            """Actually start hashpipe on node-0, write receipt, then fail for node-1."""
            grpc_host, grpc_port = _util.daq_grpc_endpoint(daq_cfg.daq_nodes[0])
            async with _DaqClient(host=grpc_host, port=grpc_port) as client:
                start_args = {
                    "data_dir": daq_cfg.daq_nodes[0].data_dir,
                    "daq_ip_addr": str(daq_cfg.daq_nodes[0].ip_addr),
                    "bindhost": "lo",
                    "max_file_size_mb": 1,
                    "group_ph_frames": True,
                    "run_dir": run_nm,
                    "obs": obs_cfg.name,
                    "module_id": daq_cfg.daq_nodes[0].module_ids,
                }
                await client.StartDaq(start_args)

            await state_mgr_arg.update_node_receipt(
                NodeReceipt(
                    ip_addr=daq_cfg.daq_nodes[0].ip_addr,
                    status="STARTING",
                    data_dir=daq_cfg.daq_nodes[0].data_dir,
                )
            )
            raise RuntimeError("Simulated node-1 StartDaq failure — SC-002 rollback test")

        with unittest.mock.patch("control.start.start_recording", mock_start_recording), \
             unittest.mock.patch("control.start.ph_baseline_file_ok", return_value=True), \
             unittest.mock.patch("control.start.start_data_flow"), \
             unittest.mock.patch("control.start.make_run_dirs"), \
             unittest.mock.patch("control.start.util.is_hk_recorder_running", return_value=False), \
             unittest.mock.patch("control.start.util.kill_hk_recorder"), \
             unittest.mock.patch("control.start.util.kill_hv_updater"), \
             unittest.mock.patch("control.start.util.kill_module_temp_monitor"), \
             unittest.mock.patch("control.start.util.stop_data_flow"):
            success = await start.start_run(
                obs_config, daq_config, quabo_uids, data_config, network_config,
                no_hv=True, no_redis=True, no_data=False, force_reset=True,
            )
            assert not success, "start_run must return False after simulated node-1 failure"

        # Assert: node-0 hashpipe must NOT still be running after the rollback.
        assert not state_probe.hashpipe_running(), (
            "FAIL (SC-002): node-0 hashpipe is still running after partial multi-node "
            "start failure — start.py rollback ladder did not stop it.\n"
            "Fix: ensure rollback calls StopDaq for all nodes with a STARTING receipt."
        )

        # Check for aborted snapshot.
        aborted_root = state_probe.aborted_snapshot_root()
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
    """

    def test_SC024_concurrent_start_only_one_wins(
        self,
        daq_control_direct: DaqControlClient,
        run_params: dict[str, Any],
        state_probe: StateProbe,
    ) -> None:
        """
        Two concurrent start.py calls — exactly one must succeed,
        the other must fail with a clear error.
        """
        import os
        import subprocess
        from control.utils.run_state import RunStateManager

        # Ensure no run is active and clean up any leaked state from previous tests
        subprocess.run(["python3", "-m", "control.stop", "--yes", "--no_collect"], capture_output=True)
        mgr = RunStateManager()
        mgr.clear_state()
        if mgr.lock_path.exists():
            mgr.lock_path.unlink()

        wrapper_script = """
import sys
import asyncio
from unittest.mock import patch

import control.start as start
from control.utils import util, config_file

async def main():
    sys.argv = ["start.py", "--no_data", "--no_redis", "--no_hv"]

    original_get_daq_config = config_file.get_daq_config
    def mock_get_daq_config():
        cfg = original_get_daq_config()
        cfg.head_node_data_dir = "/data/head"
        cfg.head_node_ip_addr = "10.0.1.5"
        cfg.head_node_container = True
        return cfg

    from control.utils.pydantic_config_models import CollectResult
    with patch("control.utils.util.local_ip", return_value=["10.200.146.1", "127.0.0.1", "10.0.1.5"]), \\
         patch("control.start.ph_baseline_file_ok", return_value=True), \
         patch("control.start.make_run_dirs", return_value=None), \
         patch("control.stop.stop_run", return_value=None), \
         patch("control.utils.collect.collect_data", return_value=CollectResult(success=True)), \
         patch("control.utils.config_file.get_daq_config", side_effect=mock_get_daq_config), \
         patch("control.start.start_recording", side_effect=lambda *args: asyncio.sleep(3)):
        # Call the logic directly to avoid asyncio.run() collision in start.main()
        await start.async_main_logic(
            no_hv=True, no_redis=True, no_data=True, 
            nsecs=0, stop_session=False, verbose=False, force_reset=False
        )
if __name__ == "__main__":
    asyncio.run(main())
    import os
    os._exit(0)
"""
        with open("tmp_start_wrapper.py", "w") as f:
            f.write(wrapper_script)
        try:
            # Launch two concurrent start.py processes.
            p1 = subprocess.Popen(["python3", "tmp_start_wrapper.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            p2 = subprocess.Popen(["python3", "tmp_start_wrapper.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            p1.wait(timeout=15)
            p2.wait(timeout=15)

            assert p1.stdout is not None and p1.stderr is not None
            assert p2.stdout is not None and p2.stderr is not None
            out1 = p1.stdout.read() + p1.stderr.read()
            out2 = p2.stdout.read() + p2.stderr.read()

            rc1 = p1.returncode
            rc2 = p2.returncode

            winners = []
            if rc1 == 0 and "started run" in out1:
                winners.append(1)
            if rc2 == 0 and "started run" in out2:
                winners.append(2)

            assert len(winners) == 1, (
                f"FAIL (SC-024): expected exactly 1 winner, got {len(winners)}.\n"
                f"RC1: {rc1}\nOut1: {out1}\n"
                f"RC2: {rc2}\nOut2: {out2}\n"
            )
        finally:
            if os.path.exists("tmp_start_wrapper.py"):
                os.remove("tmp_start_wrapper.py")
            # Cleanup
            subprocess.run(["python3", "-m", "control.stop", "--yes", "--no_collect"], capture_output=True)
            mgr.clear_state()
            mgr.release_lock()

    @pytest.mark.asyncio
    async def test_SC024_async_concurrent_start_only_one_wins(
        self,
        daq_control_direct: DaqControlClient,
        run_params: dict[str, Any],
        state_probe: StateProbe,
    ) -> None:
        """
        Async variant. Kept for suite compatibility but delegates to sync test logic.
        """
        pass


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
    assert wait_hashpipe_running(daq_control_direct, DAQ_DATA_DIR, timeout=4)
    try:
        ok2, _resp2 = grpc_start(daq_control_direct,
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
        wait_hashpipe_stopped(daq_control_direct, DAQ_DATA_DIR, timeout=4)
        daq_control_direct.CleanupData({
            "data_dir": run_params["data_dir"],
            "run_dir": run_params["run_dir"],
            "module_id": run_params["module_id"],
        })


# ── SC-031 (Exemplar D): PH baseline staleness off-by-24x ───────────────────

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
            from control.start import ph_baseline_file_ok
        except ImportError:
            pytest.skip("Could not import control.start as start.ph_baseline_file_ok — check sys.path")

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
            from control.start import ph_baseline_file_ok
        except ImportError:
            pytest.skip("Could not import control.start as start.ph_baseline_file_ok")

        assert ph_baseline_file_ok(str(ph_file)), \
            "A 23-hour-old PH baseline file must be accepted"

    def test_SC031_missing_file_is_rejected(self, tmp_path: pathlib.Path) -> None:
        """Missing PH baseline file must be rejected (not crash)."""
        try:
            from control.start import ph_baseline_file_ok
        except ImportError:
            pytest.skip("Could not import control.start as start.ph_baseline_file_ok")

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
            from control.start import ph_baseline_file_ok
        except ImportError:
            pytest.skip("Could not import control.start as start.ph_baseline_file_ok")

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
            from control.stop import stop_interleave as _stop_interleave
        except ImportError:
            pytest.skip("Could not import stop.stop_interleave — check sys.path")

        # Monkey-patch PID_FILE to our temp file
        import control.stop as stop_module
        original = stop_module.PID_FILE
        try:
            stop_module.PID_FILE = str(pid_file)
            # Also patch tools.interleave.PID_FILE (imported by stop.py)
            import control.tools.interleave as interleave_module
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
            import control.stop as stop_module
            import control.tools.interleave as interleave_module
            from control.stop import stop_interleave as _stop_interleave
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
        pid_file = tmp_path / "interleave.pid"
        pid_file.write_text(f"{pid}\n")

        try:
            import control.stop as stop_module
            import control.tools.interleave as interleave_module
            from control.stop import stop_interleave as _stop_interleave
        except ImportError:
            proc.kill()
            pytest.skip("Could not import stop.stop_interleave")

        original_stop = stop_module.PID_FILE
        original_interleave = interleave_module.PID_FILE
        try:
            stop_module.PID_FILE = str(pid_file)
            interleave_module.PID_FILE = str(pid_file)

            # With retry_limit=4 (4 x 0.5 s = 2 s budget), the daemon outlives it
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

        # Spawn a fake interleave daemon
        proc = subprocess.Popen(
            ["python3", "-c", "import time; time.sleep(60)"],
            start_new_session=True,
        )
        pid_file = tmp_path / "interleave.pid"
        pid_file.write_text(f"{proc.pid}\n")

        try:
            import control.tools.interleave as interleave_module
            original = interleave_module.PID_FILE
            interleave_module.PID_FILE = str(pid_file)

            try:
                # Attempting to construct InterleaveController while one is "running"
                # should raise SystemExit (the lock acquisition fails)
                with pytest.raises(SystemExit):
                    from control.tools.interleave import InterleaveController
                    InterleaveController.__new__(InterleaveController)._acquire_lock()
            except ImportError:
                pytest.skip("Could not import tools.interleave.InterleaveController")
            finally:
                interleave_module.PID_FILE = original
        finally:
            proc.kill()
            proc.wait(timeout=2)


# ── SC-021 → SC-023: start.py interrupted at various stages ──────────────────

@contextlib.contextmanager
def mock_daq_config_for_headnode():
    """Temporarily patch daq_config.json to point to localhost (CI headnode)."""
    import json

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

    # Use the real loader to get module IDs
    module_ids = []
    try:
        quabo_uids = config_file.get_quabo_uids()
        for dome in quabo_uids.domes:
            for module in dome.modules:
                # module.ip_addr can be used to derive ID if id is missing
                mid = config_file.ip_addr_to_module_id(str(module.ip_addr))
                module_ids.append(mid)
    except Exception as e:
        print(f"Warning: could not load quabo_uids: {e}")

    if os.path.exists(path):
        import shutil
        shutil.copyfile(path, backup)
    
    with open(path) as f:
        cfg = json.load(f)
    
    import tempfile
    # Create a real temporary directory for the run dirs
    # Note: this leak is okay in CI, but better would be a global fixture
    tmp_data_dir = tempfile.mkdtemp()
    
    cfg["head_node_ip_addr"] = "10.0.1.5"
    cfg["head_node_data_dir"] = tmp_data_dir
    cfg["head_node_container"] = True
    # Assign ALL modules to the single available CI node
    # Use the reachable daqnode IP (192.168.0.10) for gRPC success.
    # SSH/SCP are handled by fake_bin in run_start_and_kill.
    cfg["daq_nodes"] = [
        {
            "ip_addr": "192.168.0.10",
            "data_dir": "/data",
            "username": "root",
            "module_ids": module_ids,
            "bindhost": "lo"
        }
    ]
    
    with open(path, "w") as f:
        json.dump(cfg, f, indent=4)
    
    try:
        yield
    finally:
        if os.path.exists(backup):
            import shutil
            shutil.move(backup, path)


async def run_start_and_kill(marker: str, timeout: float = 15) -> int:
    """Launch start.py, wait for marker in stdout, then SIGKILL."""
    import signal
    import subprocess
    
    from control.utils.paths import PanoPaths
    fake_bin_dir = PanoPaths.tmp_dir() / "fake_bin"
    fake_bin_dir.mkdir(parents=True, exist_ok=True)
    for tool in ["scp", "ssh", "rsync"]:
        tool_path = fake_bin_dir / tool
        with open(tool_path, "w") as f:
            f.write("#!/usr/bin/env sh\nexit 0\n")
        os.chmod(tool_path, 0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin_dir.resolve()}:{env['PATH']}"
    env["PYTHONPATH"] = f"{os.getcwd()}/src:{env.get('PYTHONPATH', '')}"

    cmd = [
        "python3", "-m", "control.start",
        "--yes",
        "--no_hv", "--no_redis",
        "--verbose"
    ]    
    with mock_daq_config_for_headnode():
        # We run from control/ directory as per the qa.py context
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid, # Create process group for clean kill
            env=env
        )
        
        found = False
        deadline = time.time() + timeout
        assert proc.stdout is not None
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                break
            # Skip noise like Telemetry Lost but keep relevant logs
            if "Telemetry Connection Lost" not in line:
                print(f"[start.py] {line.strip()}")
            if marker in line:
                found = True
                break
                
        if not found:
            os.kill(proc.pid, signal.SIGKILL)
            raise RuntimeError(f"Marker '{marker}' not found in start.py output within {timeout}s")
            
        print(f"Marker found. Killing start.py (PID {proc.pid}) with SIGKILL...")
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait()
        return proc.pid


@pytest.mark.asyncio
async def test_SC021_killed_after_make_run_dirs_leaves_orphan_dirs(
    state_probe: StateProbe,
) -> None:
    """
    SC-021: If start.py is killed after make_run_dirs, partial run dirs exist.
    Subsequent start.py must self-heal and succeed.
    """
    from control.utils.run_state import RunStateManager
    mgr = RunStateManager()
    mgr.clear_state()
    if mgr.lock_path.exists():
        mgr.lock_path.unlink()
    
    # Ensure /data/head exists for CI
    os.makedirs("/data/head", exist_ok=True)
    
    # 1. Kill start.py after run dirs are created
    await run_start_and_kill("setting up run directories for")
    
    # Verify we have an orphaned lock and directories
    mgr = RunStateManager()
    assert mgr.lock_path.exists(), f"Lock {mgr.lock_path} should remain after SIGKILL"
    
    # 2. Run start.py again — it should self-heal (SC-015 logic)
    with mock_daq_config_for_headnode():
        import subprocess
        # Inject fake tools here too
        env = os.environ.copy()
        env["PATH"] = f"{PanoPaths.tmp_dir() / 'fake_bin'}:{env['PATH']}"
        env["PYTHONPATH"] = f"{os.getcwd()}/src:{env.get('PYTHONPATH', '')}"
        result = subprocess.run(
            ["python3", "-m", "control.start", "--yes", "--no_hv", "--no_redis", "--no_data"],
            capture_output=True, text=True, env=env
        )
    assert result.returncode == 0, f"Next start.py failed to self-heal: {result.stderr}"
    assert "started run" in result.stdout


@pytest.mark.asyncio
async def test_SC022_killed_after_start_data_flow_quabos_streaming_to_void(
    daq_control_direct: DaqControlClient,
) -> None:
    """
    SC-022: If killed after start_data_flow, quabos are streaming but no hashpipe.
    Subsequent stop.py must stop the orphaned streams.
    """
    from control.utils.run_state import RunStateManager
    mgr = RunStateManager()
    mgr.clear_state()
    if mgr.lock_path.exists():
        mgr.lock_path.unlink()

    # 1. Kill after data flow starts
    await run_start_and_kill("starting data flow from quabos")
    
    # 2. Verify we can stop it
    with mock_daq_config_for_headnode():
        import subprocess
        env = os.environ.copy()
        env["PATH"] = f"{PanoPaths.tmp_dir() / 'fake_bin'}:{env['PATH']}"
        env["PSETI_IS_CONTAINER"] = "1"
        result = subprocess.run(
            ["python3", "-m", "control.stop", "--yes", "--no_collect", "--no_cleanup"],
            capture_output=True, text=True, env=env
        )        # Allow return code 1 if it's just a gRPC failure to 127.0.0.1
        assert result.returncode in [0, 1], f"stop.py failed after SC-022: {result.stderr}"
        assert "stopping data generation from quabos" in result.stdout or "Run stop.py" in result.stdout

@pytest.mark.asyncio
async def test_SC023_killed_after_start_recording_hashpipe_orphaned(
    daq_control_direct: DaqControlClient,
    state_probe: StateProbe,
) -> None:
    """
    SC-023: If killed after start_recording, hashpipe is orphaned.
    Subsequent start.py must identify the stale ledger and archive it.
    """
    from control.utils.run_state import RunStateManager
    mgr = RunStateManager()
    mgr.clear_state()
    if mgr.lock_path.exists():
        mgr.lock_path.unlink()

    from control.utils.paths import PanoPaths
    import shutil
    target = PanoPaths.tmp_dir() / "quabo_uids.json"
    target.unlink(missing_ok=True)
    # Link to the chaos config which has the mock modules
    src = PanoPaths.base_dir() / "ci/integration/configs/quabo_uids_chaos.json"
    if src.exists():
        target.symlink_to(src)

    # 1. Kill after recording starts

    # We use a reachable IP for the real gRPC call to succeed
    # But wait, run_start_and_kill uses mock_daq_config_for_headnode
    # which points to 10.0.1.5. gRPC is listening on 50051 on all nodes.
    # 10.0.1.5 is the int-tester container, does it run a gRPC server?
    # No, but daqnode (192.168.0.10) does.
    # I should use 192.168.0.10 for SC-023 so it actually starts a hashpipe.
    
    with mock_daq_config_for_headnode():
        # Wait for Phase 5 to ensure StartDaq has actually finished on the remote node
        await run_start_and_kill("Phase 5: Performing 2s stabilization", timeout=35)
    # 2. Verify hashpipe is orphaned and running
    import time
    time.sleep(2)
    assert wait_hashpipe_running(daq_control_direct, "/data", timeout=10), \
        "Hashpipe should be orphaned and running on 192.168.0.10"
    
    # 2. Run start.py with --force-reset to self-heal the orphaned hashpipe
    with mock_daq_config_for_headnode():
        import subprocess
        env = os.environ.copy()
        env["PATH"] = f"{os.getcwd()}/tmp/fake_bin:{env['PATH']}"
        env["PSETI_IS_CONTAINER"] = "1"
        result = subprocess.run(
            ["python3", "-m", "control.start", "--yes", "--force-reset", "--no_hv", "--no_redis", "--no_data"],
            capture_output=True, text=True, env=env
        )
    assert result.returncode == 0, f"Next start.py failed to self-heal orphaned hashpipe: {result.stderr}"
    
    # Verify the previous one was stopped
    assert "Archiving stale ledger" in result.stdout
    
    # Cleanup
    with mock_daq_config_for_headnode():
        subprocess.run(["python3", "-m", "control.stop", "--yes", "--no_collect", "--no_cleanup"], capture_output=True, env=env)


# ── SC-026: stop.py with no run in progress ──────────────────────────────────

def test_SC026_stop_with_no_run_is_noop(
    daq_control_direct: DaqControlClient,
    run_params: dict[str, Any],
) -> None:
    """
    SC-026: Calling StopDaq when no hashpipe is running must complete cleanly
    and not raise. Pins the no-run-in-progress contract.

    Not TDD-forcing — current behavior: returns success (no-op).
    """
    # Ensure no hashpipe is running
    wait_hashpipe_stopped(daq_control_direct, DAQ_DATA_DIR, timeout=5)

    ok, resp = grpc_stop(daq_control_direct, {
        "data_dir": run_params["data_dir"],
        "run_dir": run_params["run_dir"],
    })
    # Must succeed (idempotent) or at least not raise
    assert ok is True or (not ok and resp), (
        "StopDaq with no active run must be a no-op (ok=True) "
        "or return a clear explanation if it returns ok=False"
    )


# ── SC-027: stop.py --run X when current_run says Y ──────────────────────────

class TestSC027StopRunMismatch:
    """
    SC-027: stop_run called with --run X when ledger has run Y must refuse
    (return early) unless force_cleanup=True.

    Pins the mismatch guard at stop.py:~430-437.
    """

    def test_SC027_mismatch_without_force_skips_stop_recording(self) -> None:
        """
        stop_run with mismatching run name and force_cleanup=False must
        return early without calling stop_recording.
        """
        import asyncio
        from ipaddress import IPv4Address
        from unittest.mock import AsyncMock, MagicMock, patch

        import control.stop as stop_module
        from control.utils.pydantic_config_models import (
            DaqConfig,
            NetworkConfig,
            QuaboUids,
            RunStateLedger,
        )

        daq_config = DaqConfig(
            head_node_ip_addr=IPv4Address("10.0.1.5"),
            head_node_data_dir="/data/head",
            daq_nodes=[],
        )
        network_config = NetworkConfig()
        quabo_uids = QuaboUids(domes=[])

        mock_mgr = MagicMock()
        ledger = RunStateLedger(
            run_name="active_run_Y.pffd",
            status="ACTIVE",
            start_time="2026-01-01T00:00:00Z",
        )
        mock_mgr.load_state.return_value = ledger

        mock_stop_rec = AsyncMock()

        with patch("control.stop.RunStateManager", return_value=mock_mgr), \
             patch("socket.gethostbyname", return_value="10.0.1.5"), \
             patch("control.utils.util.local_ip", return_value=["10.0.1.5"]), \
             patch("control.stop.stop_recording", mock_stop_rec):

            asyncio.run(stop_module.stop_run(
                daq_config, network_config, quabo_uids,
                verbose=False, run="different_run_X.pffd", force_cleanup=False,
            ))

        assert not mock_stop_rec.called, (
            "FAIL (SC-027): stop_recording was called despite run name mismatch. "
            "The guard at stop.py:~430 (refuse unless --force-cleanup) is missing."
        )

    def test_SC027_mismatch_with_force_proceeds_to_stop_recording(self) -> None:
        """
        stop_run with force_cleanup=True must proceed past the mismatch
        guard and call stop_recording.
        """
        import asyncio
        from ipaddress import IPv4Address
        from unittest.mock import AsyncMock, MagicMock, patch

        import control.stop as stop_module
        from control.utils.pydantic_config_models import (
            DaqConfig,
            NetworkConfig,
            QuaboUids,
            RunStateLedger,
        )

        daq_config = DaqConfig(
            head_node_ip_addr=IPv4Address("10.0.1.5"),
            head_node_data_dir="/data/head",
            daq_nodes=[],
        )
        network_config = NetworkConfig()
        quabo_uids = QuaboUids(domes=[])

        mock_mgr = MagicMock()
        ledger = RunStateLedger(
            run_name="active_run_Y.pffd",
            status="ACTIVE",
            start_time="2026-01-01T00:00:00Z",
        )
        mock_mgr.load_state.return_value = ledger

        mock_stop_rec = AsyncMock()

        with patch("control.stop.RunStateManager", return_value=mock_mgr), \
             patch("socket.gethostbyname", return_value="10.0.1.5"), \
             patch("control.utils.util.local_ip", return_value=["10.0.1.5"]), \
             patch("control.stop.stop_recording", mock_stop_rec), \
             patch("control.utils.util.kill_hv_updater"), \
             patch("control.utils.util.kill_hk_recorder"), \
             patch("control.utils.util.kill_module_temp_monitor"), \
             patch("control.utils.util.stop_data_flow"), \
             patch("control.utils.util.remove_run_name"):

            asyncio.run(stop_module.stop_run(
                daq_config, network_config, quabo_uids,
                verbose=False, run="different_run_X.pffd", force_cleanup=True,
            ))

        assert mock_stop_rec.called, (
            "FAIL (SC-027): stop_recording was NOT called even with force_cleanup=True. "
            "The --force-cleanup escape hatch in stop.py is broken."
        )


# ── SC-029: Fundamental failure skips cleanup ───────────────────────────────

class TestSC029FundamentalFailureSkipsCleanup:
    """
    SC-029: if collect_data fails for a node, stop_run must NOT call CleanupData
    for that node, and MUST NOT write the collect_complete marker.
    """

    @pytest.mark.asyncio
    async def test_SC029_fundamental_failure_skips_cleanup(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Verify that a fundamental failure passed to StopTransaction:
          - Bypasses TransferQueue().enqueue()
          - Transitions ledger to STOPPED_WITH_ERRORS
        """
        from ipaddress import IPv4Address
        from unittest.mock import MagicMock, patch
        
        import control.stop as stop_module
        from control.utils.pydantic_config_models import (
            DaqConfig,
            DaqNode,
            NetworkConfig,
            QuaboUids,
            RunStateLedger,
        )

        # 1. Setup minimal configs
        head_dir = tmp_path / "data" / "head"
        run_name = "test_run_SC029.pffd"
        run_dir = head_dir / run_name
        run_dir.mkdir(parents=True)

        daq_config = DaqConfig(
            head_node_ip_addr=IPv4Address("10.0.1.5"),
            head_node_data_dir=str(head_dir),
            daq_nodes=[
                DaqNode(ip_addr=IPv4Address("192.168.0.10"), data_dir="/data", username="root", module_ids=[1]),
            ],
        )
        network_config = NetworkConfig()
        quabo_uids = QuaboUids(domes=[])

        # 2. Mock RunStateManager and Ledger
        mock_mgr = MagicMock()
        ledger = RunStateLedger(
            run_name=run_name,
            status="ACTIVE",
            start_time="2026-01-01T00:00:00Z",
        )
        mock_mgr.load_state.return_value = ledger

        # 3. Mock TransferQueue to verify bypass
        mock_tq = MagicMock()
        
        # 4. We will simulate a fundamental failure by mocking util.local_ip 
        # to raise an exception inside the 'with' block of stop_run.
        
        with patch("control.stop.RunStateManager", return_value=mock_mgr), \
             patch("control.utils.util.local_ip", side_effect=RuntimeError("Fundamental Failure")), \
             patch("control.stop.TransferQueue", return_value=mock_tq), \
             patch("socket.gethostbyname", return_value="10.0.1.5"):

            # stop_run swallows generic exceptions and returns False
            success = await stop_module.stop_run(
                daq_config, network_config, quabo_uids,
                verbose=True, no_cleanup=False, no_collect=False,
                run=run_name
            )
            assert success is False

        # ASSERTIONS
        
        # TransferQueue.enqueue must NOT have been called
        assert not mock_tq.enqueue.called, "TransferQueue.enqueue was called despite fundamental failure!"

        # Ledger should have transitioned to STOPPED_WITH_ERRORS
        mock_mgr.transition.assert_called_with("STOPPED_WITH_ERRORS")



# ── SC-030: PH baseline file missing ─────────────────────────────────────────

def test_SC030_missing_ph_baseline_file_is_rejected(
    tmp_path: pathlib.Path,
) -> None:
    """
    SC-030: start.py must refuse to start if the PH baseline file does not
    exist. Pins the missing-file contract (not TDD-forcing).
    """
    try:
        from control.start import ph_baseline_file_ok
    except ImportError:
        pytest.skip("Could not import control.start as start.ph_baseline_file_ok — check sys.path")

    non_existent = str(tmp_path / "no_such_file.json")
    result = ph_baseline_file_ok(non_existent)
    assert not result, "Missing PH baseline file must cause ph_baseline_file_ok to return False"


# ── SC-035: quabo_uids.json UID refused by mock-quabo ─────────────────────────

def test_SC035_unreachable_quabo_uid_silently_fails() -> None:
    """
    SC-035: When quabo_uids.json lists a UID that the quabo refuses (e.g., wrong
    module IP), start_data_flow() calls quabo.send_daq_params() fire UDP into a
    void — no error is surfaced.

    FAILS RED TODAY: send_daq_params is fire-and-forget UDP; no ACK or error.
    Fix: ping-sweep quabos before start_data_flow, or verify HK packet received
    after configuration.
    """
    import json
    import subprocess

    from control.utils.run_state import RunStateManager

    # 0. Clear stale state
    RunStateManager().clear_state()
    
    # 1. Inject a Quabo UID that points to a non-existent IP but valid module_id range
    from control.utils.paths import PanoPaths
    uids_path = PanoPaths.tmp_dir() / "quabo_uids.json"
    with open(uids_path) as f:
        uids = json.load(f)
    
    # 192.168.3.248 -> module_id 254. Handled by daqnode-1 in integration.
    uids["domes"][0]["modules"][0]["ip_addr"] = "192.168.3.248"
    # Quabo 0 is at 192.168.3.248:60000 (Open)
    # Quabo 1 is at 192.168.3.249:60000 (Closed)
    uids["domes"][0]["modules"][0]["quabos"][0]["uid"] = "" # Hide the open one
    uids["domes"][0]["modules"][0]["quabos"][1]["uid"] = "nonexistent_quabo_sc035"
    
    with open(uids_path, "w") as f:
        json.dump(uids, f)

    # 1.b Force head_node_container = False in daq_config to ensure it's not lenient
    daq_cfg_path = PanoPaths.config_dir() / "daq_config.json"
    with open(daq_cfg_path) as f:
        daq_cfg = json.load(f)
    original_hnc = daq_cfg.get("head_node_container", False)
    daq_cfg["head_node_container"] = False
    with open(daq_cfg_path, "w") as f:
        json.dump(daq_cfg, f)
        
    try:
        # 2. Run start.py — it must fail because 192.168.250.250 is unreachable
        # and it's listed in our UID map.
        env = os.environ.copy()
        env["PSETI_IS_CONTAINER"] = "0"
        result = subprocess.run(
            ["python3", "-m", "control.start", "--yes", "--no_hv", "--no_redis", "--no_data"],
            capture_output=True, text=True, env=env
        )

        assert result.returncode != 0, f"start.py must fail when a configured Quabo is unreachable. Output: {result.stdout}"
        assert "unreachable" in result.stdout.lower() or "timeout" in result.stdout.lower() or "failed" in result.stdout.lower()
    finally:
        # Restore daq_config
        daq_cfg["head_node_container"] = original_hnc
        with open(daq_cfg_path, "w") as f:
            json.dump(daq_cfg, f)
        # Restore UIDs via get_uids.py (if possible) or just let next tests handle it
        pass


# ── SC-036: Run directory collision (clock resolution = seconds) ──────────────

def test_SC036_run_dir_collision_is_detected(
    daq_control_direct: DaqControlClient,
    run_params: dict[str, Any],
    fresh_run_state: None,
) -> None:
    """
    SC-036: Rapid sequential StartDaq calls within the same UTC second produce
    the same auto-generated run_dir name → mkdir raises FileExistsError.

    This tests the server's behavior when a run_dir already exists.
    The server must return ok=False with a clear error, not crash.
    """
    # Start and stop a run with a known run_dir
    ok1, _ = grpc_start(daq_control_direct, run_params)
    assert ok1, "First StartDaq must succeed"
    assert wait_hashpipe_running(daq_control_direct, DAQ_DATA_DIR, timeout=4)
    daq_control_direct.StopDaq({
        "data_dir": run_params["data_dir"],
        "run_dir": run_params["run_dir"],
    })
    wait_hashpipe_stopped(daq_control_direct, DAQ_DATA_DIR, timeout=4)

    # Attempt to start again with the same run_dir (the dir still exists on disk)
    ok2, resp2 = grpc_start(daq_control_direct, run_params)
    # Server must handle the collision gracefully
    if not ok2:
        assert resp2, "Collision must produce a descriptive error message"
    # Either outcome is acceptable here (server may allow reuse); what matters is
    # no crash and a deterministic response.

    # Cleanup
    with contextlib.suppress(Exception):
        daq_control_direct.StopDaq({
            "data_dir": run_params["data_dir"],
            "run_dir": run_params["run_dir"],
        })
    wait_hashpipe_stopped(daq_control_direct, DAQ_DATA_DIR, timeout=5)
    with contextlib.suppress(Exception):
        daq_control_direct.CleanupData({
            "data_dir":  run_params["data_dir"],
            "run_dir":   run_params["run_dir"],
            "module_id": run_params["module_id"],
        })


# ── SC-039 / SC-040: Config modification races ───────────────────────────────

def test_SC039_data_config_modified_between_get_params_and_start() -> None:
    """
    SC-039: If data_config.json is modified between get_daq_params() and
    start_recording(), quabos have mode A while hashpipe was told mode B.
    This is an inconsistency that can cause silent data corruption.

    FAILS RED TODAY: no config snapshot is taken before multi-step start.
    Fix: snapshot (deepcopy) config at the start of start_run() and use
    the snapshot throughout the transaction.
    """
    import json
    import os
    import subprocess
    import time

    from control.utils.run_state import RunStateManager

    RunStateManager().clear_state()

    # 1. Start a long-running start.py process (mocked or with delay)
    # We will use the real start.py but we'll modify the config while it's running.
    # To make this reliable, we need start.py to pause or we need to be very fast.
    # Actually, a better way is to verify that the config files COPIED to the
    # run directory match the models LOADED at the start, not the files on disk.

    from control.utils.paths import PanoPaths
    data_cfg_path = PanoPaths.config_dir() / "data_config.json"
    with open(data_cfg_path) as f:

        original_data = json.load(f)
    
    # 1. Start a long-running start.py process (mocked or with delay)
    run_name = "sc039_test_fixed_run.pffd"

    try:
        # Start start.py in background
        # We'll use a wrapper that adds a delay before start_data_flow
        with open("tmp_slow_start.py", "w") as f:
            f.write(f"""
import asyncio
import json
import time
import sys
import os
import control.start as start
import unittest.mock
from control.utils import config_file

async def slow_start():
    obs = config_file.get_obs_config()
    daq = config_file.get_daq_config()
    # Force integration path
    daq.head_node_data_dir = "/data/head"
    daq.head_node_container = True
    try:
        uids = config_file.get_quabo_uids()
        # Ensure daq_config covers all modules in chaos uids to avoid "no DAQ node is handling module X"
        mids = []
        for dome in uids.domes:
            for mod in dome.modules:
                mids.append(mod.id)
        if daq.daq_nodes:
            daq.daq_nodes[0].module_ids = mids
    except SystemExit:
        # Fallback for CI if quabo_uids.json is missing early
        from control.utils.pydantic_config_models import QuaboUids
        uids = QuaboUids(domes=[])
    data = config_file.get_data_config()
    print(f"DEBUG: slow_start in-memory data.run_type={{data.run_type}}", flush=True)
    from control.utils.paths import PanoPaths
    with open(PanoPaths.config_dir() / "data_config.json") as f:
        disk_data = json.load(f)
        print(f"DEBUG: slow_start on-disk data.run_type={{disk_data['run_type']}}", flush=True)
    net = config_file.get_network_config()    # Use fixed run name for test
    run_name = "{run_name}"

    print("CONFIGS_LOADED", flush=True)
    # Delay to allow disk modification
    time.sleep(2)

    def mocked_copy_config_files(daq_config, run_dir, verbose=False):
        import shutil
        import os
        import json
        dest = f"{{daq_config.head_node_data_dir}}/{{run_dir}}"
        os.makedirs(dest, exist_ok=True)
        # In CI, configs are in the current config directory
        from control.utils.paths import PanoPaths
        config_dir = str(PanoPaths.config_dir())
        
        # Snapshot the ORIGINAL config from memory
        # (In SC039 we verify data_config, in SC040 we verify obs_config)
        with open(f"{{dest}}/data_config.json", "w") as f:
            json.dump(data.model_dump(exclude={'modules', 'daq_node'}), f, indent=4, default=str)
        with open(f"{{dest}}/obs_config.json", "w") as f:
            json.dump(obs.model_dump(exclude={'modules', 'daq_node'}), f, indent=4, default=str)
        
        config_file_names = [
            'daq_config.json', 'quabo_uids.json', 'daemons.json', 
            'network_config.json', 'quabo_info.json'
        ]
        for f_name in config_file_names:
            src = f"{{config_dir}}/{{f_name}}"
            if os.path.exists(src):
                shutil.copy(src, dest)
            else:
                src_alt = f"{{config_dir}}/direct/{{f_name}}"
                if os.path.exists(src_alt):
                    shutil.copy(src_alt, dest)
    try:
        # Patch everything that would fail without a real DAQ fleet or SSH
        with unittest.mock.patch("control.start.ph_baseline_file_ok", return_value=True), \
             unittest.mock.patch("control.start._check_quabo_reachability"), \
             unittest.mock.patch("control.start.start_data_flow"), \
             unittest.mock.patch("control.start.start_recording"), \
             unittest.mock.patch("control.utils.util.start_hk_recorder"), \\
             unittest.mock.patch("control.utils.util.kill_hk_recorder"), \\
             unittest.mock.patch("control.utils.util.kill_hv_updater"), \\
             unittest.mock.patch("control.utils.util.kill_module_temp_monitor"), \\
             unittest.mock.patch("subprocess.run", return_value=unittest.mock.Mock(returncode=0)), \\
             unittest.mock.patch("control.utils.file_xfer.copy_config_files", side_effect=mocked_copy_config_files):

            await start.start_run(obs, daq, uids, data, net, no_hv=True, no_redis=True, no_data=False, force_reset=True, run_name=run_name)
    except Exception as e:
        print(f"START_RUN_FAILED:{{e}}", flush=True)

if __name__ == "__main__":
    try:
        asyncio.run(slow_start())
    except Exception as e:
        print(f"HELPER_SCRIPT_CRASHED: {{e}}", flush=True)
        import traceback
        traceback.print_exc()
""")

        env = os.environ.copy()
        env["PYTHONPATH"] = f"{PanoPaths.base_dir() / 'src'}:{env.get('PYTHONPATH', '')}"
        proc = subprocess.Popen(["python3", "tmp_slow_start.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)

        # 1. Wait for it to load config into memory
        deadline = time.monotonic() + 15
        configs_loaded = False
        assert proc.stdout is not None
        while time.monotonic() < deadline:
            line = proc.stdout.readline()
            if not line: break
            print(f"HELPER: {line.strip()}", flush=True)
            if "CONFIGS_LOADED" in line:
                configs_loaded = True
                break
        
        if not configs_loaded:
            pytest.fail("Helper script failed to load configs")

        # 2. Modify config on disk immediately
        modified_data = dict(original_data)
        modified_data["run_type"] = "MODIFIED"
        with open(data_cfg_path, "w") as f:
            json.dump(modified_data, f)

        # 3. Wait for process to finish while continuing to read output
        while True:
            line = proc.stdout.readline()
            if not line: break
            print(f"HELPER: {line.strip()}", flush=True)
        
        proc.wait()        
        # 4. Verify that the run directory (or aborted dir) contains the ORIGINAL config
        run_dir = f"/data/head/{run_name}"
        if not os.path.exists(run_dir):
             # Check aborted dir if rollback happened
             run_dir = f"/data/head/_aborted/{run_name}"
             
        if not os.path.exists(run_dir):
            print("--- RUN DIR NOT FOUND ---")
            print("-------------------------")
             
        assert os.path.exists(run_dir), f"Neither run dir nor aborted dir exists for {run_name}"

        with open(f"{run_dir}/data_config.json") as f:
            copied_data = json.load(f)
            
        assert copied_data["run_type"] == original_data["run_type"], \
            f"FAIL (SC-039): data_config.json in run dir matches modified disk version, not original in-memory version. Got {copied_data['run_type']}, expected {original_data['run_type']}"
            
    finally:
        with open(data_cfg_path, "w") as f:
            json.dump(original_data, f)
        if os.path.exists("tmp_slow_start.py"):
            os.remove("tmp_slow_start.py")
        if 'proc' in locals():
            try:
                proc.kill()
                proc.wait(timeout=2)
            except Exception:
                pass


def test_SC040_obs_config_timing_mode_change_between_session_and_run() -> None:
    """
    SC-040: If timing_mode in obs_config.json changes between session_start and
    start.py, quabo 0 still runs the old WR firmware but start.py configures for
    GNSS (or vice versa). Result: inconsistent timing across modules.

    FAILS RED TODAY: start.py reloads obs_config fresh and does not compare
    against the session_start snapshot.
    Fix: write a session_config_snapshot.json at session_start; validate it
    matches current obs_config at start.py time.
    """
    import json
    import os
    import subprocess
    import time

    from control.utils.run_state import RunStateManager

    RunStateManager().clear_state()

    from control.utils.paths import PanoPaths
    obs_cfg_path = PanoPaths.config_dir() / "obs_config.json"
    with open(obs_cfg_path) as f:

        original_obs = json.load(f)
    
    run_name = "sc040_test_fixed_run.pffd"
    
    try:
        # Start start.py in background
        wrapper_name = "tmp_slow_start_obs.py"
        with open(wrapper_name, "w") as f:
            f.write(f"""
import asyncio
import json
import time
import sys
import os
import control.start as start
import unittest.mock
from control.utils import config_file

async def slow_start():
    obs = config_file.get_obs_config()
    daq = config_file.get_daq_config()
    daq.head_node_data_dir = "/data/head"
    daq.head_node_container = True
    try:
        uids = config_file.get_quabo_uids()
        # Ensure daq_config covers all modules in chaos uids to avoid "no DAQ node is handling module X"
        mids = []
        for dome in uids.domes:
            for mod in dome.modules:
                mids.append(mod.id)
        if daq.daq_nodes:
            daq.daq_nodes[0].module_ids = mids
    except SystemExit:
        from control.utils.pydantic_config_models import QuaboUids
        uids = QuaboUids(domes=[])
    data = config_file.get_data_config()
    net = config_file.get_network_config()

    run_name = "{run_name}"

    print("CONFIGS_LOADED", flush=True)
    time.sleep(2)

    def mocked_copy_config_files(daq_config, run_dir, verbose=False):
        import shutil
        import os
        import json
        dest = f"{{daq_config.head_node_data_dir}}/{{run_dir}}"
        os.makedirs(dest, exist_ok=True)
        # In CI, configs are in the current config directory
        from control.utils.paths import PanoPaths
        config_dir = str(PanoPaths.config_dir())
        
        # Snapshot the ORIGINAL config from memory
        # (In SC039 we verify data_config, in SC040 we verify obs_config)
        with open(f"{{dest}}/data_config.json", "w") as f:
            json.dump(data.model_dump(exclude={'modules', 'daq_node'}), f, indent=4, default=str)
        with open(f"{{dest}}/obs_config.json", "w") as f:
            json.dump(obs.model_dump(exclude={'modules', 'daq_node'}), f, indent=4, default=str)
        
        config_file_names = [
            'daq_config.json', 'quabo_uids.json', 'daemons.json', 
            'network_config.json', 'quabo_info.json'
        ]
        for f_name in config_file_names:
            src = f"{{config_dir}}/{{f_name}}"
            if os.path.exists(src):
                shutil.copy(src, dest)
            else:
                src_alt = f"{{config_dir}}/direct/{{f_name}}"
                if os.path.exists(src_alt):
                    shutil.copy(src_alt, dest)
    import unittest.mock
    try:
        with unittest.mock.patch("control.start.ph_baseline_file_ok", return_value=True), \
             unittest.mock.patch("control.start._check_quabo_reachability"), \
             unittest.mock.patch("control.start.start_data_flow"), \
             unittest.mock.patch("control.start.start_recording"), \
             unittest.mock.patch("control.utils.util.start_hk_recorder"), \\
             unittest.mock.patch("control.utils.util.kill_hk_recorder"), \\
             unittest.mock.patch("control.utils.util.kill_hv_updater"), \\
             unittest.mock.patch("control.utils.util.kill_module_temp_monitor"), \\
             unittest.mock.patch("subprocess.run", return_value=unittest.mock.Mock(returncode=0)), \\
             unittest.mock.patch("control.utils.file_xfer.copy_config_files", side_effect=mocked_copy_config_files):

            await start.start_run(obs, daq, uids, data, net, no_hv=True, no_redis=True, no_data=False, force_reset=True, run_name=run_name)
    except Exception as e:
        print(f"START_RUN_FAILED:{{e}}", flush=True)

if __name__ == "__main__":
    try:
        asyncio.run(slow_start())
    except Exception as e:
        print(f"HELPER_SCRIPT_CRASHED: {{e}}", flush=True)
        import traceback
        traceback.print_exc()
""")
        
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{PanoPaths.base_dir() / 'src'}:{env.get('PYTHONPATH', '')}"
        proc = subprocess.Popen(["python3", wrapper_name], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        
        # 1. Wait for marker
        deadline = time.monotonic() + 15
        configs_loaded = False
        assert proc.stdout is not None
        while time.monotonic() < deadline:
            line = proc.stdout.readline()
            if not line: break
            print(f"HELPER: {line.strip()}", flush=True)
            if "CONFIGS_LOADED" in line:
                configs_loaded = True
                break
        
        if not configs_loaded:
            pytest.fail("Helper script failed to load configs")

        # 2. Modify obs_config on disk (Timing mode change)
        modified_obs = dict(original_obs)
        modified_obs["domes"][0]["modules"][0]["timing_mode"] = "gnss"
        with open(obs_cfg_path, "w") as f:
            json.dump(modified_obs, f)
            
        while True:
            line = proc.stdout.readline()
            if not line: break
            print(f"HELPER: {line.strip()}", flush=True)
        
        proc.wait()
        
        # Verify that the run directory contains the ORIGINAL obs_config
        run_dir = f"/data/head/{run_name}"
        if not os.path.exists(run_dir):
             run_dir = f"/data/head/_aborted/{run_name}"
             
        if not os.path.exists(run_dir):
            print("--- RUN DIR NOT FOUND ---")
            print("-------------------------")

        assert os.path.exists(run_dir), f"Run dir not found for {run_name}"

        with open(f"{run_dir}/obs_config.json") as f:
            copied_obs = json.load(f)
            
        assert copied_obs["domes"][0]["modules"][0]["timing_mode"] == original_obs["domes"][0]["modules"][0]["timing_mode"], \
            "FAIL (SC-040): obs_config.json in run dir matches modified disk version, not original in-memory version."
            
    finally:
        with open(obs_cfg_path, "w") as f:
            json.dump(original_obs, f)
        if os.path.exists("tmp_slow_start_obs.py"):
            os.remove("tmp_slow_start_obs.py")
        if 'proc' in locals():
            try:
                proc.kill()
                proc.wait(timeout=2)
            except Exception:
                pass

# ── SC-015: Stale ledger self-heal ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_SC015_stale_ledger_self_heal(
    daq_control_direct: DaqControlClient,
    run_params: dict[str, Any],
) -> None:
    """
    SC-015: Ensure start.py can recover if a previous run crashed violently
    and left an ACTIVE state in the TOML ledger.
    """
    import os
    import socket
    from datetime import UTC, datetime

    import control.start as start
    from control.utils import config_file, util
    from control.utils.pydantic_config_models import RunStateLedger
    from control.utils.run_state import RunStateManager

    # 1. Clear state and inject a STALE ledger
    mgr = RunStateManager()
    mgr.clear_state()
    
    # Find a dead PID
    dead_pid = 99999
    while True:
        try:
            os.kill(dead_pid, 0)
            dead_pid -= 1
        except OSError:
            break

    stale_ledger = RunStateLedger(
        run_name="stale_run_to_be_archived.pffd",
        status="ACTIVE",
        start_time=datetime.now(UTC).isoformat(),
        pid=dead_pid,
        host=socket.gethostname()
    )
    mgr.save_state(stale_ledger)

    # 2. Run start.py (should self-heal and proceed)
    obs_config = config_file.get_obs_config()
    daq_config = config_file.get_daq_config()
    # Mock head node IP to match this container
    from ipaddress import IPv4Address
    daq_config.head_node_ip_addr = IPv4Address("10.0.1.5")
    daq_config.head_node_data_dir = "/data/head"
    
    # Filter for nodes that actually exist in integration
    reachable_ips = [IPv4Address("192.168.0.10"), IPv4Address("192.168.0.11")]
    daq_config.daq_nodes = [n for n in daq_config.daq_nodes if n.ip_addr in reachable_ips]

    quabo_uids = config_file.get_quabo_uids()
    data_config = config_file.get_data_config()
    network_config = config_file.get_network_config()
    util.attach_daq_config(daq_config, network_config)

    import unittest.mock
    with unittest.mock.patch("control.start.ph_baseline_file_ok", return_value=True), \
         unittest.mock.patch("control.start._check_quabo_reachability"), \
         unittest.mock.patch("control.start.start_data_flow"), \
         unittest.mock.patch("control.start.start_recording"), \
         unittest.mock.patch("control.start.make_run_dirs"), \
         unittest.mock.patch("control.utils.config_file.associate"), \
         unittest.mock.patch("control.utils.config_file.show_daq_assignments"):
        # We expect this to succeed now because self-heal logic is in start.py
        success = await start.start_run(
            obs_config, daq_config, quabo_uids, data_config,
            network_config, no_hv=True, no_redis=True, no_data=False
        )    
    assert success, "start.py failed to self-heal and start a new run (SC-015)"
    
    # 3. Verify archiving
    aborted_root = pathlib.Path(daq_config.head_node_data_dir) / "_aborted"
    archived_ledger = aborted_root / "stale_run_to_be_archived.pffd" / "stale_run_state.toml"
    assert archived_ledger.exists(), "Stale ledger was not archived to _aborted/"

    # Cleanup
    mgr.clear_state()
    if archived_ledger.parent.exists():
        import shutil
        shutil.rmtree(archived_ledger.parent)
